"""
New MPC Local Path Planner for drone navigation (new_mpc.py).

Dynamics: 2D double-integrator + yaw
  State:   x = [px, py, vx, vy, yaw]   (5)
  Control: u = [ax, ay, yaw_rate]       (3)

Solver: CasADi + IPOPT (interior-point NLP solver)

Author: Lorenzo Ortolani
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass
from typing import Optional, List
import threading
import time


try:
    from mujoco_sim.gaussian_grid_map import GaussianGridMap
except ImportError:
    from gaussian_grid_map import GaussianGridMap


# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

@dataclass
class NewMPCConfig:
    """All tunable parameters for the new MPC planner."""

    # ── Horizon ──
    N: int = 50                      # prediction steps, before: 10
    dt: float = 0.05                  # discretisation time [s]  → before: 0.2

    # ── Dynamics limits ──
    v_max: float = 5.0               # max velocity  [m/s], before: 2.5
    a_max: float = 0.1               # max acceleration [m/s²], before: 0.5
    yaw_rate_max: float = 1.0        # max yaw rate [rad/s]

    # ── Waypoint sequencing ──
    waypoint_threshold: float = 0.3  # advance to next A* waypoint when within this distance [m]

    # ── Cost weights ──
    Q_pos: float = 400.0              # position tracking weight
    Q_vel: float = 1.0               # velocity tracking weight
    Q_yaw: float = 0.5               # heading tracking weight
    Q_terminal: float = 10.0          # terminal weight multiplier, before: 3.0
    R_acc: float = 0.8               # acceleration effort penalty
    R_yaw_rate: float = 0.05         # yaw-rate effort penalty
    R_jerk: float = 10.0              # jerk (Δu) smoothness penalty

    # ── Obstacle avoidance ──
    W_obs_grid: float = 20.0         # Gaussian grid cost weight (soft)
    W_wall_slack: float = 500.0      # half-space slack penalty weight (hard-ish)
    d_safe: float = 0.6              # safety clearance from obstacle surface [m]

    # ── Terminal path constraint ──
    terminal_path_radius: float = 10.0   # before: 0.5 -> terminal state must lie within this distance of the A* path [m]
    W_terminal_path_slack: float = 1000.0  # penalty weight on terminal path slack (soft constraint)

    # ── LiDAR preprocessing ──
    lidar_max_range: float = 5.0     # discard returns beyond this range [m]  (sensor limit)
    obs_hs_radius: float = 3.5       # search radius for half-space linearisation per step [m]

    # ── Gaussian grid map ──
    grid_reso: float = 0.25          # grid resolution [m]
    grid_std: float = 0.5            # Gaussian spread [m]
    grid_extend: float = 2.0         # map extension around scan points [m]

    # ── Solver ──
    max_iter: int = 150
    warm_start: bool = True
    print_level: int = 0             # IPOPT verbosity (0 = silent)


# ═══════════════════════════════════════════════
# Waypoint sequencer
# ═══════════════════════════════════════════════

class WaypointSequencer:
    """
    Tracks the current active A* waypoint and advances to the next one
    once the drone is within ``threshold`` metres.

    Usage::
        seq = WaypointSequencer(threshold=0.3)
        seq.set_path([[0,0,1.5], [1,0,1.5], [2,1,1.5]])
        seq.advance_if_close(pos_xy)
        goal = seq.current_target()   # [x, y]
    """

    def __init__(self, threshold: float = 0.3):
        self._waypoints: List[np.ndarray] = []  # each element: [x, y]
        self._idx: int = 0
        self.threshold = threshold

    def set_path(self, waypoints) -> None:
        """
        Set the A* waypoint list.

        Args:
            waypoints: list/array of [x, y], [x, y, z], or [x, y, z, yaw]
        """
        self._waypoints = [np.array(wp[:2], dtype=float) for wp in waypoints]
        self._idx = 0

    def has_path(self) -> bool:
        return len(self._waypoints) > 0

    def advance_if_close(self, pos_xy: np.ndarray) -> None:
        """
        Move to the next waypoint while the drone is within ``threshold``
        of the current one.  Stops at the last waypoint.
        """
        while self._idx < len(self._waypoints) - 1:
            dist = float(np.linalg.norm(pos_xy - self._waypoints[self._idx]))
            if dist < self.threshold:
                self._idx += 1
            else:
                break

    def current_target(self) -> Optional[np.ndarray]:
        """Return the current active waypoint [x, y], or None if no path set."""
        if not self._waypoints:
            return None
        return self._waypoints[min(self._idx, len(self._waypoints) - 1)].copy()

    def current_index(self) -> int:
        return self._idx

    def total_waypoints(self) -> int:
        return len(self._waypoints)

    def is_complete(self) -> bool:
        """True when the last waypoint has been reached."""
        return self._idx >= len(self._waypoints) - 1 and len(self._waypoints) > 0

    def get_progress(self) -> float:
        """Fraction of waypoints visited, in [0, 1]."""
        if not self._waypoints:
            return 1.0
        return self._idx / len(self._waypoints)


# ═══════════════════════════════════════════════
# Grid cost interpolant helper
# ═══════════════════════════════════════════════

def _build_grid_interp(grid_map: GaussianGridMap) -> Optional[ca.Function]:
    """
    Build a CasADi B-spline interpolant from the Gaussian grid map.

    Returns ca.Function cost = f(px, py) ∈ [0, 1], or None if the grid
    has not been initialised yet.
    """
    if grid_map.gmap is None:
        return None

    xw, yw = grid_map.gmap.shape
    x_grid = np.linspace(
        grid_map.minx, grid_map.minx + (xw - 1) * grid_map.xyreso, xw
    )
    y_grid = np.linspace(
        grid_map.miny, grid_map.miny + (yw - 1) * grid_map.xyreso, yw
    )
    # CasADi interpolant expects column-major (Fortran) flat data:
    #   data[i + j*nx]  =  gmap[i, j]
    data_flat = grid_map.gmap.ravel(order="F").tolist()
    return ca.interpolant(
        "obs_cost", "bspline",
        [x_grid.tolist(), y_grid.tolist()],
        data_flat,
    )


# ═══════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════

@dataclass
class NewMPCResult:
    """Solver output from one NewMPCPlanner.solve() call."""
    success: bool
    u_opt: np.ndarray               # optimal control sequence  (N, 3)
    x_pred: np.ndarray              # predicted state trajectory (N+1, 5)
    cost: float
    solve_time_ms: float
    first_control: np.ndarray       # u_opt[0]  →  [ax, ay, yaw_rate]


# ═══════════════════════════════════════════════
# MPC Planner
# ═══════════════════════════════════════════════

class NewMPCPlanner:
    """
    CasADi / IPOPT MPC planner that tracks the full A* waypoint path.

    Obstacle safety is provided by two complementary mechanisms:
      1.  Gaussian grid map  → smooth soft penalty in the NLP objective.
      2.  half-space constraints with slack

    Usage::
        planner = NewMPCPlanner()
        planner.set_path(astar_waypoints)
        result  = planner.solve(pos_xy, vel_xy, yaw)
        cmd     = result.first_control   # [ax, ay, yaw_rate]
    """

    NX = 5   # [px, py, vx, vy, yaw]
    NU = 3   # [ax, ay, yaw_rate]

    def __init__(self, config: Optional[NewMPCConfig] = None):
        self.cfg = config or NewMPCConfig()

        # A* waypoint sequencer (advances index when close to current target)
        self.sequencer = WaypointSequencer(threshold=self.cfg.waypoint_threshold)

        # Full A* path stored as 2-D xy points for terminal path constraint
        self._full_path_xy: List[np.ndarray] = []

        # Gaussian grid map for obstacle field
        self.grid_map = GaussianGridMap(
            xyreso=self.cfg.grid_reso,
            std=self.cfg.grid_std,
            extend_area=self.cfg.grid_extend,
        )
        self._grid_interp: Optional[ca.Function] = None

        # Warm-start cache
        self._prev_u_flat: Optional[np.ndarray] = None
        self._prev_x_flat: Optional[np.ndarray] = None

        # 2-D obstacle point cloud (world frame) — updated each LiDAR scan
        self._obs_points_2d: Optional[np.ndarray] = None

    # ─────────────────────────────────────────
    # Public setters
    # ─────────────────────────────────────────
    def set_path(self, waypoints) -> None:
        """
        Set the A* planned path to follow.

        Args:
            waypoints: list/array of [x, y], [x, y, z], or [x, y, z, yaw]
        """
        self.sequencer.set_path(waypoints)
        # Store full path as 2-D xy for terminal path constraint
        self._full_path_xy = [np.array(wp[:2], dtype=float) for wp in waypoints]
        # Invalidate warm-start when path changes
        self._prev_u_flat = None
        self._prev_x_flat = None

    def update_obstacles(
        self,
        lidar_points: np.ndarray,
        drone_xy: np.ndarray,
    ) -> None:
        """
        Process a new LiDAR scan: update Gaussian grid.

        Args:
            lidar_points: (M, 3) or (M, 2) point cloud in world frame
            drone_xy:     drone [x, y] position
        """
        # Ensure 3-D array for GaussianGridMap
        if lidar_points.shape[1] == 2:
            z_col = np.full((len(lidar_points), 1), 1.5)
            pts3d = np.hstack([lidar_points, z_col])
        else:
            pts3d = lidar_points

        z_mean = float(pts3d[:, 2].mean()) if len(pts3d) > 0 else 1.5
        drone_pos3d = np.array([drone_xy[0], drone_xy[1], z_mean])

        # ── LiDAR preprocessing: range filter ────────────────────────────────
        # Drop returns beyond the sensor's reliable range and trivially close
        # ghost returns (typically < 0.05 m, caused by the drone body itself).
        if len(pts3d) > 0:
            ranges = np.linalg.norm(pts3d[:, :2] - drone_xy, axis=1)
            valid  = (ranges >= 0.05) & (ranges <= self.cfg.lidar_max_range)
            pts3d  = pts3d[valid]

        self.grid_map.update_from_lidar_points(pts3d, drone_pos3d)
        self._grid_interp = _build_grid_interp(self.grid_map)

        # Store 2-D xy obstacle points for half-space constraint linearisation
        self._obs_points_2d = pts3d[:, :2].copy() if len(pts3d) > 0 else None

    # ─────────────────────────────────────────
    # A* path geometry helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _closest_point_on_path(query: np.ndarray, waypoints: List[np.ndarray]) -> np.ndarray:
        """
        Find the closest point on the piecewise-linear A* path to ``query``.

        Each segment [waypoints[i], waypoints[i+1]] is projected onto; the
        result is the projection with the smallest Euclidean distance.

        Args:
            query:     2-D query position [x, y]
            waypoints: list of 2-D path points

        Returns:
            Closest point [x, y] on the path (a NumPy array).
        """
        if len(waypoints) == 0:
            return query.copy()
        if len(waypoints) == 1:
            return waypoints[0].copy()

        best_dist = float("inf")
        best_pt   = waypoints[0].copy()

        for i in range(len(waypoints) - 1):
            a  = waypoints[i]
            b  = waypoints[i + 1]
            ab = b - a
            ab_sq = float(np.dot(ab, ab))
            if ab_sq < 1e-10:
                pt = a.copy()
            else:
                t  = float(np.clip(np.dot(query - a, ab) / ab_sq, 0.0, 1.0))
                pt = a + t * ab
            dist = float(np.linalg.norm(query - pt))
            if dist < best_dist:
                best_dist = dist
                best_pt   = pt.copy()

        return best_pt

    # ─────────────────────────────────────────
    # Reference toward current A* waypoint
    # ─────────────────────────────────────────

    def _make_reference(self, x0: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        """
        Build a straight-line reference from the current state to goal_xy.

        Args:
            x0:      current state [px, py, vx, vy, yaw]
            goal_xy: current target A* waypoint [x, y]

        Returns:
            x_ref: (N+1, 5)
        """
        N, dt = self.cfg.N, self.cfg.dt
        pos_xy = x0[:2]
        diff   = goal_xy - pos_xy
        dist   = float(np.linalg.norm(diff))

        if dist < 1e-3:
            x_ref = np.zeros((N + 1, self.NX))
            for k in range(N + 1):
                x_ref[k, :2] = goal_xy
                x_ref[k, 4]  = x0[4]
            return x_ref

        direction = diff / dist
        goal_yaw  = float(np.arctan2(diff[1], diff[0]))
        # Cruise at v_max, capped to reach goal within horizon
        speed = min(self.cfg.v_max, dist / (N * dt))

        x_ref = np.zeros((N + 1, self.NX))
        for k in range(N + 1):
            travel = min(speed * k * dt, dist)
            x_ref[k, 0] = pos_xy[0] + direction[0] * travel
            x_ref[k, 1] = pos_xy[1] + direction[1] * travel
            x_ref[k, 2] = direction[0] * speed
            x_ref[k, 3] = direction[1] * speed
            x_ref[k, 4] = goal_yaw
        return x_ref

    # ─────────────────────────────────────────
    # NLP solver
    # ─────────────────────────────────────────

    def solve(
        self,
        pos_xy: np.ndarray,
        vel_xy: np.ndarray,
        yaw: float,
    ) -> NewMPCResult:
        """
        Formulate and solve the MPC optimisation problem.

        NLP variables:
            X  (NX × N+1)   — state trajectory
            U  (NU × N)     — control sequence

        Args:
            pos_xy: current horizontal position [x, y]
            vel_xy: current horizontal velocity [vx, vy]
            yaw:    current yaw angle [rad]

        Returns:
            NewMPCResult
        """
        t_start = time.perf_counter()
        cfg = self.cfg
        N, dt = cfg.N, cfg.dt
        NX, NU = self.NX, self.NU

        x0 = np.array([pos_xy[0], pos_xy[1], vel_xy[0], vel_xy[1], yaw])

        # Advance to next waypoint if close enough
        self.sequencer.advance_if_close(pos_xy)

        # Use current A* waypoint as MPC goal
        goal_xy = self.sequencer.current_target()
        if goal_xy is None:
            goal_xy = pos_xy.copy()   # no path → hover in place

        # Build reference toward the current A* waypoint
        x_ref = self._make_reference(x0, goal_xy)

        # ── CasADi opti stack ──────────────────
        opti = ca.Opti()

        X = opti.variable(NX, N + 1)
        U = opti.variable(NU, N)

        # ── Obstacle-aware half-space pre-computation ──────────────────────
        # For each prediction step k, find the closest obstacle surface point
        # p_min_k to the linearisation position (warm-start or reference).
        # The outward unit normal  n_k = (p_ref_k − p_min_k) / ‖…‖  is a
        # fixed NumPy scalar → the resulting constraint
        #   n_k^T · pos_k + σ_k  ≥  n_k^T · p_min_k + d_safe
        # is *linear* (affine) in the optimisation variable pos_k,
        # i.e. a convex half-space.  Re-linearising every MPC call is the
        # standard Successive Convex Approximation (SCA) approach.
        _obs_hs: list = [None] * N   # per-step: (n_k: np.ndarray, rhs_k: float) | None
        _has_obs_hs = False

        if self._obs_points_2d is not None and len(self._obs_points_2d) > 0:
            obs_pts = self._obs_points_2d
            # Secondary filter from current drone position (cloud already
            # range-filtered at scan time; this narrows to locally relevant pts)
            nearby_mask = np.linalg.norm(obs_pts - pos_xy, axis=1) <= cfg.obs_hs_radius
            obs_near = obs_pts[nearby_mask]

            if len(obs_near) > 0:
                # Linearisation trajectory: previous MPC solution or reference
                if self._prev_x_flat is not None:
                    lin_xy = self._prev_x_flat.reshape(NX, N + 1)[:2, :N].T  # (N,2)
                else:
                    lin_xy = x_ref[:N, :2]  # (N,2)

                for k in range(N):
                    p_ref = lin_xy[k]                     # (2,) fixed point
                    diffs = obs_near - p_ref              # (M,2)
                    dists = np.linalg.norm(diffs, axis=1) # (M,)
                    idx   = int(np.argmin(dists))
                    d_min = float(dists[idx])

                    if d_min < 1e-3:   # linearisation point is inside obstacle
                        continue

                    p_min  = obs_near[idx]
                    n_k    = (p_ref - p_min) / d_min           # unit outward normal
                    rhs_k  = float(n_k @ p_min) + cfg.d_safe  # n^T p_min + r_safe
                    _obs_hs[k] = (n_k, rhs_k)
                    _has_obs_hs = True

        # Slack variable per prediction step (softens infeasibility)
        Sigma_obs = opti.variable(N) if _has_obs_hs else None

        # ── Weight matrices ─────────────────────
        q = np.array([cfg.Q_pos, cfg.Q_pos, cfg.Q_vel, cfg.Q_vel, cfg.Q_yaw])
        Q   = np.diag(q)
        Q_t = np.diag(q * cfg.Q_terminal)
        R   = np.diag([cfg.R_acc, cfg.R_acc, cfg.R_yaw_rate])

        # ── Objective ───────────────────────────
        cost = 0.0

        for k in range(N):
            # State tracking
            e = X[:, k] - x_ref[k]
            cost += ca.mtimes([e.T, Q, e])

            # Control effort
            u_k = U[:, k]
            cost += ca.mtimes([u_k.T, R, u_k])

            # Jerk / smoothness penalty
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                cost += cfg.R_jerk * ca.dot(du, du)

            # Soft obstacle cost from Gaussian grid
            if self._grid_interp is not None:
                pos_k = ca.vertcat(X[0, k], X[1, k])
                cost += cfg.W_obs_grid * self._grid_interp(pos_k)

        # Terminal tracking cost
        e_T = X[:, N] - x_ref[N]
        cost += ca.mtimes([e_T.T, Q_t, e_T])

        if self._grid_interp is not None:
            pos_T = ca.vertcat(X[0, N], X[1, N])
            cost += cfg.W_obs_grid * self._grid_interp(pos_T)

        # Obstacle half-space slack penalty (keeps NLP feasible while strongly
        # discouraging constraint violation)
        if Sigma_obs is not None:
            cost += cfg.W_wall_slack * ca.dot(Sigma_obs, Sigma_obs)

        # ── Terminal path constraint ───────────────────────────────────────────
        # The terminal predicted position X[:2, N] must lie within
        # ``terminal_path_radius`` of the closest point on the full A* path.
        # Implemented as a soft constraint via a scalar slack σ_T ≥ 0:
        #
        #   ‖ X[:2,N] − p_path_T ‖² ≤ r_T² + σ_T,   σ_T ≥ 0
        #
        # p_path_T is fixed (NumPy) → the constraint is a convex (ellipsoidal)
        # inequality in X[:2,N].  σ_T is penalised as W_terminal_path_slack·σ_T
        # in the objective so IPOPT always has a feasible point.
        sigma_T = None
        p_path_T = None
        if len(self._full_path_xy) >= 1:
            # Closest point on A* path to the terminal reference position
            p_path_T = self._closest_point_on_path(x_ref[N, :2], self._full_path_xy)
            sigma_T  = opti.variable()   # scalar slack
            # Penalty: linear in σ_T (L1-like) → pushes it to zero
            cost += cfg.W_terminal_path_slack * sigma_T

        opti.minimize(cost)

        # ── Dynamics constraints (double-integrator + yaw) ──
        for k in range(N):
            opti.subject_to(X[0, k+1] == X[0, k] + X[2, k] * dt + 0.5 * U[0, k] * dt**2)
            opti.subject_to(X[1, k+1] == X[1, k] + X[3, k] * dt + 0.5 * U[1, k] * dt**2)
            opti.subject_to(X[2, k+1] == X[2, k] + U[0, k] * dt)
            opti.subject_to(X[3, k+1] == X[3, k] + U[1, k] * dt)
            opti.subject_to(X[4, k+1] == X[4, k] + U[2, k] * dt)

        # ── Initial state ────────────────────────
        opti.subject_to(X[:, 0] == x0)

        # ── Control box constraints ──────────────
        opti.subject_to(opti.bounded(-cfg.a_max,         U[0, :], cfg.a_max))
        opti.subject_to(opti.bounded(-cfg.a_max,         U[1, :], cfg.a_max))
        opti.subject_to(opti.bounded(-cfg.yaw_rate_max,  U[2, :], cfg.yaw_rate_max))

        # ── Velocity magnitude constraint ────────
        for k in range(N + 1):
            opti.subject_to(X[2, k]**2 + X[3, k]**2 <= cfg.v_max**2)

        # ── Terminal path proximity constraint ───────────────────────────────
        #   ‖ X[:2,N] − p_path_T ‖²  ≤  r_T²  +  σ_T,   σ_T ≥ 0
        if sigma_T is not None and p_path_T is not None:
            r_T_sq  = cfg.terminal_path_radius ** 2
            e_T_xy  = X[:2, N] - p_path_T          # fixed NumPy offset → affine
            opti.subject_to(sigma_T >= 0)
            opti.subject_to(ca.dot(e_T_xy, e_T_xy) <= r_T_sq + sigma_T)

        # ── Obstacle-aware half-space constraints with slack ──────────────────
        #   n_k^T · pos_k + σ_k  ≥  n_k^T · p_min_k + d_safe,   σ_k ≥ 0
        #
        #   n_k      : unit outward normal (closest obstacle point → linearisation
        #              trajectory point at step k).  Fixed NumPy scalar ⟹
        #              constraint is linear (affine) in X[:2, k] ⟹ convex.
        #   p_min_k  : closest LiDAR surface point to the linearisation position.
        #   σ_k      : per-step slack variable, penalised as W_wall_slack·σ_k²
        #              in the objective (keeps NLP always feasible).
        if Sigma_obs is not None:
            opti.subject_to(Sigma_obs >= 0)
            for k in range(N):
                hs = _obs_hs[k]
                if hs is None:
                    continue
                n_k, rhs_k = hs
                opti.subject_to(
                    n_k[0] * X[0, k] + n_k[1] * X[1, k] + Sigma_obs[k] >= rhs_k
                )

        # ── IPOPT solver options ─────────────────
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter":              cfg.max_iter,
            "print_level":           cfg.print_level,
            "sb":                    "yes",
            "warm_start_init_point": "yes" if cfg.warm_start else "no",
        }
        opti.solver("ipopt", p_opts, s_opts)

        # ── Warm-start / initial guess ───────────
        if cfg.warm_start and self._prev_u_flat is not None:
            try:
                opti.set_initial(U, self._prev_u_flat.reshape(NU, N))
                opti.set_initial(X, self._prev_x_flat.reshape(NX, N + 1))
            except Exception:
                self._default_initial(opti, X, U, x_ref)
        else:
            self._default_initial(opti, X, U, x_ref)

        # Always initialise obstacle slacks at zero (feasible starting point)
        if Sigma_obs is not None:
            opti.set_initial(Sigma_obs, np.zeros(N))
        if sigma_T is not None:
            opti.set_initial(sigma_T, 0.0)

        # ── Solve ────────────────────────────────
        try:
            sol = opti.solve()
            success = True
            cost_val = float(sol.value(cost))
        except RuntimeError:
            # Retrieve the best (possibly non-converged) iterate
            sol = opti.debug
            success = False
            try:
                cost_val = float(sol.value(cost))
            except Exception:
                cost_val = float("inf")

        # ── Extract solution ─────────────────────
        U_opt = np.array(sol.value(U))    # (NU, N)
        X_opt = np.array(sol.value(X))    # (NX, N+1)

        u_seq  = U_opt.T                  # (N, NU)
        x_pred = X_opt.T                  # (N+1, NX)

        # ── Update warm-start (shift by 1 step) ──
        u_shifted = np.vstack([u_seq[1:], u_seq[-1:]])
        x_shifted = np.vstack([x_pred[1:], x_pred[-1:]])
        self._prev_u_flat = u_shifted.T.ravel()
        self._prev_x_flat = x_shifted.T.ravel()

        solve_time_ms = (time.perf_counter() - t_start) * 1000.0

        return NewMPCResult(
            success=success,
            u_opt=u_seq,
            x_pred=x_pred,
            cost=cost_val,
            solve_time_ms=solve_time_ms,
            first_control=u_seq[0],
        )

    def _default_initial(self, opti, X, U, x_ref: np.ndarray) -> None:
        """Set a reference-based default initial guess."""
        opti.set_initial(X, x_ref.T)
        opti.set_initial(U, np.zeros((self.NU, self.cfg.N)))


# ═══════════════════════════════════════════════
# Drop-in drone controller
# ═══════════════════════════════════════════════

class NewMPCDroneController:
    """
    Cascaded drone controller driven by NewMPCPlanner.

    Drop-in replacement for MPCDroneController (controllers.py) with an
    additional ``set_path(waypoints)`` method for full A* path following.

    Architecture:
        background thread:   snapshot state → IPOPT solve → cache [ax, ay, yr]
        main sim loop:       read cache → altitude PD → attitude cmd → actuators

    Typical usage::
        ctrl = NewMPCDroneController(model, data)
        ctrl.set_target_altitude(1.5)
        ctrl.set_path(astar_waypoints)
        ctrl.start()
        while running:
            state = ctrl.update(lidar_points=scan)
            mujoco.mj_step(model, data)
        ctrl.stop()
    """

    def __init__(self, model, data, mpc_config: Optional[NewMPCConfig] = None):
        try:
            from mujoco_sim.controllers import AttitudeController
        except ImportError:
            from controllers import AttitudeController

        self.model = model
        self.data  = data

        # Inner-loop attitude controller (unchanged from DroneController)
        self.attitude_ctrl = AttitudeController(model, data)

        # New MPC planner
        self.mpc_planner = NewMPCPlanner(mpc_config)

        # Physical parameters
        self.mass = 0.027       # kg
        self.g    = 9.81        # m/s²

        # Altitude PD (z is outside the 2-D MPC scope)
        self.Kp_z      = 2.0
        self.Kd_z      = 2.0
        self.max_vel_z = 0.3    # m/s
        self.target_z  = 1.5   # m

        # Limits
        self.max_tilt   = np.radians(40)
        self.max_thrust = 1.5   # N

        # Aerodynamic drag coefficients
        self.k_drag_linear  = 0.01
        self.k_drag_angular = 0.0002

        # Motor dynamics (first-order lag filter)
        self.motor_tau     = 0.02   # 20 ms time constant
        self.ctrl_filtered = np.zeros(4)

        # Propeller visual spin
        self.hover_thrust     = self.mass * self.g
        self.prop_speed_hover = 1500.0   # rad/s (~14 000 RPM)

        # ── Async MPC thread ──────────────────────
        # _cmd_lock   : protects cached [ax, ay, yr] and lidar ptr
        # _state_lock : protects state snapshot written by the main thread
        self._cmd_lock   = threading.Lock()
        self._state_lock = threading.Lock()

        # Cached MPC horizontal command (written by MPC thread, read by main)
        self._cached_ax       = 0.0
        self._cached_ay       = 0.0
        self._cached_yaw_rate = 0.0
        self._desired_yaw     = 0.0  # integrated in MPC thread with wall-clock time

        # State snapshot (written by main thread, read by MPC thread)
        self._snap_pos  = np.zeros(3)
        self._snap_vel  = np.zeros(3)
        self._snap_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Latest LiDAR scan (main thread writes, MPC thread reads)
        self._lidar_pts: Optional[np.ndarray] = None

        # Thread control
        self._stop_event = threading.Event()
        self._mpc_thread: Optional[threading.Thread] = None

        # Diagnostics
        self.last_mpc_result: Optional[NewMPCResult] = None

    # ─────────────────────────────────────────
    # Setters
    # ─────────────────────────────────────────

    def set_path(self, waypoints) -> None:
        """Set the A* planned path for the MPC to follow."""
        self.mpc_planner.set_path(waypoints)

    def set_target_altitude(self, z: float) -> None:
        """Set desired altitude for the altitude PD loop."""
        self.target_z = z

    # ─────────────────────────────────────────
    # Thread management
    # ─────────────────────────────────────────

    def start(self) -> None:
        """Start the background MPC solver thread."""
        self._stop_event.clear()
        self._mpc_thread = threading.Thread(
            target=self._mpc_loop, daemon=True, name="new_mpc_solver"
        )
        self._mpc_thread.start()

    def stop(self) -> None:
        """Signal the MPC thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._mpc_thread is not None:
            self._mpc_thread.join(timeout=5.0)
            self._mpc_thread = None

    def _mpc_loop(self) -> None:
        """
        Background MPC solver loop.

        Each iteration:
          0. Integrate yaw_rate over actual wall-clock elapsed time.
          1. Snapshot state (fast, locked).
          2. Update obstacle map with latest LiDAR scan.
          3. Solve IPOPT (slow, no lock held).
          4. Update cached command (fast, locked).
          5. Sleep remainder of the MPC period.
        """
        from scipy.spatial.transform import Rotation as R_scipy

        cfg   = self.mpc_planner.cfg
        a_max = cfg.a_max
        _fail_count = 0
        t_prev = time.perf_counter()

        while not self._stop_event.is_set():
            t_now = time.perf_counter()
            # Actual elapsed time since previous solve (capped to avoid spike)
            actual_dt = min(t_now - t_prev, 1.0)
            t_prev = t_now

            # ── 0. Integrate yaw with real elapsed time ──
            with self._cmd_lock:
                self._desired_yaw += self._cached_yaw_rate * actual_dt
                self._desired_yaw = float(np.arctan2(
                    np.sin(self._desired_yaw), np.cos(self._desired_yaw)
                ))

            # ── 1. Snapshot state ────────────────────────
            with self._state_lock:
                pos  = self._snap_pos.copy()
                vel  = self._snap_vel.copy()
                quat = self._snap_quat.copy()
            with self._cmd_lock:
                lidar_pts = self._lidar_pts   # may be None

            # Compute yaw from quaternion
            scipy_quat = [quat[1], quat[2], quat[3], quat[0]]
            euler = R_scipy.from_quat(scipy_quat).as_euler('xyz', degrees=False)
            yaw    = float(euler[2])
            pos_xy = pos[:2].copy()
            vel_xy = vel[:2].copy()

            # ── 2. Update obstacle map ───────────────────
            if lidar_pts is not None:
                try:
                    self.mpc_planner.update_obstacles(lidar_pts, pos_xy)
                except Exception:
                    pass

            # ── 3. Solve MPC (no lock held) ──────────────
            try:
                result = self.mpc_planner.solve(pos_xy, vel_xy, yaw)
                self.last_mpc_result = result

                if result.success:
                    ax, ay, yr = result.first_control
                    _fail_count = 0
                else:
                    # Non-converged best iterate: brake to stop drift
                    _fail_count += 1
                    ax = float(np.clip(-2.0 * vel[0], -a_max, a_max))
                    ay = float(np.clip(-2.0 * vel[1], -a_max, a_max))
                    yr = 0.0
            except Exception:
                _fail_count += 1
                ax = float(np.clip(-2.0 * vel[0], -a_max, a_max))
                ay = float(np.clip(-2.0 * vel[1], -a_max, a_max))
                yr = 0.0

            # Reset warm-start after repeated failures
            if _fail_count >= 3:
                self.mpc_planner._prev_u_flat = None
                self.mpc_planner._prev_x_flat = None
                _fail_count = 0

            # ── 4. Cache command ─────────────────────────
            with self._cmd_lock:
                self._cached_ax       = float(ax)
                self._cached_ay       = float(ay)
                self._cached_yaw_rate = float(yr)

            # ── 5. Sleep remainder of MPC period ─────────
            elapsed    = time.perf_counter() - t_now
            sleep_time = cfg.dt - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    # ─────────────────────────────────────────
    # Main update (sim frequency)
    # ─────────────────────────────────────────

    def update(self, lidar_points: Optional[np.ndarray] = None) -> dict:
        """
        Run one control cycle at MuJoCo's simulation frequency.

        Args:
            lidar_points: (M, 3) LiDAR scan in world frame, or None to
                          reuse the last scan.

        Returns:
            State dict with keys: position, orientation, velocity,
            angular_velocity, euler, mpc_success, mpc_solve_ms,
            mpc_first_control.
        """
        dt = self.model.opt.timestep

        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]

        # ── Push state snapshot to MPC thread (fast) ──
        with self._state_lock:
            self._snap_pos[:]  = pos
            self._snap_vel[:]  = vel
            self._snap_quat[:] = self.data.qpos[3:7]

        # ── Push new LiDAR scan if available ──────────
        if lidar_points is not None:
            with self._cmd_lock:
                self._lidar_pts = lidar_points.copy()

        # ── Read cached MPC command ────────────────────
        with self._cmd_lock:
            ax_mpc      = self._cached_ax
            ay_mpc      = self._cached_ay
            desired_yaw = self._desired_yaw

        # ── Altitude PD ───────────────────────────────
        z_err     = self.target_z - float(pos[2])
        vel_cmd_z = np.clip(self.Kp_z * z_err, -self.max_vel_z, self.max_vel_z)
        az        = self.Kd_z * (vel_cmd_z - float(vel[2])) + self.g

        # ── Full 3-D desired acceleration ─────────────
        acc_des = np.array([ax_mpc, ay_mpc, az])

        # ── Thrust ────────────────────────────────────
        thrust = float(np.clip(self.mass * acc_des[2], 0.0, self.max_thrust))

        # ── Roll / pitch from horizontal MPC acceleration ──
        # The MPC outputs world-frame accelerations [ax_mpc, ay_mpc].
        # The attitude controller interprets desired_pitch / desired_roll in the
        # BODY frame.  When yaw ≠ 0 these differ: rotate into body frame first.
        #   ax_body =  ax_world * cos(ψ) + ay_world * sin(ψ)
        #   ay_body = -ax_world * sin(ψ) + ay_world * cos(ψ)
        if acc_des[2] > 0.1:
            cos_yaw = np.cos(desired_yaw)
            sin_yaw = np.sin(desired_yaw)
            ax_body =  ax_mpc * cos_yaw + ay_mpc * sin_yaw
            ay_body = -ax_mpc * sin_yaw + ay_mpc * cos_yaw
            desired_pitch = float(np.clip( ax_body / self.g, -self.max_tilt, self.max_tilt))
            desired_roll  = float(np.clip(-ay_body / self.g, -self.max_tilt, self.max_tilt))
        else:
            desired_pitch = 0.0
            desired_roll  = 0.0

        # ── Attitude controller → body moments ────────
        moments = self.attitude_ctrl.compute(desired_roll, desired_pitch, desired_yaw)

        # ── Motor lag filter (first-order) ────────────
        alpha    = dt / (self.motor_tau + dt)
        ctrl_raw = np.array([thrust, moments[0], moments[1], moments[2]])
        self.ctrl_filtered += alpha * (ctrl_raw - self.ctrl_filtered)

        self.data.ctrl[0] = self.ctrl_filtered[0]   # thrust
        self.data.ctrl[1] = self.ctrl_filtered[1]   # mx
        self.data.ctrl[2] = self.ctrl_filtered[2]   # my
        self.data.ctrl[3] = self.ctrl_filtered[3]   # mz

        # ── Propeller visual spin ──────────────────────
        thrust_ratio = max(self.ctrl_filtered[0], 0.0) / self.hover_thrust
        prop_speed   = self.prop_speed_hover * np.sqrt(max(thrust_ratio, 0.0))
        self.data.ctrl[4] = -prop_speed   # FR  (CW)
        self.data.ctrl[5] =  prop_speed   # FL  (CCW)
        self.data.ctrl[6] = -prop_speed   # BL  (CW)
        self.data.ctrl[7] =  prop_speed   # BR  (CCW)

        # ── Aerodynamic drag ──────────────────────────
        self.data.qfrc_applied[:3]  = -self.k_drag_linear  * vel
        self.data.qfrc_applied[3:6] = -self.k_drag_angular * self.data.qvel[3:6]

        return self.get_state()

    def get_state(self) -> dict:
        """Get current drone state plus MPC diagnostics."""
        return {
            'position':          self.data.qpos[:3].copy(),
            'orientation':       self.data.qpos[3:7].copy(),
            'velocity':          self.data.qvel[:3].copy(),
            'angular_velocity':  self.data.qvel[3:6].copy(),
            'euler':             self.attitude_ctrl.quaternion_to_euler(
                                     self.data.qpos[3:7]),
            'mpc_success':       self.last_mpc_result.success
                                 if self.last_mpc_result else False,
            'mpc_solve_ms':      self.last_mpc_result.solve_time_ms
                                 if self.last_mpc_result else 0.0,
            'mpc_first_control': self.last_mpc_result.first_control.copy()
                                 if self.last_mpc_result is not None
                                 else np.zeros(3),
        }
