"""
CasADi-based MPC Local Path Planner for drone navigation.

Uses LiDAR-based Gaussian occupancy grid as a smooth obstacle cost field.
Handles both convex obstacles and long walls via:
  1. Gaussian grid map interpolated as a soft cost inside the NLP
  2. RANSAC-extracted line segments as half-plane constraints
  3. Rolling local map from scan history

Dynamics: 2D double-integrator + yaw
  State:   x = [px, py, vx, vy, yaw]   (5)
  Control: u = [ax, ay, yaw_rate]       (3)

Solver: CasADi + IPOPT (interior-point NLP solver)

Dependencies: casadi, numpy, scipy
Author: Lorenzo Ortolani
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time

from mujoco_sim.gaussian_grid_map import GaussianGridMap


# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════
@dataclass
class MPCConfig:
    """All tunable parameters for the MPC planner."""

    # ── Horizon ──
    N: int = 50                      # prediction steps
    dt: float = 0.01                  # discretisation time [s]

    # ── Dynamics limits ──
    v_max: float = 2.0               # max velocity  [m/s]
    a_max: float = 1.5               # max acceleration [m/s²]
    yaw_rate_max: float = 1.0        # max yaw rate [rad/s]

    # ── Cost weights  (Q: tracking, R: input, S: smoothness) ──
    Q_pos: float = 10.0               # position error
    Q_vel: float = 1.0               # velocity error
    Q_yaw: float = 0.5               # heading error
    Q_terminal: float = 100.0         # terminal position weight multiplier
    R_acc: float = 2.0               # acceleration penalty
    R_yaw_rate: float = 0.05         # yaw-rate penalty
    R_jerk: float = 0.5              # jerk (Δu) smoothness penalty

    # ── Obstacle avoidance ──
    W_obs_grid: float = 20.0         # Gaussian grid map cost weight
    W_obs_line: float = 20.0        # line-constraint violation weight
    d_safe: float = 0.8              # safety distance [m]

    # ── Goal-progress cost (prevents local minima) ──
    W_goal: float = 50.0              # direct distance-to-goal weight per step
    W_goal_terminal: float = 40.0    # terminal distance-to-goal weight

    # ── Subgoal / obstacle bypass ──
    subgoal_lookahead: float = 3.0   # look-ahead distance to detect blocking [m]
    subgoal_lateral: float = 1.5     # lateral offset to steer around obstacle [m]

    # ── Gaussian grid map ──
    grid_reso: float = 0.25          # [m]
    grid_std: float = 0.6            # Gaussian spread [m]
    grid_extend: float = 2.0         # extension around points [m]

    # ── RANSAC line extraction ──
    ransac_iterations: int = 60
    ransac_threshold: float = 0.15   # inlier distance [m]
    ransac_min_points: int = 10
    max_lines: int = 8

    # ── Solver ──
    max_iter: int = 150
    warm_start: bool = True
    print_level: int = 0             # IPOPT verbosity (0=silent)

    # Further parameters for reference generation
    perturb_delta: float = 1.0          # lateral perturbation


# ═══════════════════════════════════════════════
# Helper dataclasses
# ═══════════════════════════════════════════════
@dataclass
class LineSegment:
    """A wall / line extracted from LiDAR via RANSAC."""
    p1: np.ndarray              # endpoint 1  [x, y]
    p2: np.ndarray              # endpoint 2  [x, y]
    normal: np.ndarray          # outward unit normal (toward free-space)
    offset: float               # n^T p >= offset  defines the free half-plane
    length: float = 0.0
    num_inliers: int = 0


@dataclass
class DroneState:
    """Drone state vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 1.5              # altitude (fixed for 2-D planning)
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0

    def vec(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy, self.yaw])

    @staticmethod
    def from_vec(v: np.ndarray) -> "DroneState":
        return DroneState(x=v[0], y=v[1], vx=v[2], vy=v[3], yaw=v[4])


@dataclass
class MPCResult:
    """Solver output."""
    success: bool
    u_opt: np.ndarray               # optimal control sequence  [N, 3]
    x_pred: np.ndarray              # predicted state trajectory [N+1, 5]
    cost: float
    solve_time_ms: float
    first_control: np.ndarray       # u_opt[0] — the command to apply now


# ═══════════════════════════════════════════════
# RANSAC line extraction
# ═══════════════════════════════════════════════
def extract_lines_ransac(
    points_2d: np.ndarray,
    drone_xy: np.ndarray,
    cfg: MPCConfig,
) -> List[LineSegment]:
    """
    Extract line segments from a 2-D point cloud using iterative RANSAC.

    The outward normal of each line is chosen to point *toward* the drone
    (i.e. toward free space).

    Args:
        points_2d: (M, 2) obstacle points in world frame
        drone_xy:  [x, y] drone position (used to orient normals)
        cfg:       planner config

    Returns:
        List of LineSegment
    """
    if len(points_2d) < cfg.ransac_min_points:
        return []

    remaining = points_2d.copy()
    lines: List[LineSegment] = []

    for _ in range(cfg.max_lines):
        if len(remaining) < cfg.ransac_min_points:
            break

        best_inlier_idx = None
        best_count = 0

        for _ in range(cfg.ransac_iterations):
            # pick two random points
            idx = np.random.choice(len(remaining), 2, replace=False)
            p1, p2 = remaining[idx[0]], remaining[idx[1]]
            seg = p2 - p1
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-6:
                continue

            # line normal (not yet oriented)
            n = np.array([-seg[1], seg[0]]) / seg_len

            # signed distances of all points to this line
            dists = np.abs((remaining - p1) @ n)
            inlier_mask = dists < cfg.ransac_threshold
            count = int(np.sum(inlier_mask))

            if count > best_count:
                best_count = count
                best_inlier_idx = inlier_mask

        if best_count < cfg.ransac_min_points:
            break

        inlier_pts = remaining[best_inlier_idx]

        # fit line to all inliers via PCA (first principal component)
        centroid = inlier_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid)
        direction = Vt[0]  # principal direction

        # project inliers onto direction to get endpoints
        projections = (inlier_pts - centroid) @ direction
        p1 = centroid + projections.min() * direction # leftmost point along the wall
        p2 = centroid + projections.max() * direction # rightmost point along the wall

        # compute outward normal (pointing toward drone)
        normal = np.array([-direction[1], direction[0]])
        if normal @ (drone_xy - centroid) < 0:
            normal = -normal

        offset = float(normal @ centroid) + cfg.d_safe

        seg = LineSegment(
            p1=p1, p2=p2,
            normal=normal,
            offset=offset,
            length=np.linalg.norm(p2 - p1),
            num_inliers=best_count,
        )
        lines.append(seg)

        # remove inliers for next iteration
        remaining = remaining[~best_inlier_idx]

    return lines


# ═══════════════════════════════════════════════
# Obstacle cost interpolation (CasADi-compatible)
# ═══════════════════════════════════════════════
def _build_grid_cost_function(
    grid_map: GaussianGridMap,
) -> Optional[ca.Function]:
    """
    Build a CasADi interpolant from the Gaussian grid map.

    Returns a ca.Function:  cost = f(px, py)  ∈ [0, 1]
    or None if the grid map has not been initialised.
    """
    if grid_map.gmap is None:
        return None

    xw, yw = grid_map.gmap.shape
    x_grid = np.linspace(
        grid_map.minx,
        grid_map.minx + (xw - 1) * grid_map.xyreso,
        xw,
    )
    y_grid = np.linspace(
        grid_map.miny,
        grid_map.miny + (yw - 1) * grid_map.xyreso,
        yw,
    )

    # CasADi 2-D interpolant expects data in column-major (Fortran) order:
    # data[i + j*nx] = gmap[i, j]
    data_flat = grid_map.gmap.ravel(order="F").tolist()

    interp = ca.interpolant(
        "obs_cost", "bspline", [x_grid.tolist(), y_grid.tolist()], data_flat
    )
    return interp


# ═══════════════════════════════════════════════
# MPC Local Planner
# ═══════════════════════════════════════════════
class MPCLocalPlanner:
    """
    CasADi / IPOPT  MPC local planner for a drone navigating with LiDAR.

    Workflow every control cycle:
        1.  Receive LiDAR point cloud  →  update Gaussian grid map
        2.  Extract line segments (walls) via RANSAC
        3.  Build & solve the NLP with CasADi
        4.  Return first control command  [ax, ay, yaw_rate]

    Usage:
        planner = MPCLocalPlanner()
        result  = planner.plan(drone_state, goal, lidar_points)
        cmd     = result.first_control   # [ax, ay, yaw_rate]
    """

    NX = 5   # state dimension  [px, py, vx, vy, yaw]
    NU = 3   # control dimension [ax, ay, yaw_rate]

    def __init__(self, config: Optional[MPCConfig] = None):
        self.cfg = config or MPCConfig()

        # Gaussian grid map
        self.grid_map = GaussianGridMap(
            xyreso=self.cfg.grid_reso,
            std=self.cfg.grid_std,
            extend_area=self.cfg.grid_extend,
        )

        # Extracted wall lines (updated each cycle)
        self.lines: List[LineSegment] = []

        # CasADi interpolant (rebuilt when grid changes)
        self._grid_interp: Optional[ca.Function] = None

        # Warm-start storage
        self._prev_u_flat: Optional[np.ndarray] = None
        self._prev_x_flat: Optional[np.ndarray] = None

    # ─────────────────────────────────────────
    # 1.  Perception update
    # ─────────────────────────────────────────
    def update_obstacles(
        self,
        lidar_points: np.ndarray,
        drone_state: DroneState,
    ) -> None:
        """
        Process a new LiDAR scan and update the obstacle representation.

        Args:
            lidar_points: (M, 3) point cloud in *world frame*
            drone_state:  current drone state
        """
        drone_pos = np.array([drone_state.x, drone_state.y, drone_state.z])
        self.grid_map.update_from_lidar_points(lidar_points, drone_pos)

        # rebuild the CasADi interpolant from the new grid
        self._grid_interp = _build_grid_cost_function(self.grid_map)

        # extract wall lines from 2-D projection
        pts_2d = lidar_points[:, :2]
        self.lines = extract_lines_ransac(
            pts_2d,
            drone_xy=np.array([drone_state.x, drone_state.y]),
            cfg=self.cfg,
        )

    # ─────────────────────────────────────────
    # 2.  Reference trajectory (obstacle-aware)
    # ─────────────────────────────────────────
    def _compute_subgoal(
        self, x0: np.ndarray, goal: np.ndarray,
    ) -> np.ndarray:
        """
        If the straight line toward the goal is blocked by the Gaussian
        grid cost, pick a lateral subgoal that steers around the obstacle.

        Returns:
            subgoal  [gx, gy]  — either the original goal or a bypass point.
        """
        cfg = self.cfg
        goal_xy = goal[:2]
        drone_xy = x0[:2]
        diff = goal_xy - drone_xy
        dist = np.linalg.norm(diff)

        if dist < 0.5 or self._grid_interp is None:
            return goal_xy  # close enough or no grid — go direct

        direction = diff / dist

        # Sample points along the straight line up to lookahead distance
        lookahead = min(cfg.subgoal_lookahead, dist)
        n_samples = int(lookahead / cfg.grid_reso) + 1
        blocked = False
        block_point = None

        for i in range(1, n_samples + 1):
            d = i * cfg.perturb_delta           # before: grid_reso
            sample = drone_xy + direction * d
            try:
                cost_val = float(self._grid_interp(sample))
            except Exception:
                continue
            if cost_val > 0.15:          # significant obstacle probability
                blocked = True
                block_point = sample
                break

        if not blocked:
            return goal_xy  # path is clear

        # Compute a lateral offset to steer around the obstacle
        # Perpendicular direction (try both sides, pick the one with lower cost)
        perp = np.array([-direction[1], direction[0]])
        offset = cfg.subgoal_lateral

        # Candidate subgoals: pass left or right of the obstacle
        candidates = [
            block_point + perp * offset,
            block_point - perp * offset,
        ]

        best = goal_xy
        best_cost = float('inf')
        for cand in candidates:
            try:
                c = float(self._grid_interp(cand))
            except Exception:
                c = 1.0
            # Prefer the side that is clearer AND closer to goal
            total = c + 0.1 * np.linalg.norm(cand - goal_xy)
            if total < best_cost:
                best_cost = total
                best = cand

        return best

    def _make_reference(
        self, x0: np.ndarray, goal: np.ndarray,
    ) -> np.ndarray:
        """
        Build a reference trajectory toward a (possibly adjusted) subgoal.

        If the straight path to the goal is blocked by an obstacle in the
        Gaussian grid, the reference curves toward a lateral subgoal.

        Args:
            x0:   current state [5]
            goal:  [gx, gy]  or  [gx, gy, g_yaw]

        Returns:
            x_ref  [N+1, NX]
        """
        N, dt = self.cfg.N, self.cfg.dt
        goal_xy = goal[:2]

        # Pick subgoal (may differ from goal if path is blocked)
        subgoal_xy = self._compute_subgoal(x0, goal)

        diff = goal_xy - x0[:2]     # before: subgoal_xy - x0[:2]
        dist = np.linalg.norm(diff)
        # Yaw always points toward the *final* goal for consistency
        diff_goal = goal_xy - x0[:2]
        goal_yaw = goal[2] if len(goal) >= 3 else np.arctan2(diff_goal[1], diff_goal[0])

        speed = min(self.cfg.v_max, dist / (N * dt)) if dist > 0.01 else 0.0
        direction = diff / max(dist, 1e-6)

        ref = np.zeros((N + 1, self.NX))
        for k in range(N + 1):
            t = k * dt
            travel = min(speed * t, dist)
            ref[k, 0] = x0[0] + direction[0] * travel   # px
            ref[k, 1] = x0[1] + direction[1] * travel   # py
            ref[k, 2] = direction[0] * speed             # vx
            ref[k, 3] = direction[1] * speed             # vy
            ref[k, 4] = goal_yaw                         # yaw
        return ref

    # ─────────────────────────────────────────
    # 3.  Build & solve the NLP
    # ─────────────────────────────────────────
    def solve(
        self,
        drone_state: DroneState,
        goal: np.ndarray,
    ) -> MPCResult:
        """
        Formulate and solve the MPC optimisation.

        Args:
            drone_state: current state
            goal:        [gx, gy] or [gx, gy, g_yaw]

        Returns:
            MPCResult with optimal controls, predicted trajectory, etc.
        """
        t_start = time.perf_counter()
        cfg = self.cfg
        N, dt = cfg.N, cfg.dt
        NX, NU = self.NX, self.NU

        x0 = drone_state.vec()
        x_ref = self._make_reference(x0, goal)

        # ── CasADi opti stack ──
        opti = ca.Opti()

        X = opti.variable(NX, N + 1)   # states  as columns
        U = opti.variable(NU, N)        # controls as columns

        # ── Weight matrices (diagonal) ──
        q = np.array([cfg.Q_pos, cfg.Q_pos, cfg.Q_vel, cfg.Q_vel, cfg.Q_yaw])
        Q = np.diag(q)
        Q_t = np.diag(q * cfg.Q_terminal)     # heavier terminal weight
        R = np.diag([cfg.R_acc, cfg.R_acc, cfg.R_yaw_rate])

        # ── Build objective ──
        cost = 0.0

        for k in range(N):
            # ---- state tracking ----
            e = X[:, k] - x_ref[k]
            cost += ca.mtimes([e.T, Q, e])

            # ---- control effort ----
            u_k = U[:, k]
            cost += ca.mtimes([u_k.T, R, u_k])

            # ---- smoothness / jerk penalty ----
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                cost += cfg.R_jerk * ca.dot(du, du)

            # ---- Gaussian grid map obstacle cost (soft) ----
            if self._grid_interp is not None:
                pos_k = ca.vertcat(X[0, k], X[1, k])
                obs_val = self._grid_interp(pos_k)
                cost += cfg.W_obs_grid * obs_val

            # ---- Line / wall constraints (soft quadratic penalty) ----
            for line in self.lines:
                n = line.normal
                # signed distance: positive = safe side
                signed_dist = n[0] * X[0, k] + n[1] * X[1, k] - line.offset
                violation = ca.fmax(0.0, -signed_dist)
                cost += cfg.W_obs_line * violation ** 2

            # ---- Direct goal-progress cost (prevents local minima) ----
            dx_g = X[0, k] - goal[0]
            dy_g = X[1, k] - goal[1]
            cost += cfg.W_goal * (dx_g**2 + dy_g**2)

        # ── Terminal cost ──
        e_T = X[:, N] - x_ref[N]
        cost += ca.mtimes([e_T.T, Q_t, e_T])

        # ── Terminal goal-progress (strong pull toward goal) ──
        dx_gT = X[0, N] - goal[0]
        dy_gT = X[1, N] - goal[1]
        cost += cfg.W_goal_terminal * (dx_gT**2 + dy_gT**2)

        if self._grid_interp is not None:
            pos_T = ca.vertcat(X[0, N], X[1, N])
            cost += cfg.W_obs_grid * self._grid_interp(pos_T)

        opti.minimize(cost)

        # ── Dynamics constraints (double integrator + yaw) ──
        for k in range(N):
            px_k  = X[0, k];  py_k  = X[1, k]
            vx_k  = X[2, k];  vy_k  = X[3, k]
            yaw_k = X[4, k]
            ax_k  = U[0, k];  ay_k  = U[1, k];  yr_k = U[2, k]

            opti.subject_to(X[0, k+1] == px_k  + vx_k * dt + 0.5 * ax_k * dt**2)
            opti.subject_to(X[1, k+1] == py_k  + vy_k * dt + 0.5 * ay_k * dt**2)
            opti.subject_to(X[2, k+1] == vx_k  + ax_k * dt)
            opti.subject_to(X[3, k+1] == vy_k  + ay_k * dt)
            opti.subject_to(X[4, k+1] == yaw_k + yr_k * dt)

        # ── Initial state ──
        opti.subject_to(X[:, 0] == x0)

        # ── Box constraints on controls ──
        for k in range(N):
            opti.subject_to(opti.bounded(-cfg.a_max,       U[0, k], cfg.a_max))
            opti.subject_to(opti.bounded(-cfg.a_max,       U[1, k], cfg.a_max))
            opti.subject_to(opti.bounded(-cfg.yaw_rate_max, U[2, k], cfg.yaw_rate_max))

        # ── Velocity magnitude constraint ──
        for k in range(N + 1):
            opti.subject_to(X[2, k]**2 + X[3, k]**2 <= cfg.v_max**2)

        # ── IPOPT solver options ──
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": cfg.max_iter,
            "print_level": cfg.print_level,
            "sb": "yes",
            "warm_start_init_point": "yes" if cfg.warm_start else "no",
        }
        opti.solver("ipopt", p_opts, s_opts)

        # ── Initial guess (warm-start or default) ──
        if cfg.warm_start and self._prev_u_flat is not None:
            try:
                opti.set_initial(U, self._prev_u_flat.reshape(NU, N))
                opti.set_initial(X, self._prev_x_flat.reshape(NX, N + 1))
            except Exception:
                self._set_default_initial(opti, X, U, x_ref)
        else:
            self._set_default_initial(opti, X, U, x_ref)

        # ── Solve ──
        try:
            sol = opti.solve()
            success = True
            cost_val = float(sol.value(cost))
        except RuntimeError:
            # IPOPT may fail to converge; retrieve best iterate
            sol = opti.debug
            success = False
            try:
                cost_val = float(sol.value(cost))
            except Exception:
                cost_val = float("inf")

        # ── Extract solution ──
        U_opt = np.array(sol.value(U))    # (NU, N)
        X_opt = np.array(sol.value(X))    # (NX, N+1)

        u_seq  = U_opt.T                  # (N, NU)
        x_pred = X_opt.T                  # (N+1, NX)

        # ── Store warm-start for next call (shift by 1 step) ──
        u_shifted = np.vstack([u_seq[1:], u_seq[-1:]])
        x_shifted = np.vstack([x_pred[1:], x_pred[-1:]])
        self._prev_u_flat = u_shifted.T.ravel()
        self._prev_x_flat = x_shifted.T.ravel()

        solve_time = (time.perf_counter() - t_start) * 1000.0

        return MPCResult(
            success=success,
            u_opt=u_seq,
            x_pred=x_pred,
            cost=cost_val,
            solve_time_ms=solve_time,
            first_control=u_seq[0],
        )

    def _set_default_initial(self, opti, X, U, x_ref):
        """Set a default initial guess from the reference trajectory."""
        opti.set_initial(X, x_ref.T)
        opti.set_initial(U, np.zeros((self.NU, self.cfg.N)))

    # ─────────────────────────────────────────
    # 4.  One-call planning step
    # ─────────────────────────────────────────
    def plan(
        self,
        drone_state: DroneState,
        goal: np.ndarray,
        lidar_points: Optional[np.ndarray] = None,
    ) -> MPCResult:
        """
        Full planning cycle: update obstacles (if new scan) → solve MPC.

        Args:
            drone_state:  current drone state
            goal:         [gx, gy] or [gx, gy, g_yaw]
            lidar_points: (M, 3) new scan in world frame, None to reuse last

        Returns:
            MPCResult
        """
        if lidar_points is not None:
            self.update_obstacles(lidar_points, drone_state)
        return self.solve(drone_state, goal)

    # ─────────────────────────────────────────
    # 5.  Visualisation
    # ─────────────────────────────────────────
    def plot(
        self,
        result: MPCResult,
        drone_state: DroneState,
        goal: np.ndarray,
        lidar_points: Optional[np.ndarray] = None,
        ax=None,
    ):
        """
        Visualise the MPC solution on top of the Gaussian grid map.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Gaussian heatmap
        if self.grid_map.gmap is not None:
            self.grid_map.draw_heatmap(ax=ax, cmap="Reds", alpha=0.5)

        # extracted wall lines
        for i, line in enumerate(self.lines):
            label = "wall" if i == 0 else None
            ax.plot(
                [line.p1[0], line.p2[0]],
                [line.p1[1], line.p2[1]],
                "b-", linewidth=2.5, label=label,
            )
            mid = (line.p1 + line.p2) / 2.0
            ax.arrow(
                mid[0], mid[1],
                line.normal[0] * 0.3, line.normal[1] * 0.3,
                head_width=0.1, color="blue", alpha=0.6,
            )

        # raw LiDAR
        if lidar_points is not None:
            ax.scatter(
                lidar_points[:, 0], lidar_points[:, 1],
                s=4, c="grey", alpha=0.4, label="LiDAR",
            )

        # MPC predicted trajectory
        traj = result.x_pred
        ax.plot(traj[:, 0], traj[:, 1], "g.-", linewidth=2, label="MPC traj")

        # drone
        ax.plot(drone_state.x, drone_state.y, "ko", markersize=8, label="drone")
        L = 0.5
        ax.arrow(
            drone_state.x, drone_state.y,
            L * np.cos(drone_state.yaw), L * np.sin(drone_state.yaw),
            head_width=0.15, color="black",
        )

        # goal
        ax.plot(goal[0], goal[1], "r*", markersize=15, label="goal")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(
            f"MPC  |  cost={result.cost:.1f}  |  "
            f"solve={result.solve_time_ms:.1f} ms  |  ok={result.success}"
        )
        ax.legend(loc="upper right")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        return ax
