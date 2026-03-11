"""
CasADi / IPOPT MPC trajectory tracker for drone navigation (3-D).

Design
------
- Tracks a path produced by AStarPlanner.
  Path waypoints may be 2-D (x, y) or 3-D (x, y, z).
  When z is absent from the path, the MPC tracks the constant altitude
  passed via the 'z_ref' argument of solve().

- Uses a 3-D double-integrator + yaw dynamics model.

- The reference trajectory is built by advancing along the A* path at a
  desired cruising speed, producing a smooth N-step reference for the MPC.

- Soft obstacle avoidance is added via the FixedGaussianGridMap interpolated
  as a CasADi bspline cost (applied to the x-y plane only, where the grid lives).

- MPCResult.next_position  gives [x, y, z] one step ahead — direct PID setpoint.

State    x  = [px, py, pz, vx, vy, vz, yaw]     (NX = 7)
Control  u  = [ax, ay, az, yaw_rate]            (NU = 4)

Euler integration over dt:
    px_{k+1}  = px_k + vx_k*dt + 0.5*ax_k*dt^2
    py_{k+1}  = py_k + vy_k*dt + 0.5*ay_k*dt^2
    pz_{k+1}  = pz_k + vz_k*dt + 0.5*az_k*dt^2
    vx_{k+1}  = vx_k + ax_k*dt
    vy_{k+1}  = vy_k + ay_k*dt
    vz_{k+1}  = vz_k + az_k*dt
    yaw_{k+1} = yaw_k + yaw_rate_k*dt

Dependencies: casadi, numpy, scipy
author: Lorenzo Ortolani
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import casadi as ca

from new_mujoco.gaussian_grid_map import FixedGaussianGridMap


# ============================================================
# Configuration
# ============================================================

@dataclass
class MPCConfig:
    """All tunable MPC parameters."""

    # Horizon
    N: int = 30                     # prediction steps
    dt: float = 0.1                 # discretisation step [s]

    # Dynamics limits
    v_max_xy: float = 2.0           # max horizontal speed [m/s]
    v_max_z: float = 1.0            # max vertical speed [m/s]
    a_max_xy: float = 2.0           # max horizontal acceleration [m/s²]
    a_max_z: float = 1.5            # max vertical acceleration [m/s²]
    yaw_rate_max: float = 1.5       # max yaw rate [rad/s]

    # Desired cruise speed along A* path for reference generation
    v_ref: float = 1.0              # [m/s] — horizontal component

    # Tracking cost weights
    Q_xy: float = 15.0              # horizontal position error
    Q_z: float = 20.0               # altitude error (higher to keep constant altitude)
    Q_vel_xy: float = 1.0           # horizontal velocity error
    Q_vel_z: float = 2.0            # vertical velocity error
    Q_yaw: float = 0.2              # heading error
    Q_terminal: float = 50.0        # terminal state weight multiplier
    R_acc_xy: float = 1.0           # horizontal acceleration effort
    R_acc_z: float = 1.5            # vertical acceleration effort
    R_yaw_rate: float = 0.1         # yaw-rate effort
    R_jerk: float = 0.3             # delta-u smoothness (jerk = d(acceleration)/dt)

    # Obstacle avoidance (applied to x-y plane)
    W_obs: float = 30.0             # Gaussian grid soft penalty weight

    # Half-space constraints from raw LiDAR points
    d_safe_pts: float = 0.5         # minimum clearance from each LiDAR point [m]
    W_obs_pts: float = 50.0         # quadratic penalty weight for half-space violation
    max_obs_constraints: int = 15   # maximum LiDAR points used as half-space constraints
    obs_check_radius: float = 3.0   # only consider points within this radius [m]

    # IPOPT
    max_iter: int = 100
    warm_start: bool = True
    print_level: int = 0            # 0 = silent


# ============================================================
# Result
# ============================================================

@dataclass
class MPCResult:
    success: bool
    x_pred: np.ndarray              # (N+1, 7)  predicted states
    u_opt: np.ndarray               # (N,   4)  optimal controls
    cost: float
    solve_time_ms: float

    @property
    def next_position(self) -> np.ndarray:
        """[x, y, z] one MPC step ahead — use directly as cascaded-PID setpoint."""
        return self.x_pred[1, :3]

    @property
    def next_velocity(self) -> np.ndarray:
        """[vx, vy, vz] one step ahead."""
        return self.x_pred[1, 3:6]

    @property
    def predicted_xy(self) -> np.ndarray:
        """(N+1, 2) predicted horizontal positions — useful for plotting."""
        return self.x_pred[:, :2]

    @property
    def predicted_z(self) -> np.ndarray:
        """(N+1, 1) predicted altitude."""
        return self.x_pred[:, 2]


# ============================================================
# MPC Tracker
# ============================================================

class MPCTracker:
    """
    3-D path-tracking MPC for the Skydio drone.

    Typical usage
    -------------
    tracker = MPCTracker()

    # every planning cycle (after updating the grid map):
    tracker.update_grid(grid_map)
    result = tracker.solve(drone_state, a_star_path, z_ref=1.5)

    # feed into cascaded PID:
    ctrl.set_target(*result.next_position)
    """

    NX = 7   # [px, py, pz, vx, vy, vz, yaw]
    NU = 4   # [ax, ay, az, yaw_rate]

    def __init__(self, config: Optional[MPCConfig] = None):
        self.cfg = config or MPCConfig()

        # CasADi bspline interpolant of the current occupancy map
        self._grid_interp: Optional[ca.Function] = None

        # Warm-start storage
        self._prev_u: Optional[np.ndarray] = None   # (N, NU)
        self._prev_x: Optional[np.ndarray] = None   # (N+1, NX)

    # ------------------------------------------------------------------
    # Obstacle point selection for half-space constraints
    # ------------------------------------------------------------------

    def _select_constraint_points(
        self,
        pts_2d: np.ndarray,
        drone_xy: np.ndarray,
    ) -> np.ndarray:
        """
        Return up to max_obs_constraints LiDAR points within obs_check_radius,
        sorted nearest-first.  Used to build per-point half-space penalty terms.

        Parameters
        ----------
        pts_2d   : (M, 2) obstacle point positions in world x-y
        drone_xy : (2,)   current drone x-y position

        Returns
        -------
        (K, 2) selected points, K <= max_obs_constraints
        """
        if len(pts_2d) == 0:
            return np.empty((0, 2))

        dists = np.linalg.norm(pts_2d - drone_xy, axis=1)  # distances from drone to each obstacle point
        mask = dists < self.cfg.obs_check_radius  # only consider points within a check radius
        if not np.any(mask):
            return np.empty((0, 2))

        pts_close  = pts_2d[mask]  # select only points within the check radius
        dist_close = dists[mask]  # corresponding distances
        n_sel = min(len(pts_close), self.cfg.max_obs_constraints)  # select points with the lower distances
        idx   = np.argsort(dist_close)[:n_sel]  # take indices of closest points 
        return pts_close[idx]

    # ------------------------------------------------------------------
    # Perception update
    # ------------------------------------------------------------------

    def update_grid(self, grid_map: FixedGaussianGridMap) -> None:
        """
        Rebuild the CasADi interpolant from the current occupancy map.
        Call this after grid_map.update() every planning cycle.
        """
        if grid_map.gmap is None:
            self._grid_interp = None
            return

        xw, yw = grid_map.gmap.shape  # grid dimensions (cells)
        x_grid = (np.arange(xw) * grid_map.reso + grid_map.minx).tolist()  # convert the grid array (if 10 cells is [0,1,...,9]) rescaling it by the  
        y_grid = (np.arange(yw) * grid_map.reso + grid_map.miny).tolist()  # resolution, then sum the offset of the grid minx, finally convert to NumPy array
        data_flat = grid_map.gmap.ravel(order='F').tolist()   # flatten the final 2D array (x and y) following the column-major syntax expected by CasADi

        self._grid_interp = ca.interpolant(
            'obs_cost', 'bspline', [x_grid, y_grid], data_flat  # build a CasADi bspline interpolant of the occupancy grid, it is a function differentiable 2-times  

        )

    # ------------------------------------------------------------------
    # Reference trajectory from A* path
    # ------------------------------------------------------------------

    def _build_reference(
        self,
        drone_state: np.ndarray,
        path_world: list,
        z_ref: float,
    ) -> np.ndarray:
        """
        Build an (N+1, NX) reference trajectory by advancing along the A* path
        at v_ref m/s from the closest waypoint.

        Parameters
        ----------
        drone_state : (7,) [px, py, pz, vx, vy, vz, yaw]
        path_world  : list of (x, y) or (x, y, z) tuples from A*
        z_ref       : fallback constant altitude when path has no z component

        Returns
        -------
        x_ref : (N+1, NX)
        """
        N, dt, v_ref = self.cfg.N, self.cfg.dt, self.cfg.v_ref
        x_ref = np.zeros((N + 1, self.NX))  # initialise state array

        if not path_world or len(path_world) < 2:
            # No path: hold current pose
            for k in range(N + 1):
                x_ref[k] = drone_state
            return x_ref

        path = np.array(path_world, dtype=float)  # (M, 2) or (M, 3)
        has_z = path.shape[1] >= 3

        path_xy = path[:, :2]   # (M, 2)

        # Arc-length parameterisation along horizontal plane
        diffs_xy = np.diff(path_xy, axis=0)   # vectors between consecutive waypoints (M-1, 2)
        seg_len  = np.hypot(diffs_xy[:, 0], diffs_xy[:, 1])  # euclidean distances between waypoints (M-1,)
        arc      = np.concatenate([[0.0], np.cumsum(seg_len)])  # cumulative arc length (0, s1, s1+s2, ...)
        total_arc = arc[-1]  # total path length

        if has_z:
            diffs_z = np.diff(path[:, 2])  # (M-1,)

        # Find closest waypoint to current drone position (horizontal only)
        drone_xy = drone_state[:2]
        i_closest = int(np.argmin(np.linalg.norm(path_xy - drone_xy, axis=1)))  # index of closest waypoint wrt the drone
        s0 = arc[i_closest]  # starting point

        for k in range(N + 1):
            s_k = min(s0 + v_ref * k * dt, total_arc)  # proceed along the arc at v_ref = constant 

            idx = int(np.searchsorted(arc, s_k, side='right')) - 1  # find the index for s_k
            idx = np.clip(idx, 0, len(path_xy) - 2)  # ensure idx is within valid range (not out of bounds)

            seg_l = seg_len[idx]  # length of the current segment
            t = (s_k - arc[idx]) / (seg_l + 1e-9)  # normalised position along the segment, add small epsilon to avoid division by zero
            t = np.clip(t, 0.0, 1.0)

            # Reference x, y
            pos_xy = path_xy[idx] + t * diffs_xy[idx]  # linear interpolation between the two consecutive waypoints idx and idx+1
            seg_dir = diffs_xy[idx] / (seg_l + 1e-9)  # unit direction vector of the segment
            yaw_k = np.arctan2(seg_dir[1], seg_dir[0])  # yaw aligned with the segment direction

            # Reference z
            if has_z:
                ref_z = path[idx, 2] + t * diffs_z[idx]  # linear interpolation of z between the two consecutive waypoints
            else:
                ref_z = z_ref

            x_ref[k, 0] = pos_xy[0]            # x
            x_ref[k, 1] = pos_xy[1]            # y
            x_ref[k, 2] = ref_z                # z
            x_ref[k, 3] = seg_dir[0] * v_ref   # vx
            x_ref[k, 4] = seg_dir[1] * v_ref   # vy
            x_ref[k, 5] = 0.0                  # vz reference is zero (cruise)
            x_ref[k, 6] = yaw_k                # yaw

        return x_ref

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        drone_state: np.ndarray,
        path_world: list,
        z_ref: float = 1.5,
        obstacle_points_2d: Optional[np.ndarray] = None,
    ) -> MPCResult:
        """
        Formulate and solve the MPC optimisation.

        Parameters
        ----------
        drone_state        : (7,) [px, py, pz, vx, vy, vz, yaw]
        path_world         : list of (x, y) or (x, y, z) waypoints from A*
        z_ref              : target altitude [m] — used when path has no z component
        obstacle_points_2d : (M, 2) raw LiDAR obstacle positions in world x-y.
                             Used to add half-space soft penalties that push the
                             predicted trajectory away from nearby obstacles.

        Returns
        -------
        MPCResult
        """
        t0 = time.perf_counter()
        cfg = self.cfg
        N, dt = cfg.N, cfg.dt
        NX, NU = self.NX, self.NU

        x0 = np.asarray(drone_state, dtype=float)
        x_ref = self._build_reference(x0, path_world, z_ref)  # reference trajectory from A* path

        # Select nearby LiDAR points for half-space penalty
        drone_xy_np = x0[:2]
        if obstacle_points_2d is not None and len(obstacle_points_2d) > 0:
            obs_pts = self._select_constraint_points(obstacle_points_2d, drone_xy_np)
        else:
            obs_pts = np.empty((0, 2))

        # Precompute outward normals (obstacle → drone, pointing away from each obstacle)
        # n_i = (drone_xy - pt_i) / ||drone_xy - pt_i||
        half_space = []   # list of (ox, oy, nx, ny)
        for pt in obs_pts:
            diff = drone_xy_np - pt
            d = float(np.linalg.norm(diff))
            if d < 1e-6:
                continue  # skip points that are too close to the drone (due to sensor noise)
            half_space.append((float(pt[0]), float(pt[1]), diff[0] / d, diff[1] / d))

        # ── CasADi Opti ──────────────────────────────────────────────
        opti = ca.Opti()
        X = opti.variable(NX, N + 1)   # states — columns are time steps
        U = opti.variable(NU, N)       # controls (for each time step)

        # Weight matrices
        q = np.array([
            cfg.Q_xy, cfg.Q_xy, cfg.Q_z,
            cfg.Q_vel_xy, cfg.Q_vel_xy, cfg.Q_vel_z,
            cfg.Q_yaw,
        ])
        Q   = np.diag(q)
        Q_T = np.diag(q * cfg.Q_terminal)
        R   = np.diag([cfg.R_acc_xy, cfg.R_acc_xy, cfg.R_acc_z, cfg.R_yaw_rate])

        # ── Objective ────────────────────────────────────────────────
        cost = 0.0

        for k in range(N):
            # State tracking
            e = X[:, k] - x_ref[k]  # penalize distance from reference trajectory
            cost += ca.mtimes([e.T, Q, e])

            # Control effort
            u_k = U[:, k]  # penalize control effort (acceleration and yaw rate)
            cost += ca.mtimes([u_k.T, R, u_k])

            # Jerk (smoothness)
            if k > 0:
                du = U[:, k] - U[:, k - 1]  # penalize big changes in control inputs
                cost += cfg.R_jerk * ca.dot(du, du)

            # Soft obstacle cost from Gaussian grid map (x-y plane only)
            if self._grid_interp is not None:
                pos_xy = ca.vertcat(X[0, k], X[1, k])  
                cost += cfg.W_obs * self._grid_interp(pos_xy)  # penalize proximity to obstacles for actual position

            # Half-space soft penalty: push predicted state away from each nearby
            # LiDAR point.  For obstacle at (ox, oy) with outward normal (nx, ny):
            #   signed_dist = nx*(X[0,k]-ox) + ny*(X[1,k]-oy)
            # Penalise max(0, d_safe - signed_dist)^2 when the predicted position
            # is closer than d_safe to the obstacle along the normal direction.
            for ox, oy, nx, ny in half_space:
                signed_dist = nx * (X[0, k] - ox) + ny * (X[1, k] - oy)
                cost += cfg.W_obs_pts * ca.fmax(0.0, cfg.d_safe_pts - signed_dist) ** 2  # penalize proximity of next position to any obstacle

        # Terminal cost
        e_T = X[:, N] - x_ref[N]
        cost += ca.mtimes([e_T.T, Q_T, e_T])

        if self._grid_interp is not None:
            pos_xy_T = ca.vertcat(X[0, N], X[1, N])
            cost += cfg.W_obs * self._grid_interp(pos_xy_T)

        # Half-space penalty at terminal step
        for ox, oy, nx, ny in half_space:
            signed_dist_T = nx * (X[0, N] - ox) + ny * (X[1, N] - oy)
            cost += cfg.W_obs_pts * ca.fmax(0.0, cfg.d_safe_pts - signed_dist_T) ** 2

        opti.minimize(cost)

        # ── Dynamics constraints (Euler integration) ──────────────────
        for k in range(N):
            px_k, py_k, pz_k = X[0, k], X[1, k], X[2, k]
            vx_k, vy_k, vz_k = X[3, k], X[4, k], X[5, k]
            yaw_k             = X[6, k]
            ax_k, ay_k, az_k  = U[0, k], U[1, k], U[2, k]
            yr_k              = U[3, k]

            opti.subject_to(X[0, k+1] == px_k  + vx_k * dt + 0.5 * ax_k * dt**2)
            opti.subject_to(X[1, k+1] == py_k  + vy_k * dt + 0.5 * ay_k * dt**2)
            opti.subject_to(X[2, k+1] == pz_k  + vz_k * dt + 0.5 * az_k * dt**2)
            opti.subject_to(X[3, k+1] == vx_k  + ax_k * dt)
            opti.subject_to(X[4, k+1] == vy_k  + ay_k * dt)
            opti.subject_to(X[5, k+1] == vz_k  + az_k * dt)
            opti.subject_to(X[6, k+1] == yaw_k + yr_k * dt)

        # Initial state
        opti.subject_to(X[:, 0] == x0)

        # ── Box constraints on controls (limit accelerations and yaw rate)
        for k in range(N):
            opti.subject_to(opti.bounded(-cfg.a_max_xy,      U[0, k],  cfg.a_max_xy))
            opti.subject_to(opti.bounded(-cfg.a_max_xy,      U[1, k],  cfg.a_max_xy))
            opti.subject_to(opti.bounded(-cfg.a_max_z,       U[2, k],  cfg.a_max_z))
            opti.subject_to(opti.bounded(-cfg.yaw_rate_max,  U[3, k],  cfg.yaw_rate_max))

        # ── Speed constraints ────────────────────────────────────────
        for k in range(N + 1):
            # Horizontal speed
            opti.subject_to(X[3, k]**2 + X[4, k]**2 <= cfg.v_max_xy**2)  # (vx^2 + vy^2) limited to v_max_xy^2
            # Vertical speed
            opti.subject_to(opti.bounded(-cfg.v_max_z, X[5, k], cfg.v_max_z))  # limited to +- v_max_z

        # ── IPOPT solver ─────────────────────────────────────────────
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {
            'max_iter': cfg.max_iter,
            'print_level': cfg.print_level,
            'sb': 'yes',
            'warm_start_init_point': 'yes' if cfg.warm_start else 'no',
        }
        opti.solver('ipopt', p_opts, s_opts)

        # ── Initial guess ────────────────────────────────────────────
        if cfg.warm_start and self._prev_u is not None:
            try:
                opti.set_initial(U, self._prev_u.T)   # in case of warm start available, employ
                opti.set_initial(X, self._prev_x.T)   # previous solutions as initial guess
            except Exception:
                self._default_guess(opti, X, U, x_ref)
        else:
            self._default_guess(opti, X, U, x_ref)  # X = x_ref, U = 0 as default guess

        # ── Solve ────────────────────────────────────────────────────
        try:
            sol = opti.solve()
            success = True
            cost_val = float(sol.value(cost))
        except RuntimeError:
            sol = opti.debug        # best iterate on convergence failure
            success = False
            try:
                cost_val = float(sol.value(cost))
            except Exception:
                cost_val = float('inf')

        # Results
        U_opt  = np.array(sol.value(U))    # (NU, N)
        X_opt  = np.array(sol.value(X))    # (NX, N+1)
        u_seq  = U_opt.T                   # (N,  NU)
        x_pred = X_opt.T                   # (N+1, NX)

        # Shift warm-start buffer by one step
        self._prev_u = np.vstack([u_seq[1:], u_seq[-1:]])    # (N, NU)
        self._prev_x = np.vstack([x_pred[1:], x_pred[-1:]])  # (N+1, NX)

        return MPCResult(
            success=success,
            x_pred=x_pred,
            u_opt=u_seq,
            cost=cost_val,
            solve_time_ms=(time.perf_counter() - t0) * 1e3,
        )

    def _default_guess(self, opti, X, U, x_ref: np.ndarray) -> None:
        opti.set_initial(X, x_ref.T)
        opti.set_initial(U, np.zeros((self.NU, self.cfg.N)))
