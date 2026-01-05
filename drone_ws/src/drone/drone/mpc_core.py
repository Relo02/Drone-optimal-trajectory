"""
MPC Core - Clean Model Predictive Controller for Drone Obstacle Avoidance

This module provides a robust MPC implementation with:
- Quadrotor double-integrator dynamics
- Distance-based obstacle avoidance constraints
- Proper safety margins and emergency behavior
- Efficient CasADi-based optimization

Author: Refactored implementation
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class MPCConfig:
    """MPC configuration parameters."""
    # Horizon
    dt: float = 0.1          # Time step [s]
    N: int = 20              # Prediction horizon steps
    
    # Safety
    safety_radius: float = 0.6    # Minimum distance from obstacles [m]
    emergency_radius: float = 0.3  # Emergency brake distance [m]
    
    # Limits
    v_max: float = 2.0       # Max velocity [m/s]
    a_max: float = 2.5       # Max acceleration [m/s²]
    yaw_rate_max: float = 1.0    # Max yaw rate [rad/s]
    yaw_accel_max: float = 2.0   # Max yaw acceleration [rad/s²]
    z_min: float = 0.3       # Min altitude [m]
    z_max: float = 10.0      # Max altitude [m]
    
    # Cost weights
    Q_pos: float = 50.0      # Position tracking weight
    Q_goal: float = 100.0    # Goal attraction weight
    Q_vel: float = 1.0       # Velocity regularization
    Q_yaw: float = 2.0       # Yaw tracking weight
    R_acc: float = 0.1       # Acceleration effort weight
    R_yaw_acc: float = 0.5   # Yaw acceleration effort weight
    Q_terminal: float = 200.0  # Terminal position weight
    
    # Obstacle avoidance
    obstacle_weight: float = 1000.0  # Slack penalty for obstacle constraints
    max_obstacles: int = 15   # Max obstacles to consider per step
    obstacle_range: float = 5.0  # Max range to consider obstacles [m]


@dataclass 
class MPCState:
    """Drone state vector."""
    position: np.ndarray      # [x, y, z] in world frame
    velocity: np.ndarray      # [vx, vy, vz] in world frame
    yaw: float               # Heading angle [rad]
    yaw_rate: float          # Yaw rate [rad/s]
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat state vector [px, py, pz, yaw, vx, vy, vz, yaw_dot]."""
        return np.array([
            self.position[0], self.position[1], self.position[2],
            self.yaw,
            self.velocity[0], self.velocity[1], self.velocity[2],
            self.yaw_rate
        ], dtype=float)
    
    @classmethod
    def from_vector(cls, x: np.ndarray) -> 'MPCState':
        """Create from flat state vector."""
        return cls(
            position=np.array([x[0], x[1], x[2]]),
            velocity=np.array([x[4], x[5], x[6]]),
            yaw=float(x[3]),
            yaw_rate=float(x[7])
        )


@dataclass
class MPCResult:
    """MPC solution result."""
    success: bool
    acceleration: np.ndarray   # [ax, ay, az] command
    yaw_acceleration: float    # Yaw acceleration command
    predicted_trajectory: np.ndarray  # (N+1, 8) state trajectory
    predicted_controls: np.ndarray    # (N, 4) control trajectory
    cost: float
    solve_time_ms: float
    status: str
    emergency_stop: bool = False


class ObstacleSet:
    """Efficient obstacle representation for MPC constraints."""
    
    def __init__(self, points: np.ndarray, config: MPCConfig):
        """
        Initialize obstacle set from point cloud.
        
        Args:
            points: (M, 3) array of obstacle positions in world frame
            config: MPC configuration
        """
        self.config = config
        self.all_points = np.asarray(points).reshape(-1, 3) if points.size > 0 else np.zeros((0, 3))
        
    def get_relevant_obstacles(self, position: np.ndarray, direction: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get obstacles relevant for constraint generation.
        
        Uses a combination of:
        1. Distance-based selection (closest obstacles)
        2. Angular coverage (ensure 360° coverage)
        3. Direction weighting (prioritize obstacles in movement direction)
        
        Args:
            position: Current/reference position
            direction: Optional movement direction for prioritization
            
        Returns:
            (K, 3) array of selected obstacle positions
        """
        if self.all_points.shape[0] == 0:
            return np.zeros((0, 3))
        
        # Filter by range
        rel_pos = self.all_points - position
        distances = np.linalg.norm(rel_pos, axis=1)
        in_range = distances <= self.config.obstacle_range
        
        if not np.any(in_range):
            return np.zeros((0, 3))
        
        candidates = self.all_points[in_range]
        rel_pos = rel_pos[in_range]
        distances = distances[in_range]
        
        if candidates.shape[0] <= self.config.max_obstacles:
            return candidates
        
        # Score-based selection
        scores = np.zeros(candidates.shape[0])
        
        # 1. Distance score (closer = higher priority)
        scores += 1.0 / (distances + 0.1)
        
        # 2. Angular coverage - ensure we don't miss any direction
        angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        n_sectors = 8
        sector_size = 2 * np.pi / n_sectors
        
        selected_mask = np.zeros(candidates.shape[0], dtype=bool)
        
        # First pass: select closest in each sector
        for i in range(n_sectors):
            sector_min = -np.pi + i * sector_size
            sector_max = sector_min + sector_size
            in_sector = (angles >= sector_min) & (angles < sector_max)
            
            if np.any(in_sector):
                sector_indices = np.where(in_sector)[0]
                closest_idx = sector_indices[np.argmin(distances[sector_indices])]
                selected_mask[closest_idx] = True
        
        # Second pass: fill remaining slots with highest-scoring obstacles
        remaining_slots = self.config.max_obstacles - np.sum(selected_mask)
        if remaining_slots > 0:
            unselected_scores = scores.copy()
            unselected_scores[selected_mask] = -np.inf
            top_indices = np.argsort(unselected_scores)[-remaining_slots:]
            selected_mask[top_indices] = True
        
        return candidates[selected_mask]
    
    def check_emergency(self, position: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if emergency braking is needed.
        
        Returns:
            (emergency_needed, escape_direction)
        """
        if self.all_points.shape[0] == 0:
            return False, None
        
        rel_pos = self.all_points - position
        distances = np.linalg.norm(rel_pos, axis=1)
        min_dist = np.min(distances)
        
        if min_dist < self.config.emergency_radius:
            # Find escape direction (away from closest obstacle)
            closest_idx = np.argmin(distances)
            escape_dir = -rel_pos[closest_idx]
            escape_dir = escape_dir / (np.linalg.norm(escape_dir) + 1e-6)
            return True, escape_dir
        
        return False, None


class MPCSolver:
    """
    CasADi-based MPC solver for obstacle avoidance.
    
    Uses a double-integrator model with soft obstacle constraints.
    """
    
    def __init__(self, config: MPCConfig):
        self.config = config
        self._solver = None
        self._last_solution = None
        
    def _build_dynamics(self) -> ca.Function:
        """Build discrete-time dynamics function."""
        # State: [px, py, pz, yaw, vx, vy, vz, yaw_dot]
        x = ca.MX.sym('x', 8)
        # Control: [ax, ay, az, yaw_ddot]
        u = ca.MX.sym('u', 4)
        
        dt = self.config.dt
        
        # Double integrator dynamics
        x_next = ca.vertcat(
            x[0] + dt * x[4],           # px
            x[1] + dt * x[5],           # py
            x[2] + dt * x[6],           # pz
            x[3] + dt * x[7],           # yaw
            x[4] + dt * u[0],           # vx
            x[5] + dt * u[1],           # vy
            x[6] + dt * u[2],           # vz
            x[7] + dt * u[3],           # yaw_dot
        )
        
        return ca.Function('dynamics', [x, u], [x_next])
    
    def solve(
        self,
        state: MPCState,
        goal: np.ndarray,
        obstacles: ObstacleSet,
        reference_trajectory: Optional[np.ndarray] = None,
        yaw_reference: Optional[np.ndarray] = None,
    ) -> MPCResult:
        """
        Solve MPC optimization problem.
        
        Args:
            state: Current drone state
            goal: Goal position [x, y, z]
            obstacles: Obstacle set for avoidance
            reference_trajectory: Optional (N, 3) reference positions
            yaw_reference: Optional (N,) reference yaw angles
            
        Returns:
            MPCResult with solution
        """
        import time
        start_time = time.perf_counter()
        
        cfg = self.config
        N = cfg.N
        
        x0 = state.to_vector()
        goal = np.asarray(goal).flatten()[:3]
        
        # Check for emergency
        emergency, escape_dir = obstacles.check_emergency(state.position)
        if emergency:
            # Emergency braking with escape direction
            brake_acc = -state.velocity * 2.0  # Proportional braking
            if escape_dir is not None:
                brake_acc[:3] += escape_dir * cfg.a_max * 0.5
            brake_acc = np.clip(brake_acc, -cfg.a_max, cfg.a_max)
            
            return MPCResult(
                success=True,
                acceleration=brake_acc[:3],
                yaw_acceleration=0.0,
                predicted_trajectory=np.tile(x0, (N+1, 1)),
                predicted_controls=np.zeros((N, 4)),
                cost=0.0,
                solve_time_ms=(time.perf_counter() - start_time) * 1000,
                status="EMERGENCY_BRAKE",
                emergency_stop=True
            )
        
        # Build reference if not provided
        if reference_trajectory is None:
            # Try warm-start from previous solution first (best option)
            reference_trajectory = self._build_reference_from_previous(x0, goal)
            
            if reference_trajectory is None:
                # Use obstacle-aware reference (potential field)
                obs_points = obstacles.all_points if obstacles.all_points.shape[0] > 0 else np.zeros((0, 3))
                reference_trajectory = self._build_obstacle_aware_reference(
                    state.position, goal, obs_points
                )
        
        if yaw_reference is None:
            # Point toward goal
            goal_dir = goal[:2] - state.position[:2]
            if np.linalg.norm(goal_dir) > 0.1:
                target_yaw = np.arctan2(goal_dir[1], goal_dir[0])
            else:
                target_yaw = state.yaw
            yaw_reference = np.full(N, target_yaw)
        
        # Get relevant obstacles for each horizon step
        obstacle_sets = []
        for k in range(N):
            ref_pos = reference_trajectory[k] if k < reference_trajectory.shape[0] else goal
            obs_k = obstacles.get_relevant_obstacles(ref_pos)
            obstacle_sets.append(obs_k)
        
        # Build and solve NLP
        try:
            result = self._solve_nlp(x0, goal, reference_trajectory, yaw_reference, obstacle_sets)
        except Exception as e:
            # Fallback: simple braking
            brake_acc = -state.velocity * 0.5
            brake_acc = np.clip(brake_acc, -cfg.a_max, cfg.a_max)
            
            return MPCResult(
                success=False,
                acceleration=brake_acc[:3] if len(brake_acc) >= 3 else np.zeros(3),
                yaw_acceleration=0.0,
                predicted_trajectory=np.tile(x0, (N+1, 1)),
                predicted_controls=np.zeros((N, 4)),
                cost=0.0,
                solve_time_ms=(time.perf_counter() - start_time) * 1000,
                status=f"SOLVER_ERROR: {str(e)}"
            )
        
        result.solve_time_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    def _build_straight_line_reference(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Build straight-line reference trajectory (fallback, ignores obstacles)."""
        N = self.config.N
        dt = self.config.dt
        
        direction = goal - start
        dist = np.linalg.norm(direction)
        
        if dist < 0.1:
            return np.tile(goal, (N, 1))
        
        direction = direction / dist
        speed = min(self.config.v_max * 0.8, dist / (N * dt))
        
        ref = np.zeros((N, 3))
        for k in range(N):
            t = (k + 1) * dt
            pos = start + direction * speed * t
            if np.linalg.norm(pos - start) >= dist:
                ref[k:] = goal
                break
            ref[k] = pos
        
        return ref
    
    def _build_obstacle_aware_reference(
        self, 
        start: np.ndarray, 
        goal: np.ndarray, 
        obstacles: np.ndarray,
        repulsion_gain: float = 1.5,
        attraction_gain: float = 1.0,
    ) -> np.ndarray:
        """
        Build reference trajectory using artificial potential field.
        
        This creates a reference that bends around obstacles instead of
        going straight through them - critical for walled environments.
        
        Args:
            start: Starting position
            goal: Goal position
            obstacles: (M, 3) obstacle positions
            repulsion_gain: Strength of obstacle repulsion
            attraction_gain: Strength of goal attraction
            
        Returns:
            (N, 3) reference trajectory
        """
        N = self.config.N
        dt = self.config.dt
        cfg = self.config
        
        ref = np.zeros((N, 3))
        pos = start.copy()
        
        goal_dist = np.linalg.norm(goal - start)
        if goal_dist < 0.1:
            return np.tile(goal, (N, 1))
        
        # Desired speed
        speed = min(cfg.v_max * 0.7, goal_dist / (N * dt))
        
        for k in range(N):
            # Attractive force toward goal
            to_goal = goal - pos
            dist_to_goal = np.linalg.norm(to_goal)
            
            if dist_to_goal < 0.1:
                ref[k:] = goal
                break
            
            f_attractive = attraction_gain * to_goal / dist_to_goal
            
            # Repulsive force from obstacles
            f_repulsive = np.zeros(3)
            
            if obstacles.shape[0] > 0:
                rel_pos = pos - obstacles  # Vector from obstacle to pos
                distances = np.linalg.norm(rel_pos, axis=1)
                
                # Only consider nearby obstacles
                influence_radius = cfg.safety_radius * 4.0
                
                for i, d in enumerate(distances):
                    if d < influence_radius and d > 0.01:
                        # Repulsive force: stronger when closer
                        # F = gain * (1/d - 1/d0) * (1/d^2) * direction
                        strength = repulsion_gain * (1.0/d - 1.0/influence_radius) * (1.0/(d*d))
                        direction = rel_pos[i] / d
                        f_repulsive += strength * direction
            
            # Combine forces
            f_total = f_attractive + f_repulsive
            f_norm = np.linalg.norm(f_total)
            
            if f_norm > 0.01:
                # Move in direction of total force
                velocity = (f_total / f_norm) * speed
            else:
                # Fallback: move toward goal
                velocity = (to_goal / dist_to_goal) * speed
            
            # Integrate position
            pos = pos + velocity * dt
            
            # Clamp altitude
            pos[2] = np.clip(pos[2], cfg.z_min + 0.1, cfg.z_max - 0.1)
            
            ref[k] = pos
            
            # Check if reached goal
            if np.linalg.norm(pos - goal) < 0.2:
                ref[k:] = goal
                break
        
        return ref
    
    def _build_reference_from_previous(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Build reference from previous MPC solution (warm start).
        
        This is often the best reference as it's already obstacle-aware
        from the previous optimization.
        
        Returns:
            (N, 3) reference trajectory or None if not available
        """
        if self._last_solution is None:
            return None
        
        X_prev, _ = self._last_solution
        N = self.config.N
        
        # Check if previous solution is still relevant
        prev_start = X_prev[0, :3]
        start_error = np.linalg.norm(prev_start - x0[:3])
        
        if start_error > 1.0:  # Too far from previous trajectory
            return None
        
        # Shift trajectory forward by one step
        ref = np.zeros((N, 3))
        ref[:N-1] = X_prev[2:N+1, :3]  # Shift by 1 step
        
        # Extend last part toward goal
        last_pos = ref[N-2]
        to_goal = goal - last_pos
        dist = np.linalg.norm(to_goal)
        
        if dist > 0.1:
            ref[N-1] = last_pos + (to_goal / dist) * min(0.5, dist)
        else:
            ref[N-1] = goal
        
        return ref
    
    def _solve_nlp(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        p_ref: np.ndarray,
        yaw_ref: np.ndarray,
        obstacle_sets: List[np.ndarray],
    ) -> MPCResult:
        """Build and solve the NLP."""
        cfg = self.config
        N = cfg.N
        
        # Decision variables
        X = ca.MX.sym('X', N + 1, 8)  # States
        U = ca.MX.sym('U', N, 4)       # Controls
        
        # Slack variables for obstacle constraints
        n_slacks = sum(obs.shape[0] for obs in obstacle_sets)
        S = ca.MX.sym('S', n_slacks) if n_slacks > 0 else ca.MX.sym('S', 1)
        
        # Build dynamics function
        dynamics = self._build_dynamics()
        
        # Constraints and cost
        g = []  # Constraints
        lbg = []
        ubg = []
        J = 0   # Cost
        
        # Initial condition
        g.append(X[0, :].T - ca.DM(x0))
        lbg += [0.0] * 8
        ubg += [0.0] * 8
        
        slack_idx = 0
        
        for k in range(N):
            xk = X[k, :].T
            uk = U[k, :].T
            xk_next = X[k + 1, :].T
            
            # Dynamics constraint
            x_pred = dynamics(xk, uk)
            g.append(xk_next - x_pred)
            lbg += [0.0] * 8
            ubg += [0.0] * 8
            
            # Extract state components
            pk = xk[0:3]
            yawk = xk[3]
            vk = xk[4:7]
            
            # Reference for this step
            p_ref_k = ca.DM(p_ref[min(k, p_ref.shape[0]-1)])
            yaw_ref_k = float(yaw_ref[min(k, len(yaw_ref)-1)])
            
            # Stage cost
            J += cfg.Q_pos * ca.sumsqr(pk - p_ref_k)
            J += cfg.Q_goal * ca.sumsqr(pk - ca.DM(goal))
            J += cfg.Q_vel * ca.sumsqr(vk)
            J += cfg.Q_yaw * ca.sumsqr(yawk - yaw_ref_k)
            J += cfg.R_acc * ca.sumsqr(uk[0:3])
            J += cfg.R_yaw_acc * ca.sumsqr(uk[3])
            
            # Altitude constraint
            g.append(pk[2])
            lbg.append(cfg.z_min)
            ubg.append(cfg.z_max)
            
            # Obstacle avoidance constraints
            obs_k = obstacle_sets[k]
            for i in range(obs_k.shape[0]):
                obs_pos = ca.DM(obs_k[i])
                
                # Distance squared (avoid sqrt for better numerics)
                dist_sq = ca.sumsqr(pk - obs_pos)
                min_dist_sq = cfg.safety_radius ** 2
                
                # Soft constraint: dist_sq >= min_dist_sq - slack
                s_i = S[slack_idx] if n_slacks > 0 else 0
                g.append(min_dist_sq - dist_sq - s_i)
                lbg.append(-ca.inf)
                ubg.append(0.0)
                
                # Slack penalty
                if n_slacks > 0:
                    J += cfg.obstacle_weight * s_i * s_i
                    slack_idx += 1
        
        # Terminal cost
        pk_N = X[N, 0:3].T
        J += cfg.Q_terminal * ca.sumsqr(pk_N - ca.DM(goal))
        
        # Terminal altitude constraint
        g.append(X[N, 2])
        lbg.append(cfg.z_min)
        ubg.append(cfg.z_max)
        
        # Stack decision variables
        w = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1),
            S if n_slacks > 0 else ca.DM([])
        )
        
        # Variable bounds
        lbx, ubx = self._build_variable_bounds(N, n_slacks)
        
        # Build NLP
        nlp = {'x': w, 'f': J, 'g': ca.vertcat(*g)}
        
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 80,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive',
        }
        
        solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        
        # Initial guess (warm start from previous solution or straight line)
        w0 = self._build_initial_guess(x0, goal, p_ref, yaw_ref, N, n_slacks)
        
        # Solve
        sol = solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        stats = solver.stats()
        
        # Extract solution
        w_opt = np.array(sol['x']).flatten()
        
        X_opt = w_opt[:(N+1)*8].reshape((N+1, 8), order='F')
        U_opt = w_opt[(N+1)*8:(N+1)*8 + N*4].reshape((N, 4), order='F')
        
        # Store for warm starting
        self._last_solution = (X_opt, U_opt)
        
        return MPCResult(
            success=stats['success'],
            acceleration=U_opt[0, :3],
            yaw_acceleration=float(U_opt[0, 3]),
            predicted_trajectory=X_opt,
            predicted_controls=U_opt,
            cost=float(sol['f']),
            solve_time_ms=0.0,  # Will be filled by caller
            status=stats.get('return_status', 'unknown')
        )
    
    def _build_variable_bounds(self, N: int, n_slacks: int) -> Tuple[List, List]:
        """Build variable bounds for the NLP."""
        cfg = self.config
        lbx = []
        ubx = []
        
        # State bounds (column-major order)
        # px, py
        lbx += [-1000.0] * (N + 1) * 2
        ubx += [1000.0] * (N + 1) * 2
        # pz
        lbx += [cfg.z_min] * (N + 1)
        ubx += [cfg.z_max] * (N + 1)
        # yaw
        lbx += [-1000.0] * (N + 1)
        ubx += [1000.0] * (N + 1)
        # vx, vy, vz
        lbx += [-cfg.v_max] * (N + 1) * 3
        ubx += [cfg.v_max] * (N + 1) * 3
        # yaw_dot
        lbx += [-cfg.yaw_rate_max] * (N + 1)
        ubx += [cfg.yaw_rate_max] * (N + 1)
        
        # Control bounds (column-major order)
        # ax, ay, az
        lbx += [-cfg.a_max] * N * 3
        ubx += [cfg.a_max] * N * 3
        # yaw_ddot
        lbx += [-cfg.yaw_accel_max] * N
        ubx += [cfg.yaw_accel_max] * N
        
        # Slack bounds (non-negative)
        lbx += [0.0] * n_slacks
        ubx += [1000.0] * n_slacks  # Upper bound on slack
        
        return lbx, ubx
    
    def _build_initial_guess(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        p_ref: np.ndarray,
        yaw_ref: np.ndarray,
        N: int,
        n_slacks: int
    ) -> np.ndarray:
        """Build initial guess for warm starting."""
        cfg = self.config
        
        # Use previous solution if available
        if self._last_solution is not None:
            X_prev, U_prev = self._last_solution
            # Shift forward
            X0 = np.zeros((N + 1, 8))
            X0[0] = x0
            X0[1:N] = X_prev[2:N+1]
            X0[N] = X_prev[N]
            
            U0 = np.zeros((N, 4))
            U0[:N-1] = U_prev[1:]
            U0[N-1] = U_prev[N-1]
        else:
            # Build from reference
            X0 = np.zeros((N + 1, 8))
            X0[0] = x0
            
            for k in range(N):
                X0[k + 1, :3] = p_ref[min(k, p_ref.shape[0]-1)]
                X0[k + 1, 3] = yaw_ref[min(k, len(yaw_ref)-1)]
                if k < p_ref.shape[0] - 1:
                    X0[k + 1, 4:7] = (p_ref[k + 1] - p_ref[k]) / cfg.dt
            
            U0 = np.zeros((N, 4))
        
        # Stack
        w0 = np.concatenate([
            X0.flatten(order='F'),
            U0.flatten(order='F'),
            np.zeros(n_slacks)
        ])
        
        return w0
