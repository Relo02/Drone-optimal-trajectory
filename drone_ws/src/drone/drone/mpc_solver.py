"""
CasADi MPC with LiDAR-based obstacle avoidance constraints (safe halfspaces + slack)

This example is intentionally "drop-in" as a template:
- preprocess LaserScan -> obstacle points z_i in WORLD frame (Nx3 numpy array)
- provide a reference predicted trajectory p_ref[k] in WORLD frame (N x 3).
- We build linear halfspace constraints per horizon step:
      n^T p_k >= n^T z + r_s
  (with slack to avoid infeasibility)
- We solve with CasADi (IPOPT). Constraints are linear in p_k, but the overall MPC
  can be nonlinear depending on the dynamics.
"""

import numpy as np
import casadi as ca


def build_halfspaces_for_step(
    p_ref: np.ndarray,
    obstacles_world: np.ndarray,
    r_s: float,
    m_planes: int = 8,
    max_range: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build M halfspace constraints for a single step"""
    if obstacles_world.size == 0:
        return np.zeros((0, 3)), np.zeros((0,))

    d = np.linalg.norm(obstacles_world - p_ref[None, :], axis=1)
    mask = d <= max_range
    cand = obstacles_world[mask]
    if cand.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0,))

    d_c = np.linalg.norm(cand - p_ref[None, :], axis=1)
    idx = np.argsort(d_c)[: min(m_planes, cand.shape[0])]
    pts = cand[idx]

    A_rows = []
    b_rows = []

    for z in pts:
        v = p_ref - z
        dist = np.linalg.norm(v)
        if dist < 1e-6:
            continue

        n = v / dist
        A_rows.append(-n.reshape(1, 3))
        b_rows.append(-(float(n @ z) + float(r_s)))

    if len(A_rows) == 0:
        return np.zeros((0, 3)), np.zeros((0,))

    A = np.vstack(A_rows)
    b = np.array(b_rows).reshape(-1)
    return A, b


def build_halfspaces_over_horizon(
    p_ref_traj: np.ndarray,
    obstacles_world: np.ndarray,
    r_s: float,
    m_planes: int = 8,
    max_range: float = 6.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build (A_k, b_k) for each k in 0..N-1"""
    N = p_ref_traj.shape[0]
    out = []
    for k in range(N):
        A_k, b_k = build_halfspaces_for_step(
            p_ref=p_ref_traj[k],
            obstacles_world=obstacles_world,
            r_s=r_s,
            m_planes=m_planes,
            max_range=max_range,
        )
        out.append((A_k, b_k))
    return out


def mpc_solve(
    x0: np.ndarray,
    p_goal: np.ndarray,
    obstacles_world: np.ndarray,
    p_ref_traj: np.ndarray,
    yaw_ref_traj=None,
    dt: float = 0.1,
    N: int = 20,
    r_s: float = 0.5,
    m_planes: int = 3,
    max_obs_range: float = 4.0,
):
    """
    FIXED MPC solver with proper variable bounds
    """
    x0 = np.asarray(x0, dtype=float).reshape(8)
    x0_cas = ca.DM(x0).reshape((8, 1))
    p_goal = np.asarray(p_goal, dtype=float).reshape(3)
    p_goal_cas = ca.DM(p_goal).reshape((3, 1))
    p_ref_traj = np.asarray(p_ref_traj, dtype=float).reshape(N, 3)
    
    if yaw_ref_traj is None:
        yaw_ref_traj = np.full((N,), float(x0[3]))
    yaw_ref_traj = np.asarray(yaw_ref_traj, dtype=float).reshape(N)
    yaw_ref_cas = ca.DM(yaw_ref_traj).reshape((N, 1))
    
    p_ref_cas = ca.DM(p_ref_traj.T).T  # (N, 3)

    # Build halfspace constraints
    halfspaces = build_halfspaces_over_horizon(
        p_ref_traj=p_ref_traj,
        obstacles_world=np.asarray(obstacles_world).reshape(-1, 3) if obstacles_world is not None else np.zeros((0, 3)),
        r_s=r_s,
        m_planes=m_planes,
        max_range=max_obs_range,
    )

    # Decision variables
    X = ca.MX.sym("X", N + 1, 8)
    U = ca.MX.sym("U", N, 4)

    # Slack variables
    S_list = []
    for k in range(N):
        A_k, b_k = halfspaces[k]
        m_k = A_k.shape[0]
        S_list.append(ca.MX.sym(f"S_{k}", m_k))

    # Dynamics
    def step(xk, uk):
        xk = ca.reshape(xk, (8, 1))
        uk = ca.reshape(uk, (4, 1))
        pk = xk[0:3]
        yawk = xk[3]
        vk = xk[4:7]
        yaw_dotk = xk[7]
        ax, ay, az, yaw_ddot = ca.vertsplit(uk)
        pk_next = pk + dt * vk
        yaw_next = yawk + dt * yaw_dotk
        vk_next = vk + dt * ca.vertcat(ax, ay, az)
        yaw_dot_next = yaw_dotk + dt * yaw_ddot
        return ca.vertcat(pk_next, yaw_next, vk_next, yaw_dot_next)

    g = []
    lbg = []
    ubg = []
    J = 0

    # === TUNED WEIGHTS ===
    Qp_ref = 100.0     # Track reference trajectory
    Qp_goal = 150.0    # Pull toward goal
    Qyaw = 0.5        # Yaw tracking
    Qv = 0.5          # Velocity penalty
    Qyaw_rate = 0.3
    Ru = 0.1          # Control effort
    Rr = 0.1
    rho_slack = 2000.0  # Slack penalty

    # === HARD LIMITS ===
    u_max = 2.5       # Max acceleration (reduced from 3.0)
    v_max = 2.0       # Max velocity (reduced from 2.5)
    yaw_rate_max = 1.5  # Reduced
    yaw_accel_max = 2.0  # Reduced
    z_min = 0.0       # Min altitude
    z_max = 5.0      # Max altitude

    # Initial condition constraint
    g.append(X[0, :].T - x0_cas)
    lbg += [0.0] * 8
    ubg += [0.0] * 8

    # Loop over horizon
    for k in range(N):
        xk = X[k, :].T
        uk = U[k, :].T

        # Dynamics constraint
        x_next = step(xk, uk)
        g.append(X[k + 1, :].T - x_next)
        lbg += [0.0] * 8
        ubg += [0.0] * 8

        # Extract states
        pk = xk[0:3]
        yawk = xk[3]
        vk = xk[4:7]
        yaw_dotk = xk[7]
        yaw_ref_k = yaw_ref_cas[k]
        p_ref_k = p_ref_cas[k, :].T
        
        # Progressive goal weight
        # alpha = (k + 1) / N
        # Qp_goal_k = Qp_goal * alpha
        Qp_goal_k = Qp_goal
        
        yaw_err = yawk - yaw_ref_k
        
        # Cost function
        J += (
            Qp_ref * ca.sumsqr(pk - p_ref_k)
            + Qp_goal_k * ca.sumsqr(pk - p_goal_cas)
            + Qyaw * ca.sumsqr(yaw_err)
            + Qv * ca.sumsqr(vk)
            + Qyaw_rate * ca.sumsqr(yaw_dotk)
            + Ru * ca.sumsqr(uk[0:3])
            + Rr * ca.sumsqr(uk[3])
        )

        # === Z ALTITUDE CONSTRAINT (CRITICAL) ===
        g.append(pk[2])
        lbg += [z_min]
        ubg += [z_max]

        # Obstacle avoidance
        A_k, b_k = halfspaces[k]
        m_k = A_k.shape[0]
        if m_k > 0:
            A_cas = ca.DM(A_k)
            b_cas = ca.DM(b_k).reshape((m_k, 1))
            s_k = S_list[k].reshape((m_k, 1))

            g.append(A_cas @ pk.reshape((3, 1)) - b_cas - s_k)
            lbg += [-ca.inf] * m_k
            ubg += [0.0] * m_k

            g.append(s_k)
            lbg += [0.0] * m_k
            ubg += [ca.inf] * m_k

            J += rho_slack * ca.sumsqr(s_k)

    # Terminal cost
    yaw_err_terminal = X[N, 3] - yaw_ref_cas[N - 1]
    J += 100.0 * ca.sumsqr(X[N, 0:3].T - p_goal_cas) + 5.0 * ca.sumsqr(
        yaw_err_terminal
    )

    # Terminal Z constraint
    g.append(X[N, 2])
    lbg += [z_min]
    ubg += [z_max]

    # === STACK VARIABLES WITH PROPER BOUNDS ===
    w = [ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)]
    for s in S_list:
        w.append(ca.reshape(s, -1, 1))
    w = ca.vertcat(*w) # Final decision variable vector

    # === VARIABLE BOUNDS (lbx, ubx) - THE FIX ===
    lbx = []
    ubx = []
    
    # State bounds: [px, py, pz, yaw, vx, vy, vz, yaw_dot]
    for k in range(N + 1):
        lbx += [-ca.inf, -ca.inf, z_min, -ca.inf]  # position + yaw
        ubx += [ca.inf, ca.inf, z_max, ca.inf]
        lbx += [-v_max, -v_max, -v_max, -yaw_rate_max]  # velocities
        ubx += [v_max, v_max, v_max, yaw_rate_max]
    
    # Control bounds: [ax, ay, az, yaw_ddot]
    for k in range(N):
        lbx += [-u_max, -u_max, -u_max, -yaw_accel_max]
        ubx += [u_max, u_max, u_max, yaw_accel_max]
    
    # Slack bounds (non-negative)
    for k in range(N):
        m_k = halfspaces[k][0].shape[0]
        lbx += [0.0] * m_k
        ubx += [ca.inf] * m_k

    nlp = {"x": w, "f": J, "g": ca.vertcat(*g)}

    # Solver with tighter settings
    solver = ca.nlpsol(
        "solver",
        "ipopt",
        nlp,
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 80,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
        },
    )

    # Warm start
    w0 = np.zeros((int(w.size1()),))
    X0 = np.zeros((N + 1, 8))
    X0[0, :] = x0
    
    for k in range(N):
        X0[k + 1, 0:3] = p_ref_traj[k]
        X0[k + 1, 2] = max(X0[k + 1, 2], z_min + 0.1)  # Ensure above ground
        X0[k + 1, 3] = yaw_ref_traj[k]
        if k < N - 1:
            v_desired = (p_ref_traj[k + 1] - p_ref_traj[k]) / dt
            X0[k + 1, 4:7] = np.clip(v_desired, -v_max * 0.5, v_max * 0.5)
        else:
            X0[k + 1, 4:7] = 0.0
        X0[k + 1, 7] = 0.0
    
    offset = 0
    w0[offset : offset + (N + 1) * 8] = X0.reshape(-1, order="F")
    offset += (N + 1) * 8
    offset += N * 4

    # Solve with bounds
    sol = solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    try:
        stats["objective"] = float(sol["f"])
    except Exception:
        stats["objective"] = None

    w_opt = np.array(sol["x"]).reshape(-1)

    # Unpack
    offset = 0
    X_opt = w_opt[offset : offset + (N + 1) * 8].reshape((N + 1, 8), order="F")
    offset += (N + 1) * 8
    U_opt = w_opt[offset : offset + N * 4].reshape((N, 4), order="F")
    
    return U_opt, X_opt, halfspaces, stats
