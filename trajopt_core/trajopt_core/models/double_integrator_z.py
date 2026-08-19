"""
Aerial instantiation: 3-D double integrator with yaw.

    x = [px, py, pz, vx, vy, vz, psi]        (NX = 7)
    u = [ax, ay, az, psi_dot]                (NU = 4)

Relative degree 2 between the planned output (CoM position) and the commanded
input, because the inner attitude loop cannot change velocity instantaneously:
the tilt limit imposes a_max ~ g*tan(phi_max), a genuine dynamic constraint that
belongs inside the OCP.

Discretisation
--------------
With acceleration held constant over the interval (zero-order hold, which is
exactly how the input is applied by a digital controller), the double integrator
admits an EXACT discretisation:

    p_{k+1} = p_k + v_k dt + 1/2 a_k dt^2
    v_{k+1} = v_k + a_k dt

This is not forward Euler: the truncation error is identically zero.  Choosing
the model so that discretisation is exact is a modelling decision, not an
accident — see Sec. 2.1.3 of the course notes on Euler vs higher-order schemes.

The heading channel is a single integrator and its Euler step is likewise exact
for a piecewise-constant yaw rate.

Altitude note
-------------
The three position channels are dynamically decoupled and the obstacle cost acts
only on (x, y): this model is a planar planner plus a decoupled altitude channel
tracking a constant reference height.
"""

from __future__ import annotations

import numpy as np

from trajopt_core.models.base import CostWeights, ModelLimits, MotionModel


class DoubleIntegratorZ(MotionModel):
    name = "double_integrator_z"
    NX = 7
    NU = 4
    POS_IDX = (0, 1, 2)
    PLANAR_IDX = (0, 1)
    YAW_IDX = 6
    RELATIVE_DEGREE = 2
    PLANS_ALTITUDE = True

    STATE_LAYOUT = ("px", "py", "pz", "vx", "vy", "vz", "psi")
    INPUT_LAYOUT = ("ax", "ay", "az", "psi_dot")

    Z_IDX = 2
    VEL_IDX = (3, 4, 5)

    def __init__(self, limits: ModelLimits | None = None):
        super().__init__(limits)

    # ------------------------------------------------------------------
    # Dynamics — exact ZOH discretisation of the double integrator
    # ------------------------------------------------------------------
    def step(self, x, u, dt: float):
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        psi = x[6]
        ax, ay, az = u[0], u[1], u[2]
        psi_dot = u[3]

        half_dt2 = 0.5 * dt * dt
        return [
            px + vx * dt + half_dt2 * ax,
            py + vy * dt + half_dt2 * ay,
            pz + vz * dt + half_dt2 * az,
            vx + ax * dt,
            vy + ay * dt,
            vz + az * dt,
            psi + psi_dot * dt,
        ]

    # ------------------------------------------------------------------
    # Admissible sets
    # ------------------------------------------------------------------
    def input_bounds(self):
        L = self.limits
        lb = np.array([-L.a_max_xy, -L.a_max_xy, -L.a_max_z, -L.yaw_rate_max])
        ub = np.array([L.a_max_xy, L.a_max_xy, L.a_max_z, L.yaw_rate_max])
        return lb, ub

    def add_state_constraints(self, opti, X) -> None:
        """
        Horizontal speed limited by a convex quadratic constraint (a disc, not a
        box: an axis-aligned box would allow sqrt(2)*v_max on the diagonal), and
        a box on the vertical speed.
        """
        L = self.limits
        n_steps = X.shape[1]
        for k in range(n_steps):
            opti.subject_to(X[3, k] ** 2 + X[4, k] ** 2 <= L.v_max_xy ** 2)
            opti.subject_to(opti.bounded(-L.v_max_z, X[5, k], L.v_max_z))

    # ------------------------------------------------------------------
    # Reference lifting
    # ------------------------------------------------------------------
    def lift_reference(self, p_xy, tangent_xy, v_ref: float, z_ref: float):
        yaw = float(np.arctan2(tangent_xy[1], tangent_xy[0]))
        return np.array(
            [
                float(p_xy[0]),
                float(p_xy[1]),
                float(z_ref),
                float(tangent_xy[0]) * v_ref,
                float(tangent_xy[1]) * v_ref,
                0.0,          # cruise: no vertical velocity reference
                yaw,
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Cost mapping
    # ------------------------------------------------------------------
    def state_weight_vector(self, w: CostWeights) -> np.ndarray:
        return np.array(
            [w.q_pos_xy, w.q_pos_xy, w.q_pos_z,
             w.q_vel_xy, w.q_vel_xy, w.q_vel_z,
             w.q_yaw],
            dtype=float,
        )

    def input_weight_vector(self, w: CostWeights) -> np.ndarray:
        return np.array([w.r_lin_xy, w.r_lin_xy, w.r_lin_z, w.r_ang], dtype=float)

    # ------------------------------------------------------------------
    # Reference inner loop (simulation only) — a PD on position, mirroring the
    # position/velocity cascade that consumes the TrajectorySetpoint on the
    # real vehicle.  Yaw is relative degree 1 here, so a proportional law is
    # enough for it.
    # ------------------------------------------------------------------
    def tracking_input(self, x_row, target_pos, target_yaw, gains):
        from trajopt_core.util.angles import wrap_angle

        x_row = np.asarray(x_row, dtype=float)
        p = x_row[list(self.POS_IDX)]
        v = x_row[list(self.VEL_IDX)]
        target = np.asarray(target_pos, dtype=float).ravel()

        acc = gains.kp_pos * (target - p) - gains.kd_pos * v
        yaw_rate = gains.kp_yaw * wrap_angle(float(target_yaw) - x_row[self.YAW_IDX])

        u = np.array([acc[0], acc[1], acc[2], yaw_rate], dtype=float)
        lb, ub = self.input_bounds()
        return np.clip(u, lb, ub)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @staticmethod
    def make_state(position, velocity=(0.0, 0.0, 0.0), yaw: float = 0.0) -> np.ndarray:
        p = np.asarray(position, dtype=float).ravel()
        v = np.asarray(velocity, dtype=float).ravel()
        return np.array([p[0], p[1], p[2], v[0], v[1], v[2], float(yaw)], dtype=float)
