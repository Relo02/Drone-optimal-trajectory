"""
Ground / legged instantiation: holonomic kinematic model on SE(2).

    x = [px, py, psi]            (NX = 3)
    u = [vx, vy, omega]          (NU = 3)   body-frame velocities

Relative degree 1 between the planned output (CoM position) and the commanded
input: the gait controller accepts velocity commands and tracks them on a time
scale far faster than the planner, so modelling the extra integrator would add
variables without adding information.

Discretisation
--------------
Unlike the double integrator, this model is genuinely nonlinear (the rotation
R(psi) multiplies the input), so no exact discretisation exists for a
piecewise-constant input.  Two schemes are offered:

  'euler'     p_{k+1} = p_k + R(psi_k) v_k dt
              Forward Euler, local error O(dt^2), global O(dt).
              The heading is evaluated at the START of the interval, so the
              displacement is systematically misaligned by omega*dt/2.
              With omega = 1.5 rad/s and dt = 0.1 s that is 0.075 rad per step.

  'midpoint'  psi_bar = psi_k + 1/2 omega_k dt
              p_{k+1} = p_k + R(psi_bar) v_k dt
              Second-order Runge-Kutta (mid-point rule, eq. 2.10 of the course
              notes).  Local error O(dt^3), global O(dt^2), at the cost of one
              extra addition — the computational overhead is nil because the
              heading update is needed anyway.

'midpoint' is the default.  Keeping 'euler' selectable is deliberate: it makes
the discretisation-order experiment a configuration switch rather than a code
change (see trajopt_core.bench).

Nonholonomic platforms
----------------------
A differential-drive or car-like robot fits this same model by shrinking the
admissible set: setting v_max_lat = 0 in ModelLimits removes the lateral degree
of freedom without touching the dynamics.

A biped that must not walk backward fits it the same way, on the other axis:
ModelLimits.v_min_xy = 0.0 makes U_Sigma one-sided in the forward direction
without changing f_Sigma at all — the same "restrict the admissible set, not
the model" pattern, applied to a different coordinate.
"""

from __future__ import annotations

import numpy as np

from trajopt_core.models.base import CostWeights, ModelLimits, MotionModel

try:                     # CasADi is optional for pure-kinematics unit tests
    import casadi as ca
except ImportError:      # pragma: no cover
    ca = None


def _cos(v):
    return ca.cos(v) if (ca is not None and isinstance(v, (ca.MX, ca.SX, ca.DM))) else np.cos(v)


def _sin(v):
    return ca.sin(v) if (ca is not None and isinstance(v, (ca.MX, ca.SX, ca.DM))) else np.sin(v)


class KinematicSE2(MotionModel):
    name = "kinematic_se2"
    NX = 3
    NU = 3
    POS_IDX = (0, 1)
    PLANAR_IDX = (0, 1)
    YAW_IDX = 2
    RELATIVE_DEGREE = 1
    PLANS_ALTITUDE = False

    STATE_LAYOUT = ("px", "py", "psi")
    INPUT_LAYOUT = ("vx", "vy", "omega")

    VALID_INTEGRATORS = ("midpoint", "euler")

    def __init__(self, limits: ModelLimits | None = None, integrator: str = "midpoint"):
        super().__init__(limits)
        if integrator not in self.VALID_INTEGRATORS:
            raise ValueError(
                f"integrator must be one of {self.VALID_INTEGRATORS}, got {integrator!r}"
            )
        self.integrator = integrator
        self.name = f"kinematic_se2[{integrator}]"

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def step(self, x, u, dt: float):
        px, py, psi = x[0], x[1], x[2]
        vx, vy, omega = u[0], u[1], u[2]

        if self.integrator == "midpoint":
            psi_eval = psi + 0.5 * omega * dt      # RK2 / mid-point rule
        else:
            psi_eval = psi                          # forward Euler

        c = _cos(psi_eval)
        s = _sin(psi_eval)

        return [
            px + (vx * c - vy * s) * dt,
            py + (vx * s + vy * c) * dt,
            psi + omega * dt,
        ]

    # ------------------------------------------------------------------
    # Admissible sets
    # ------------------------------------------------------------------
    def input_bounds(self):
        L = self.limits
        lb = np.array([L.resolved_v_min_xy, -L.v_max_lat, -L.yaw_rate_max])
        ub = np.array([L.v_max_xy, L.v_max_lat, L.yaw_rate_max])
        return lb, ub

    # No state constraints: for a relative-degree-1 platform the kinematic
    # limits live entirely in U_Sigma, which keeps the NLP free of nonlinear
    # inequality constraints on the state.

    # ------------------------------------------------------------------
    # Reference lifting
    # ------------------------------------------------------------------
    def lift_reference(self, p_xy, tangent_xy, v_ref: float, z_ref: float):
        yaw = float(np.arctan2(tangent_xy[1], tangent_xy[0]))
        return np.array([float(p_xy[0]), float(p_xy[1]), yaw], dtype=float)

    # ------------------------------------------------------------------
    # Cost mapping
    # ------------------------------------------------------------------
    def state_weight_vector(self, w: CostWeights) -> np.ndarray:
        return np.array([w.q_pos_xy, w.q_pos_xy, w.q_yaw], dtype=float)

    def input_weight_vector(self, w: CostWeights) -> np.ndarray:
        return np.array([w.r_lin_xy, w.r_lin_xy, w.r_ang], dtype=float)

    # ------------------------------------------------------------------
    # Reference inner loop (simulation only) — the body-frame proportional law
    # that `setpoint_to_cmd_vel_node` applies before handing /cmd_vel to the
    # gait controller.
    # ------------------------------------------------------------------
    def tracking_input(self, x_row, target_pos, target_yaw, gains):
        from trajopt_core.util.angles import wrap_angle

        x_row = np.asarray(x_row, dtype=float)
        psi = float(x_row[2])
        e = np.asarray(target_pos, dtype=float).ravel()[:2] - x_row[:2]

        c, s = np.cos(psi), np.sin(psi)
        e_body = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]])

        u = np.array(
            [
                gains.kp_pos * e_body[0],
                gains.kp_pos * e_body[1],
                gains.kp_yaw * wrap_angle(float(target_yaw) - psi),
            ],
            dtype=float,
        )
        lb, ub = self.input_bounds()
        return np.clip(u, lb, ub)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @staticmethod
    def make_state(position, yaw: float = 0.0) -> np.ndarray:
        p = np.asarray(position, dtype=float).ravel()
        return np.array([p[0], p[1], float(yaw)], dtype=float)

    # ------------------------------------------------------------------
    # Reference solution, used to quantify the discretisation error
    # ------------------------------------------------------------------
    @staticmethod
    def exact_step(x, u, dt: float) -> np.ndarray:
        """
        Closed-form solution for constant body-frame (vx, vy, omega).

        For omega != 0 the trajectory is a circular arc; integrating
        R(psi_0 + omega*t) v over [0, dt] gives, with a = omega*dt,

            Delta_world = (1/omega) * R(psi_0) * [[ sin a,   cos a - 1],
                                                  [1 - cos a, sin a   ]] @ v

        Used only in tests/benchmarks as the ground truth against which the
        Euler and mid-point schemes are compared.
        """
        px, py, psi = float(x[0]), float(x[1]), float(x[2])
        vx, vy, omega = float(u[0]), float(u[1]), float(u[2])

        if abs(omega) < 1e-9:
            c, s = np.cos(psi), np.sin(psi)
            return np.array(
                [px + (vx * c - vy * s) * dt,
                 py + (vx * s + vy * c) * dt,
                 psi],
                dtype=float,
            )

        a = omega * dt
        sa, ca_ = np.sin(a), np.cos(a)
        m = np.array([[sa, ca_ - 1.0], [1.0 - ca_, sa]]) / omega
        rot = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        delta = rot @ (m @ np.array([vx, vy]))
        return np.array([px + delta[0], py + delta[1], psi + a], dtype=float)
