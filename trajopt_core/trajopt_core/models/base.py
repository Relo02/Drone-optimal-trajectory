"""
Platform abstraction for the trajectory-optimisation layer.

------------------------------------------------------------------------------
The central claim of this package
------------------------------------------------------------------------------
Every layer of the navigation stack — occupancy grid, A* global planning,
obstacle cost, reference generation, lookahead extraction — depends on the state
ONLY through the planar position of the robot centre of mass,

    p_k = C x_k      (the "output map", MotionModel.PLANAR_IDX)

The robotic platform therefore enters the optimal control problem through
exactly four objects:

    f_Sigma   discrete-time dynamics          -> step()
    U_Sigma   input admissible set            -> input_bounds()
    X_Sigma   state admissible set            -> add_state_constraints()
    C         output map (CoM position)       -> PLANAR_IDX / POS_IDX

Everything else in `trajopt_core.mpc` is written once and reused unchanged.

------------------------------------------------------------------------------
What actually differs between platforms
------------------------------------------------------------------------------
The two instantiations shipped here differ in the RELATIVE DEGREE between the
planned output (CoM position) and the commanded input:

  DoubleIntegratorZ (aerial)   relative degree 2:  p'' = a
      The inner attitude loop cannot change velocity instantaneously; the tilt
      limit imposes a_max ~ g*tan(phi_max), a genuine dynamic constraint that
      must be represented inside the OCP.

  KinematicSE2 (legged/ground) relative degree 1:  p' = R(psi) v
      The gait controller accepts velocity commands and tracks them on a time
      scale far faster than the planner, so the extra integrator carries no
      information for the planner and is omitted.

Validity assumption (state it explicitly in the report): the abstraction holds
as long as the inner loop bandwidth is well above the planner bandwidth — the
classical time-scale separation of hierarchical control architectures.

------------------------------------------------------------------------------
Note on dimensionality
------------------------------------------------------------------------------
Both instantiations plan in the HORIZONTAL PLANE.  The aerial model carries an
additional altitude channel which is dynamically decoupled from (x, y) and does
not participate in obstacle avoidance; its reference is a constant height.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Semantic cost weights, shared by every platform
# ---------------------------------------------------------------------------
@dataclass
class CostWeights:
    """
    Weights expressed in PLATFORM-INDEPENDENT semantic terms.

    Each MotionModel maps these onto its own diagonal Q and R vectors, so a
    single YAML schema drives both robots.  Note that `r_lin_*` / `r_ang` weight
    the *commanded input*, whatever its physical nature: an acceleration for a
    relative-degree-2 platform, a velocity for a relative-degree-1 one.
    """

    # --- state tracking -----------------------------------------------------
    q_pos_xy: float = 30.0      # horizontal position error
    q_pos_z: float = 20.0       # altitude error       (ignored if no z state)
    q_vel_xy: float = 10.0      # horizontal velocity  (ignored if no v state)
    q_vel_z: float = 2.0        # vertical velocity    (ignored if no v state)
    q_yaw: float = 0.2          # heading error

    # --- terminal -----------------------------------------------------------
    q_terminal: float = 10.0    # multiplier applied on top of the stage Q

    # --- control effort -----------------------------------------------------
    r_lin_xy: float = 1.0       # horizontal input effort
    r_lin_z: float = 1.5        # vertical input effort (ignored if planar)
    r_ang: float = 0.1          # angular input effort
    r_jerk: float = 0.3         # penalty on delta-u (smoothness)


@dataclass
class InnerLoopGains:
    """
    Gains of the *reference* inner loop used for closed-loop simulation.

    The deployed stacks do NOT apply the first optimal input to the actuators:
    the MPC publishes a lookahead setpoint and a platform-specific inner loop
    tracks it — a cascaded PID driving motor thrusts on the aerial platform, a
    proportional controller feeding /cmd_vel to the gait controller on the legged
    one.  `MotionModel.tracking_input` reproduces that structure so that the
    simulated closed loop matches the deployed one.

    This is deliberately the only place in the package where a controller
    appears: it is a stand-in for hardware, not part of the optimal control
    problem, and it plays no role when `run_mission(..., closed_loop="direct")`
    applies the optimal input directly.
    """

    kp_pos: float = 1.2
    kd_pos: float = 0.8      # relative-degree-2 platforms only
    kp_yaw: float = 1.5


@dataclass
class ModelLimits:
    """Kinematic / dynamic limits.  Only the entries relevant to a given model
    are consumed by it; the rest are ignored."""

    v_max_xy: float = 2.0       # [m/s]  forward (body-x) speed bound, SE(2) only
    v_min_xy: float | None = None
    #: Backward speed bound, SE(2) only.  None (default) means symmetric,
    #: i.e. -v_max_xy.  A legged or humanoid platform that must not walk
    #: backward sets this to 0.0: the admissible input set U_Sigma becomes
    #: one-sided without touching the dynamics, exactly as a differential-drive
    #: base is expressed by zeroing v_max_lat instead.
    v_max_z: float = 1.0        # [m/s]
    a_max_xy: float = 2.0       # [m/s^2]   relative-degree-2 platforms only
    a_max_z: float = 1.5        # [m/s^2]   relative-degree-2 platforms only
    yaw_rate_max: float = 1.5   # [rad/s]
    v_max_lat: float = 0.5      # [m/s]     lateral (body-y) speed, SE(2) only

    extra: dict = field(default_factory=dict)

    @property
    def resolved_v_min_xy(self) -> float:
        return -self.v_max_xy if self.v_min_xy is None else self.v_min_xy


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class MotionModel(ABC):
    """Abstract platform model consumed by :class:`trajopt_core.mpc.ocp.TrajectoryOCP`."""

    name: str = "abstract"
    NX: int = 0                     # state dimension
    NU: int = 0                     # input dimension
    POS_IDX: tuple = ()             # indices of the CoM position inside x (2 or 3)
    PLANAR_IDX: tuple = (0, 1)      # indices of the (x, y) CoM position inside x
    YAW_IDX: int | None = None      # index of the heading inside x, if any
    RELATIVE_DEGREE: int = 1        # order of the integrator chain output <- input
    PLANS_ALTITUDE: bool = False    # True if z is part of the optimised state

    def __init__(self, limits: ModelLimits | None = None):
        self.limits = limits or ModelLimits()

    # -- dynamics -----------------------------------------------------------
    @abstractmethod
    def step(self, x, u, dt: float):
        """One discrete step x_{k+1} = f(x_k, u_k).  Symbolic (CasADi) or numeric."""

    # -- admissible sets ----------------------------------------------------
    @abstractmethod
    def input_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Box bounds (lb, ub) on the input, each of shape (NU,)."""

    def add_state_constraints(self, opti, X) -> None:
        """
        Optional state constraints (e.g. a speed norm), added to the Opti stack
        for every column of X.  Default: none.
        """
        return None

    # -- reference ----------------------------------------------------------
    @abstractmethod
    def lift_reference(
        self,
        p_xy: np.ndarray,
        tangent_xy: np.ndarray,
        v_ref: float,
        z_ref: float,
    ) -> np.ndarray:
        """
        Lift a PLANAR reference sample to a full state reference of shape (NX,).

        Parameters
        ----------
        p_xy       : (2,) reference position on the A* path
        tangent_xy : (2,) unit tangent of the path at that point
        v_ref      : desired cruise speed [m/s]
        z_ref      : desired altitude [m] (ignored by planar models)
        """

    # -- cost mapping -------------------------------------------------------
    @abstractmethod
    def state_weight_vector(self, w: CostWeights) -> np.ndarray:
        """Diagonal of Q, shape (NX,), built from the semantic weights."""

    @abstractmethod
    def input_weight_vector(self, w: CostWeights) -> np.ndarray:
        """Diagonal of R, shape (NU,), built from the semantic weights."""

    # -- reference inner loop (simulation only) ------------------------------
    def tracking_input(
        self,
        x_row: np.ndarray,
        target_pos: np.ndarray,
        target_yaw: float,
        gains: "InnerLoopGains",
    ) -> np.ndarray:
        """
        Admissible input that drives the platform towards a pose setpoint.

        Stands in for the platform's real inner loop during closed-loop
        simulation; see :class:`InnerLoopGains`.
        """
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------
    def planar_position(self, x):
        """Extract the (x, y) CoM position from a state vector or matrix column."""
        i, j = self.PLANAR_IDX
        return x[i], x[j]

    def position(self, x_row: np.ndarray) -> np.ndarray:
        """Numeric CoM position (2 or 3 components) from a state row."""
        return np.asarray([x_row[i] for i in self.POS_IDX], dtype=float)

    def heading(self, x_row: np.ndarray) -> float:
        """Numeric heading from a state row (0.0 if the model has no heading)."""
        if self.YAW_IDX is None:
            return 0.0
        return float(x_row[self.YAW_IDX])

    def describe(self) -> dict:
        """Summary used to auto-generate the 'instantiation' table of the report."""
        lb, ub = self.input_bounds()
        return {
            "name": self.name,
            "nx": self.NX,
            "nu": self.NU,
            "relative_degree": self.RELATIVE_DEGREE,
            "plans_altitude": self.PLANS_ALTITUDE,
            "state_layout": self.STATE_LAYOUT,
            "input_layout": self.INPUT_LAYOUT,
            "input_lb": np.asarray(lb).tolist(),
            "input_ub": np.asarray(ub).tolist(),
        }

    # Human-readable layouts, overridden by concrete models
    STATE_LAYOUT: tuple = ()
    INPUT_LAYOUT: tuple = ()
