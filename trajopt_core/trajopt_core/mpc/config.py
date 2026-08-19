"""Configuration objects for the trajectory OCP (platform-independent)."""

from __future__ import annotations

from dataclasses import dataclass, field

from trajopt_core.models.base import CostWeights, ModelLimits


@dataclass
class SolverOptions:
    """IPOPT / CasADi options.

    IPOPT is an interior-point method.  The course notes (Sec. 6.2.2) put the
    choice on a quantitative footing: active-set strategies win when the number
    of inequality constraints is small, interior-point ones when it is large.
    Here every horizon step carries up to `max_points` obstacle terms plus the
    input box and the speed constraints, i.e. several hundred inequalities — well
    inside the interior-point regime.
    """

    max_iter: int = 100
    print_level: int = 0
    expand: bool = True             # scalarise the graph (SX) — faster evaluation

    warm_start: bool = True
    #: Shift the previous solution by one step and reuse it as the initial guess.

    cold_start: bool = False
    #: When `warm_start` is off, start from ZERO instead of from the reference
    #: trajectory.  Only useful to establish a baseline: the reference is
    #: available for free and is already an excellent guess, so it is what the
    #: solver falls back to in normal operation.  Quoting a large warm-start
    #: speed-up without saying which of the two baselines it is measured against
    #: is meaningless — the gap to the reference guess and the gap to zero differ
    #: by an order of magnitude.

    warm_start_init_point: bool = True
    tol: float | None = None
    acceptable_tol: float | None = None
    mu_strategy: str | None = None  # e.g. "adaptive"
    extra: dict = field(default_factory=dict)

    def casadi_options(self) -> tuple[dict, dict]:
        p_opts = {"expand": self.expand, "print_time": False}
        s_opts = {
            "max_iter": self.max_iter,
            "print_level": self.print_level,
            "sb": "yes",
            "warm_start_init_point": "yes" if self.warm_start_init_point else "no",
        }
        if self.tol is not None:
            s_opts["tol"] = self.tol
        if self.acceptable_tol is not None:
            s_opts["acceptable_tol"] = self.acceptable_tol
        if self.mu_strategy is not None:
            s_opts["mu_strategy"] = self.mu_strategy
        s_opts.update(self.extra)
        return p_opts, s_opts


@dataclass
class OCPConfig:
    """Horizon, reference and cost settings shared by every platform."""

    # --- horizon ------------------------------------------------------------
    N: int = 30                     # prediction steps
    dt: float = 0.1                 # discretisation step [s]

    # --- reference ----------------------------------------------------------
    v_ref: float = 1.0              # cruise speed along the A* path [m/s]
    z_ref: float = 1.5              # planning height [m] (aerial only)
    anchor_first_to_state: bool = False

    # --- path conditioning ---------------------------------------------------
    path_resample_ds: float = 0.20  # [m] uniform arc-length resampling (0 = off)
    path_smooth_window: int = 5     # moving-average width (0 or <3 = off)
    #: An 8-connected A* can only emit headings that are multiples of 45 deg, so
    #: its tangent — and therefore the yaw reference — flips at the cell scale.
    #: A world-frame relative-degree-2 platform merely tracks that badly; a
    #: body-frame relative-degree-1 platform stalls, because the optimiser
    #: rotates instead of advancing.  Setting both to 0 recovers the raw grid
    #: polyline and is used by the benchmark to quantify the effect.
    #: When True the first reference sample is forced to the current state.  It
    #: removes a discontinuity at the start of the horizon, at the price of
    #: biasing the tracking error term towards zero at k = 0.

    # --- successive convex approximation ------------------------------------
    sca_iterations: int = 1
    #: Number of re-linearisations per control cycle for the half-space obstacle
    #: terms.  With 1 the half-spaces are built around the previous solution (or,
    #: on the very first cycle, around the reference).  That is cheap but it has
    #: a documented failure mode: if the linearisation trajectory pierces an
    #: obstacle, the steps lying beyond it generate normals pointing the opposite
    #: way, and the resulting half-spaces are mutually contradictory — the slack
    #: then saturates at a value no penalty weight can reduce.  Iterating the
    #: linearisation (the standard SCA remedy, and the same idea that underpins
    #: SQP in Sec. 6.3 of the course notes) resolves it at the cost of one extra
    #: solve per iteration.  Terms that carry no linearisation (the grid B-spline
    #: and the isotropic sigmoid barrier) are unaffected.

    # --- terminal ingredients ----------------------------------------------
    terminal_zero_velocity: bool = False
    #: Terminal equality constraint forcing the final predicted state to be an
    #: equilibrium (v_N = 0).  This is the cheapest of the terminal ingredients
    #: discussed in Sec. 7.2.5 of the course notes and it buys nominal recursive
    #: feasibility: a feasible input sequence always exists at the next step (the
    #: tail of the previous one, completed with the equilibrium input).  For a
    #: navigating robot it has a direct physical reading: a safe braking
    #: trajectory always exists inside the horizon.  Only meaningful for
    #: platforms whose state contains a velocity.

    # --- weights and limits -------------------------------------------------
    weights: CostWeights = field(default_factory=CostWeights)
    limits: ModelLimits = field(default_factory=ModelLimits)

    # --- output -------------------------------------------------------------
    lookahead_dist: float = 1.5     # [m] minimum distance of the published setpoint

    # --- solver -------------------------------------------------------------
    solver: SolverOptions = field(default_factory=SolverOptions)

    @property
    def horizon_seconds(self) -> float:
        return self.N * self.dt
