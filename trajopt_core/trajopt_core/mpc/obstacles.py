"""
Pluggable obstacle-avoidance terms for the trajectory OCP.

Every term is platform-agnostic: it only ever sees the PLANAR position of the
robot centre of mass, so the same objects serve the aerial and the legged
instantiation unchanged.

Why several implementations
---------------------------
The way obstacles enter the optimisation problem is the single most consequential
modelling decision in this stack, and the alternatives differ along axes that the
course material makes precise:

  smoothness      Newton-type solvers (IPOPT) use second derivatives.  A penalty
                  built on max(0, .)^2 is C^1 but NOT C^2: its Hessian jumps
                  every time an obstacle crosses the activation surface.  The
                  logistic barrier is C^infinity everywhere.

  convexity       The raw collision constraint ||p - o|| >= d is non-convex.
                  Linearising it about a reference trajectory yields an AFFINE
                  half-space — a convex constraint — which is the Successive
                  Convex Approximation strategy.

  exactness       A quadratic penalty rho*s^2 leaves a residual violation
                  s* ~ mu*/(2 rho) for any finite weight.  An L1 penalty rho*|s|
                  is EXACT: it drives s* to zero as soon as rho exceeds the
                  Lagrange multiplier of the corresponding hard constraint
                  (Thm 6.3.1 of the course notes).  Implemented with an explicit
                  non-negative slack and a LINEAR cost, so the NLP stays smooth.

Keeping all of them behind one interface turns the corresponding experiments
into configuration switches instead of code forks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

SENTINEL = 1.0e3   # parked position for unused obstacle slots


# ---------------------------------------------------------------------------
# Context handed to the terms once per solve
# ---------------------------------------------------------------------------
@dataclass
class ObstacleContext:
    """Everything the obstacle terms may need at value-setting time."""

    robot_xy: np.ndarray                      # (2,) current CoM planar position
    points_xy: np.ndarray | None = None       # (M, 2) LiDAR hits, world frame
    grid: object | None = None                # FixedGaussianGridMap
    lin_traj: np.ndarray | None = None        # (N+1, 2) linearisation trajectory
    extra: dict = field(default_factory=dict)


def _select_points_per_step(
    points_xy: np.ndarray | None,
    lin_traj: np.ndarray,
    max_points: int,
    check_radius: float,
):
    """
    For every horizon step, pick the `max_points` nearest LiDAR hits within
    `check_radius` of that step's linearisation point.

    Returns
    -------
    obs   : (S, K, 2) selected obstacle positions (sentinel-padded)
    nrm   : (S, K, 2) outward unit normals, from the obstacle toward the
                      linearisation point (the SCA half-space normal)
    mask  : (S, K)    1.0 for a real point, 0.0 for a padded slot

    The mask makes padded slots exactly inert: both the value and the gradient of
    every term are multiplied by it, so the choice of sentinel cannot leak into
    the solution.
    """
    S = lin_traj.shape[0]
    K = max_points
    obs = np.full((S, K, 2), SENTINEL, dtype=float)
    nrm = np.zeros((S, K, 2), dtype=float)
    nrm[:, :, 0] = 1.0
    mask = np.zeros((S, K), dtype=float)

    if points_xy is None or len(points_xy) == 0:
        return obs, nrm, mask

    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return obs, nrm, mask

    for k in range(S):
        q = lin_traj[k]
        d = np.linalg.norm(pts - q, axis=1)
        sel = np.where(d < check_radius)[0]
        if sel.size == 0:
            continue
        order = sel[np.argsort(d[sel])][:K]
        chosen = pts[order]
        n = len(chosen)

        obs[k, :n] = chosen
        diff = q - chosen                       # obstacle -> linearisation point
        norms = np.linalg.norm(diff, axis=1)
        ok = norms > 1e-6
        nrm[k, :n][ok] = diff[ok] / norms[ok, None]
        mask[k, :n] = ok.astype(float)

    return obs, nrm, mask


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class ObstacleTerm(ABC):
    """Base class for an obstacle-avoidance contribution to the OCP."""

    name: str = "abstract"
    #: True if the CasADi graph must be rebuilt whenever the perception input
    #: changes (i.e. the term cannot be expressed with Opti parameters).
    requires_rebuild: bool = False

    def prepare(self, ctx: ObstacleContext) -> None:
        """Called before a (re)build for terms that bake data into the graph."""

    @abstractmethod
    def declare(self, opti, N: int, model) -> None:
        """Create the parameters / variables owned by this term."""

    def stage_cost(self, k: int, p_sym):
        """Cost contribution at horizon step k, given the symbolic planar position."""
        return 0

    def add_constraints(self, opti, k: int, p_sym) -> None:
        """Optional constraints at horizon step k."""

    def global_cost(self):
        """Cost contribution not attached to a specific step (e.g. slack penalty)."""
        return 0

    def set_values(self, opti, ctx: ObstacleContext) -> None:
        """Push the numeric values of this term's parameters."""

    def describe(self) -> dict:
        return {"name": self.name, "requires_rebuild": self.requires_rebuild}


# ---------------------------------------------------------------------------
# 1. Gaussian grid, evaluated through a C^2 B-spline interpolant
# ---------------------------------------------------------------------------
class GaussianGridCost(ObstacleTerm):
    """
    Soft cost read off the Gaussian occupancy grid.

        J_k = W_obs * f_grid(p_k)

    `f_grid` is a tensor-product B-spline interpolant, hence twice continuously
    differentiable: IPOPT gets exact gradients AND an exact Hessian through the
    obstacle field.  A bilinear interpolant would be only C^0 and would break the
    second-order model the solver relies on.

    Parametric mode — available, but NOT the default.  Here is why.
    ---------------------------------------------------------------
    The occupancy grid translates with the robot, so its world-frame knots move
    every cycle, which forces a rebuild of the whole NLP.  That can be avoided by
    placing the knots in the LOCAL frame, where they are constant forever, and
    evaluating the spline at (p_k - origin):

        J_k = W_obs * f_local(p_k - origin;  coefficients)

    with `origin` and `coefficients` both Opti parameters.  It is bit-identical
    to the world-frame interpolant (see tests) and it does keep the graph fixed.

    It is also catastrophically slow.  Measured on the corridor scenario,
    N = 25, 40x40 grid, identical iteration count (9):

        variant             solve      f     grad_f   hess_L
        parametric        13582 ms   2872 ms  6189 ms  5023 ms
        baked (rebuild)      35 ms      0.2     0.6      1.0

    With the coefficients supplied as a symbolic input CasADi cannot exploit the
    spline structure, so every function, gradient and Hessian evaluation
    recomputes the tensor-product basis over all 1600 coefficients — and the term
    is evaluated at each of the N+1 horizon points.  The build-once saving is
    three orders of magnitude smaller than the evaluation penalty it buys.

    The general lesson, worth stating in the report: eliminating a rebuild is
    only a win when the parametrisation leaves the per-evaluation cost intact.
    For the point-based terms in this module it does; for a spline whose
    coefficients become symbolic, it does not.  Hence the defaults:

        grid B-spline  ->  baked coefficients, rebuilt when the map changes
        point barriers ->  parametric, graph built exactly once

    Note that the deployed legged stack had already reached the same conclusion
    empirically, by dropping the grid from the MPC cost and keeping it for A*.
    """

    def __init__(
        self,
        weight: float = 100.0,
        half_width: float = 5.0,
        reso: float = 0.25,
        parametric: bool = False,
        terminal: bool = True,
    ):
        self.weight = float(weight)
        self.half_width = float(half_width)
        self.reso = float(reso)
        self.parametric = bool(parametric)
        self.terminal = bool(terminal)
        self.cells = int(round(2.0 * self.half_width / self.reso))
        self.requires_rebuild = not self.parametric
        self.name = f"grid_bspline[{'param' if self.parametric else 'rebuild'}]"

        self._f = None
        self._p_origin = None
        self._p_coef = None
        self._baked = None       # (axes, coefficients) for rebuild mode

    def prepare(self, ctx: ObstacleContext) -> None:
        if self.parametric or ctx.grid is None or not ctx.grid.is_initialised:
            return
        self._baked = (ctx.grid.local_axes(), ctx.grid.origin(), ctx.grid.coefficients())

    def declare(self, opti, N: int, model) -> None:
        axis = (np.arange(self.cells, dtype=float) * self.reso).tolist()

        if self.parametric:
            self._f = ca.interpolant("grid_cost", "bspline", [axis, axis], 1)
            self._p_origin = opti.parameter(2)
            self._p_coef = opti.parameter(self.cells * self.cells)
        else:
            if self._baked is None:
                # no map yet: a flat zero field keeps the graph well defined
                axes = [axis, axis]
                coef = np.zeros(self.cells * self.cells)
                origin = np.zeros(2)
            else:
                axes, origin, coef = self._baked
            self._f = ca.interpolant("grid_cost", "bspline", axes, coef.tolist())
            self._p_origin = ca.DM(origin)

    def stage_cost(self, k: int, p_sym):
        if self._f is None:
            return 0
        local = p_sym - self._p_origin
        if self.parametric:
            return self.weight * self._f(local, self._p_coef)
        return self.weight * self._f(local)

    def set_values(self, opti, ctx: ObstacleContext) -> None:
        if not self.parametric or ctx.grid is None:
            return
        if not ctx.grid.is_initialised:
            opti.set_value(self._p_origin, ctx.robot_xy - self.half_width)
            opti.set_value(self._p_coef, np.zeros(self.cells * self.cells))
            return
        opti.set_value(self._p_origin, ctx.grid.origin())
        opti.set_value(self._p_coef, ctx.grid.coefficients())

    def describe(self) -> dict:
        d = super().describe()
        d.update(weight=self.weight, cells=self.cells, smoothness="C2")
        return d


# ---------------------------------------------------------------------------
# Shared machinery for the point-based terms
# ---------------------------------------------------------------------------
class _PointBasedTerm(ObstacleTerm):
    """Common parameter layout for terms built on selected LiDAR points."""

    def __init__(self, max_points: int, check_radius: float, per_step: bool):
        self.max_points = int(max_points)
        self.check_radius = float(check_radius)
        self.per_step = bool(per_step)
        self._N = None
        self._p_obs = None
        self._p_nrm = None
        self._p_mask = None

    def declare(self, opti, N: int, model) -> None:
        self._N = N
        S = N + 1
        K = self.max_points
        self._p_obs = opti.parameter(2, S * K)
        self._p_nrm = opti.parameter(2, S * K)
        self._p_mask = opti.parameter(1, S * K)

    def _col(self, k: int, j: int) -> int:
        return k * self.max_points + j

    def set_values(self, opti, ctx: ObstacleContext) -> None:
        S = self._N + 1
        lin = ctx.lin_traj
        if lin is None:
            lin = np.tile(np.asarray(ctx.robot_xy, dtype=float), (S, 1))
        lin = np.asarray(lin, dtype=float).reshape(-1, 2)
        if lin.shape[0] < S:
            lin = np.vstack([lin, np.tile(lin[-1], (S - lin.shape[0], 1))])
        if not self.per_step:
            # legacy behaviour: one linearisation point for the whole horizon
            lin = np.tile(lin[0], (S, 1))

        obs, nrm, mask = _select_points_per_step(
            ctx.points_xy, lin[:S], self.max_points, self.check_radius
        )
        opti.set_value(self._p_obs, obs.reshape(-1, 2).T)
        opti.set_value(self._p_nrm, nrm.reshape(-1, 2).T)
        opti.set_value(self._p_mask, mask.reshape(1, -1))

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            max_points=self.max_points,
            check_radius=self.check_radius,
            per_step_linearisation=self.per_step,
        )
        return d


# ---------------------------------------------------------------------------
# 2. Half-space penalty, quadratic hinge  (C^1, legacy aerial behaviour)
# ---------------------------------------------------------------------------
class HalfSpaceQuadratic(_PointBasedTerm):
    """
        J_k = W * sum_j mask_j * max(0, d_safe - n_j.(p_k - o_j))^2

    Convex in p (a squared hinge of an affine function), but only C^1: the
    Hessian is discontinuous on the activation surface d = d_safe.  Kept as the
    baseline against which the smooth barrier is measured.
    """

    def __init__(
        self,
        weight: float = 100.0,
        d_safe: float = 0.5,
        max_points: int = 15,
        check_radius: float = 3.0,
        per_step: bool = False,
    ):
        super().__init__(max_points, check_radius, per_step)
        self.weight = float(weight)
        self.d_safe = float(d_safe)
        self.name = "halfspace_hinge2"

    def stage_cost(self, k: int, p_sym):
        total = 0
        for j in range(self.max_points):
            c = self._col(k, j)
            o = self._p_obs[:, c]
            n = self._p_nrm[:, c]
            m = self._p_mask[0, c]
            signed = ca.dot(n, p_sym - o)
            total = total + m * ca.fmax(0.0, self.d_safe - signed) ** 2
        return self.weight * total

    def describe(self) -> dict:
        d = super().describe()
        d.update(weight=self.weight, d_safe=self.d_safe, smoothness="C1")
        return d


# ---------------------------------------------------------------------------
# 3. Logistic (sigmoid) barrier  (C^infinity, legged behaviour)
# ---------------------------------------------------------------------------
class SigmoidBarrier(_PointBasedTerm):
    """
        J_k = W * sum_j mask_j / (1 + exp(alpha (d_j - r)))
            = W * sum_j mask_j * 0.5 * (1 - tanh(0.5 alpha (d_j - r)))

    with d_j = ||p_k - o_j||.  The tanh form is used because it is numerically
    stable (no exp overflow) and analytically identical.

    Smooth everywhere, so the Newton model IPOPT builds stays valid across the
    activation region.  It is isotropic — no normal is required — so it carries
    no linearisation error, at the price of being non-convex.

    `alpha` plays the role of the inverse barrier parameter of an interior-point
    method: the steeper the barrier, the worse the conditioning.  Sweeping it is
    the cleanest way to expose that trade-off experimentally.
    """

    def __init__(
        self,
        weight: float = 200.0,
        alpha: float = 4.0,
        r_safe: float = 0.55,
        max_points: int = 12,
        check_radius: float = 3.0,
        eps: float = 1e-6,
    ):
        super().__init__(max_points, check_radius, per_step=True)
        self.weight = float(weight)
        self.alpha = float(alpha)
        self.r_safe = float(r_safe)
        self.eps = float(eps)
        self.name = "sigmoid_barrier"

    def stage_cost(self, k: int, p_sym):
        total = 0
        for j in range(self.max_points):
            c = self._col(k, j)
            o = self._p_obs[:, c]
            m = self._p_mask[0, c]
            d = ca.sqrt((p_sym[0] - o[0]) ** 2 + (p_sym[1] - o[1]) ** 2 + self.eps)
            s = self.alpha * (d - self.r_safe)
            total = total + m * 0.5 * (1.0 - ca.tanh(0.5 * s))
        return self.weight * total

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            weight=self.weight, alpha=self.alpha, r_safe=self.r_safe,
            smoothness="Cinf", convex=False,
        )
        return d


# ---------------------------------------------------------------------------
# 4. Slacked half-space constraint  (SCA + exact / quadratic penalty)
# ---------------------------------------------------------------------------
class SlackedHalfSpace(_PointBasedTerm):
    """
    Successive Convex Approximation of the collision constraint, with slack:

        mask_j * ( n_j.(p_k - o_j) + s_jk - d_safe ) >= 0,     s_jk >= 0

    The constraint is AFFINE in p (the normal is a parameter at solve time), so
    it is convex; the slack guarantees the problem is never infeasible.

    penalty='l1'  ->  rho * sum(s)        EXACT penalty: s* = 0 whenever
                                          rho > |mu*|, the multiplier of the
                                          corresponding hard constraint.
                                          Linear in s, so the NLP stays smooth.
    penalty='l2'  ->  rho * sum(s^2)      residual violation s* ~ mu*/(2 rho),
                                          never exactly zero for finite rho.

    Comparing the two at equal rho is the cleanest experimental demonstration of
    the exact-penalty property, and it turns the choice of rho from a tuning knob
    into a quantity read off the dual solution.
    """

    VALID_PENALTIES = ("l1", "l2")

    def __init__(
        self,
        d_safe: float = 0.6,
        rho: float = 100.0,
        penalty: str = "l1",
        max_points: int = 12,
        check_radius: float = 3.0,
        per_step: bool = True,
    ):
        super().__init__(max_points, check_radius, per_step)
        if penalty not in self.VALID_PENALTIES:
            raise ValueError(f"penalty must be one of {self.VALID_PENALTIES}")
        self.d_safe = float(d_safe)
        self.rho = float(rho)
        self.penalty = penalty
        self.name = f"slacked_halfspace[{penalty}]"
        self._S = None

    def declare(self, opti, N: int, model) -> None:
        super().declare(opti, N, model)
        self._S = opti.variable(self.max_points, N + 1)
        opti.subject_to(ca.vec(self._S) >= 0)
        opti.set_initial(self._S, np.zeros((self.max_points, N + 1)))

    def add_constraints(self, opti, k: int, p_sym) -> None:
        for j in range(self.max_points):
            c = self._col(k, j)
            o = self._p_obs[:, c]
            n = self._p_nrm[:, c]
            m = self._p_mask[0, c]
            signed = ca.dot(n, p_sym - o)
            opti.subject_to(m * (signed + self._S[j, k] - self.d_safe) >= 0)

    def global_cost(self):
        if self._S is None:
            return 0
        if self.penalty == "l1":
            return self.rho * ca.sum1(ca.sum2(self._S))   # slacks are >= 0
        return self.rho * ca.sumsqr(self._S)

    def slack_value(self, sol) -> np.ndarray:
        return np.array(sol.value(self._S))

    def describe(self) -> dict:
        d = super().describe()
        d.update(d_safe=self.d_safe, rho=self.rho, penalty=self.penalty, convex=True)
        return d


# ---------------------------------------------------------------------------
# Registry, so a YAML string can select a strategy
# ---------------------------------------------------------------------------
OBSTACLE_TERMS = {
    "grid_bspline": GaussianGridCost,
    "halfspace_hinge2": HalfSpaceQuadratic,
    "sigmoid_barrier": SigmoidBarrier,
    "slacked_halfspace": SlackedHalfSpace,
}


def build_obstacle_terms(specs) -> list:
    """
    Instantiate terms from a list of dicts, e.g.

        [{"type": "grid_bspline", "weight": 100.0},
         {"type": "sigmoid_barrier", "alpha": 4.0}]
    """
    terms = []
    for spec in specs or []:
        s = dict(spec)
        kind = s.pop("type")
        if kind not in OBSTACLE_TERMS:
            raise KeyError(f"unknown obstacle term {kind!r}; known: {sorted(OBSTACLE_TERMS)}")
        terms.append(OBSTACLE_TERMS[kind](**s))
    return terms
