"""
Behaviour of the platform-agnostic optimal control problem.

The same `TrajectoryOCP` class is exercised with both platform models: these
tests are the operative form of the claim that the optimisation layer is
independent of the robot.
"""

from __future__ import annotations

import numpy as np
import pytest

from trajopt_core.mapping import FixedGaussianGridMap
from trajopt_core.models import CostWeights, DoubleIntegratorZ, KinematicSE2, ModelLimits
from trajopt_core.mpc import (
    GaussianGridCost,
    HalfSpaceQuadratic,
    OCPConfig,
    SigmoidBarrier,
    SlackedHalfSpace,
    SolverOptions,
    TrajectoryOCP,
    select_lookahead,
)

RESO, HALF_WIDTH, STD = 0.25, 5.0, 0.7


def straight_path(n=25, dx=0.4):
    return [(i * dx, 0.0) for i in range(n)]


def wall_points(x=3.0, y_lo=-2.0, y_hi=2.0, n=40):
    return np.stack([np.full(n, x), np.linspace(y_lo, y_hi, n)], axis=1)


def make_cfg(N=12, dt=0.1, **kw):
    cfg = OCPConfig(
        N=N,
        dt=dt,
        v_ref=1.0,
        z_ref=1.5,
        weights=CostWeights(),
        limits=ModelLimits(),
        solver=SolverOptions(max_iter=80, print_level=0),
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def make_grid(points, robot_xy):
    grid = FixedGaussianGridMap(reso=RESO, half_width=HALF_WIDTH, std=STD)
    grid.update(np.hstack([points, np.zeros((len(points), 1))]), np.append(robot_xy, 0.0))
    return grid


PLATFORMS = {
    "aerial": (DoubleIntegratorZ(), np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0])),
    "legged": (KinematicSE2(), np.array([0.0, 0.0, 0.0])),
}


# ---------------------------------------------------------------------------
# The same class solves for both platforms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_solves_on_both_platforms(platform):
    model, x0 = PLATFORMS[platform]
    ocp = TrajectoryOCP(model, make_cfg(), [SigmoidBarrier(max_points=5)])
    pts = wall_points()

    res = ocp.solve(x0, straight_path(), points_xy=pts)

    assert res.success, res.status
    assert res.x_pred.shape == (ocp.cfg.N + 1, model.NX)
    assert res.u_opt.shape == (ocp.cfg.N, model.NU)
    assert np.isfinite(res.cost)
    # initial condition is imposed as an equality constraint
    assert np.allclose(res.x_pred[0], x0, atol=1e-6)


@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_input_bounds_are_respected(platform):
    model, x0 = PLATFORMS[platform]
    ocp = TrajectoryOCP(model, make_cfg(), [SigmoidBarrier(max_points=5)])
    res = ocp.solve(x0, straight_path(), points_xy=wall_points())

    lb, ub = model.input_bounds()
    assert np.all(res.u_opt >= lb - 1e-6)
    assert np.all(res.u_opt <= ub + 1e-6)


# ---------------------------------------------------------------------------
# Build-once: the central performance claim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_graph_is_built_once_when_parametric(platform):
    """
    The point-based terms parametrise cleanly, so the graph is constructed once
    and every subsequent cycle only refreshes parameter values.  (The grid
    B-spline is deliberately excluded here: it is parametrisable but its
    parametric form is far more expensive to evaluate — see
    `GaussianGridCost` and the build benchmark.)
    """
    model, x0 = PLATFORMS[platform]
    terms = [SigmoidBarrier(max_points=5), HalfSpaceQuadratic(max_points=4)]
    ocp = TrajectoryOCP(model, make_cfg(), terms)
    assert ocp.is_parametric

    pts = wall_points()
    state = x0.copy()
    for step in range(5):
        grid = make_grid(pts, state[list(model.PLANAR_IDX)])
        res = ocp.solve(state, straight_path(), points_xy=pts, grid=grid)
        assert res.success, res.status
        assert res.rebuilt == (step == 0)
        assert res.build_ms == 0.0 or step == 0
        state = res.x_pred[1]

    assert ocp.build_count == 1, "a parametric OCP must build its graph exactly once"


def test_non_parametric_grid_forces_a_rebuild_every_cycle():
    """The legacy behaviour, kept selectable so the cost can be measured."""
    model, x0 = PLATFORMS["aerial"]
    terms = [GaussianGridCost(half_width=HALF_WIDTH, reso=RESO, parametric=False)]
    ocp = TrajectoryOCP(model, make_cfg(N=8), terms)
    assert not ocp.is_parametric

    pts = wall_points()
    for _ in range(3):
        grid = make_grid(pts, x0[list(model.PLANAR_IDX)])
        res = ocp.solve(x0, straight_path(), points_xy=pts, grid=grid)
        assert res.rebuilt and res.build_ms > 0.0

    assert ocp.build_count == 3


def test_parametric_grid_matches_the_baked_one():
    """
    Moving the B-spline knots into the local frame must not change the cost that
    the solver sees — otherwise the build-once optimisation would be buying speed
    by changing the problem.
    """
    model, x0 = PLATFORMS["aerial"]
    pts = wall_points()
    grid = make_grid(pts, x0[list(model.PLANAR_IDX)])
    path = straight_path()

    res_par = TrajectoryOCP(
        model, make_cfg(N=8), [GaussianGridCost(half_width=HALF_WIDTH, reso=RESO, parametric=True)]
    ).solve(x0, path, points_xy=pts, grid=grid)

    res_bak = TrajectoryOCP(
        model, make_cfg(N=8), [GaussianGridCost(half_width=HALF_WIDTH, reso=RESO, parametric=False)]
    ).solve(x0, path, points_xy=pts, grid=grid)

    assert res_par.cost == pytest.approx(res_bak.cost, rel=1e-7)
    assert np.allclose(res_par.x_pred, res_bak.x_pred, atol=1e-6)


# ---------------------------------------------------------------------------
# Obstacle terms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "term",
    [
        SigmoidBarrier(max_points=5),
        HalfSpaceQuadratic(max_points=5),
        SlackedHalfSpace(max_points=5, penalty="l1"),
        SlackedHalfSpace(max_points=5, penalty="l2"),
    ],
    ids=lambda t: t.name,
)
def test_padded_slots_are_inert(term):
    """
    Unused obstacle slots are parked at a sentinel and masked out.  With no
    perception input at all the SOLUTION must coincide with the obstacle-free
    one, otherwise the padding would be leaking into the optimum.

    The comparison is made on the trajectory rather than on the cost: an
    interior-point method keeps the slack variables a small distance inside their
    bound, so a slack-carrying term shows a residual cost offset of order
    rho * n_slacks * mu_final even when every slack is inactive.  That offset is
    an artefact of the barrier, not a perturbation of the optimiser.
    """
    model, x0 = PLATFORMS["legged"]
    path = straight_path()

    free = TrajectoryOCP(model, make_cfg(N=10), []).solve(x0, path)
    padded = TrajectoryOCP(model, make_cfg(N=10), [term]).solve(x0, path, points_xy=None)

    assert padded.success
    assert np.allclose(padded.x_pred, free.x_pred, atol=1e-6)
    assert np.allclose(padded.u_opt, free.u_opt, atol=1e-6)
    assert padded.cost == pytest.approx(free.cost, rel=1e-3, abs=1e-3)


def test_obstacle_term_increases_clearance():
    """
    With a wall sitting on the reference, the predicted trajectory must keep more
    clearance than the obstacle-free solution.

    Clearance is the right quantity to assert on, not lateral deviation: faced
    with a wall spanning the whole corridor the optimiser is free to avoid it
    either by going around it or by stopping short of it, and with a symmetric
    obstacle it legitimately chooses the latter.
    """
    model, x0 = PLATFORMS["legged"]
    pts = wall_points(x=2.0, y_lo=-0.6, y_hi=0.6, n=20)
    cfg = make_cfg(N=20)

    free = TrajectoryOCP(model, cfg, []).solve(x0, straight_path())
    avoid = TrajectoryOCP(
        model, cfg, [SigmoidBarrier(weight=400.0, r_safe=0.8, max_points=8)]
    ).solve(x0, straight_path(), points_xy=pts)

    def clearance(res):
        p = res.x_pred[:, list(model.PLANAR_IDX)]
        return float(np.min(np.linalg.norm(p[:, None, :] - pts[None, :, :], axis=2)))

    assert clearance(free) < 0.2, "the free solution should drive into the wall"
    assert clearance(avoid) > 1.0, "the barrier should keep the horizon clear"


# ---------------------------------------------------------------------------
# Exact vs quadratic penalty  (Thm 6.3.1 of the course notes)
# ---------------------------------------------------------------------------
def _max_slack(penalty: str, rho: float, points, n=15) -> float:
    model, x0 = PLATFORMS["legged"]
    term = SlackedHalfSpace(
        d_safe=0.5, rho=rho, penalty=penalty, max_points=4, check_radius=3.0
    )
    ocp = TrajectoryOCP(model, make_cfg(N=n), [term])
    res = ocp.solve(x0, straight_path(), points_xy=points)
    assert res.success, res.status
    return max(0.0, float(np.max(term.slack_value(ocp._opti.debug))))


def test_l1_slack_penalty_is_exact_while_l2_is_not():
    """
    Exact-penalty property (Thm 6.3.1 of the course notes).

    On a geometry where the constraint IS satisfiable, an L1 slack penalty drives
    the violation to exactly zero as soon as rho exceeds the multiplier of the
    corresponding hard constraint, whereas a quadratic penalty always leaves a
    residual of order mu*/(2 rho) — decaying with rho but never vanishing.

    This is the experiment quoted in the report to justify choosing the slack
    weight from the dual solution rather than by trial and error.
    """
    side_obstacle = np.array([[1.0, 0.30]])   # beside the path, not across it

    for rho in (20.0, 200.0, 2000.0):
        assert _max_slack("l1", rho, side_obstacle) < 1e-8, f"L1 not exact at rho={rho}"

    l2 = [_max_slack("l2", rho, side_obstacle) for rho in (20.0, 200.0, 2000.0)]
    assert all(v > 1e-4 for v in l2), "L2 should always leave a residual violation"
    assert l2[0] > l2[1] > l2[2], "the L2 residual should decay with rho"
    assert l2[1] / l2[2] > 3.0, "L2 residual should decay roughly like 1/rho"


def test_sca_resolves_contradictory_half_spaces():
    """
    Documented failure mode of a single-shot linearisation.

    When the linearisation trajectory pierces the obstacle, the horizon steps
    lying beyond it produce normals pointing the opposite way, so the half-spaces
    ask for p_x <= a AND p_x >= b with b > a.  The problem is then genuinely
    infeasible and the slack saturates at a value no penalty weight can reduce —
    increasing rho does nothing, which is the diagnostic signature.

    Re-linearising (SCA) removes the contradiction.
    """
    model, x0 = PLATFORMS["legged"]
    wall = wall_points(x=1.5, y_lo=-0.4, y_hi=0.4, n=12)
    kwargs = dict(d_safe=0.8, penalty="l1", max_points=6, check_radius=4.0)

    def run(sca_iterations, rho):
        term = SlackedHalfSpace(rho=rho, **kwargs)
        ocp = TrajectoryOCP(model, make_cfg(N=15, sca_iterations=sca_iterations), [term])
        ocp.solve(x0, straight_path(), points_xy=wall)
        return max(0.0, float(np.max(term.slack_value(ocp._opti.debug))))

    # one shot: the slack is insensitive to the penalty weight -> infeasible
    s_low, s_high = run(1, 50.0), run(1, 50_000.0)
    assert s_low > 0.1 and s_high > 0.1
    assert abs(s_low - s_high) / s_low < 0.05, "slack should be insensitive to rho here"

    # re-linearising restores feasibility
    assert run(4, 500.0) < 0.5 * s_high


# ---------------------------------------------------------------------------
# Structure and terminal ingredients
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_structure_reports_a_sparse_problem(platform):
    model, _ = PLATFORMS[platform]
    ocp = TrajectoryOCP(model, make_cfg(N=20), [SigmoidBarrier(max_points=5)])
    s = ocp.structure()

    assert s["n_variables"] == model.NX * 21 + model.NU * 20
    assert s["n_constraints"] > 0
    # multiple shooting yields a block-banded Jacobian: it must be very sparse
    assert s["jac_density"] < 0.05
    assert s["is_parametric"] is True


def test_terminal_zero_velocity_constraint():
    """The cheapest terminal ingredient: the horizon must end at an equilibrium."""
    model, x0 = PLATFORMS["aerial"]
    cfg = make_cfg(N=15, terminal_zero_velocity=True)
    res = TrajectoryOCP(model, cfg, []).solve(x0, straight_path())

    assert res.success
    assert np.allclose(res.x_pred[-1, list(model.VEL_IDX)], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------
def test_warm_start_reduces_iterations():
    model, x0 = PLATFORMS["legged"]
    ocp = TrajectoryOCP(model, make_cfg(N=20), [SigmoidBarrier(max_points=6)])
    pts = wall_points(x=2.5)
    path = straight_path()

    first = ocp.solve(x0, path, points_xy=pts)
    later = [ocp.solve(x0, path, points_xy=pts).iterations for _ in range(3)]

    assert first.iterations > 0
    assert min(later) <= first.iterations


def test_failed_solve_does_not_poison_the_warm_start():
    model, x0 = PLATFORMS["legged"]
    ocp = TrajectoryOCP(model, make_cfg(N=10), [])
    ocp.solve(x0, straight_path())

    bad = ocp.solve(np.array([np.nan, np.nan, np.nan]), straight_path())
    assert np.isfinite(bad.x_pred).all()

    good = ocp.solve(x0, straight_path())
    assert good.success and np.isfinite(good.x_pred).all()


# ---------------------------------------------------------------------------
# Lookahead
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_lookahead_respects_the_minimum_distance(platform):
    model, x0 = PLATFORMS[platform]
    ocp = TrajectoryOCP(model, make_cfg(N=20), [])
    res = ocp.solve(x0, straight_path())

    i, j = model.PLANAR_IDX
    look = select_lookahead(model, res.x_pred, np.array([x0[i], x0[j]]), 1.0)

    assert look.found
    assert look.distance >= 1.0 - 1e-9
    assert look.position.shape == (len(model.POS_IDX),)


def test_lookahead_falls_back_to_the_goal_when_near():
    model, x0 = PLATFORMS["legged"]
    x_pred = np.tile(np.array([0.05, 0.0, 0.0]), (10, 1))
    look = select_lookahead(model, x_pred, np.zeros(2), 1.5, fallback_waypoint=(0.2, 0.1))

    assert not look.found
    assert np.allclose(look.position, [0.2, 0.1])
