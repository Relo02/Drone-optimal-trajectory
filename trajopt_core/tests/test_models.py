"""
Properties of the two platform models.

The discretisation tests are the numerical counterpart of Sec. 2.1.3 of the
course notes and produce the figures quoted in the report: the double integrator
under a zero-order-hold input is discretised EXACTLY, while the SE(2) kinematic
model is not, and the mid-point rule buys one order of accuracy for free.
"""

from __future__ import annotations

import numpy as np
import pytest

from trajopt_core.models import CostWeights, DoubleIntegratorZ, KinematicSE2, ModelLimits


# ---------------------------------------------------------------------------
# Structural invariants required by the platform-agnostic layer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model",
    [DoubleIntegratorZ(), KinematicSE2(), KinematicSE2(integrator="euler")],
)
def test_interface_contract(model):
    lb, ub = model.input_bounds()
    assert lb.shape == (model.NU,) and ub.shape == (model.NU,)
    assert np.all(lb <= ub)

    assert len(model.PLANAR_IDX) == 2
    assert all(0 <= i < model.NX for i in model.PLANAR_IDX)
    assert all(0 <= i < model.NX for i in model.POS_IDX)
    assert len(model.STATE_LAYOUT) == model.NX
    assert len(model.INPUT_LAYOUT) == model.NU

    w = CostWeights()
    assert model.state_weight_vector(w).shape == (model.NX,)
    assert model.input_weight_vector(w).shape == (model.NU,)
    assert np.all(model.state_weight_vector(w) >= 0)
    assert np.all(model.input_weight_vector(w) >= 0)


@pytest.mark.parametrize("model", [DoubleIntegratorZ(), KinematicSE2()])
def test_lift_reference_is_consistent(model):
    tangent = np.array([np.cos(0.6), np.sin(0.6)])
    x_ref = model.lift_reference(np.array([3.0, -2.0]), tangent, v_ref=1.0, z_ref=1.5)

    assert x_ref.shape == (model.NX,)
    i, j = model.PLANAR_IDX
    assert x_ref[i] == pytest.approx(3.0)
    assert x_ref[j] == pytest.approx(-2.0)
    assert model.heading(x_ref) == pytest.approx(0.6)
    if model.PLANS_ALTITUDE:
        assert x_ref[model.Z_IDX] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Double integrator: the ZOH discretisation is exact
# ---------------------------------------------------------------------------
def test_double_integrator_discretisation_is_exact():
    """
    Integrating p'' = a with a held constant, using RK4 at a very fine step, must
    reproduce the one-shot discrete update to machine precision: the truncation
    error is identically zero, not merely small.
    """
    model = DoubleIntegratorZ()
    dt = 0.1
    x0 = np.array([1.0, -2.0, 1.5, 0.7, -0.3, 0.2, 0.4])
    u = np.array([1.3, -0.9, 0.5, 0.8])

    x_disc = np.array(model.step(x0, u, dt), dtype=float)

    # reference: 20000 RK4 sub-steps of the continuous system
    n_sub = 20000
    h = dt / n_sub

    def rhs(x):
        return np.array([x[3], x[4], x[5], u[0], u[1], u[2], u[3]])

    x = x0.copy()
    for _ in range(n_sub):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * h * k1)
        k3 = rhs(x + 0.5 * h * k2)
        k4 = rhs(x + h * k3)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    assert np.max(np.abs(x_disc - x)) < 1e-12


# ---------------------------------------------------------------------------
# SE(2): Euler is first order, mid-point is second order
# ---------------------------------------------------------------------------
def _local_error(integrator: str, dt: float) -> float:
    model = KinematicSE2(integrator=integrator)
    x0 = np.array([0.0, 0.0, 0.3])
    u = np.array([0.8, 0.15, 1.2])
    approx = np.array(model.step(x0, u, dt), dtype=float)
    exact = KinematicSE2.exact_step(x0, u, dt)
    return float(np.linalg.norm(approx[:2] - exact[:2]))


@pytest.mark.parametrize(
    "integrator,expected_order",
    [("euler", 2.0), ("midpoint", 3.0)],
)
def test_local_truncation_order(integrator, expected_order):
    """Local error ~ C dt^q: the fitted slope on a log-log plot must match q."""
    steps = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    errs = np.array([_local_error(integrator, float(h)) for h in steps])
    slope = np.polyfit(np.log(steps), np.log(errs), 1)[0]
    assert slope == pytest.approx(expected_order, abs=0.15), (
        f"{integrator}: fitted order {slope:.3f}, expected {expected_order}"
    )


def test_midpoint_beats_euler_at_nominal_settings():
    """
    At the settings actually used on the robot (dt = 0.1 s, omega up to
    1.5 rad/s) the mid-point rule must be materially more accurate, otherwise
    there is no reason to prefer it.
    """
    dt = 0.1
    e_euler = _local_error("euler", dt)
    e_mid = _local_error("midpoint", dt)
    assert e_mid < e_euler / 10.0


def test_exact_step_degenerates_to_straight_line():
    """omega -> 0 must recover the straight-line displacement without blowing up."""
    x0 = np.array([1.0, 2.0, 0.5])
    u = np.array([0.9, 0.2, 0.0])
    exact = KinematicSE2.exact_step(x0, u, 0.1)
    euler = np.array(KinematicSE2(integrator="euler").step(x0, u, 0.1), dtype=float)
    assert np.allclose(exact, euler, atol=1e-12)


# ---------------------------------------------------------------------------
# Admissible sets
# ---------------------------------------------------------------------------
def test_nonholonomic_platform_is_a_restriction_of_the_same_model():
    """A differential-drive robot is the same model with a shrunken input set."""
    limits = ModelLimits(v_max_xy=1.0, v_max_lat=0.0, yaw_rate_max=1.5)
    model = KinematicSE2(limits=limits)
    lb, ub = model.input_bounds()
    assert lb[1] == 0.0 and ub[1] == 0.0


def test_symmetric_vx_bound_is_the_default():
    """Without v_min_xy, U_Sigma is symmetric: lb = -ub, as before this field existed."""
    limits = ModelLimits(v_max_xy=1.3)
    lb, ub = KinematicSE2(limits=limits).input_bounds()
    assert lb[0] == pytest.approx(-1.3)
    assert limits.resolved_v_min_xy == pytest.approx(-1.3)


def test_no_reverse_platform_is_a_one_sided_restriction():
    """
    A biped that must not walk backward is the same model and the same dynamics
    (f_Sigma untouched), with U_Sigma restricted on one side only — the same
    "shrink the admissible set" pattern as the nonholonomic case above, applied
    to the forward/backward axis instead of the lateral one.
    """
    limits = ModelLimits(v_max_xy=0.8, v_min_xy=0.0)
    model = KinematicSE2(limits=limits)
    lb, ub = model.input_bounds()
    assert lb[0] == pytest.approx(0.0)
    assert ub[0] == pytest.approx(0.8)

    # the dynamics are unaffected: forward motion still integrates the same way
    x1 = np.array(model.step([0.0, 0.0, 0.0], [0.8, 0.0, 0.0], 0.1), dtype=float)
    x2 = np.array(
        KinematicSE2(limits=ModelLimits(v_max_xy=0.8)).step(
            [0.0, 0.0, 0.0], [0.8, 0.0, 0.0], 0.1
        ),
        dtype=float,
    )
    assert np.allclose(x1, x2)


def test_aerial_limits_map_to_accelerations():
    limits = ModelLimits(a_max_xy=2.0, a_max_z=1.5, yaw_rate_max=1.5)
    lb, ub = DoubleIntegratorZ(limits=limits).input_bounds()
    assert np.allclose(ub, [2.0, 2.0, 1.5, 1.5])
    assert np.allclose(lb, -ub)
