"""
Declarative configuration and A* polyline conditioning.

Both are shared infrastructure: a single YAML schema drives the two platforms,
and the path conditioning is what makes the shared reference builder usable by a
body-frame platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trajopt_core.config_io import load_planner
from trajopt_core.models import DoubleIntegratorZ, KinematicSE2
from trajopt_core.mpc import condition_path, resample_polyline, smooth_polyline
from trajopt_core.mpc.reference import build_path_reference

CONFIG = Path(__file__).resolve().parents[1] / "config" / "planner_params.yaml"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_shipped_config_loads_the_aerial_profile():
    pytest.importorskip("yaml")
    setup = load_planner(CONFIG)

    assert isinstance(setup.model, DoubleIntegratorZ)
    assert setup.model.RELATIVE_DEGREE == 2
    assert setup.cfg.N == 25 and setup.cfg.dt == pytest.approx(0.1)
    # limits declared once in `common` must reach the model
    assert setup.model.limits.a_max_xy == pytest.approx(2.0)

    # The shipped default must be build-once: no term may bake perception data
    # into the graph.  This is the property the configuration exists to deliver,
    # so assert it rather than the number of terms.
    assert setup.obstacle_terms, "the default profile must carry an obstacle term"
    assert not any(t.requires_rebuild for t in setup.obstacle_terms)


def test_same_schema_switches_platform():
    """
    Only the `model:` block and a handful of numeric overrides differ between the
    two robots — the structure of the file is identical.
    """
    pytest.importorskip("yaml")
    import yaml

    data = yaml.safe_load(CONFIG.read_text())
    data["model"] = {"type": "kinematic_se2", "integrator": "midpoint"}
    data["common"]["limits"]["v_max_xy"] = 1.0

    setup = load_planner(data)
    assert isinstance(setup.model, KinematicSE2)
    assert setup.model.RELATIVE_DEGREE == 1
    assert setup.model.limits.v_max_xy == pytest.approx(1.0)
    # everything else is untouched
    assert setup.cfg.N == 25
    assert [t.name for t in setup.obstacle_terms] == [
        t.name for t in load_planner(CONFIG).obstacle_terms
    ]


def test_legged_profile_is_a_small_delta_on_the_shared_base():
    """
    The legged profile is expressed as overrides merged onto the aerial base, so
    "how much of the stack is platform-specific?" is answered by reading one short
    file rather than by diffing two near-identical configurations.
    """
    pytest.importorskip("yaml")
    overrides = CONFIG.parent / "legged_overrides.yaml"
    assert overrides.exists()

    base = load_planner(CONFIG)
    legged = load_planner(CONFIG, overrides)

    # the model is swapped
    assert isinstance(base.model, DoubleIntegratorZ)
    assert isinstance(legged.model, KinematicSE2)
    assert (base.model.NX, base.model.NU) == (7, 4)
    assert (legged.model.NX, legged.model.NU) == (3, 3)
    assert base.model.RELATIVE_DEGREE == 2 and legged.model.RELATIVE_DEGREE == 1

    # the declared deltas took effect
    assert legged.cfg.v_ref == pytest.approx(0.7)
    assert legged.cfg.lookahead_dist == pytest.approx(1.0)
    assert legged.model.limits.v_max_xy == pytest.approx(1.0)

    # everything NOT declared in the delta file is shared verbatim
    for attr in ("N", "dt", "path_resample_ds", "path_smooth_window",
                 "sca_iterations", "terminal_zero_velocity", "anchor_first_to_state"):
        assert getattr(legged.cfg, attr) == getattr(base.cfg, attr), attr
    assert legged.cfg.solver.max_iter == base.cfg.solver.max_iter
    assert legged.cfg.weights.q_pos_xy == base.cfg.weights.q_pos_xy


def test_g1_profile_reuses_the_legged_model_and_obstacle_term():
    """
    The G1 (humanoid) profile is deliberately built to be comparable with the
    other two, not a fourth bespoke stack: same MotionModel class and
    integrator, same obstacle-avoidance term, as the Go2 profile — only the
    numeric limits and weights differ, plus the one genuinely new admissible-set
    restriction a biped needs (no reverse walking).
    """
    pytest.importorskip("yaml")
    overrides = CONFIG.parent / "g1_overrides.yaml"
    assert overrides.exists()

    legged = load_planner(CONFIG, CONFIG.parent / "legged_overrides.yaml")
    g1 = load_planner(CONFIG, overrides)

    # same model family and integrator as the Go2 profile
    assert isinstance(g1.model, KinematicSE2)
    assert g1.model.integrator == legged.model.integrator == "midpoint"
    assert g1.model.RELATIVE_DEGREE == legged.model.RELATIVE_DEGREE == 1

    # same obstacle-avoidance strategy, no new term type introduced
    assert [t.name for t in g1.obstacle_terms] == [t.name for t in legged.obstacle_terms]
    assert [type(t) for t in g1.obstacle_terms] == [type(t) for t in legged.obstacle_terms]

    # the one genuinely new admissible-set restriction: no reverse walking
    assert legged.model.limits.v_min_xy is None       # Go2 can reverse
    assert g1.model.limits.v_min_xy == pytest.approx(0.0)   # G1 cannot
    lb, _ = g1.model.input_bounds()
    assert lb[0] == pytest.approx(0.0)

    # dynamics, solver machinery and everything platform-independent is shared
    assert g1.cfg.dt == legged.cfg.dt
    assert g1.cfg.path_resample_ds == legged.cfg.path_resample_ds
    assert g1.cfg.sca_iterations == legged.cfg.sca_iterations


def test_g1_config_declares_no_capability_absent_from_the_other_profiles():
    """
    The port must not smuggle in a capability the aerial/Go2 profiles don't
    have (dynamic obstacles, yield behaviour, ...): every obstacle term the G1
    profile declares must be one already offered by the shared registry, using
    only its documented fields.
    """
    from trajopt_core.mpc.obstacles import OBSTACLE_TERMS

    g1 = load_planner(CONFIG, CONFIG.parent / "g1_overrides.yaml")
    assert len(g1.obstacle_terms) >= 1
    for term in g1.obstacle_terms:
        assert type(term) in OBSTACLE_TERMS.values()


def test_demo_g1_matches_the_shipped_profile():
    """
    `examples/cross_platform_demo.py` spells its platforms out in Python so the
    report can read them as code, while the deployed system reads YAML.  Two
    spellings of the same robot is exactly the kind of thing that drifts, so the
    numbers that decide the trajectory are compared here.
    """
    pytest.importorskip("yaml")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from cross_platform_demo import g1 as demo_g1          # noqa: PLC0415

    _, model, cfg, terms, gains = demo_g1()
    shipped = load_planner(CONFIG, CONFIG.parent / "g1_overrides.yaml")

    assert type(model) is type(shipped.model)
    assert model.integrator == shipped.model.integrator
    for f in ("v_max_xy", "v_min_xy", "v_max_lat", "yaw_rate_max"):
        assert getattr(model.limits, f) == pytest.approx(
            getattr(shipped.model.limits, f)
        ), f

    for f in ("N", "dt", "v_ref", "lookahead_dist"):
        assert getattr(cfg, f) == pytest.approx(getattr(shipped.cfg, f)), f
    for f in ("q_pos_xy", "q_yaw", "q_terminal", "r_lin_xy", "r_ang", "r_jerk"):
        assert getattr(cfg.weights, f) == pytest.approx(
            getattr(shipped.cfg.weights, f)
        ), f
    for f in ("kp_pos", "kd_pos", "kp_yaw"):
        assert getattr(gains, f) == pytest.approx(getattr(shipped.gains, f)), f

    assert [t.name for t in terms] == [t.name for t in shipped.obstacle_terms]
    assert terms[0].weight == pytest.approx(shipped.obstacle_terms[0].weight)


def test_deep_merge_replaces_lists_wholesale():
    """
    A half-merged list of obstacle terms would be far more surprising than a
    replaced one, so lists are replaced rather than merged element by element.
    """
    setup = load_planner(
        {"model": {"type": "kinematic_se2"},
         "common": {"obstacles": [{"type": "sigmoid_barrier"},
                                  {"type": "halfspace_hinge2"}]}},
        {"common": {"obstacles": [{"type": "sigmoid_barrier", "weight": 1.0}]}},
    )
    assert [t.name for t in setup.obstacle_terms] == ["sigmoid_barrier"]
    assert setup.obstacle_terms[0].weight == pytest.approx(1.0)


def test_overrides_do_not_mutate_the_base_document():
    """Merging must be pure: loading the base again afterwards yields the base."""
    pytest.importorskip("yaml")
    import yaml

    doc = yaml.safe_load(CONFIG.read_text())
    before = doc["common"]["v_ref"]
    load_planner(doc, {"common": {"v_ref": 99.0}})
    assert doc["common"]["v_ref"] == before


def test_unknown_field_is_rejected():
    """A silently ignored typo in a weight would be a debugging nightmare."""
    with pytest.raises(KeyError, match="q_pos_XY"):
        load_planner({"model": {"type": "kinematic_se2"},
                      "common": {"weights": {"q_pos_XY": 1.0}}})


def test_minimal_config_falls_back_to_defaults():
    setup = load_planner({"model": {"type": "kinematic_se2"}})
    assert setup.cfg.N == 25 or setup.cfg.N > 0
    assert setup.obstacle_terms == []


def test_ros_parameter_wrapper_is_accepted():
    """ROS param files nest everything under /**: ros__parameters:."""
    setup = load_planner({"/**": {"ros__parameters": {
        "model": {"type": "kinematic_se2"},
        "common": {"N": 12},
    }}})
    assert setup.cfg.N == 12


# ---------------------------------------------------------------------------
# Path conditioning
# ---------------------------------------------------------------------------
def _zigzag(n=20, step=0.25):
    """A polyline of the kind an 8-connected grid search emits."""
    pts = [(0.0, 0.0)]
    for i in range(n):
        x, y = pts[-1]
        pts.append((x + step, y + (step if i % 2 == 0 else 0.0)))
    return np.asarray(pts)


def test_resampling_gives_uniform_arc_length_spacing():
    """
    Samples are uniform in ARC LENGTH, so the chord between two consecutive ones
    equals ds on a straight stretch and is strictly shorter whenever a corner
    falls between them — the chord cuts the corner.  Asserting equal chords would
    be asserting something false about polylines.
    """
    ds = 0.2
    out = resample_polyline(_zigzag(), ds)
    chords = np.linalg.norm(np.diff(out, axis=0), axis=1)

    assert np.all(chords <= ds + 1e-9)
    assert chords.max() == pytest.approx(ds, abs=1e-9)   # straight stretches
    assert np.mean(chords) > 0.9 * ds                    # corners are the exception


def test_smoothing_preserves_the_endpoints():
    raw = _zigzag()
    out = smooth_polyline(raw, 5)
    assert np.allclose(out[0], raw[0])
    assert np.allclose(out[-1], raw[-1])
    assert out.shape == raw.shape


def test_conditioning_reduces_heading_chatter():
    """
    The quantity that matters is the variation of the TANGENT, because that is
    what becomes the yaw reference.  A raw 8-connected path flips by 45 deg from
    one waypoint to the next.
    """
    def max_heading_step(pts):
        d = np.diff(pts[:, :2], axis=0)
        h = np.arctan2(d[:, 1], d[:, 0])
        return float(np.max(np.abs(np.diff(h))))

    raw = _zigzag()
    conditioned = condition_path(raw, resample_ds=0.2, smooth_window=5)

    assert max_heading_step(raw) > np.deg2rad(40.0)
    assert max_heading_step(conditioned) < 0.5 * max_heading_step(raw)


def test_conditioning_is_off_when_disabled():
    raw = _zigzag()
    assert np.allclose(condition_path(raw, 0.0, 0), raw)


def test_short_paths_survive_conditioning():
    for path in (np.zeros((0, 2)), np.array([[1.0, 2.0]]), np.array([[0.0, 0.0], [0.1, 0.0]])):
        out = condition_path(path, 0.2, 5)
        assert out.shape[0] == max(path.shape[0], 0)


@pytest.mark.parametrize("model", [DoubleIntegratorZ(), KinematicSE2()])
def test_reference_builder_applies_conditioning(model):
    raw = _zigzag()
    state = model.make_state((0.0, 0.0, 1.5)) if model.PLANS_ALTITUDE else model.make_state((0.0, 0.0))

    rough = build_path_reference(model, state, raw, 20, 0.1, 1.0, 1.5)
    smooth = build_path_reference(model, state, raw, 20, 0.1, 1.0, 1.5,
                                  resample_ds=0.2, smooth_window=5)

    yaw_idx = model.YAW_IDX
    chatter_rough = float(np.max(np.abs(np.diff(rough.x_ref[:, yaw_idx]))))
    chatter_smooth = float(np.max(np.abs(np.diff(smooth.x_ref[:, yaw_idx]))))
    assert chatter_smooth < chatter_rough
