"""
The unified mapping / planning layers must reproduce the two original
implementations exactly.

This is the safety net of the deduplication: the layers were already identical
across the aerial and legged repositories, and this test turns that observation
into a regression guard.  Golden vectors are checked in, so the test also runs on
a machine where neither platform workspace is available; when a workspace *is*
present, the original code is imported and compared directly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from trajopt_core.mapping import FixedGaussianGridMap
from trajopt_core.planning import AStarPlanner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_golden import (  # noqa: E402
    DEFAULT_DRONE_SRC,
    DEFAULT_GO2_SRC,
    SCENARIO,
    make_scenario,
    run_case,
)

GOLDEN = Path(__file__).resolve().parent / "golden" / "grid_astar.npz"


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("golden file missing; run tests/generate_golden.py")
    return np.load(GOLDEN)


@pytest.fixture(scope="module")
def core_result():
    return run_case(FixedGaussianGridMap, AStarPlanner, make_scenario())


def test_grid_matches_golden(golden, core_result):
    assert core_result["gmap"].shape == tuple(golden["gmap"].shape)
    assert np.array_equal(core_result["gmap"], golden["gmap"])
    assert core_result["minx"] == pytest.approx(float(golden["minx"]))
    assert core_result["miny"] == pytest.approx(float(golden["miny"]))
    assert core_result["cells"] == int(golden["cells"])


def test_path_matches_golden(golden, core_result):
    assert core_result["path"].shape == tuple(golden["path"].shape)
    assert np.array_equal(core_result["path"], golden["path"])
    assert core_result["path"].shape[0] > 1, "scenario must produce a non-trivial path"


def test_occupancy_saturates_at_one_half(core_result):
    """P = 1 - Phi(d/sigma) cannot exceed 0.5; any threshold above that never fires."""
    assert core_result["gmap"].max() <= 0.5 + 1e-9
    assert core_result["gmap"].min() >= 0.0


@pytest.mark.parametrize(
    "src,pkg,label",
    [
        (DEFAULT_DRONE_SRC, "new_mujoco", "aerial"),
        (DEFAULT_GO2_SRC, "a_star_mpc_planner", "legged"),
    ],
)
def test_matches_original_implementation(core_result, src, pkg, label):
    """Compare against the original source tree when it is available."""
    if not (Path(src) / pkg).is_dir():
        pytest.skip(f"{label} workspace not available at {src}")
    sys.path.insert(0, str(src))
    try:
        grid_mod = importlib.import_module(f"{pkg}.gaussian_grid_map")
        astar_mod = importlib.import_module(f"{pkg}.a_star_planner")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot import {pkg}: {exc}")

    ref = run_case(grid_mod.FixedGaussianGridMap, astar_mod.AStarPlanner, make_scenario())

    assert np.array_equal(ref["gmap"], core_result["gmap"])
    assert np.array_equal(ref["path"], core_result["path"])
    assert ref["minx"] == core_result["minx"]
    assert ref["miny"] == core_result["miny"]
    assert ref["hit"] == core_result["hit"]


def test_blocking_radius_matches_theory():
    """
    The hard-blocking radius implied by (sigma, threshold) is
    d = sigma * Phi^-1(1 - tau).  Verifying it keeps the report's numbers honest.
    """
    from scipy.stats import norm

    sigma, tau = SCENARIO["std"], SCENARIO["obstacle_threshold"]
    d_block = sigma * norm.ppf(1.0 - tau)

    grid = FixedGaussianGridMap(reso=0.01, half_width=2.0, std=sigma)
    grid.update(np.array([[0.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0]))
    planner = AStarPlanner(obstacle_threshold=tau, obstacle_cost_weight=1.0)

    inside = grid.world_to_index(0.5 * d_block, 0.0)
    outside = grid.world_to_index(1.5 * d_block, 0.0)
    assert not planner._is_free(grid, *inside)
    assert planner._is_free(grid, *outside)
