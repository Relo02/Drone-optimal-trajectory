#!/usr/bin/env python3
"""
Regenerate the golden vectors used by the equivalence tests.

The mapping and planning layers were *already* byte-identical in the two
platform repositories before the unification (the diff was limited to the import
path, a docstring and a matplotlib helper).  Deduplicating them is therefore a
zero-risk refactor, but "zero-risk" is a claim that should be checked rather than
asserted: this script runs a deterministic scenario through

    - new_mujoco            (aerial repository)
    - a_star_mpc_planner    (legged repository)
    - trajopt_core          (unified core)

and stores the outputs.  `tests/test_golden_equivalence.py` then verifies that
the core reproduces them exactly.

Usage
-----
    python3 tests/generate_golden.py [--drone-src PATH] [--go2-src PATH]

Reference sources that cannot be imported are skipped with a warning, so the
script still produces a usable golden file on a machine that only has one of the
two workspaces checked out.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

DEFAULT_DRONE_SRC = REPO / "mujoco" / "ros2_ws" / "src" / "new_mujoco"
DEFAULT_GO2_SRC = Path("/mnt/c/Users/franc/Desktop/Tesi/Go2_navigation/src/a_star_mpc_planner")

# --------------------------------------------------------------------------
# Deterministic scenario
# --------------------------------------------------------------------------
SCENARIO = dict(
    reso=0.25,
    half_width=5.0,
    std=0.7,
    obstacle_threshold=0.1,
    obstacle_cost_weight=15.0,
    robot=np.array([2.0, -1.0, 1.5]),
    goal=np.array([20.0, 1.0]),
    seed=20260803,
    n_points=180,
)


def make_scenario():
    """
    A reproducible LiDAR cloud: a wall with a gate, plus scattered clutter.

    The gate is 3.0 m wide.  With sigma = 0.7 and an obstacle threshold of 0.1
    the hard-blocking radius is sigma*Phi^-1(0.9) ~ 0.90 m, so roughly 1.2 m of
    genuinely free corridor remains: wide enough for A* to find a way through,
    narrow enough that the soft cost actually shapes the path.
    """
    p = SCENARIO
    rng = np.random.default_rng(p["seed"])
    robot = p["robot"]

    wall_x = robot[0] + 2.6
    lower = np.stack(
        [np.full(50, wall_x), np.linspace(robot[1] - 4.0, robot[1] - 0.6, 50)], axis=1
    )
    upper = np.stack(
        [np.full(40, wall_x), np.linspace(robot[1] + 2.4, robot[1] + 4.5, 40)], axis=1
    )
    # clutter kept behind the wall so it cannot randomly seal the gate
    clutter = np.stack(
        [
            rng.uniform(robot[0] - 4.5, robot[0] + 1.4, p["n_points"] - 90),
            rng.uniform(robot[1] - 4.5, robot[1] + 4.5, p["n_points"] - 90),
        ],
        axis=1,
    )

    pts2d = np.vstack([lower, upper, clutter])
    z = np.full((pts2d.shape[0], 1), 0.4)
    return np.hstack([pts2d, z])


# --------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------
def load_reference(src: Path, pkg: str):
    """Import `pkg.gaussian_grid_map` / `pkg.a_star_planner` from a source tree."""
    if not (src / pkg).is_dir():
        return None
    sys.path.insert(0, str(src))
    try:
        grid_mod = importlib.import_module(f"{pkg}.gaussian_grid_map")
        astar_mod = importlib.import_module(f"{pkg}.a_star_planner")
        return grid_mod.FixedGaussianGridMap, astar_mod.AStarPlanner
    except Exception as exc:                      # pragma: no cover
        print(f"  ! cannot import {pkg} from {src}: {type(exc).__name__}: {exc}")
        return None


def run_case(GridCls, AStarCls, points):
    p = SCENARIO
    grid = GridCls(reso=p["reso"], half_width=p["half_width"], std=p["std"])
    hit = grid.update(points, p["robot"])
    planner = AStarCls(
        obstacle_threshold=p["obstacle_threshold"],
        obstacle_cost_weight=p["obstacle_cost_weight"],
    )
    path = planner.plan(grid, p["robot"][:2], p["goal"])
    return {
        "gmap": np.asarray(grid.gmap, dtype=np.float64),
        "minx": float(grid.minx),
        "miny": float(grid.miny),
        "cells": int(grid.cells),
        "hit": bool(hit),
        "path": np.asarray(path, dtype=np.float64) if path else np.zeros((0, 2)),
    }


def compare(name_a, a, name_b, b) -> bool:
    ok = True
    for key in ("minx", "miny", "cells", "hit"):
        if a[key] != b[key]:
            print(f"  MISMATCH {key}: {name_a}={a[key]} {name_b}={b[key]}")
            ok = False
    dg = np.max(np.abs(a["gmap"] - b["gmap"])) if a["gmap"].shape == b["gmap"].shape else np.inf
    if not (dg == 0.0):
        print(f"  MISMATCH gmap: max|diff| = {dg:g}")
        ok = False
    if a["path"].shape != b["path"].shape:
        print(f"  MISMATCH path shape: {a['path'].shape} vs {b['path'].shape}")
        ok = False
    else:
        dp = np.max(np.abs(a["path"] - b["path"])) if a["path"].size else 0.0
        if not (dp == 0.0):
            print(f"  MISMATCH path: max|diff| = {dp:g}")
            ok = False
    if ok:
        print(f"  {name_a} == {name_b}  (grid {a['gmap'].shape}, path {a['path'].shape[0]} wpts)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone-src", type=Path, default=DEFAULT_DRONE_SRC)
    ap.add_argument("--go2-src", type=Path, default=DEFAULT_GO2_SRC)
    ap.add_argument("--out", type=Path, default=HERE / "golden" / "grid_astar.npz")
    args = ap.parse_args()

    points = make_scenario()
    print(f"scenario: {points.shape[0]} lidar points, robot {SCENARIO['robot'][:2]}, "
          f"goal {SCENARIO['goal']}")

    from trajopt_core.mapping import FixedGaussianGridMap
    from trajopt_core.planning import AStarPlanner

    print("\ntrajopt_core:")
    core = run_case(FixedGaussianGridMap, AStarPlanner, points)
    print(f"  grid {core['gmap'].shape}, P in [{core['gmap'].min():.4f}, "
          f"{core['gmap'].max():.4f}], path {core['path'].shape[0]} wpts")

    all_ok = True
    for label, src, pkg in (
        ("new_mujoco (aerial)", args.drone_src, "new_mujoco"),
        ("a_star_mpc_planner (legged)", args.go2_src, "a_star_mpc_planner"),
    ):
        print(f"\n{label}:  {src}")
        ref = load_reference(src, pkg)
        if ref is None:
            print("  (skipped: source tree not available)")
            continue
        res = run_case(ref[0], ref[1], points)
        all_ok &= compare("core", core, label, res)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        points=points,
        gmap=core["gmap"],
        path=core["path"],
        minx=core["minx"],
        miny=core["miny"],
        cells=core["cells"],
        hit=core["hit"],
    )
    print(f"\ngolden written -> {args.out}")
    print("RESULT:", "all reference implementations reproduced exactly" if all_ok else "MISMATCHES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
