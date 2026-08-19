#!/usr/bin/env python3
"""
The same optimal control problem, three robots.

This script is the operative form of the claim made in the report: one
`TrajectoryOCP` class, one occupancy grid, one A* planner, one reference builder,
one lookahead rule — instantiated with `MotionModel` objects whose state and
input spaces need have nothing in common.

    aerial   x = [px py pz vx vy vz psi] in R^7,  u = [ax ay az psi_dot] in R^4
    legged   x = [px py psi]             in R^3,  u = [vx vy omega]      in R^3
    g1       x = [px py psi]             in R^3,  u = [vx vy omega]      in R^3

Nothing outside `trajopt_core.models` changes between the runs.  What differs is
the RELATIVE DEGREE between the planned output (the centre-of-mass position) and
the commanded input: two for the aerial platform, whose inner attitude loop
cannot change velocity instantaneously, one for the legged platform, whose gait
controller accepts velocity commands directly.  The humanoid shares the legged
model exactly — it differs only in its numbers and in a one-sided admissible set
(no reverse walking), which is the point: two platforms that share a relative
degree share an instantiation.

Run:
    PYTHONPATH=. python3 examples/cross_platform_demo.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajopt_core.bench import corridor_with_pillars, open_field, run_mission, table
from trajopt_core.models import (
    CostWeights,
    DoubleIntegratorZ,
    InnerLoopGains,
    KinematicSE2,
    ModelLimits,
)
from trajopt_core.mpc import OCPConfig, SigmoidBarrier, SolverOptions, TrajectoryOCP

GRID = dict(grid_reso=0.25, grid_half_width=5.0, grid_std=0.7)


def aerial():
    """Relative degree 2: the input is an acceleration, bounded by the tilt limit."""
    model = DoubleIntegratorZ(
        ModelLimits(v_max_xy=2.0, v_max_z=1.0, a_max_xy=2.0, a_max_z=1.5, yaw_rate_max=1.5)
    )
    cfg = OCPConfig(
        N=25, dt=0.1, v_ref=1.0, z_ref=1.5,
        weights=CostWeights(q_pos_xy=30.0, q_pos_z=20.0, q_vel_xy=10.0, q_vel_z=2.0,
                            q_yaw=0.2, q_terminal=10.0,
                            r_lin_xy=1.0, r_lin_z=1.5, r_ang=0.1, r_jerk=0.3),
        lookahead_dist=1.2,
        solver=SolverOptions(max_iter=100),
    )
    terms = [SigmoidBarrier(weight=200.0, alpha=4.0, r_safe=0.7, max_points=10)]
    gains = InnerLoopGains(kp_pos=0.9, kd_pos=1.2, kp_yaw=1.5)
    return "aerial", model, cfg, terms, gains


def legged():
    """Relative degree 1: the input is a velocity, bounded by the gait envelope."""
    model = KinematicSE2(
        ModelLimits(v_max_xy=1.0, v_max_lat=0.5, yaw_rate_max=1.5), integrator="midpoint"
    )
    cfg = OCPConfig(
        N=25, dt=0.1, v_ref=0.7,
        weights=CostWeights(q_pos_xy=30.0, q_yaw=1.0, q_terminal=10.0,
                            r_lin_xy=1.0, r_ang=0.5, r_jerk=0.3),
        lookahead_dist=1.0,
        solver=SolverOptions(max_iter=100),
    )
    terms = [SigmoidBarrier(weight=200.0, alpha=4.0, r_safe=0.55, max_points=10)]
    gains = InnerLoopGains(kp_pos=1.0, kp_yaw=1.5)
    return "legged", model, cfg, terms, gains


def g1():
    """
    Relative degree 1 as well, and deliberately the SAME model class as the Go2:
    the humanoid is a re-tuning of the legged instantiation, not a third stack.
    The one structural difference is the admissible set — a biped does not walk
    backward, so U_Sigma is one-sided in body-x (`v_min_xy = 0`) while f_Sigma is
    untouched.

    These numbers mirror `config/g1_overrides.yaml`; the test
    `test_demo_g1_matches_the_shipped_profile` keeps the two from drifting.
    """
    model = KinematicSE2(
        ModelLimits(v_max_xy=0.8, v_min_xy=0.0, v_max_lat=0.4, yaw_rate_max=1.0),
        integrator="midpoint",
    )
    cfg = OCPConfig(
        N=20, dt=0.1, v_ref=0.5,
        weights=CostWeights(q_pos_xy=200.0, q_yaw=1.0, q_terminal=100.0,
                            r_lin_xy=1.0, r_ang=0.5, r_jerk=0.5),
        lookahead_dist=0.9,
        solver=SolverOptions(max_iter=100),
    )
    terms = [SigmoidBarrier(weight=200.0, alpha=4.0, r_safe=0.5, max_points=12)]
    gains = InnerLoopGains(kp_pos=1.0, kd_pos=0.0, kp_yaw=1.5)
    return "g1", model, cfg, terms, gains


def instantiation_table(platforms) -> str:
    """
    The single place in the report where the state spaces appear side by side.
    Generated from the code so it can never drift from what actually runs.
    """
    rows = []
    for name, model, cfg, terms, _ in platforms:
        d = model.describe()
        rows.append({
            "platform": name,
            "model": d["name"],
            "state x": "[" + ", ".join(d["state_layout"]) + "]",
            "input u": "[" + ", ".join(d["input_layout"]) + "]",
            "nx": d["nx"],
            "nu": d["nu"],
            "rel. degree": d["relative_degree"],
            "N": cfg.N,
            "dt [s]": cfg.dt,
            "v_ref [m/s]": cfg.v_ref,
            "lookahead [m]": cfg.lookahead_dist,
        })
    keys = list(rows[0].keys())
    head = "| " + " | ".join(keys) + " |"
    sep = "|" + "---|" * len(keys)
    body = ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="directory for CSV/JSON output")
    args = ap.parse_args()

    platforms = [aerial(), legged(), g1()]
    scenarios = [open_field(), corridor_with_pillars()]

    print("=" * 78)
    print("INSTANTIATION — the only place the models are compared side by side")
    print("=" * 78)
    print(instantiation_table(platforms))

    # --- the shared problem structure, reported per platform ---------------
    print("\n" + "=" * 78)
    print("NLP STRUCTURE — same builder, same formulation, different model")
    print("=" * 78)
    struct_keys = ["model", "n_variables", "n_constraints", "n_parameters",
                   "jac_nnz", "jac_density", "hess_density", "is_parametric"]
    structs = []
    for _, model, cfg, terms, _ in platforms:
        s = TrajectoryOCP(model, cfg, terms).structure()
        structs.append({k: s[k] for k in struct_keys})
    head = "| " + " | ".join(struct_keys) + " |"
    print(head)
    print("|" + "---|" * len(struct_keys))
    for s in structs:
        print("| " + " | ".join(
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in s.values()
        ) + " |")

    # --- closed-loop missions ---------------------------------------------
    results = []
    for scen in scenarios:
        print(f"\n{'=' * 78}\nMISSION: {scen.name}  "
              f"({len(scen.obstacles)} obstacle points, goal {scen.goal_xy})\n{'=' * 78}")
        for name, model, cfg, terms, gains in platforms:
            res = run_mission(model, cfg, terms, scen, label=f"{name}/{scen.name}",
                              gains=gains, **GRID)
            results.append(res)
            print(f"  {name:7s} reached={str(res.reached):5s} "
                  f"t={res.mission_time_s:5.1f}s  len={res.path_length_m:5.2f}m  "
                  f"clr_min={res.min_clearance_m:5.2f}m  "
                  f"err_mean={res.mean_track_err_m:5.3f}m  "
                  f"solve p95={res.recorder.summary()['solve_ms_p95']:6.1f}ms  "
                  f"builds={res.recorder.summary()['n_rebuilds']}")

    print("\n" + "=" * 78)
    print("MISSION METRICS — identical code path for every platform")
    print("=" * 78)
    print(table(results, [
        "label", "reached", "mission_time_s", "path_length_m", "min_clearance_m",
        "mean_track_err_m", "max_track_err_m",
    ]))

    print("\n" + "=" * 78)
    print("SOLVER METRICS — the distribution, not the mean: the tail decides "
          "real-time admissibility")
    print("=" * 78)
    print(table([r.recorder.summary() for r in results], [
        "label", "n", "success_rate", "iter_mean", "iter_max",
        "solve_ms_mean", "solve_ms_p95", "solve_ms_max", "deadline_miss_rate",
    ]))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for r in results:
            stem = r.label.replace("/", "_")
            r.recorder.to_csv(args.out / f"solves_{stem}.csv")
            np.savetxt(args.out / f"traj_{stem}.csv", r.trajectory,
                       delimiter=",", header="x,y", comments="")
        (args.out / "summary.json").write_text(
            json.dumps([r.summary() for r in results], indent=2, default=str)
        )
        print(f"\nartefacts written to {args.out}")

    ok = all(r.reached for r in results)
    print(f"\nRESULT: {'every mission reached its goal' if ok else 'SOME MISSIONS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
