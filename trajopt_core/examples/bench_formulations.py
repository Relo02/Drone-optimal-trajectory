#!/usr/bin/env python3
"""
Formulation benchmarks — the numbers quoted in the report.

Every experiment isolates ONE modelling decision and measures its cost, running
the identical closed-loop mission on both platforms wherever that is meaningful.
Because the decisions live behind interfaces (`MotionModel`, `ObstacleTerm`,
`OCPConfig`), each variant is a configuration switch rather than a code fork,
which is what makes the comparisons controlled.

Experiments
-----------
  build        parametric NLP vs rebuild-every-cycle          (Sec. 7.2.2)
  horizon      solve time and sparsity vs N                   (multiple shooting)
  barrier      C^1 hinge^2 vs C^inf logistic barrier          (Sec. 4.2.5, 4.4.4)
  penalty      L1 (exact) vs L2 slack penalty, rho sweep      (Thm 6.3.1)
  discretise   forward Euler vs mid-point rule                (Sec. 2.1.3)
  warmstart    warm start on/off                              (Sec. 7.1.1, 7.2.5)
  pathcond     A* polyline conditioning on/off

Run:
    PYTHONPATH=. python3 examples/bench_formulations.py [--only build,barrier] [--out DIR]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from trajopt_core.bench import corridor_with_pillars, open_field, run_mission
from trajopt_core.models import KinematicSE2
from trajopt_core.mpc import (
    GaussianGridCost,
    HalfSpaceQuadratic,
    OCPConfig,
    SigmoidBarrier,
    SlackedHalfSpace,
    TrajectoryOCP,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_platform_demo import GRID, aerial, legged  # noqa: E402

SCEN = corridor_with_pillars()


# ---------------------------------------------------------------------------
def md_table(rows, keys, title=None) -> str:
    out = []
    if title:
        out += ["", "### " + title, ""]
    out.append("| " + " | ".join(keys) + " |")
    out.append("|" + "---|" * len(keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def mission(factory, *, terms=None, cfg_patch=None, label=None, max_steps=None, **kw):
    """Run the corridor mission for one platform with a patched configuration."""
    name, model, cfg, default_terms, gains = factory()
    if cfg_patch:
        for k, v in cfg_patch.items():
            setattr(cfg, k, v)

    scen = SCEN
    if max_steps is not None:
        scen = replace(SCEN, max_steps=max_steps)

    res = run_mission(
        model, cfg, terms if terms is not None else default_terms, scen,
        label=label or name, gains=gains, **GRID, **kw
    )
    s = res.recorder.summary()
    # Distance still to go when the loop ended.  Under a step cap this is the
    # only honest progress metric: `reached` and `mission_time_s` both saturate.
    goal_dist = (
        float(np.linalg.norm(res.trajectory[-1] - scen.goal_xy))
        if len(res.trajectory) else float("nan")
    )
    return {
        "variant": label or name,
        "reached": res.reached,
        "time_s": res.mission_time_s,
        "goal_dist_m": goal_dist,
        "len_m": res.path_length_m,
        "clr_min_m": res.min_clearance_m,
        "err_mean_m": res.mean_track_err_m,
        "success_rate": s["success_rate"],
        "iter_mean": s["iter_mean"],
        "iter_max": s["iter_max"],
        "solve_ms_mean": s["solve_ms_mean"],
        "solve_ms_p50": s["solve_ms_p50"],
        "solve_ms_p95": s["solve_ms_p95"],
        "build_ms_total": s["build_ms_total"],
        "n_builds": s["n_rebuilds"],
        "total_ms_mean": s["total_ms_mean"],
    }


# ---------------------------------------------------------------------------
# 1. Parametric NLP vs rebuild-every-cycle
# ---------------------------------------------------------------------------
def bench_build() -> str:
    """
    Does eliminating the rebuild actually pay?

    Three strategies, same mission, same solver.  The occupancy grid translates
    with the robot, so a B-spline read off it has moving knots and would force a
    rebuild of the whole CasADi graph on every cycle.  That can be avoided by
    putting the knots in the local frame and passing the cell values as a
    parameter — bit-identical, and the graph stays fixed.

    The measurement shows that this is nonetheless the wrong trade: with the
    coefficients symbolic, CasADi can no longer exploit the spline structure and
    the derivative callbacks dominate.  The point-based barrier parametrises
    cleanly and is the configuration that actually delivers build-once.
    """
    rows = []
    variants = [
        ("point barrier, parametric",
         [SigmoidBarrier(weight=200.0, alpha=4.0, r_safe=0.55, max_points=10)]),
        ("grid B-spline, baked (rebuild)",
         [GaussianGridCost(weight=100.0, half_width=GRID["grid_half_width"],
                           reso=GRID["grid_reso"], parametric=False)]),
        ("grid B-spline, parametric",
         [GaussianGridCost(weight=100.0, half_width=GRID["grid_half_width"],
                           reso=GRID["grid_reso"], parametric=True)]),
    ]
    # capped: we are measuring the PER-CYCLE cost of each strategy, not the
    # mission outcome, and two of the three variants are very expensive.
    for tag, terms in variants:
        row = mission(legged, terms=terms, max_steps=15, label=tag)
        rows.append(row)

    return md_table(
        rows,
        # median rather than mean: with a short capped run the first solve
        # (graph construction plus the first factorisation) dominates an average
        ["variant", "n_builds", "build_ms_total", "solve_ms_p50", "solve_ms_p95",
         "total_ms_mean", "iter_mean"],
        "1. Does build-once pay? Three obstacle-term strategies (legged, corridor)",
    )


# ---------------------------------------------------------------------------
# 2. Horizon scaling and sparsity
# ---------------------------------------------------------------------------
def bench_horizon() -> str:
    """
    Multiple shooting keeps both states and inputs as decision variables, so the
    variable count grows linearly with N while the Jacobian stays block-banded.
    The density column is the quantitative form of that argument.
    """
    rows = []
    for factory in (aerial, legged):
        name, model, cfg, terms, _ = factory()
        for N in (10, 20, 30, 50):
            cfg.N = N
            s = TrajectoryOCP(model, cfg, terms).structure()
            rows.append({
                "platform": name, "N": N,
                "n_variables": s["n_variables"],
                "n_constraints": s["n_constraints"],
                "jac_nnz": s["jac_nnz"],
                "jac_density": s["jac_density"],
                "hess_density": s["hess_density"],
            })
    return md_table(
        rows,
        ["platform", "N", "n_variables", "n_constraints", "jac_nnz",
         "jac_density", "hess_density"],
        "2. NLP size and sparsity vs horizon (multiple shooting)",
    )


# ---------------------------------------------------------------------------
# 3. Barrier smoothness
# ---------------------------------------------------------------------------
def bench_barrier() -> str:
    """
    Smoothness or convexity — which one actually buys solver performance?

    The hypothesis going in was that smoothness would dominate: IPOPT is a
    Newton-type method, and a penalty built on max(0,.)^2 is C^1 but not C^2, so
    its Hessian jumps whenever an obstacle crosses the activation surface, while
    the logistic barrier is C^infinity everywhere.

    The measurement says otherwise, and the reason is instructive:

      - the squared hinge is a hinge of an AFFINE function (the SCA half-space),
        hence CONVEX, and it is exactly inactive whenever the predicted point is
        clear of the safety margin — most of the horizon, most of the time;
      - the logistic barrier is isotropic and needs no normal, which spares it
        any linearisation error, but it is NON-CONVEX and contributes a non-zero
        value and gradient at every single predicted point.

    Convexity plus activation sparsity beat the extra derivative.  What the
    logistic barrier does buy is clearance, and that trade-off is real: it is a
    genuine cost/safety Pareto choice rather than a dominated option.
    """
    rows = []
    for factory in (aerial, legged):
        for term, tag in (
            (HalfSpaceQuadratic(weight=200.0, d_safe=0.6, max_points=10, per_step=True), "hinge^2 (C1)"),
            (SigmoidBarrier(weight=200.0, alpha=4.0, r_safe=0.6, max_points=10), "logistic (Cinf)"),
        ):
            name = factory()[0]
            rows.append(mission(factory, terms=[term], label=f"{name} / {tag}"))
    return md_table(
        rows,
        ["variant", "reached", "success_rate", "iter_mean", "iter_max",
         "solve_ms_mean", "solve_ms_p95", "clr_min_m"],
        "3. Obstacle penalty smoothness: C^1 hinge^2 vs C^inf logistic barrier",
    )


# ---------------------------------------------------------------------------
# 4. Exact vs quadratic penalty
# ---------------------------------------------------------------------------
def bench_penalty() -> str:
    """
    An L1 slack penalty is EXACT: the residual violation is identically zero once
    rho exceeds the multiplier of the corresponding hard constraint.  A quadratic
    penalty leaves a residual of order mu*/(2 rho) — decaying, never vanishing.

    Measured open loop on a fixed, feasible geometry so that the only thing
    varying is the penalty.
    """
    model = KinematicSE2()
    x0 = model.make_state((0.0, 0.0), 0.0)
    path = [(i * 0.4, 0.0) for i in range(25)]
    obstacle = np.array([[1.0, 0.30]])

    rows = []
    for rho in (20.0, 200.0, 2000.0, 20000.0):
        row = {"rho": rho}
        for pen in ("l1", "l2"):
            term = SlackedHalfSpace(
                d_safe=0.5, rho=rho, penalty=pen, max_points=4, check_radius=3.0
            )
            ocp = TrajectoryOCP(model, OCPConfig(N=15, dt=0.1, v_ref=0.7), [term])
            res = ocp.solve(x0, path, points_xy=obstacle)
            row[f"max_slack_{pen}"] = max(0.0, float(np.max(term.slack_value(ocp._opti.debug))))
            row[f"iter_{pen}"] = res.iterations
        rows.append(row)
    return md_table(
        rows,
        ["rho", "max_slack_l1", "max_slack_l2", "iter_l1", "iter_l2"],
        "4. Exact (L1) vs quadratic (L2) slack penalty",
    )


# ---------------------------------------------------------------------------
# 5. Discretisation order
# ---------------------------------------------------------------------------
def bench_discretise() -> str:
    """
    The aerial double integrator under a zero-order-hold input is discretised
    EXACTLY; the SE(2) kinematic model is not, and the mid-point rule buys one
    order of accuracy for the price of one addition.
    """
    rows = []
    x0 = np.array([0.0, 0.0, 0.3])
    u = np.array([0.8, 0.15, 1.2])
    for dt in (0.2, 0.1, 0.05, 0.025):
        exact = KinematicSE2.exact_step(x0, u, dt)
        row = {"dt_s": dt}
        for integ in ("euler", "midpoint"):
            approx = np.array(KinematicSE2(integrator=integ).step(x0, u, dt), dtype=float)
            row[f"err_{integ}_m"] = float(np.linalg.norm(approx[:2] - exact[:2]))
        rows.append(row)

    steps = np.array([r["dt_s"] for r in rows])
    orders = {
        integ: float(np.polyfit(np.log(steps),
                                np.log([r[f"err_{integ}_m"] for r in rows]), 1)[0])
        for integ in ("euler", "midpoint")
    }
    rows.append({
        "dt_s": "fitted order",
        "err_euler_m": round(orders["euler"], 3),
        "err_midpoint_m": round(orders["midpoint"], 3),
    })

    table = md_table(rows, ["dt_s", "err_euler_m", "err_midpoint_m"],
                     "5a. SE(2) local truncation error vs step size")

    closed = [
        mission(legged, cfg_patch=None, label=f"legged / {integ}")
        for integ in ("midpoint",)
    ]
    name, _, cfg, terms, gains = legged()
    euler_model = KinematicSE2(cfg.limits, integrator="euler")
    res = run_mission(euler_model, cfg, terms, SCEN, label="legged / euler",
                      gains=gains, **GRID)
    s = res.recorder.summary()
    closed.insert(0, {
        "variant": "legged / euler", "reached": res.reached,
        "time_s": res.mission_time_s, "len_m": res.path_length_m,
        "clr_min_m": res.min_clearance_m, "err_mean_m": res.mean_track_err_m,
        "solve_ms_mean": s["solve_ms_mean"], "iter_mean": s["iter_mean"],
    })
    return table + "\n" + md_table(
        closed,
        ["variant", "reached", "time_s", "len_m", "clr_min_m", "err_mean_m",
         "iter_mean", "solve_ms_mean"],
        "5b. Closed-loop effect (legged, corridor)",
    )


# ---------------------------------------------------------------------------
# 6. Warm start
# ---------------------------------------------------------------------------
def bench_warmstart() -> str:
    """
    Warm starting against WHICH baseline?

    "Warm start cuts iterations by 5x" is not a statement about warm starting
    unless the fallback is named, because there are two very different fallbacks:

      reference trajectory  the reference is built every cycle anyway and is an
                            excellent guess — it is the trajectory the optimiser
                            is trying to track;
      cold (zeros)          the textbook baseline, and the only one against which
                            large speed-ups are observed.

    What the measurement robustly shows, on both platforms and in both scenarios
    (4 cases out of 4), is that the SHIFTED PREVIOUS SOLUTION is never the best
    guess — it is the worst of the three every single time.  The reference
    trajectory is the best default overall.  The cold start swaps places with it
    between scenarios, so its ranking is not stable.

    A plausible explanation for the first, stable finding is structural rather
    than a tuning artefact: IPOPT is an interior-point method whose iterates must
    stay strictly inside the feasible set (Sec. 6.2.2 of the course notes — the
    central path), and the previous optimum sits ON the boundary, with obstacle
    and bound constraints active.  That is the worst place to restart a barrier
    method from.  It should be reported as a plausible mechanism, not a
    demonstrated one: it accounts for the shift being uniformly worst, but not for
    the cold/reference ranking flipping between the two scenarios, which would
    need a dedicated experiment to isolate.

    Warm starting remains the right default for ACTIVE-SET and SQP solvers, where
    the previous active set is precisely the information being reused.  The
    transferable lesson is that the advice does not carry across solver families
    unexamined — and that any quoted speed-up is meaningless until the baseline it
    is measured against is named.
    """
    rows = []
    for scen in (open_field(), corridor_with_pillars()):
        for factory in (aerial, legged):
            name = factory()[0]
            for tag, warm, cold in (
                ("shifted previous solution", True, False),
                ("reference trajectory", False, False),
                ("cold (zeros)", False, True),
            ):
                _, model, cfg, terms, gains = factory()
                cfg.solver.warm_start = warm
                cfg.solver.cold_start = cold
                res = run_mission(model, cfg, terms, scen,
                                  label=f"{name} / {tag}", gains=gains, **GRID)
                s = res.recorder.summary()
                rows.append({
                    "variant": f"{scen.name:22s} / {name} / {tag}",
                    "iter_mean": s["iter_mean"], "iter_max": s["iter_max"],
                    "solve_ms_mean": s["solve_ms_mean"],
                    "success_rate": s["success_rate"], "reached": res.reached,
                })
    return md_table(
        rows,
        ["variant", "iter_mean", "iter_max", "solve_ms_mean",
         "success_rate", "reached"],
        "6. Initial guess: shifted solution vs reference vs cold start",
    )


# ---------------------------------------------------------------------------
# 7. A* polyline conditioning
# ---------------------------------------------------------------------------
def bench_pathcond() -> str:
    """
    An 8-connected grid search emits headings that are multiples of 45 degrees,
    so the yaw reference derived from its tangent chatters at the cell scale.

    Whether that matters turns out to depend entirely on how the loop is closed,
    which is why both modes are measured here:

      direct     the first optimal input goes straight to the plant.  Only the
                 opening fraction of each plan is ever executed, and for a
                 body-frame relative-degree-1 platform that fraction is often the
                 "rotate first" part.
      lookahead  the deployed architecture: the published setpoint sits ~1 m ahead
                 on the predicted trajectory, which acts as a spatial low-pass and
                 absorbs most of the cell-scale chatter by itself.

    Two things come out of it, and the second is the larger.

    Conditioning helps modestly and consistently — roughly 6 to 17 % less distance
    left to the goal at a fixed compute budget, with one flat case.

    But the CLOSED-LOOP MODE dominates it completely: applying the first optimal
    input leaves the robot 8-11 m from the goal after 150 cycles, where publishing
    a lookahead setpoint leaves it 1.9-2.9 m.  The architectural decision to use
    the optimiser as a reference generator rather than as a direct controller is
    worth several times more than any tuning of the reference it generates.  That
    is worth stating plainly in the report, because it is easy to present the
    lookahead as an implementation detail when it is in fact load-bearing.
    """
    rows = []
    for loop in ("lookahead", "direct"):
        for factory in (aerial, legged):
            name = factory()[0]
            for tag, patch in (
                ("raw", {"path_resample_ds": 0.0, "path_smooth_window": 0}),
                ("conditioned", {"path_resample_ds": 0.20, "path_smooth_window": 5}),
            ):
                rows.append(mission(
                    factory, cfg_patch=patch, max_steps=150, closed_loop=loop,
                    label=f"{loop:9s} / {name} / {tag}",
                ))
    return md_table(
        rows,
        # capped at 150 cycles, so `reached` saturates: read goal_dist_m, the
        # distance still to go after a fixed compute budget (lower = better)
        ["variant", "goal_dist_m", "len_m", "err_mean_m", "clr_min_m",
         "iter_mean", "solve_ms_mean"],
        "7. A* polyline conditioning, both closed-loop modes (150-cycle budget)",
    )


BENCHES = {
    "build": bench_build,
    "horizon": bench_horizon,
    "barrier": bench_barrier,
    "penalty": bench_penalty,
    "discretise": bench_discretise,
    "warmstart": bench_warmstart,
    "pathcond": bench_pathcond,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of: " + ", ".join(BENCHES))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    names = args.only.split(",") if args.only else list(BENCHES)
    unknown = [n for n in names if n not in BENCHES]
    if unknown:
        raise SystemExit(f"unknown benchmark(s): {unknown}; known: {list(BENCHES)}")

    chunks = ["# Formulation benchmarks",
              "",
              "Scenario: `corridor_with_pillars` — 4 m corridor, three staggered pillars, "
              "no prior map (6 m LiDAR).  Closed loop matches deployment: the MPC publishes "
              "a lookahead setpoint tracked by the platform inner loop.",
              ""]
    for name in names:
        print(f"[{name}] running ...", flush=True)
        chunks.append(BENCHES[name]())

    report = "\n".join(chunks)
    print("\n" + report)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "benchmarks.md").write_text(report + "\n")
        print(f"\nwritten -> {args.out / 'benchmarks.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
