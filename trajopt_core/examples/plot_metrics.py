#!/usr/bin/env python3
"""
Figures from the recorded metrics.

`cross_platform_demo.py --out DIR` writes the raw artefacts; this script turns
them into the plots the report uses.  The two steps are kept apart on purpose:
recording is slow and deterministic, plotting is fast and a matter of taste, and
nothing should have to re-run a mission to change an axis label.

The quantities plotted are the ones the formulation actually controls:

  trajectories   what the closed loop did, against the obstacles it could see
  solve time     as a DISTRIBUTION with the deadline drawn on it — the tail is
                 what decides real-time admissibility, not the mean
  iterations     per cycle: how hard the NLP was, cycle by cycle
  cost           the optimal value along the mission, i.e. how the closed loop
                 settles as the goal is approached

Run:
    PYTHONPATH=. python3 examples/cross_platform_demo.py --out ../results/cross_platform
    PYTHONPATH=. python3 examples/plot_metrics.py ../results/cross_platform

A live ROS run records the same quantities on /mpc/diagnostics; feed the CSV
produced by `ros2 topic echo --csv` straight in:

    PYTHONPATH=. python3 examples/plot_metrics.py --diagnostics diagnostics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")            # write files, never require a display
    import matplotlib.pyplot as plt
except ImportError:                  # pragma: no cover - depends on the environment
    raise SystemExit("matplotlib is required:  python3 -m pip install --user matplotlib")

from trajopt_core.bench import corridor_with_pillars, open_field

#: Scenario name -> the object, so the obstacles can be drawn under the
#: trajectories without the recorder having to serialise them.
SCENARIOS = {s.name: s for s in (open_field(), corridor_with_pillars())}

#: One colour per platform, kept consistent across every figure.
COLOURS = {"aerial": "#1f77b4", "legged": "#d62728", "g1": "#2ca02c"}


def _colour(platform: str) -> str:
    return COLOURS.get(platform, "#7f7f7f")


def _platform(run: dict) -> str:
    """
    The recorder's `platform` field is the MODEL name, which the Go2 and the G1
    share — so it cannot tell those two apart.  The mission label carries the
    profile name (`g1/corridor_with_pillars`), which can.
    """
    return str(run.get("label", "")).split("/")[0] or run.get("platform", "?")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_runs(results_dir: Path) -> list[dict]:
    """One dict per mission: its summary, its per-solve records, its trajectory."""
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        raise SystemExit(
            f"{summary_file} not found — run:\n"
            f"    PYTHONPATH=. python3 examples/cross_platform_demo.py --out {results_dir}"
        )

    runs = []
    for entry in json.loads(summary_file.read_text()):
        stem = entry["label"].replace("/", "_")
        run = dict(entry)
        run["solves"] = _read_solves(results_dir / f"solves_{stem}.csv")
        traj_file = results_dir / f"traj_{stem}.csv"
        run["trajectory"] = (
            np.atleast_2d(np.loadtxt(traj_file, delimiter=",", skiprows=1))
            if traj_file.exists() else np.zeros((0, 2))
        )
        runs.append(run)
    return runs


def _read_solves(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}

    out: dict[str, np.ndarray] = {}
    for key in ("step", "iterations", "cost", "solve_ms", "total_ms"):
        if key in rows[0]:
            out[key] = np.array([_float(r[key]) for r in rows])
    out["success"] = np.array([r.get("success", "True") == "True" for r in rows])
    return out


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_trajectories(runs, out_dir: Path) -> Path:
    """What the closed loop did — one panel per scenario, all platforms overlaid."""
    scenarios = sorted({r["scenario"] for r in runs})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.0 * len(scenarios), 5.0),
                             squeeze=False)

    for ax, name in zip(axes[0], scenarios):
        scen = SCENARIOS.get(name)
        if scen is not None and len(scen.obstacles):
            ax.scatter(scen.obstacles[:, 0], scen.obstacles[:, 1],
                       s=3, c="0.55", label="obstacles", zorder=1)
        if scen is not None:
            ax.plot(*scen.start_xy, "ko", ms=8, label="start", zorder=4)
            ax.plot(*scen.goal_xy, "k*", ms=15, label="goal", zorder=4)

        for run in [r for r in runs if r["scenario"] == name]:
            traj = run["trajectory"]
            if len(traj):
                ax.plot(traj[:, 0], traj[:, 1], lw=2.0, zorder=3,
                        color=_colour(_platform(run)),
                        label=f"{_platform(run)} "
                              f"({run['path_length_m']:.1f} m, "
                              f"{run['mission_time_s']:.1f} s)")

        ax.set_title(name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Closed-loop trajectories — same core, three instantiations")
    return _save(fig, out_dir / "trajectories.png")


def plot_solve_time(runs, out_dir: Path) -> Path:
    """
    The distribution, with the deadline on it.  A mean hides exactly the events
    that make a controller inadmissible, so this figure deliberately shows the
    spread and the outliers rather than a bar per platform.

    Both axes are logarithmic, because the first solve of every run carries the
    CasADi graph construction and is one to two ORDERS OF MAGNITUDE above the
    steady-state cycle: on a linear axis that single point flattens everything
    else into a line.  A box plot is used rather than a violin because its
    whiskers are exact order statistics, so the log axis distorts nothing.
    """
    labelled = [(r, r["solves"].get("solve_ms")) for r in runs]
    labelled = [(r, v) for r, v in labelled if v is not None and len(v)]
    if not labelled:
        return None

    fig, (ax_dist, ax_ecdf) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    data = [v for _, v in labelled]
    names = [r["label"] for r, _ in labelled]
    box = ax_dist.boxplot(data, patch_artist=True, widths=0.6,
                          flierprops=dict(marker=".", ms=4, mfc="0.3", mec="none"),
                          medianprops=dict(color="k", lw=1.6))
    for patch, (run, _) in zip(box["boxes"], labelled):
        patch.set_facecolor(_colour(_platform(run)))
        patch.set_alpha(0.55)

    for i, (_, values) in enumerate(labelled, start=1):
        ax_dist.plot(i, np.percentile(values, 95), "kv", ms=7,
                     label="p95" if i == 1 else None)

    ax_dist.set_yscale("log")

    deadlines = {r.get("deadline_ms") for r, _ in labelled if r.get("deadline_ms")}
    for d in deadlines:
        ax_dist.axhline(d, color="k", ls="--", lw=1.0, alpha=0.7)
        ax_dist.text(0.5, d, f" deadline {d:.0f} ms", va="bottom", fontsize=8)

    ax_dist.set_xticks(range(1, len(names) + 1))
    ax_dist.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax_dist.set_ylabel("solve time [ms]")
    ax_dist.set_title("Solve-time distribution")
    ax_dist.grid(alpha=0.3, axis="y")
    ax_dist.legend(fontsize=8)

    for run, values in labelled:
        ordered = np.sort(values)
        ax_ecdf.plot(ordered, np.linspace(0, 1, len(ordered)), lw=1.8,
                     color=_colour(_platform(run)),
                     ls="-" if "corridor" in run["scenario"] else "--",
                     label=run["label"])
    for d in deadlines:
        ax_ecdf.axvline(d, color="k", ls="--", lw=1.0, alpha=0.7)
    ax_ecdf.set_xscale("log")
    ax_ecdf.set_xlabel("solve time [ms]")
    ax_ecdf.set_ylabel("fraction of cycles below")
    ax_ecdf.set_title("Empirical CDF — the tail is the whole question")
    ax_ecdf.grid(alpha=0.3)
    ax_ecdf.legend(fontsize=7)

    return _save(fig, out_dir / "solve_time.png")


def plot_per_cycle(runs, out_dir: Path) -> Path:
    """Iterations and optimal value along the mission, cycle by cycle."""
    fig, (ax_it, ax_cost) = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)

    for run in runs:
        s = run["solves"]
        if not s or "iterations" not in s:
            continue
        style = dict(lw=1.4, color=_colour(_platform(run)),
                     ls="-" if "corridor" in run["scenario"] else "--",
                     label=run["label"])
        ax_it.plot(s["step"], s["iterations"], **style)
        if "cost" in s:
            cost = np.where(np.isfinite(s["cost"]), s["cost"], np.nan)
            ax_cost.plot(s["step"], cost, **style)

    ax_it.set_ylabel("IPOPT iterations")
    ax_it.set_title("Per-cycle solver effort — the peaks are the obstacles")
    ax_it.grid(alpha=0.3)
    ax_it.legend(fontsize=7, ncol=2)

    ax_cost.set_xlabel("MPC cycle")
    ax_cost.set_ylabel("optimal value J*")
    ax_cost.set_yscale("log")
    ax_cost.set_title("Optimal value along the mission")
    ax_cost.grid(alpha=0.3)

    return _save(fig, out_dir / "per_cycle.png")


def plot_mission_summary(runs, out_dir: Path) -> Path:
    """The trade-off, four bars at a time: speed, path length, clearance, tracking."""
    metrics = [
        ("mission_time_s", "mission time [s]", False),
        ("path_length_m", "path length [m]", False),
        ("min_clearance_m", "min clearance [m]", True),
        ("mean_track_err_m", "mean tracking error [m]", False),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.6))

    labels = [r["label"] for r in runs]
    colours = [_colour(_platform(r)) for r in runs]
    x = np.arange(len(runs))

    for ax, (key, title, higher_is_better) in zip(axes, metrics):
        values = [r.get(key, float("nan")) for r in runs]
        values = [np.nan if v in (None, float("inf")) else v for v in values]
        ax.bar(x, values, color=colours, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(title + (" ↑" if higher_is_better else " ↓"), fontsize=10)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Mission metrics — different limits, identical code path")
    return _save(fig, out_dir / "mission_summary.png")


def plot_diagnostics(csv_file: Path, out_dir: Path) -> Path:
    """
    The live counterpart: /mpc/diagnostics from a running robot, as recorded by
    `ros2 topic echo --csv`.  Layout is
    [success, cost, solve_ms, iterations, cumulative_failures].
    """
    rows = []
    for line in csv_file.read_text().splitlines():
        parts = [p for p in line.replace('"', "").split(",") if p.strip()]
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue                       # header or an interleaved message
        if len(values) >= 5:
            rows.append(values[-5:])       # tolerate a leading stamp column

    if not rows:
        raise SystemExit(f"no [success, cost, solve_ms, iters, fails] rows in {csv_file}")

    arr = np.asarray(rows)
    ok, cost, solve_ms, iters, fails = (arr[:, i] for i in range(5))
    step = np.arange(len(arr))

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.0), sharex=True)
    axes[0].plot(step, solve_ms, lw=1.3, color="#1f77b4")
    axes[0].axhline(float(np.percentile(solve_ms, 95)), color="k", ls="--", lw=1.0,
                    label=f"p95 = {np.percentile(solve_ms, 95):.1f} ms")
    axes[0].set_ylabel("solve time [ms]")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{csv_file.name} — {len(arr)} cycles, "
                      f"{100.0 * ok.mean():.1f}% solved, {int(fails[-1])} failures")

    axes[1].plot(step, iters, lw=1.3, color="#d62728")
    axes[1].set_ylabel("iterations")

    axes[2].plot(step, np.where(cost > 0, cost, np.nan), lw=1.3, color="#2ca02c")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("optimal value J*")
    axes[2].set_xlabel("MPC cycle")

    for ax in axes:
        ax.grid(alpha=0.3)

    return _save(fig, out_dir / f"{csv_file.stem}.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("results_dir", type=Path, nargs="?",
                    help="directory written by cross_platform_demo.py --out")
    ap.add_argument("--diagnostics", type=Path, default=None,
                    help="CSV from `ros2 topic echo --csv /mpc/diagnostics`")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the figures (default: alongside the input)")
    args = ap.parse_args()

    if not args.results_dir and not args.diagnostics:
        ap.error("give a results directory, --diagnostics, or both")

    if args.diagnostics:
        out_dir = args.out or args.diagnostics.parent
        print(f"plotting {args.diagnostics} -> {out_dir}")
        plot_diagnostics(args.diagnostics, out_dir)

    if args.results_dir:
        out_dir = args.out or args.results_dir / "figures"
        runs = load_runs(args.results_dir)
        print(f"plotting {len(runs)} missions -> {out_dir}")
        plot_trajectories(runs, out_dir)
        plot_solve_time(runs, out_dir)
        plot_per_cycle(runs, out_dir)
        plot_mission_summary(runs, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
