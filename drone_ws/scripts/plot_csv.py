#!/usr/bin/env python3
import argparse
import csv
import os
import sys


def _read_csv(path, expected_cols):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            vals = [float(v) for v in row[:expected_cols]]
            rows.append(vals)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot CSV data with matplotlib.")
    parser.add_argument(
        "--dir",
        default="plot_data",
        help="Directory containing CSV files (default: plot_data)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output image path (e.g., mpc_plot.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib import failed: {exc}", file=sys.stderr)
        print("If you see a NumPy ABI error, install numpy<2 or rebuild matplotlib.", file=sys.stderr)
        return 1

    data_dir = args.dir
    cost_rows = _read_csv(os.path.join(data_dir, "cost.csv"), 2)
    pos_rows = _read_csv(os.path.join(data_dir, "pos.csv"), 4)
    vel_rows = _read_csv(os.path.join(data_dir, "vel.csv"), 4)
    ref_all = _read_csv(os.path.join(data_dir, "traj_ref_all.csv"), 3)
    ref_last = _read_csv(os.path.join(data_dir, "traj_ref_last.csv"), 3)
    opt_all = _read_csv(os.path.join(data_dir, "traj_opt_all.csv"), 3)
    opt_last = _read_csv(os.path.join(data_dir, "traj_opt_last.csv"), 3)

    fig = plt.figure(figsize=(12, 9))

    ax_traj = fig.add_subplot(2, 2, 1)
    ax_cost = fig.add_subplot(2, 2, 2)
    ax_pos = fig.add_subplot(2, 2, 3)
    ax_vel = fig.add_subplot(2, 2, 4)

    if ref_all:
        ax_traj.scatter(
            [p[0] for p in ref_all],
            [p[1] for p in ref_all],
            s=4,
            alpha=0.3,
            label="ref all",
        )
    if opt_all:
        ax_traj.scatter(
            [p[0] for p in opt_all],
            [p[1] for p in opt_all],
            s=4,
            alpha=0.3,
            label="opt all",
        )
    if ref_last:
        ax_traj.plot([p[0] for p in ref_last], [p[1] for p in ref_last], label="ref last")
    if opt_last:
        ax_traj.plot([p[0] for p in opt_last], [p[1] for p in opt_last], label="opt last")
    ax_traj.set_title("Trajectories (E-N)")
    ax_traj.set_xlabel("E [m]")
    ax_traj.set_ylabel("N [m]")
    ax_traj.legend(loc="best")

    if cost_rows:
        ax_cost.plot([r[0] for r in cost_rows], [r[1] for r in cost_rows], color="#2ca02c")
    ax_cost.set_title("MPC Objective")
    ax_cost.set_xlabel("t [s]")
    ax_cost.set_ylabel("cost")

    if pos_rows:
        t = [r[0] for r in pos_rows]
        ax_pos.plot(t, [r[1] for r in pos_rows], label="E")
        ax_pos.plot(t, [r[2] for r in pos_rows], label="N")
        ax_pos.plot(t, [r[3] for r in pos_rows], label="U")
    ax_pos.set_title("Position (ENU)")
    ax_pos.set_xlabel("t [s]")
    ax_pos.set_ylabel("m")
    ax_pos.legend(loc="best")

    if vel_rows:
        t = [r[0] for r in vel_rows]
        ax_vel.plot(t, [r[1] for r in vel_rows], label="E")
        ax_vel.plot(t, [r[2] for r in vel_rows], label="N")
        ax_vel.plot(t, [r[3] for r in vel_rows], label="U")
    ax_vel.set_title("Velocity (ENU)")
    ax_vel.set_xlabel("t [s]")
    ax_vel.set_ylabel("m/s")
    ax_vel.legend(loc="best")

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Wrote plot to {args.out}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
