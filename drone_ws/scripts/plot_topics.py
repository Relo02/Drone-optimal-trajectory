#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def _ned_to_enu(vec):
    return np.array([vec[1], vec[0], -vec[2]], dtype=float)


def _read_bag(bag_path):
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }

    data = {
        "t0": None,
        "cost": [],
        "cost_t": [],
        "odom_pos": [],
        "odom_vel": [],
        "odom_t": [],
        "ref_paths": [],
        "ref_t": [],
        "opt_paths": [],
        "opt_t": [],
    }

    while reader.has_next():
        topic, msg_data, t = reader.read_next()
        if topic not in topic_types:
            continue

        if data["t0"] is None:
            data["t0"] = t
        t_rel = (t - data["t0"]) * 1e-9

        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(msg_data, msg_type)

        if topic == "/mpc/cost":
            data["cost"].append(float(msg.data))
            data["cost_t"].append(t_rel)
        elif topic == "/fmu/out/vehicle_odometry":
            pos = _ned_to_enu(msg.position)
            vel = _ned_to_enu(msg.velocity)
            data["odom_pos"].append(pos)
            data["odom_vel"].append(vel)
            data["odom_t"].append(t_rel)
        elif topic == "/mpc/reference_trajectory":
            if msg.poses:
                pts = np.array(
                    [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
                    dtype=float,
                )
                data["ref_paths"].append(pts)
                data["ref_t"].append(t_rel)
        elif topic == "/mpc/optimal_trajectory":
            if msg.poses:
                pts = np.array(
                    [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
                    dtype=float,
                )
                data["opt_paths"].append(pts)
                data["opt_t"].append(t_rel)

    return data


def _plot_matplotlib(data, out_path=None, show=True):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib import failed: {exc}") from exc

    fig_traj = plt.figure(figsize=(8, 7))
    ax_traj = fig_traj.add_subplot(1, 1, 1, projection="3d")

    if data["ref_paths"]:
        ref_all = np.vstack(data["ref_paths"])
        ax_traj.scatter(
            ref_all[:, 0],
            ref_all[:, 1],
            ref_all[:, 2],
            s=2,
            alpha=0.3,
            label="Reference horizon (all timesteps)",
        )
        ref_last = data["ref_paths"][-1]
        ax_traj.plot(
            ref_last[:, 0],
            ref_last[:, 1],
            ref_last[:, 2],
            color="tab:blue",
            label="Reference horizon (last)",
        )

    if data["opt_paths"]:
        opt_all = np.vstack(data["opt_paths"])
        ax_traj.scatter(
            opt_all[:, 0],
            opt_all[:, 1],
            opt_all[:, 2],
            s=2,
            alpha=0.3,
            label="MPC predicted (all timesteps)",
        )
        opt_last = data["opt_paths"][-1]
        ax_traj.plot(
            opt_last[:, 0],
            opt_last[:, 1],
            opt_last[:, 2],
            color="tab:orange",
            label="MPC predicted (last)",
        )

    ax_traj.set_title("Reference vs MPC Trajectories (ENU)")
    ax_traj.set_xlabel("E [m]")
    ax_traj.set_ylabel("N [m]")
    ax_traj.set_zlabel("U [m]")
    ax_traj.legend(loc="best")

    if data["odom_t"]:
        odom_pos = np.array(data["odom_pos"])
        ax_xy.plot(odom_pos[:, 0], odom_pos[:, 1], color="tab:green", label="Odometry")
    if data["ref_paths"]:
        ax_xy.plot(
            ref_last[:, 0],
            ref_last[:, 1],
            color="tab:blue",
            label="Reference (last)",
        )
    if data["opt_paths"]:
        opt_all = np.vstack(data["opt_paths"])
        ax_xy.scatter(
            opt_all[:, 0],
            opt_all[:, 1],
            s=4,
            alpha=0.2,
            color="tab:orange",
            label="MPC predicted (all)",
        )
        ax_xy.plot(
            opt_last[:, 0],
            opt_last[:, 1],
            color="tab:orange",
            label="MPC predicted (last)",
        )

    fig_metrics, (ax_cost, ax_pos, ax_vel) = plt.subplots(1, 3, figsize=(15, 4))
    fig_xy, ax_xy = plt.subplots(1, 1, figsize=(6, 6))

    if data["cost_t"]:
        ax_cost.plot(data["cost_t"], data["cost"], color="tab:green")
    ax_cost.set_title("MPC Objective")
    ax_cost.set_xlabel("t [s]")
    ax_cost.set_ylabel("cost")

    if data["odom_t"]:
        odom_t = np.array(data["odom_t"])
        odom_pos = np.array(data["odom_pos"])
        odom_vel = np.array(data["odom_vel"])

        ax_pos.plot(odom_t, odom_pos[:, 0], label="East (x)")
        ax_pos.plot(odom_t, odom_pos[:, 1], label="North (y)")
        ax_pos.plot(odom_t, odom_pos[:, 2], label="Up (z)")
        ax_pos.set_title("Position (ENU)")
        ax_pos.set_xlabel("t [s]")
        ax_pos.set_ylabel("m")
        ax_pos.legend(loc="best")

        ax_vel.plot(odom_t, odom_vel[:, 0], label="East (x)")
        ax_vel.plot(odom_t, odom_vel[:, 1], label="North (y)")
        ax_vel.plot(odom_t, odom_vel[:, 2], label="Up (z)")
        ax_vel.set_title("Velocity (ENU)")
        ax_vel.set_xlabel("t [s]")
        ax_vel.set_ylabel("m/s")
        ax_vel.legend(loc="best")

    fig_traj.tight_layout()
    fig_metrics.tight_layout()
    ax_xy.set_title("Trajectory (E-N)")
    ax_xy.set_xlabel("E [m]")
    ax_xy.set_ylabel("N [m]")
    ax_xy.axis("equal")
    ax_xy.legend(loc="best")

    if out_path:
        base, ext = os.path.splitext(out_path)
        if not ext:
            ext = ".png"
        fig_traj.savefig(f"{base}_traj{ext}", dpi=150)
        fig_metrics.savefig(f"{base}_metrics{ext}", dpi=150)
        fig_xy.savefig(f"{base}_xy{ext}", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig_traj)
        plt.close(fig_metrics)
        plt.close(fig_xy)


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def _plot_gnuplot(data, out_path):
    if not shutil.which("gnuplot"):
        raise RuntimeError("gnuplot not found on PATH")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=out_dir, prefix="mpc_plot_data_") as tmpdir:
        ref_all = np.vstack(data["ref_paths"]) if data["ref_paths"] else np.empty((0, 3))
        opt_all = np.vstack(data["opt_paths"]) if data["opt_paths"] else np.empty((0, 3))
        ref_last = data["ref_paths"][-1] if data["ref_paths"] else np.empty((0, 3))
        opt_last = data["opt_paths"][-1] if data["opt_paths"] else np.empty((0, 3))

        _write_csv(os.path.join(tmpdir, "traj_ref_all.csv"), ["x", "y", "z"], ref_all.tolist())
        _write_csv(os.path.join(tmpdir, "traj_opt_all.csv"), ["x", "y", "z"], opt_all.tolist())
        _write_csv(os.path.join(tmpdir, "traj_ref_last.csv"), ["x", "y", "z"], ref_last.tolist())
        _write_csv(os.path.join(tmpdir, "traj_opt_last.csv"), ["x", "y", "z"], opt_last.tolist())
        _write_csv(os.path.join(tmpdir, "cost.csv"), ["t", "cost"], list(zip(data["cost_t"], data["cost"])))

        if data["odom_t"]:
            odom_pos = np.array(data["odom_pos"])
            odom_vel = np.array(data["odom_vel"])
            pos_rows = np.column_stack([data["odom_t"], odom_pos]).tolist()
            vel_rows = np.column_stack([data["odom_t"], odom_vel]).tolist()
        else:
            pos_rows = []
            vel_rows = []

        _write_csv(os.path.join(tmpdir, "pos.csv"), ["t", "e", "n", "u"], pos_rows)
        _write_csv(os.path.join(tmpdir, "vel.csv"), ["t", "e", "n", "u"], vel_rows)

        script = f"""
set terminal pngcairo size 1200,900
set output '{out_path}'
set multiplot layout 2,2 title 'MPC Topics'

set view 60, 30
set xlabel 'E [m]'
set ylabel 'N [m]'
set zlabel 'U [m]'
splot \\
  'traj_ref_all.csv' using 1:2:3 with points pt 7 ps 0.3 lc rgb '#1f77b4' title 'ref all', \\
  'traj_ref_last.csv' using 1:2:3 with lines lw 2 lc rgb '#1f77b4' title 'ref last', \\
  'traj_opt_all.csv' using 1:2:3 with points pt 7 ps 0.3 lc rgb '#ff7f0e' title 'opt all', \\
  'traj_opt_last.csv' using 1:2:3 with lines lw 2 lc rgb '#ff7f0e' title 'opt last'

unset zlabel
set xlabel 't [s]'
set ylabel 'cost'
plot 'cost.csv' using 1:2 with lines lc rgb '#2ca02c' title 'cost'

set xlabel 't [s]'
set ylabel 'm'
plot 'pos.csv' using 1:2 with lines title 'E', \\
     '' using 1:3 with lines title 'N', \\
     '' using 1:4 with lines title 'U'

set xlabel 't [s]'
set ylabel 'm/s'
plot 'vel.csv' using 1:2 with lines title 'E', \\
     '' using 1:3 with lines title 'N', \\
     '' using 1:4 with lines title 'U'

unset multiplot
"""

        script_path = os.path.join(tmpdir, "plot.gp")
        with open(script_path, "w") as handle:
            handle.write(script.strip() + "\n")

        subprocess.run(["gnuplot", script_path], check=True, cwd=tmpdir)


def main():
    parser = argparse.ArgumentParser(description="Plot MPC bag topics.")
    parser.add_argument(
        "--bag",
        default=os.path.join(os.path.dirname(__file__), "mpc_bag_0"),
        help="Path to rosbag2 directory (default: drone_ws/scripts/mpc_bag_0)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output image path (e.g., plot.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "matplotlib", "gnuplot", "csv"],
        default="auto",
        help="Plot backend (default: auto)",
    )
    args = parser.parse_args()

    data = _read_bag(args.bag)
    if data["t0"] is None:
        raise RuntimeError(f"No messages found in bag: {args.bag}")

    backend = args.backend
    if backend == "auto":
        try:
            _plot_matplotlib(data, out_path=args.out, show=not args.no_show)
            return
        except Exception as exc:
            if shutil.which("gnuplot"):
                backend = "gnuplot"
            else:
                print(f"matplotlib failed: {exc}")
                print("gnuplot not found; falling back to CSV export")
                backend = "csv"

    if backend == "matplotlib":
        _plot_matplotlib(data, out_path=args.out, show=not args.no_show)
        return

    if backend == "gnuplot":
        if not shutil.which("gnuplot"):
            raise RuntimeError("gnuplot not found on PATH (use --backend matplotlib or --backend csv)")
        out_path = args.out
        if out_path is None:
            out_path = os.path.join(args.bag, "mpc_plot.png")
        _plot_gnuplot(data, out_path=out_path)
        print(f"Wrote plot to {out_path}")
        return

    out_dir = args.out or os.path.join(args.bag, "plot_data")
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "cost.csv"), ["t", "cost"], list(zip(data["cost_t"], data["cost"])))
    if data["odom_t"]:
        odom_pos = np.array(data["odom_pos"])
        odom_vel = np.array(data["odom_vel"])
        _write_csv(
            os.path.join(out_dir, "pos.csv"),
            ["t", "e", "n", "u"],
            np.column_stack([data["odom_t"], odom_pos]).tolist(),
        )
        _write_csv(
            os.path.join(out_dir, "vel.csv"),
            ["t", "e", "n", "u"],
            np.column_stack([data["odom_t"], odom_vel]).tolist(),
        )
    if data["ref_paths"]:
        ref_all = np.vstack(data["ref_paths"])
        _write_csv(os.path.join(out_dir, "traj_ref_all.csv"), ["x", "y", "z"], ref_all.tolist())
        _write_csv(os.path.join(out_dir, "traj_ref_last.csv"), ["x", "y", "z"], data["ref_paths"][-1].tolist())
    if data["opt_paths"]:
        opt_all = np.vstack(data["opt_paths"])
        _write_csv(os.path.join(out_dir, "traj_opt_all.csv"), ["x", "y", "z"], opt_all.tolist())
        _write_csv(os.path.join(out_dir, "traj_opt_last.csv"), ["x", "y", "z"], data["opt_paths"][-1].tolist())
    print(f"Wrote CSV data to {out_dir}")


if __name__ == "__main__":
    main()
