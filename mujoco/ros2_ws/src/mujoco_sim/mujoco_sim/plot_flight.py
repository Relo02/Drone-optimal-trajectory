"""
Flight data visualization for drone simulation.

Usage:
    python plot_flight.py                    # Plot most recent flight
    python plot_flight.py <session_id>       # Plot specific session
    python plot_flight.py --list             # List available sessions
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import 3D plotting (may fail on some matplotlib versions)
try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False
    print("Warning: 3D plotting not available")


def load_flight_data(log_dir: Path, session_id: str = None) -> dict:
    """Load flight data from JSON file."""
    if session_id is None:
        # Find most recent session
        json_files = sorted(log_dir.glob("flight_data_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No flight data found in {log_dir}")
        json_file = json_files[-1]
    else:
        json_file = log_dir / f"flight_data_{session_id}.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Flight data not found: {json_file}")

    with open(json_file) as f:
        return json.load(f)


def list_sessions(log_dir: Path):
    """List available flight sessions."""
    json_files = sorted(log_dir.glob("flight_data_*.json"))
    if not json_files:
        print("No flight sessions found.")
        return

    print(f"Available flight sessions in {log_dir}:\n")
    for f in json_files:
        with open(f) as fp:
            data = json.load(fp)
        meta = data['metadata']
        stats = meta.get('stats', {})
        print(f"  {meta['session_id']}")
        print(f"    Start: {meta['start_time']}")
        print(f"    Duration: {stats.get('duration_s', 0):.1f}s")
        print(f"    Distance: {stats.get('total_distance', 0):.2f}m")
        print(f"    Completed: {meta.get('trajectory_completed', False)}")
        print()


def plot_3d_trajectory(data: dict, ax: plt.Axes = None, show_waypoints: bool = True):
    """Plot 3D trajectory with waypoints."""
    if not HAS_3D:
        return None

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Extract position data
    positions = np.array(data['state_history']['position'])
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Plot trajectory
    ax.plot(x, y, z, 'b-', linewidth=1.5, label='Actual trajectory', alpha=0.8)

    # Start and end markers
    ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', label='End')

    # Plot A* waypoints (preferred) or fallback waypoints
    if show_waypoints and 'trajectory' in data:
        astar_wps = data['trajectory'].get('astar_waypoints', [])
        fallback_wps = data['trajectory'].get('waypoints', [])
        waypoints_to_plot = astar_wps if astar_wps else fallback_wps
        label_prefix = 'A*' if astar_wps else 'Planned'

        if len(waypoints_to_plot) > 0:
            wp_x = [wp[0] for wp in waypoints_to_plot]
            wp_y = [wp[1] for wp in waypoints_to_plot]
            wp_z = [wp[2] for wp in waypoints_to_plot]
            ax.scatter(wp_x, wp_y, wp_z, c='orange', s=60, marker='o',
                      label=f'{label_prefix} waypoints', edgecolors='black', linewidths=0.5)
            ax.plot(wp_x, wp_y, wp_z, '--', color='orange', linewidth=1.5,
                    alpha=0.8, label=f'{label_prefix} path')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Flight Trajectory')
    ax.legend(loc='upper left')

    # Equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, max(z.max() + 0.5, mid_z + max_range))

    return ax


def plot_position_vs_time(data: dict, axes: np.ndarray = None):
    """Plot x, y, z position over time."""
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    time = np.array(data['state_history']['time'])
    positions = np.array(data['state_history']['position'])
    targets = np.array(data['state_history']['target'])

    labels = ['X', 'Y', 'Z']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time, positions[:, i], color=color, linewidth=1.5, label=f'{label} actual')
        ax.plot(time, targets[:, i], '--', color=color, linewidth=1, alpha=0.7, label=f'{label} target')
        ax.set_ylabel(f'{label} (m)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Position vs Time')

    return axes


def plot_orientation_vs_time(data: dict, axes: np.ndarray = None):
    """Plot roll, pitch, yaw over time."""
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    time = np.array(data['state_history']['time'])
    euler = np.array(data['state_history']['euler'])
    euler_deg = np.degrees(euler)

    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['#ff7f0e', '#9467bd', '#8c564b']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time, euler_deg[:, i], color=color, linewidth=1.5)
        ax.set_ylabel(f'{label} (deg)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Orientation vs Time')

    return axes


def plot_velocity_vs_time(data: dict, axes: np.ndarray = None):
    """Plot linear velocities over time."""
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    time = np.array(data['state_history']['time'])
    velocity = np.array(data['state_history']['velocity'])

    labels = ['Vx', 'Vy', 'Vz']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time, velocity[:, i], color=color, linewidth=1.5)
        ax.set_ylabel(f'{label} (m/s)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Linear Velocity vs Time')

    return axes


def plot_control_inputs(data: dict, axes: np.ndarray = None):
    """Plot control inputs over time."""
    if axes is None:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    time = np.array(data['state_history']['time'])
    control = np.array(data['state_history']['control'])

    labels = ['Thrust (N)', 'Mx (Nm)', 'My (Nm)', 'Mz (Nm)']
    colors = ['#17becf', '#bcbd22', '#e377c2', '#7f7f7f']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time, control[:, i], color=color, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Control Inputs vs Time')

    return axes


def plot_tracking_error(data: dict, ax: plt.Axes = None):
    """Plot position tracking error over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    time = np.array(data['state_history']['time'])
    error = np.array(data['state_history']['error'])

    ax.fill_between(time, 0, error, alpha=0.3, color='red')
    ax.plot(time, error, 'r-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Tracking Error vs Time')
    ax.grid(True, alpha=0.3)

    # Add waypoint markers
    if 'trajectory' in data:
        for wp_info in data['trajectory'].get('waypoint_times', []):
            ax.axvline(x=wp_info['time'], color='green', linestyle='--',
                      alpha=0.7, linewidth=1)

    # Statistics annotation
    stats = data['metadata'].get('stats', {})
    stats_text = f"Mean: {stats.get('mean_error', 0):.3f}m\nMax: {stats.get('max_error', 0):.3f}m"
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def plot_2d_trajectory_xy(data: dict, ax: plt.Axes = None):
    """Plot 2D trajectory (top-down view)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    positions = np.array(data['state_history']['position'])
    x, y = positions[:, 0], positions[:, 1]

    # Color trajectory by time
    time = np.array(data['state_history']['time'])
    points = ax.scatter(x, y, c=time, cmap='viridis', s=2, alpha=0.7)
    plt.colorbar(points, ax=ax, label='Time (s)')

    # Start and end
    ax.plot(x[0], y[0], 'go', markersize=12, label='Start')
    ax.plot(x[-1], y[-1], 'rs', markersize=12, label='End')

    # A* waypoints (preferred) or fallback
    if 'trajectory' in data:
        astar_wps = data['trajectory'].get('astar_waypoints', [])
        fallback_wps = data['trajectory'].get('waypoints', [])
        waypoints_to_plot = astar_wps if astar_wps else fallback_wps
        label_prefix = 'A*' if astar_wps else 'Planned'

        if waypoints_to_plot:
            wp_x = [wp[0] for wp in waypoints_to_plot]
            wp_y = [wp[1] for wp in waypoints_to_plot]
            ax.plot(wp_x, wp_y, '--', color='orange', linewidth=2,
                    alpha=0.9, label=f'{label_prefix} path')
            ax.plot(wp_x, wp_y, 'o', color='orange', markersize=7,
                    markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('2D Trajectory (Top View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return ax


def plot_trajectory_comparison(data: dict, ax: plt.Axes = None):
    """
    Debug plot: overlay actual trajectory, A* planned path, and MPC predictions.

    Shows:
      - Actual drone path (blue, colored by time)
      - A* planned waypoints + connecting line (orange)
      - MPC predicted horizons sampled at regular intervals (light green)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    positions = np.array(data['state_history']['position'])
    time = np.array(data['state_history']['time'])
    x, y = positions[:, 0], positions[:, 1]

    # Actual trajectory (colored by time)
    sc = ax.scatter(x, y, c=time, cmap='Blues', s=3, alpha=0.8, zorder=3)
    plt.colorbar(sc, ax=ax, label='Time (s)', fraction=0.025, pad=0.02, shrink=0.8)
    ax.plot(x[0], y[0], 'go', markersize=10, zorder=5, label='Start')
    ax.plot(x[-1], y[-1], 'rs', markersize=10, zorder=5, label='End')

    # A* planned path
    astar_wps = data['trajectory'].get('astar_waypoints', [])
    if astar_wps:
        wp_x = [wp[0] for wp in astar_wps]
        wp_y = [wp[1] for wp in astar_wps]
        ax.plot(wp_x, wp_y, '--', color='#ff7f0e', linewidth=2.5,
                alpha=0.9, zorder=4, label='A* planned path')
        ax.plot(wp_x, wp_y, 'o', color='#ff7f0e', markersize=8,
                markeredgecolor='black', markeredgewidth=0.5, zorder=4)

    # MPC predicted trajectories (sampled uniformly, up to 15 shown)
    mpc_preds = data.get('mpc_predictions', [])
    if mpc_preds:
        n_total = len(mpc_preds)
        step = max(1, n_total // 15)
        indices = list(range(0, n_total, step))
        cmap = plt.cm.Greens
        for k, idx in enumerate(indices):
            pred = mpc_preds[idx]
            xy = np.array(pred['xy'])   # (N+1, 2)
            alpha = 0.25 + 0.55 * (k / max(len(indices) - 1, 1))
            label = 'MPC horizon' if k == 0 else None
            ax.plot(xy[:, 0], xy[:, 1], '-', color=cmap(0.4 + 0.5 * k / max(len(indices) - 1, 1)),
                    linewidth=1.2, alpha=alpha, zorder=2, label=label)
            ax.plot(xy[0, 0], xy[0, 1], '.', color='darkgreen',
                    markersize=5, alpha=alpha, zorder=2)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison: Actual vs A* vs MPC Predictions')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return ax


def plot_mpc_diagnostics(data: dict, axes=None):
    """
    Plot MPC cost and solve time over time.

    Creates two vertically-stacked subplots:
        1. MPC cost vs time
        2. MPC solve time vs time (with success/failure coloring)

    Args:
        data: Flight data dict (must contain 'mpc_history' key)
        axes: Optional array of 2 Axes

    Returns:
        axes or None if no MPC data available
    """
    mpc = data.get('mpc_history')
    if mpc is None or len(mpc.get('time', [])) == 0:
        return None

    time = np.array(mpc['time'])
    cost = np.array(mpc['cost'])
    solve_ms = np.array(mpc['solve_time_ms'])
    success = np.array(mpc['success'])

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # ── Cost vs time ──
    ax_cost = axes[0]
    ax_cost.plot(time, cost, 'b-', linewidth=1.2, label='MPC cost')
    ax_cost.fill_between(time, 0, cost, alpha=0.15, color='blue')
    # Mark failures
    fail_mask = ~success.astype(bool)
    if np.any(fail_mask):
        ax_cost.scatter(time[fail_mask], cost[fail_mask], c='red', s=30,
                        zorder=5, label='Solver failed')
    ax_cost.set_ylabel('Cost')
    ax_cost.set_title('MPC Cost vs Time')
    ax_cost.legend(loc='upper right')
    ax_cost.grid(True, alpha=0.3)

    # Stats annotation
    stats = data['metadata'].get('stats', {})
    stats_text = (f"Mean: {stats.get('mpc_mean_cost', 0):.1f}\n"
                  f"Max: {stats.get('mpc_max_cost', 0):.1f}\n"
                  f"Success: {stats.get('mpc_success_rate', 0)*100:.0f}%")
    ax_cost.text(0.02, 0.95, stats_text, transform=ax_cost.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # ── Solve time vs time ──
    ax_t = axes[1]
    colors = np.where(success.astype(bool), '#2ca02c', '#d62728')  # green / red
    ax_t.bar(time, solve_ms, width=np.median(np.diff(time)) * 0.8 if len(time) > 1 else 0.01,
             color=colors, alpha=0.7)
    ax_t.set_ylabel('Solve time (ms)')
    ax_t.set_xlabel('Time (s)')
    ax_t.set_title('MPC Solve Time vs Time')
    ax_t.grid(True, alpha=0.3)

    solve_text = (f"Mean: {stats.get('mpc_mean_solve_ms', 0):.1f} ms\n"
                  f"Max: {stats.get('mpc_max_solve_ms', 0):.1f} ms")
    ax_t.text(0.02, 0.95, solve_text, transform=ax_t.transAxes, fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    return axes


def create_all_figures(data: dict, save_dir: Path = None) -> list:
    """
    Create one figure window per topic for post-flight analysis.

    Returns a list of (name, figure) tuples:
      1. trajectory_comparison  — actual path vs A* plan vs MPC horizons (main debug view)
      2. trajectory_3d          — 3D flight path with waypoints
      3. time_domain            — position, orientation, velocity vs time (3 subplots)
      4. tracking_error         — position error vs time
      5. mpc_diagnostics        — MPC cost + solve time (only if MPC data present)
    """
    meta  = data['metadata']
    stats = meta.get('stats', {})
    sid   = meta['session_id']
    base_title = (f"Session {sid} | "
                  f"{stats.get('duration_s', 0):.1f}s | "
                  f"{stats.get('total_distance', 0):.2f}m | "
                  f"Completed: {meta.get('trajectory_completed', False)}")

    figures = []

    # ── 1. Trajectory comparison ──────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    plot_trajectory_comparison(data, ax1)
    fig1.suptitle(f"Trajectory Comparison | {base_title}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    figures.append(('trajectory_comparison', fig1))

    # ── 2. 3D trajectory (or altitude fallback) ───────────────────────────
    if HAS_3D:
        fig2 = plt.figure(figsize=(9, 7))
        ax2 = fig2.add_subplot(111, projection='3d')
        plot_3d_trajectory(data, ax2)
    else:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        time_arr = np.array(data['state_history']['time'])
        positions = np.array(data['state_history']['position'])
        ax2.plot(time_arr, positions[:, 2], 'b-', linewidth=1.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Altitude (m)')
        ax2.set_title('Altitude vs Time')
        ax2.grid(True, alpha=0.3)
    fig2.suptitle(f"3D Trajectory | {base_title}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    figures.append(('trajectory_3d', fig2))

    # ── 3. Time-domain states ─────────────────────────────────────────────
    time_arr  = np.array(data['state_history']['time'])
    positions = np.array(data['state_history']['position'])
    euler     = np.degrees(np.array(data['state_history']['euler']))
    velocity  = np.array(data['state_history']['velocity'])
    speed     = np.linalg.norm(velocity, axis=1)

    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes3[0].plot(time_arr, positions[:, 0], label='X')
    axes3[0].plot(time_arr, positions[:, 1], label='Y')
    axes3[0].plot(time_arr, positions[:, 2], label='Z')
    axes3[0].set_ylabel('Position (m)')
    axes3[0].set_title('Position vs Time')
    axes3[0].legend(fontsize=9, loc='upper right')
    axes3[0].grid(True, alpha=0.3)

    axes3[1].plot(time_arr, euler[:, 0], label='Roll')
    axes3[1].plot(time_arr, euler[:, 1], label='Pitch')
    axes3[1].plot(time_arr, euler[:, 2], label='Yaw')
    axes3[1].set_ylabel('Angle (deg)')
    axes3[1].set_title('Orientation vs Time')
    axes3[1].legend(fontsize=9, loc='upper right')
    axes3[1].grid(True, alpha=0.3)
    axes3[1].axhline(y=0, color='gray', linewidth=0.5)

    axes3[2].plot(time_arr, velocity[:, 0], label='Vx', alpha=0.8)
    axes3[2].plot(time_arr, velocity[:, 1], label='Vy', alpha=0.8)
    axes3[2].plot(time_arr, velocity[:, 2], label='Vz', alpha=0.8)
    axes3[2].plot(time_arr, speed, 'k-', linewidth=1.8, label='Speed')
    axes3[2].set_xlabel('Time (s)')
    axes3[2].set_ylabel('Velocity (m/s)')
    axes3[2].set_title('Velocity vs Time')
    axes3[2].legend(fontsize=9, loc='upper right')
    axes3[2].grid(True, alpha=0.3)

    fig3.suptitle(f"State Time-Domain | {base_title}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    figures.append(('time_domain', fig3))

    # ── 4. Tracking error ────────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    plot_tracking_error(data, ax4)
    fig4.suptitle(f"Tracking Error | {base_title}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    figures.append(('tracking_error', fig4))

    # ── 5. MPC diagnostics (only when data available) ────────────────────
    has_mpc = 'mpc_history' in data and len(data['mpc_history'].get('time', [])) > 0
    if has_mpc:
        fig5, axes5 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        plot_mpc_diagnostics(data, axes5)
        mpc_stats = (f" | Success: {stats.get('mpc_success_rate', 0)*100:.0f}%"
                     f" ({stats.get('mpc_total_solves', 0)} solves)"
                     f" | Mean solve: {stats.get('mpc_mean_solve_ms', 0):.1f}ms")
        fig5.suptitle(f"MPC Diagnostics | {base_title}{mpc_stats}",
                      fontsize=11, fontweight='bold')
        plt.tight_layout()
        figures.append(('mpc_diagnostics', fig5))

    # ── Save if requested ────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        for name, fig in figures:
            path = save_dir / f"{name}_{sid}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

    return figures


def create_summary_figure(data: dict, save_path: Path = None):
    """Legacy single-figure summary. Prefer create_all_figures() for debugging."""
    figs = create_all_figures(data, save_dir=save_path.parent if save_path else None)
    # Return the trajectory comparison figure for backward compatibility
    return figs[0][1] if figs else None


def main():
    parser = argparse.ArgumentParser(description='Plot flight data from drone simulation.')
    parser.add_argument('session_id', nargs='?', default=None,
                       help='Session ID to plot (default: most recent)')
    parser.add_argument('--list', action='store_true',
                       help='List available sessions')
    parser.add_argument('--save', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Log directory path')
    args = parser.parse_args()

    # Determine log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(__file__).parent.parent / "logs"

    if args.list:
        list_sessions(log_dir)
        return

    # Load data
    try:
        data = load_flight_data(log_dir, args.session_id)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    session_id = data['metadata']['session_id']
    print(f"Plotting flight session: {session_id}")

    # Create one figure per topic
    save_dir = log_dir if args.save else None
    figures = create_all_figures(data, save_dir=save_dir)
    print(f"Opened {len(figures)} figure windows: "
          f"{', '.join(name for name, _ in figures)}")

    plt.show()


if __name__ == '__main__':
    main()
