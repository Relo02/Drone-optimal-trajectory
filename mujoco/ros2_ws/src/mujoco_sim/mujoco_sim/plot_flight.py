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

    # Plot waypoints
    if show_waypoints and 'trajectory' in data:
        waypoints = np.array(data['trajectory']['waypoints'])
        if len(waypoints) > 0:
            wp_x = [wp[0] for wp in waypoints]
            wp_y = [wp[1] for wp in waypoints]
            wp_z = [wp[2] for wp in waypoints]
            ax.scatter(wp_x, wp_y, wp_z, c='orange', s=150, marker='*',
                      label='Waypoints', edgecolors='black', linewidths=0.5)
            # Connect waypoints with dashed line
            ax.plot(wp_x, wp_y, wp_z, 'k--', linewidth=1, alpha=0.5, label='Planned path')

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

    # Waypoints
    if 'trajectory' in data:
        waypoints = data['trajectory']['waypoints']
        wp_x = [wp[0] for wp in waypoints]
        wp_y = [wp[1] for wp in waypoints]
        ax.plot(wp_x, wp_y, 'k*--', markersize=15, linewidth=1,
               alpha=0.7, label='Waypoints')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('2D Trajectory (Top View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return ax


def create_summary_figure(data: dict, save_path: Path = None):
    """Create a comprehensive summary figure."""
    fig = plt.figure(figsize=(16, 12))

    # 3D trajectory (top-left, larger) - or 2D altitude plot if 3D unavailable
    if HAS_3D:
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        plot_3d_trajectory(data, ax1)
    else:
        ax1 = fig.add_subplot(2, 3, 1)
        positions = np.array(data['state_history']['position'])
        time = np.array(data['state_history']['time'])
        ax1.plot(time, positions[:, 2], 'b-', linewidth=1.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Altitude (m)')
        ax1.set_title('Altitude vs Time')
        ax1.grid(True, alpha=0.3)

    # 2D trajectory (top-middle)
    ax2 = fig.add_subplot(2, 3, 2)
    plot_2d_trajectory_xy(data, ax2)

    # Tracking error (top-right)
    ax3 = fig.add_subplot(2, 3, 3)
    plot_tracking_error(data, ax3)

    # Position vs time (bottom-left)
    ax4 = fig.add_subplot(2, 3, 4)
    time = np.array(data['state_history']['time'])
    positions = np.array(data['state_history']['position'])
    ax4.plot(time, positions[:, 0], label='X')
    ax4.plot(time, positions[:, 1], label='Y')
    ax4.plot(time, positions[:, 2], label='Z')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Position vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Orientation vs time (bottom-middle)
    ax5 = fig.add_subplot(2, 3, 5)
    euler = np.degrees(np.array(data['state_history']['euler']))
    ax5.plot(time, euler[:, 0], label='Roll')
    ax5.plot(time, euler[:, 1], label='Pitch')
    ax5.plot(time, euler[:, 2], label='Yaw')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angle (deg)')
    ax5.set_title('Orientation vs Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Velocity vs time (bottom-right)
    ax6 = fig.add_subplot(2, 3, 6)
    velocity = np.array(data['state_history']['velocity'])
    speed = np.linalg.norm(velocity, axis=1)
    ax6.plot(time, velocity[:, 0], label='Vx', alpha=0.7)
    ax6.plot(time, velocity[:, 1], label='Vy', alpha=0.7)
    ax6.plot(time, velocity[:, 2], label='Vz', alpha=0.7)
    ax6.plot(time, speed, 'k-', linewidth=2, label='Speed')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Velocity (m/s)')
    ax6.set_title('Velocity vs Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add session info
    meta = data['metadata']
    stats = meta.get('stats', {})
    title = (f"Flight Session: {meta['session_id']} | "
             f"Duration: {stats.get('duration_s', 0):.1f}s | "
             f"Distance: {stats.get('total_distance', 0):.2f}m | "
             f"Completed: {meta.get('trajectory_completed', False)}")
    fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


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

    # Create summary figure
    save_path = log_dir / f"flight_plot_{session_id}.png" if args.save else None
    create_summary_figure(data, save_path)

    # Show plots
    plt.show()


if __name__ == '__main__':
    main()
