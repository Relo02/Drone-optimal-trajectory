"""
Flight Data Visualization Tool

Generates comprehensive plots and animations from logged flight data:
1. Occupancy grid evolution (with inflated obstacles)
2. MPC performance metrics
3. Trajectory plots (2D and 3D)
4. Waypoint following analysis
5. Velocity and acceleration profiles
6. Cost function evolution

Usage:
    python visualize_flight.py /path/to/session_dir [--interactive]

    --interactive: Display plots in interactive GUI windows

Author: Flight Analysis Team
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path
import argparse

from flight_logger import load_flight_data, GridSnapshot, FlightDataPoint


class FlightVisualizer:
    """Generate comprehensive visualizations from flight data."""

    def __init__(self, session_dir: str, interactive: bool = False):
        """
        Initialize visualizer.

        Args:
            session_dir: Path to logged session directory
            interactive: If True, display plots interactively instead of saving
        """
        self.session_dir = Path(session_dir)
        self.interactive = interactive

        # Verify session directory exists
        if not self.session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        if not self.interactive:
            self.output_dir = self.session_dir / "plots"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading flight data from: {session_dir}")
        self.flight_data, self.grid_snapshots, self.metadata = load_flight_data(session_dir)

        print(f"Loaded:")
        print(f"  - {len(self.flight_data)} data points")
        print(f"  - {len(self.grid_snapshots)} grid snapshots")
        print(f"  - Duration: {self.metadata['duration']:.2f}s")

        # Extract arrays for easy plotting
        self._extract_arrays()

    def _save_or_show(self, filename: str):
        """Save plot to file or show interactively based on mode."""
        if self.interactive:
            plt.show()
        else:
            output_file = self.output_dir / filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()

    def _extract_arrays(self):
        """Extract numpy arrays from flight data."""

        self.timestamps = np.array([d.timestamp for d in self.flight_data])
        self.positions = np.array([d.position for d in self.flight_data])
        self.velocities = np.array([d.velocity for d in self.flight_data])
        self.yaws = np.array([d.yaw for d in self.flight_data])

        self.active_goals = np.array([d.active_goal for d in self.flight_data])
        self.final_goals = np.array([d.final_goal for d in self.flight_data])

        self.mpc_solve_times = np.array([d.mpc_solve_time_ms for d in self.flight_data])
        self.mpc_costs = np.array([d.mpc_cost for d in self.flight_data])

        self.distances_to_active = np.array([d.distance_to_active_goal for d in self.flight_data])
        self.distances_to_final = np.array([d.distance_to_final_goal for d in self.flight_data])

        self.num_waypoints = np.array([d.num_waypoints for d in self.flight_data])

        self.accelerations = np.array([d.acceleration_cmd for d in self.flight_data])

        # Compute speeds
        self.speeds = np.linalg.norm(self.velocities, axis=1)

    def plot_trajectory_2d(self):
        """Plot 2D trajectory with waypoints and goals."""

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot trajectory
        ax.plot(self.positions[:, 0], self.positions[:, 1],
                'b-', linewidth=2, label='Actual Trajectory', alpha=0.8)

        # Plot start and end
        ax.plot(self.positions[0, 0], self.positions[0, 1],
                'go', markersize=15, label='Start', zorder=10)
        ax.plot(self.positions[-1, 0], self.positions[-1, 1],
                'r^', markersize=15, label='End', zorder=10)

        # Plot final goal
        final_goal = self.final_goals[-1]
        ax.plot(final_goal[0], final_goal[1],
                'r*', markersize=20, label='Final Goal', zorder=10)

        # Plot active goal trajectory (shows waypoint switching)
        unique_goals = []
        for i, goal in enumerate(self.active_goals):
            if i == 0 or not np.allclose(goal, self.active_goals[i-1], atol=0.1):
                unique_goals.append((self.positions[i], goal))

        for pos, goal in unique_goals:
            ax.plot([pos[0], goal[0]], [pos[1], goal[1]],
                    'y--', alpha=0.3, linewidth=1)
            ax.plot(goal[0], goal[1], 'yo', markersize=8, alpha=0.6)

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('2D Trajectory with Waypoints', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_aspect('equal')

        plt.tight_layout()
        self._save_or_show("trajectory_2d.png")

    def plot_trajectory_3d(self):
        """Plot 3D trajectory."""

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2],
                'b-', linewidth=2, label='Trajectory', alpha=0.8)

        # Plot start and end
        ax.scatter(self.positions[0, 0], self.positions[0, 1], self.positions[0, 2],
                   c='g', marker='o', s=200, label='Start', zorder=10)
        ax.scatter(self.positions[-1, 0], self.positions[-1, 1], self.positions[-1, 2],
                   c='r', marker='^', s=200, label='End', zorder=10)

        # Plot final goal
        final_goal = self.final_goals[-1]
        ax.scatter(final_goal[0], final_goal[1], final_goal[2],
                   c='r', marker='*', s=400, label='Goal', zorder=10)

        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('Z Position (m)', fontsize=12)
        ax.set_title('3D Trajectory', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)

        plt.tight_layout()
        self._save_or_show("trajectory_3d.png")

    def plot_mpc_performance(self):
        """Plot MPC solver performance metrics."""

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Solve time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.timestamps, self.mpc_solve_times, 'b-', linewidth=1.5)
        ax1.axhline(y=np.mean(self.mpc_solve_times), color='r', linestyle='--',
                    label=f'Mean: {np.mean(self.mpc_solve_times):.2f}ms')
        ax1.axhline(y=100, color='orange', linestyle='--', alpha=0.5, label='100ms limit')
        ax1.fill_between(self.timestamps, 0, self.mpc_solve_times, alpha=0.3)
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Solve Time (ms)', fontsize=11)
        ax1.set_title('MPC Solver Time', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Cost evolution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.timestamps, self.mpc_costs, 'g-', linewidth=1.5)
        ax2.fill_between(self.timestamps, 0, self.mpc_costs, alpha=0.3, color='green')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('MPC Cost', fontsize=11)
        ax2.set_title('MPC Cost Function Evolution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Distance to goal
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.timestamps, self.distances_to_final, 'b-',
                 linewidth=2, label='Final Goal', alpha=0.8)
        ax3.plot(self.timestamps, self.distances_to_active, 'orange',
                 linewidth=2, label='Active Goal (Waypoint)', alpha=0.8)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Distance (m)', fontsize=11)
        ax3.set_title('Distance to Goals', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # 4. Speed profile
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.timestamps, self.speeds, 'purple', linewidth=1.5)
        ax4.fill_between(self.timestamps, 0, self.speeds, alpha=0.3, color='purple')
        ax4.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='v_max=1.5 m/s')
        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.set_ylabel('Speed (m/s)', fontsize=11)
        ax4.set_title('Velocity Profile', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        # 5. Waypoint count
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(self.timestamps, self.num_waypoints, 'cyan', linewidth=2, marker='o',
                 markersize=2, alpha=0.7)
        ax5.fill_between(self.timestamps, 0, self.num_waypoints, alpha=0.3, color='cyan')
        ax5.set_xlabel('Time (s)', fontsize=11)
        ax5.set_ylabel('Number of Waypoints', fontsize=11)
        ax5.set_title('Remaining Waypoints', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. Acceleration magnitude
        ax6 = fig.add_subplot(gs[2, 1])
        acc_mag = np.linalg.norm(self.accelerations, axis=1)
        ax6.plot(self.timestamps, acc_mag, 'red', linewidth=1.5)
        ax6.fill_between(self.timestamps, 0, acc_mag, alpha=0.3, color='red')
        ax6.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='a_max=3.0 m/s²')
        ax6.set_xlabel('Time (s)', fontsize=11)
        ax6.set_ylabel('Acceleration (m/s²)', fontsize=11)
        ax6.set_title('Acceleration Command Magnitude', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)

        fig.suptitle('MPC Performance Metrics', fontsize=16, fontweight='bold', y=0.995)

        self._save_or_show("mpc_performance.png")

    def plot_occupancy_grid_evolution(self, num_snapshots: int = 9):
        """
        Plot grid evolution over time.

        Args:
            num_snapshots: Number of snapshots to show
        """
        if not self.grid_snapshots:
            print("No grid snapshots available")
            return

        # Select evenly spaced snapshots
        indices = np.linspace(0, len(self.grid_snapshots) - 1, num_snapshots, dtype=int)
        selected_snapshots = [self.grid_snapshots[i] for i in indices]

        # Create subplot grid
        rows = int(np.ceil(np.sqrt(num_snapshots)))
        cols = int(np.ceil(num_snapshots / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        axes = axes.flatten()

        for idx, snapshot in enumerate(selected_snapshots):
            ax = axes[idx]

            # Plot occupancy grid
            extent = [
                snapshot.origin_x,
                snapshot.origin_x + snapshot.width * snapshot.resolution,
                snapshot.origin_y,
                snapshot.origin_y + snapshot.height * snapshot.resolution
            ]

            im = ax.imshow(snapshot.grid, origin='lower', extent=extent,
                           cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')

            # Plot drone position
            drone_pos = snapshot.drone_position
            ax.plot(drone_pos[0], drone_pos[1], 'ro', markersize=10,
                    markeredgecolor='white', markeredgewidth=2, label='Drone')

            # Plot final goal
            if idx < len(self.flight_data):
                goal = self.flight_data[int(indices[idx])].final_goal
                ax.plot(goal[0], goal[1], 'g*', markersize=15,
                        markeredgecolor='white', markeredgewidth=1.5, label='Goal')

            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_title(f't = {snapshot.timestamp:.2f}s', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3, color='yellow', linewidth=0.5)
            ax.set_aspect('equal')

        # Hide unused subplots
        for idx in range(len(selected_snapshots), len(axes)):
            axes[idx].axis('off')

        # Add colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Occupancy Probability', fontsize=12)

        fig.suptitle('Occupancy Grid Evolution (Inflated Obstacles)',
                     fontsize=18, fontweight='bold')

        self._save_or_show("occupancy_grid_evolution.png")

    def plot_detailed_occupancy_with_trajectory(self):
        """Plot final occupancy grid with full trajectory overlay."""

        if not self.grid_snapshots:
            print("No grid snapshots available")
            return

        # Use final grid snapshot
        snapshot = self.grid_snapshots[-1]

        fig, ax = plt.subplots(figsize=(16, 14))

        # Plot occupancy grid
        extent = [
            snapshot.origin_x,
            snapshot.origin_x + snapshot.width * snapshot.resolution,
            snapshot.origin_y,
            snapshot.origin_y + snapshot.height * snapshot.resolution
        ]

        im = ax.imshow(snapshot.grid, origin='lower', extent=extent,
                       cmap='gray_r', vmin=0, vmax=1, interpolation='nearest', alpha=0.7)

        # Overlay trajectory
        ax.plot(self.positions[:, 0], self.positions[:, 1],
                'b-', linewidth=3, label='Trajectory', alpha=0.9)

        # Plot start and goal
        ax.plot(self.positions[0, 0], self.positions[0, 1],
                'go', markersize=20, markeredgecolor='white',
                markeredgewidth=2, label='Start', zorder=10)

        final_goal = self.final_goals[-1]
        ax.plot(final_goal[0], final_goal[1],
                'r*', markersize=30, markeredgecolor='white',
                markeredgewidth=2, label='Goal', zorder=10)

        ax.plot(self.positions[-1, 0], self.positions[-1, 1],
                'r^', markersize=20, markeredgecolor='white',
                markeredgewidth=2, label='Final Position', zorder=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Occupancy Probability\n(Inflated for Safety)', fontsize=12)

        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        ax.set_title('Final Occupancy Grid with Complete Trajectory',
                     fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, color='yellow', linewidth=0.5)
        ax.set_aspect('equal')

        self._save_or_show("occupancy_grid_with_trajectory.png")

    def plot_statistics_summary(self):
        """Generate statistical summary plot."""

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Compute statistics
        total_distance = np.sum(np.linalg.norm(np.diff(self.positions, axis=0), axis=1))
        straight_distance = np.linalg.norm(self.positions[-1] - self.positions[0])
        path_efficiency = (straight_distance / total_distance) * 100 if total_distance > 0 else 0

        avg_speed = np.mean(self.speeds)
        max_speed = np.max(self.speeds)

        avg_solve_time = np.mean(self.mpc_solve_times)
        max_solve_time = np.max(self.mpc_solve_times)

        final_dist_to_goal = self.distances_to_final[-1]

        # 1. Summary text
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        summary_text = f"""
        FLIGHT PERFORMANCE SUMMARY

        Duration: {self.metadata['duration']:.2f} seconds

        TRAJECTORY METRICS:
        • Total Distance Traveled: {total_distance:.2f} m
        • Straight-Line Distance: {straight_distance:.2f} m
        • Path Efficiency: {path_efficiency:.1f}%
        • Final Distance to Goal: {final_dist_to_goal:.2f} m

        VELOCITY METRICS:
        • Average Speed: {avg_speed:.2f} m/s
        • Maximum Speed: {max_speed:.2f} m/s
        • Max Acceleration: {np.max(np.linalg.norm(self.accelerations, axis=1)):.2f} m/s²

        MPC PERFORMANCE:
        • Average Solve Time: {avg_solve_time:.2f} ms
        • Maximum Solve Time: {max_solve_time:.2f} ms
        • Average Cost: {np.mean(self.mpc_costs):.2f}

        OBSTACLES:
        • Avg Raw Obstacles: {np.mean([d.num_raw_obstacles for d in self.flight_data]):.1f}
        • Avg Clustered Obstacles: {np.mean([d.num_clustered_obstacles for d in self.flight_data]):.1f}
        """

        ax1.text(0.1, 0.5, summary_text, fontsize=13, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 2. Speed histogram
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.speeds, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax2.axvline(avg_speed, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_speed:.2f} m/s')
        ax2.set_xlabel('Speed (m/s)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Speed Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. Solve time histogram
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(self.mpc_solve_times, bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax3.axvline(avg_solve_time, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {avg_solve_time:.2f} ms')
        ax3.axvline(100, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='100ms limit')
        ax3.set_xlabel('Solve Time (ms)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('MPC Solve Time Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        fig.suptitle('Flight Statistics Summary', fontsize=16, fontweight='bold')

        self._save_or_show("statistics_summary.png")

    def generate_all_plots(self):
        """Generate all visualization plots."""

        if self.interactive:
            print("\nDisplaying visualizations in interactive mode...")
            print("Close each window to proceed to the next plot.")
        else:
            print("\nGenerating visualizations...")
        print("=" * 50)

        self.plot_trajectory_2d()
        self.plot_trajectory_3d()
        self.plot_mpc_performance()
        self.plot_occupancy_grid_evolution()
        self.plot_detailed_occupancy_with_trajectory()
        self.plot_statistics_summary()

        print("=" * 50)
        if not self.interactive:
            print(f"\nAll plots saved to: {self.output_dir}")
            print("\nGenerated plots:")
            print("  1. trajectory_2d.png - 2D trajectory view")
            print("  2. trajectory_3d.png - 3D trajectory view")
            print("  3. mpc_performance.png - MPC metrics over time")
            print("  4. occupancy_grid_evolution.png - Grid evolution")
            print("  5. occupancy_grid_with_trajectory.png - Final grid + trajectory")
            print("  6. statistics_summary.png - Performance summary")
        else:
            print("\nAll plots displayed.")


def main():
    """Main entry point for visualization script."""

    parser = argparse.ArgumentParser(
        description="Visualize drone flight data from logged session"
    )
    parser.add_argument(
        'session_dir',
        type=str,
        help='Path to session directory (e.g., /tmp/drone_logs/session_20260124_210333)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Display plots in interactive GUI windows instead of saving to files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory for plots (default: session_dir/plots)'
    )

    args = parser.parse_args()

    # Create visualizer
    visualizer = FlightVisualizer(args.session_dir, interactive=args.interactive)

    if args.output_dir and not args.interactive:
        visualizer.output_dir = Path(args.output_dir)
        visualizer.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    visualizer.generate_all_plots()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python visualize_flight.py <session_dir> [--interactive]")
        print("\nExamples:")
        print("  Save plots to files:")
        print("    python3 visualize_flight.py /tmp/drone_logs/session_20260124_210333")
        print("\n  Display in interactive GUI:")
        print("    python3 visualize_flight.py /tmp/drone_logs/session_20260124_210333 --interactive")
