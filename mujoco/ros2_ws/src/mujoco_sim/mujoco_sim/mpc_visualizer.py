"""
Real-time MPC Visualizer Node

Subscribes to MPC-related topics and displays a live matplotlib plot
showing:
  - Gaussian occupancy grid (heatmap)
  - MPC predicted trajectory (green line)
  - Drone position and heading (black arrow)
  - Goal position (red star)
  - LiDAR points (grey dots)
  - MPC diagnostics text (cost, solve time, success)

Run alongside the MPC simulation node:
    ros2 run mujoco_sim mpc_viz_node

author: Lorenzo Ortolani
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
import matplotlib.cm as cm

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class MPCVisualizer(Node):
    """
    ROS2 node that subscribes to MPC topics and plots in real time.
    """

    def __init__(self):
        super().__init__('mpc_visualizer')

        # Declare parameters
        self.declare_parameter('goal_x', 10.0)
        self.declare_parameter('goal_y', 1.0)
        self.declare_parameter('update_rate_hz', 5.0)

        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        update_rate = self.get_parameter('update_rate_hz').value

        # QoS for sensor data (best effort)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscribers ──
        self.create_subscription(
            PoseStamped, '/drone/pose', self._pose_cb, 10)
        self.create_subscription(
            Path, '/mpc/predicted_path', self._path_cb, 10)
        self.create_subscription(
            Float32MultiArray, '/mpc/grid_data', self._grid_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/mpc/diagnostics', self._diag_cb, 10)
        self.create_subscription(
            PointCloud2, '/lidar/points', self._lidar_cb, sensor_qos)

        # ── Data storage ──
        self.drone_xy = np.array([0.0, 0.0])
        self.drone_yaw = 0.0
        self.mpc_traj = None            # (K, 2) array
        self.grid_data = None           # dict with minx, miny, reso, xw, yw, gmap
        self.lidar_pts = None           # (M, 2) array
        self.diag = {'cost': 0.0, 'solve_ms': 0.0, 'success': True}
        self.drone_history = []         # trail of past positions

        # ── Matplotlib setup ──
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('MPC Live Visualizer')

        # Persistent plot elements (initialized to None)
        self._grid_img = None
        self._lidar_sc = None
        self._traj_line = None
        self._trail_line = None
        self._drone_marker = None
        self._drone_arrow = None
        self._goal_marker = None
        self._info_text = None

        self._setup_plot()

        # Timer for periodic plot updates
        period_sec = 1.0 / update_rate
        self.create_timer(period_sec, self._update_plot)

        self.get_logger().info(
            f'MPC Visualizer started — update rate {update_rate} Hz, '
            f'goal ({self.goal_x}, {self.goal_y})'
        )

    # ─────────────────────────────────────────
    # ROS2 callbacks
    # ─────────────────────────────────────────
    def _pose_cb(self, msg: PoseStamped):
        self.drone_xy = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
        ])
        # Quaternion → yaw
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.drone_yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Append to trail (limit length)
        self.drone_history.append(self.drone_xy.copy())
        if len(self.drone_history) > 2000:
            self.drone_history = self.drone_history[-1500:]

    def _path_cb(self, msg: Path):
        if len(msg.poses) == 0:
            return
        pts = np.array([
            [p.pose.position.x, p.pose.position.y]
            for p in msg.poses
        ])
        self.mpc_traj = pts

    def _grid_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 6:
            return
        minx = msg.data[0]
        miny = msg.data[1]
        reso = msg.data[2]
        xw = int(msg.data[3])
        yw = int(msg.data[4])
        flat = np.array(msg.data[5:], dtype=np.float32)

        if flat.size != xw * yw:
            return

        gmap = flat.reshape(xw, yw)
        self.grid_data = {
            'minx': minx, 'miny': miny,
            'reso': reso, 'xw': xw, 'yw': yw,
            'gmap': gmap,
        }

    def _lidar_cb(self, msg: PointCloud2):
        pts = list(point_cloud2.read_points(msg, field_names=('x', 'y'), skip_nans=True))
        if len(pts) > 0:
            self.lidar_pts = np.array(pts)
        else:
            self.lidar_pts = None

    def _diag_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 3:
            self.diag = {
                'cost': msg.data[0],
                'solve_ms': msg.data[1],
                'success': msg.data[2] > 0.5,
            }

    # ─────────────────────────────────────────
    # Plotting
    # ─────────────────────────────────────────
    def _setup_plot(self):
        """Initial plot setup."""
        ax = self.ax
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('MPC Live View', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')

        # Goal marker (static)
        self._goal_marker, = ax.plot(
            self.goal_x, self.goal_y, 'r*',
            markersize=20, label='Goal', zorder=10)

        # Drone trail
        self._trail_line, = ax.plot(
            [], [], '-', color='steelblue', linewidth=1.0,
            alpha=0.5, label='Trail')

        # Lidar scatter
        self._lidar_sc = ax.scatter(
            [], [], s=3, c='grey', alpha=0.4, label='LiDAR', zorder=3)

        # MPC trajectory
        self._traj_line, = ax.plot(
            [], [], 'g.-', linewidth=2.5, markersize=6,
            label='MPC trajectory', zorder=7)

        # Drone marker
        self._drone_marker, = ax.plot(
            0, 0, 'ko', markersize=10, zorder=9, label='Drone')

        # Drone heading arrow
        self._drone_arrow = ax.annotate(
            '', xy=(0.5, 0), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle='->', color='black', lw=2.0),
            zorder=9)

        # Info text box
        self._info_text = ax.text(
            0.02, 0.98, '', transform=ax.transAxes,
            fontsize=10, fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85),
            zorder=15)

        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)

        # Initial view
        ax.set_xlim(-2, 15)
        ax.set_ylim(-5, 5)

    def _update_plot(self):
        """Timer callback — refresh the plot with latest data."""
        ax = self.ax

        # ── 1. Gaussian grid heatmap ──
        if self.grid_data is not None:
            g = self.grid_data
            # Build coordinate edges for pcolormesh
            x_edges = np.linspace(
                g['minx'] - g['reso'] / 2,
                g['minx'] + g['xw'] * g['reso'] - g['reso'] / 2,
                g['xw'] + 1)
            y_edges = np.linspace(
                g['miny'] - g['reso'] / 2,
                g['miny'] + g['yw'] * g['reso'] - g['reso'] / 2,
                g['yw'] + 1)

            # Remove previous heatmap
            if self._grid_img is not None:
                self._grid_img.remove()

            self._grid_img = ax.pcolormesh(
                x_edges, y_edges, g['gmap'].T,
                vmin=0, vmax=1.0,
                cmap='YlOrRd', alpha=0.6, shading='flat', zorder=1)

        # ── 2. LiDAR points ──
        if self.lidar_pts is not None and len(self.lidar_pts) > 0:
            self._lidar_sc.set_offsets(self.lidar_pts)
        else:
            self._lidar_sc.set_offsets(np.empty((0, 2)))

        # ── 3. MPC trajectory ──
        if self.mpc_traj is not None and len(self.mpc_traj) > 0:
            self._traj_line.set_data(self.mpc_traj[:, 0], self.mpc_traj[:, 1])
        else:
            self._traj_line.set_data([], [])

        # ── 4. Drone trail ──
        if len(self.drone_history) > 1:
            trail = np.array(self.drone_history)
            self._trail_line.set_data(trail[:, 0], trail[:, 1])

        # ── 5. Drone marker + arrow ──
        dx = self.drone_xy[0]
        dy = self.drone_xy[1]
        self._drone_marker.set_data([dx], [dy])

        arrow_len = 0.6
        self._drone_arrow.set_position((dx, dy))
        self._drone_arrow.xy = (
            dx + arrow_len * np.cos(self.drone_yaw),
            dy + arrow_len * np.sin(self.drone_yaw),
        )

        # ── 6. Info text ──
        dist_to_goal = np.linalg.norm(
            self.drone_xy - np.array([self.goal_x, self.goal_y]))
        ok_str = '✓' if self.diag['success'] else '✗'
        self._info_text.set_text(
            f"Pos:   ({dx:.2f}, {dy:.2f})\n"
            f"Goal:  ({self.goal_x}, {self.goal_y})\n"
            f"Dist:  {dist_to_goal:.2f} m\n"
            f"Cost:  {self.diag['cost']:.1f}\n"
            f"Solve: {self.diag['solve_ms']:.0f} ms  {ok_str}"
        )

        # ── 7. Auto-zoom: center on drone with some margin ──
        margin = 4.0
        # Include drone, goal, and trajectory in view
        x_points = [dx, self.goal_x]
        y_points = [dy, self.goal_y]
        if self.mpc_traj is not None and len(self.mpc_traj) > 0:
            x_points.extend(self.mpc_traj[:, 0].tolist())
            y_points.extend(self.mpc_traj[:, 1].tolist())

        xmin = min(x_points) - margin
        xmax = max(x_points) + margin
        ymin = min(y_points) - margin
        ymax = max(y_points) + margin

        # Keep aspect ratio roughly equal
        x_range = xmax - xmin
        y_range = ymax - ymin
        if x_range > y_range:
            mid_y = (ymin + ymax) / 2
            ymin = mid_y - x_range / 2
            ymax = mid_y + x_range / 2
        else:
            mid_x = (xmin + xmax) / 2
            xmin = mid_x - y_range / 2
            xmax = mid_x + y_range / 2

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # ── Redraw ──
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    node = MPCVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
