"""
Real-time point cloud visualization with ROS2 integration.
Subscribes to lidar data, drone pose, goal pose, and visualizes
the drone's view with local planning grid boundaries.

author: Lorenzo Ortolani
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import threading
import copy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from mujoco_sim.gaussian_grid_map import GaussianGridMap
from mujoco_sim.A_star_planner import AStarLocalPlanner


class PointCloudVisualizer(Node):
    """ROS2 node for real-time point cloud visualization."""

    def __init__(self):
        super().__init__('pointcloud_visualizer')

        # Declare parameters
        self.declare_parameter('grid_resolution', 0.2)
        self.declare_parameter('gaussian_std', 0.3)
        self.declare_parameter('extend_area', 2.0)
        self.declare_parameter('obstacle_threshold', 0.5)
        self.declare_parameter('obstacle_cost_weight', 10.0)
        self.declare_parameter('update_rate', 20.0)  # Hz for visualization update

        # Get parameters
        self.grid_resolution = self.get_parameter('grid_resolution').value
        self.gaussian_std = self.get_parameter('gaussian_std').value
        self.extend_area = self.get_parameter('extend_area').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.obstacle_cost_weight = self.get_parameter('obstacle_cost_weight').value
        self.update_rate = self.get_parameter('update_rate').value

        # Initialize grid map and planner
        self.grid_map = GaussianGridMap(
            xyreso=self.grid_resolution,
            std=self.gaussian_std,
            extend_area=self.extend_area
        )
        self.planner = AStarLocalPlanner(
            obstacle_threshold=self.obstacle_threshold,
            obstacle_cost_weight=self.obstacle_cost_weight
        )
        self.planner.set_grid_map(self.grid_map)

        # State variables (thread-safe access with lock)
        self.lock = threading.Lock()
        self.drone_pose = None
        self.goal_pose = None
        self.lidar_points = None
        self.local_goal = None
        self.drone_trail = []  # History of drone positions for trajectory plot
        self.frame_count = 0

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/drone/pose',
            self.pose_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.pointcloud_callback,
            sensor_qos
        )

        self.get_logger().info('PointCloud Visualizer Node initialized')
        self.get_logger().info(f'  Update rate: {self.update_rate}Hz')
        self.get_logger().info(f'  Grid resolution: {self.grid_resolution}m')

    def pose_callback(self, msg: PoseStamped):
        """Handle current drone pose updates."""
        with self.lock:
            self.drone_pose = msg

    def goal_callback(self, msg: PoseStamped):
        """Handle goal pose updates."""
        with self.lock:
            # Only log if this is a genuinely new goal
            is_new = True
            if self.goal_pose is not None:
                old = self.goal_pose.pose.position
                new = msg.pose.position
                if (abs(old.x - new.x) < 0.1 and
                        abs(old.y - new.y) < 0.1 and
                        abs(old.z - new.z) < 0.1):
                    is_new = False

            self.goal_pose = msg

            if is_new:
                self.get_logger().info(
                    f'Goal received: ({msg.pose.position.x:.2f}, '
                    f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})'
                )

    def pointcloud_callback(self, msg: PointCloud2):
        """Handle lidar point cloud updates."""
        # Convert PointCloud2 to numpy array
        points_list = []
        for point in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])

        with self.lock:
            if points_list:
                self.lidar_points = np.array(points_list)
                
                # Update grid map if we have drone pose
                if self.drone_pose is not None:
                    drone_pos = [
                        self.drone_pose.pose.position.x,
                        self.drone_pose.pose.position.y,
                        self.drone_pose.pose.position.z
                    ]
                    self.grid_map.update_from_lidar_points(self.lidar_points, drone_pos)
                    self.planner.set_grid_map(self.grid_map)
                    
                    # Update local goal if we have global goal
                    if self.goal_pose is not None:
                        goal_2d = (
                            self.goal_pose.pose.position.x,
                            self.goal_pose.pose.position.y
                        )
                        local_goal_grid = self.planner.get_local_goal(drone_pos[:2], goal_2d)
                        if local_goal_grid is not None:
                            self.local_goal = self.planner.grid_to_world(
                                local_goal_grid[0], local_goal_grid[1]
                            )
            else:
                self.lidar_points = None

    def get_visualization_data(self):
        """Get current visualization data (thread-safe deep copy)."""
        with self.lock:
            # Record drone trail
            if self.drone_pose is not None:
                pos = self.drone_pose.pose.position
                self.drone_trail.append([pos.x, pos.y])
                # Keep trail limited to last 500 positions
                if len(self.drone_trail) > 500:
                    self.drone_trail = self.drone_trail[-500:]

            # Deep-copy grid map data for thread safety
            gmap_copy = None
            grid_bounds = None
            if self.grid_map.gmap is not None:
                gmap_copy = self.grid_map.gmap.copy()
                grid_bounds = {
                    'minx': self.grid_map.minx,
                    'miny': self.grid_map.miny,
                    'maxx': self.grid_map.maxx,
                    'maxy': self.grid_map.maxy,
                    'xw': self.grid_map.xw,
                    'yw': self.grid_map.yw,
                    'xyreso': self.grid_map.xyreso,
                }

            return {
                'drone_pose': self.drone_pose,
                'goal_pose': self.goal_pose,
                'lidar_points': self.lidar_points.copy() if self.lidar_points is not None else None,
                'local_goal': self.local_goal,
                'gmap': gmap_copy,
                'grid_bounds': grid_bounds,
                'drone_trail': list(self.drone_trail),
            }


class RealtimeVisualizer:
    """Real-time matplotlib visualization (2D only — avoids mpl_toolkits.mplot3d issues)."""

    def __init__(self, ros_node):
        self.ros_node = ros_node
        self.frame_count = 0

        # Setup figure with 2x2 layout (all 2D)
        self.fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Trajectory history subplot (top-left)
        self.ax1 = axes[0, 0]
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_title('Drone Trajectory')
        self.ax1.set_aspect('equal')
        self.ax1.grid(True)

        # 2D top-down lidar view subplot (top-right)
        self.ax2 = axes[0, 1]
        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Y (m)')
        self.ax2.set_title('Top-Down View (2D Lidar Scan)')
        self.ax2.set_aspect('equal')
        self.ax2.grid(True)

        # Gaussian grid map subplot (bottom-left)
        self.ax3 = axes[1, 0]
        self.ax3.set_xlabel('X (m)')
        self.ax3.set_ylabel('Y (m)')
        self.ax3.set_title('Gaussian Grid Map (Obstacle Probability)')

        # Drone's view with local planning subplot (bottom-right)
        self.ax4 = axes[1, 1]
        self.ax4.set_xlabel('X (m)')
        self.ax4.set_ylabel('Y (m)')
        self.ax4.set_title("Drone's View with Local Planning")
        self.ax4.set_aspect('equal')
        self.ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        print("Matplotlib window created, waiting for ROS2 data...")

    def _on_key(self, event):
        """Handle key press events."""
        if event.key == 'escape' or event.key == 'q':
            plt.close(self.fig)

    def update(self, frame):
        """Update function called by FuncAnimation."""
        # Process pending ROS2 callbacks in the spin thread
        data = self.ros_node.get_visualization_data()

        points = data['lidar_points']
        drone_pose = data['drone_pose']
        goal_pose = data['goal_pose']
        local_goal = data['local_goal']
        gmap = data['gmap']
        grid_bounds = data['grid_bounds']
        drone_trail = data['drone_trail']

        self.frame_count += 1

        # If no data yet, show waiting message
        if points is None or drone_pose is None:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.cla()
                ax.set_title(f'Waiting for data... (frame {self.frame_count})')
            return []

        drone_pos = np.array([
            drone_pose.pose.position.x,
            drone_pose.pose.position.y,
            drone_pose.pose.position.z
        ])

        # Update all subplots
        self._update_trajectory_plot(drone_pos, drone_trail, goal_pose)
        self._update_2d_plot(points, drone_pos)
        self._update_grid_map_plot(points, drone_pos, gmap, grid_bounds)
        self._update_local_planning_view(points, drone_pos, goal_pose, local_goal, grid_bounds)

        return []

    def _update_trajectory_plot(self, drone_pos, drone_trail, goal_pose):
        """Update trajectory history visualization."""
        self.ax1.cla()

        if len(drone_trail) > 1:
            trail = np.array(drone_trail)
            self.ax1.plot(
                trail[:, 0], trail[:, 1],
                'b-', linewidth=1.5, alpha=0.6, label='Trajectory'
            )
            # Start position
            self.ax1.scatter(
                trail[0, 0], trail[0, 1],
                c='green', s=120, marker='o', edgecolors='darkgreen',
                linewidths=2, label='Start', zorder=5
            )

        # Current drone position
        self.ax1.scatter(
            drone_pos[0], drone_pos[1],
            c='blue', s=150, marker='^', edgecolors='darkblue',
            linewidths=2, label='Drone', zorder=5
        )

        # Global goal
        if goal_pose is not None:
            self.ax1.scatter(
                goal_pose.pose.position.x, goal_pose.pose.position.y,
                c='red', s=200, marker='*', edgecolors='darkred',
                linewidths=2, label='Global Goal', zorder=5
            )

        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_title(f'Drone Trajectory (Frame {self.frame_count})')
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper left', fontsize=8)

        # Auto-adjust limits
        margin = 2.0
        all_x = [drone_pos[0]]
        all_y = [drone_pos[1]]
        if len(drone_trail) > 0:
            trail = np.array(drone_trail)
            all_x.extend(trail[:, 0])
            all_y.extend(trail[:, 1])
        if goal_pose is not None:
            all_x.append(goal_pose.pose.position.x)
            all_y.append(goal_pose.pose.position.y)
        self.ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)

    def _update_2d_plot(self, points, drone_pos):
        """Update 2D top-down lidar view."""
        self.ax2.cla()

        if len(points) > 0:
            self.ax2.scatter(
                points[:, 0], points[:, 1],
                c='green', s=10, alpha=0.7, label='Lidar hits'
            )

        self.ax2.scatter(
            drone_pos[0], drone_pos[1],
            c='blue', s=100, marker='^', label='Drone'
        )

        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Y (m)')
        self.ax2.set_title(f'Top-Down View ({len(points)} points)')
        self.ax2.set_aspect('equal')
        self.ax2.grid(True)
        self.ax2.legend()

        if len(points) > 0:
            margin = 1.0
            self.ax2.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
            self.ax2.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)

    def _update_grid_map_plot(self, points, drone_pos, gmap, grid_bounds):
        """Update Gaussian grid map visualization using thread-safe copies."""
        self.ax3.cla()

        if gmap is not None and grid_bounds is not None:
            # Render heatmap from the deep-copied grid data
            xyreso = grid_bounds['xyreso']
            xw = grid_bounds['xw']
            yw = grid_bounds['yw']
            minx = grid_bounds['minx']
            miny = grid_bounds['miny']

            x_edges = np.linspace(
                minx - xyreso / 2.0,
                minx + xw * xyreso - xyreso / 2.0,
                xw + 1
            )
            y_edges = np.linspace(
                miny - xyreso / 2.0,
                miny + yw * xyreso - xyreso / 2.0,
                yw + 1
            )

            self.ax3.pcolormesh(
                x_edges, y_edges, gmap.T,
                vmax=1.0, cmap='Reds', alpha=0.8, shading='flat'
            )

            if len(points) > 0:
                self.ax3.scatter(
                    points[:, 0], points[:, 1],
                    c='black', s=5, alpha=0.5, label='Lidar hits'
                )

            self.ax3.scatter(
                drone_pos[0], drone_pos[1],
                c='blue', s=100, marker='^', label='Drone'
            )

            self.ax3.legend(fontsize=8)
            self.ax3.set_aspect('equal')

        self.ax3.set_xlabel('X (m)')
        self.ax3.set_ylabel('Y (m)')
        self.ax3.set_title('Gaussian Grid Map (Obstacle Probability)')

    def _update_local_planning_view(self, points, drone_pos, goal_pose, local_goal, grid_bounds):
        """Update drone's view with local planning visualization."""
        self.ax4.cla()

        # Plot 2D pointcloud
        if len(points) > 0:
            self.ax4.scatter(
                points[:, 0], points[:, 1],
                c='green', s=15, alpha=0.6, label='Lidar hits', edgecolors='darkgreen'
            )

        # Plot drone position
        self.ax4.scatter(
            drone_pos[0], drone_pos[1],
            c='blue', s=150, marker='^', label='Drone', edgecolors='darkblue', linewidths=2
        )

        # Plot global goal if available
        if goal_pose is not None:
            goal_x = goal_pose.pose.position.x
            goal_y = goal_pose.pose.position.y

            self.ax4.scatter(
                goal_x, goal_y,
                c='red', s=200, marker='*', label='Global Goal',
                edgecolors='darkred', linewidths=2
            )

            # Draw line from drone to global goal
            self.ax4.plot(
                [drone_pos[0], goal_x],
                [drone_pos[1], goal_y],
                'red', linestyle=':', linewidth=1.5, alpha=0.5
            )

        # Plot local goal if available
        if local_goal is not None:
            self.ax4.scatter(
                local_goal[0], local_goal[1],
                c='orange', s=200, marker='x', label='Local Goal', linewidths=3
            )

            # Draw line from drone to local goal
            self.ax4.plot(
                [drone_pos[0], local_goal[0]],
                [drone_pos[1], local_goal[1]],
                'orange', linestyle=':', linewidth=2, alpha=0.7
            )

        # Draw grid boundary rectangle if available
        if grid_bounds is not None:
            grid_width = grid_bounds['maxx'] - grid_bounds['minx']
            grid_height = grid_bounds['maxy'] - grid_bounds['miny']

            grid_boundary = Rectangle(
                (grid_bounds['minx'], grid_bounds['miny']),
                grid_width, grid_height,
                fill=False, edgecolor='purple', linewidth=2.5,
                linestyle='--', label='Grid Boundary'
            )
            self.ax4.add_patch(grid_boundary)

        # Set plot properties
        self.ax4.set_xlabel('X (m)')
        self.ax4.set_ylabel('Y (m)')
        self.ax4.set_title("Drone's View with Local Planning")
        self.ax4.set_aspect('equal')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(loc='upper right', fontsize=8)

        # Set view limits to include all relevant points
        margin = 2.0
        all_x = [drone_pos[0]]
        all_y = [drone_pos[1]]

        if len(points) > 0:
            all_x.extend(points[:, 0])
            all_y.extend(points[:, 1])

        if goal_pose is not None:
            all_x.append(goal_pose.pose.position.x)
            all_y.append(goal_pose.pose.position.y)

        if local_goal is not None:
            all_x.append(local_goal[0])
            all_y.append(local_goal[1])

        self.ax4.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax4.set_ylim(min(all_y) - margin, max(all_y) + margin)

    def run(self):
        """Start the real-time visualization."""
        print("Starting real-time visualization...")
        print("Press 'q' or 'Escape' to quit")

        # Update interval in milliseconds
        update_interval = int(1000.0 / self.ros_node.update_rate)

        self.anim = FuncAnimation(
            self.fig,
            self.update,
            interval=update_interval,
            blit=False,
            cache_frame_data=False
        )

        plt.show()


def ros_spin_thread(node):
    """Spin ROS2 node in a separate thread."""
    try:
        rclpy.spin(node)
    except Exception:
        pass  # Silently handle shutdown


def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)

    # Create ROS2 node
    ros_node = PointCloudVisualizer()

    # Start ROS2 spinning in a separate thread
    ros_thread = threading.Thread(target=ros_spin_thread, args=(ros_node,), daemon=True)
    ros_thread.start()

    # Create and run visualizer in main thread (matplotlib must run in main thread)
    visualizer = RealtimeVisualizer(ros_node)

    try:
        visualizer.run()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nShutting down visualizer...')
        try:
            ros_node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        ros_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
