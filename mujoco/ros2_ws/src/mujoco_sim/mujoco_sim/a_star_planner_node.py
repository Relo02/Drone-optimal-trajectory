"""
A* Path Planner ROS2 Node

Provides path planning services using the A* algorithm with
Gaussian grid map obstacle representation.

author: Lorenzo Ortolani
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from mujoco_sim.gaussian_grid_map import GaussianGridMap
from mujoco_sim.A_star_planner import AStarLocalPlanner


class AStarPlannerNode(Node):
    """ROS2 node for A* path planning with Gaussian grid maps."""

    def __init__(self):
        super().__init__('a_star_planner')

        # Declare parameters
        self.declare_parameter('grid_resolution', 0.2)
        self.declare_parameter('gaussian_std', 0.3)
        self.declare_parameter('extend_area', 2.0)
        self.declare_parameter('obstacle_threshold', 0.5)
        self.declare_parameter('obstacle_cost_weight', 10.0)
        self.declare_parameter('planning_height', 1.5)
        self.declare_parameter('replan_rate', 2.0)

        # Get parameters
        self.grid_resolution = self.get_parameter('grid_resolution').value
        self.gaussian_std = self.get_parameter('gaussian_std').value
        self.extend_area = self.get_parameter('extend_area').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.obstacle_cost_weight = self.get_parameter('obstacle_cost_weight').value
        self.planning_height = self.get_parameter('planning_height').value
        self.replan_rate = self.get_parameter('replan_rate').value

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

        # State variables
        self.current_pose = None
        self.goal_pose = None
        self.lidar_points = None
        self.last_path = None
        self.path_published = False  # Only plan once per goal

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

        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # Timer for periodic replanning
        replan_period = 1.0 / self.replan_rate
        self.replan_timer = self.create_timer(replan_period, self.replan_callback)

        self.get_logger().info('A* Planner Node initialized')
        self.get_logger().info(f'  Grid resolution: {self.grid_resolution}m')
        self.get_logger().info(f'  Obstacle threshold: {self.obstacle_threshold}')
        self.get_logger().info(f'  Replan rate: {self.replan_rate}Hz')

    def pose_callback(self, msg: PoseStamped):
        """Handle current drone pose updates."""
        self.current_pose = msg

    def goal_callback(self, msg: PoseStamped):
        """Handle goal pose updates and trigger planning."""
        # Check if this is actually a new goal
        is_new_goal = True
        if self.goal_pose is not None:
            old_goal = (self.goal_pose.pose.position.x, self.goal_pose.pose.position.y, self.goal_pose.pose.position.z)
            new_goal = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
            # Consider same if within 0.1m
            if np.linalg.norm(np.array(old_goal) - np.array(new_goal)) < 0.1:
                is_new_goal = False

        self.goal_pose = msg

        if is_new_goal:
            self.path_published = False  # Reset flag only for new goal
            self.get_logger().info(
                f'New goal received: ({msg.pose.position.x:.2f}, '
                f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})'
            )
            # Trigger immediate planning
            self.plan_path()

    def pointcloud_callback(self, msg: PointCloud2):
        """Handle lidar point cloud updates."""
        # Convert PointCloud2 to numpy array
        points_list = []
        for point in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])

        if points_list:
            self.lidar_points = np.array(points_list)
        else:
            self.lidar_points = None

    def replan_callback(self):
        """Periodic replanning callback - only plans if no path published yet."""
        if self.goal_pose is not None and self.current_pose is not None and not self.path_published:
            self.plan_path()

    def plan_path(self):
        """Execute A* path planning."""
        if self.current_pose is None:
            self.get_logger().warn('Cannot plan: no current pose')
            return

        if self.goal_pose is None:
            self.get_logger().warn('Cannot plan: no goal pose')
            return

        # Get current position
        start = (
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y
        )
        drone_pos = [
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ]

        # Get goal position
        goal = (
            self.goal_pose.pose.position.x,
            self.goal_pose.pose.position.y
        )

        # Update grid map with lidar points if available
        if self.lidar_points is not None and len(self.lidar_points) > 0:
            self.grid_map.update_from_lidar_points(self.lidar_points, drone_pos)
            self.planner.set_grid_map(self.grid_map)
        else:
            # Create empty grid map centered on drone if no obstacles
            self.grid_map._calc_grid_config(
                np.array([drone_pos[0]]),
                np.array([drone_pos[1]]),
                drone_pos
            )
            self.grid_map.gmap = np.zeros((self.grid_map.xw, self.grid_map.yw))
            self.planner.set_grid_map(self.grid_map)

        # Plan path
        path_world = self.planner.plan(start, goal)

        if path_world is None:
            self.get_logger().warn('A* planner failed to find path')
            return

        # Convert to ROS Path message
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'world'

        for wx, wy in path_world:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = self.planning_height
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.last_path = path_msg
        self.path_published = True  # Don't replan until new goal

        self.get_logger().info(f'Published path with {len(path_world)} waypoints')


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
