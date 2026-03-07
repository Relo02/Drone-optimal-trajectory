"""
A* planner ROS2 node for the Skydio X2 drone.

Architecture
------------
  Subscribes:
    /skydio/pose    PoseStamped  — drone position and orientation
    /skydio/scan3d  PointCloud2  — 3-D lidar hits (world frame)
    /global_goal    PoseStamped  — runtime global goal override

  Publishes:
    /a_star/path            nav_msgs/Path           — local A* path to local goal
    /a_star/local_goal      geometry_msgs/PoseStamped — current local goal (grid-edge point)
    /a_star/occupancy_grid  nav_msgs/OccupancyGrid  — Gaussian grid map (for Foxglove)
    /a_star/grid_raw        std_msgs/Float32MultiArray — raw float32 grid + metadata (for MPC node)
    TF: world -> drone_base_link  (rebroadcast from /skydio/pose)

  Control flow:
    A timer fires at replan_rate_hz.  On each tick:
      1. Build FixedGaussianGridMap from latest LiDAR scan.
      2. Run A* from drone position toward global goal.
         If the goal is outside the local grid, the local A* target is the
         point where the ray (drone -> global_goal) intersects the grid boundary.
      3. Publish path, local_goal, occupancy_grid, grid_raw.

author: Lorenzo Ortolani
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformBroadcaster

from new_mujoco.a_star_planner import AStarPlanner
from new_mujoco.gaussian_grid_map import FixedGaussianGridMap


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class AStarNode(Node):

    def __init__(self):
        super().__init__('a_star_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('goal_x',               10.0)
        self.declare_parameter('goal_y',                1.0)
        self.declare_parameter('goal_z',                1.5)
        self.declare_parameter('grid_reso',             0.25)
        self.declare_parameter('grid_half_width',       5.0)
        self.declare_parameter('grid_std',              0.4)
        self.declare_parameter('obstacle_threshold',    0.5)
        self.declare_parameter('obstacle_cost_weight', 10.0)
        self.declare_parameter('replan_rate_hz',        5.0)
        self.declare_parameter('goal_reached_radius',   0.3)
        self.declare_parameter('max_lidar_range',       6.0)
        self.declare_parameter('planning_height',       1.5)

        self._goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value,
        ])
        self._planning_height    = float(self.get_parameter('planning_height').value)
        self._goal_reached_radius = float(self.get_parameter('goal_reached_radius').value)
        self._max_lidar_range    = float(self.get_parameter('max_lidar_range').value)

        # ── Algorithm objects ─────────────────────────────────────────
        self._grid_map = FixedGaussianGridMap(
            reso=float(self.get_parameter('grid_reso').value),
            half_width=float(self.get_parameter('grid_half_width').value),
            std=float(self.get_parameter('grid_std').value),
        )
        self._planner = AStarPlanner(
            obstacle_threshold=float(self.get_parameter('obstacle_threshold').value),
            obstacle_cost_weight=float(self.get_parameter('obstacle_cost_weight').value),
        )

        # ── State ─────────────────────────────────────────────────────
        self._pose: PoseStamped | None = None
        self._lidar_points: np.ndarray | None = None
        self._goal_reached = False

        # ── TF broadcaster ────────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── QoS ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(PoseStamped, '/skydio/pose',  self._pose_cb,  10)
        self.create_subscription(PoseStamped, '/global_goal',  self._goal_cb,  10)
        self.create_subscription(PointCloud2, '/skydio/scan3d', self._lidar_cb, sensor_qos)

        # ── Publishers ────────────────────────────────────────────────
        self._path_pub  = self.create_publisher(Path,              '/a_star/path',           10)
        self._lgpal_pub = self.create_publisher(PoseStamped,       '/a_star/local_goal',     10)
        self._grid_pub  = self.create_publisher(OccupancyGrid,     '/a_star/occupancy_grid', 10)
        self._raw_pub   = self.create_publisher(Float32MultiArray, '/a_star/grid_raw',       10)

        # ── Replan timer ──────────────────────────────────────────────
        rate = float(self.get_parameter('replan_rate_hz').value)
        self.create_timer(1.0 / rate, self._replan_cb)

        self.get_logger().info(
            f'A* node ready | goal=({self._goal[0]:.1f}, {self._goal[1]:.1f}, {self._goal[2]:.1f})'
            f' | grid={2*self._grid_map.half_width:.0f}m × {2*self._grid_map.half_width:.0f}m'
            f' @ {self._grid_map.reso}m/cell'
        )
        self.get_logger().info(
            'Topics:\n'
            '  sub  /skydio/pose\n'
            '  sub  /skydio/scan3d\n'
            '  sub  /global_goal      (runtime goal override)\n'
            '  pub  /a_star/path\n'
            '  pub  /a_star/local_goal\n'
            '  pub  /a_star/occupancy_grid\n'
            '  pub  /a_star/grid_raw\n'
            '  tf   world -> drone_base_link'
        )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        self._pose = msg

        # Rebroadcast pose as TF  world → drone_base_link
        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = 'drone_base_link'
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self._tf_broadcaster.sendTransform(t)

    def _goal_cb(self, msg: PoseStamped):
        self._goal = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        self._goal_reached = False
        self.get_logger().info(
            f'Global goal updated: ({self._goal[0]:.2f}, {self._goal[1]:.2f}, {self._goal[2]:.2f})'
        )

    def _lidar_cb(self, msg: PointCloud2):
        pts = []
        for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            pts.append([p[0], p[1], p[2]])

        if not pts:
            self._lidar_points = None
            return

        arr = np.array(pts, dtype=float)

        # Range filter relative to drone position
        if self._pose is not None:
            px = self._pose.pose.position.x
            py = self._pose.pose.position.y
            dists = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
            arr = arr[dists <= self._max_lidar_range]

        self._lidar_points = arr if len(arr) > 0 else None

    # ── Replan timer ──────────────────────────────────────────────────

    def _replan_cb(self):
        if self._pose is None:
            return

        pos   = self._pose.pose.position
        drone = np.array([pos.x, pos.y, pos.z])

        dist_to_goal = float(np.linalg.norm(drone[:2] - self._goal[:2]))
        if dist_to_goal <= self._goal_reached_radius:
            if not self._goal_reached:
                self.get_logger().info(
                    f'[A*] Global goal reached! dist={dist_to_goal:.2f} m'
                )
                self._goal_reached = True
            return
        self._goal_reached = False

        # 1. Update grid
        self._grid_map.update(self._lidar_points, drone)

        # 2. Publish occupancy grid and raw data (always, even before A*)
        stamp = self.get_clock().now().to_msg()
        self._publish_occupancy_grid(stamp)
        self._publish_grid_raw(stamp)

        # 3. Run A*
        path = self._planner.plan(self._grid_map, drone[:2], self._goal[:2])

        if path is None or len(path) == 0:
            self.get_logger().warn('[A*] No path found', throttle_duration_sec=2.0)
            return

        local_goal = path[-1]

        # 4. Publish Path
        path_msg = Path()
        path_msg.header.stamp    = stamp
        path_msg.header.frame_id = 'world'
        for wx, wy in path:
            ps = PoseStamped()
            ps.header.stamp    = stamp
            ps.header.frame_id = 'world'
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.position.z = self._planning_height
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self._path_pub.publish(path_msg)

        # 5. Publish local goal
        lg = PoseStamped()
        lg.header.stamp    = stamp
        lg.header.frame_id = 'world'
        lg.pose.position.x = float(local_goal[0])
        lg.pose.position.y = float(local_goal[1])
        lg.pose.position.z = self._planning_height
        lg.pose.orientation.w = 1.0
        self._lgpal_pub.publish(lg)

        self.get_logger().info(
            f'[A*] path={len(path)} wpts '
            f'local_goal=({local_goal[0]:.2f}, {local_goal[1]:.2f}) '
            f'dist_global={dist_to_goal:.2f} m',
            throttle_duration_sec=1.0,
        )

    # ── Publisher helpers ──────────────────────────────────────────────

    def _publish_occupancy_grid(self, stamp):
        gm = self._grid_map
        if gm.gmap is None:
            return

        msg = OccupancyGrid()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'world'
        msg.info.resolution = gm.reso
        msg.info.width      = gm.cells
        msg.info.height     = gm.cells
        msg.info.origin.position.x = gm.minx
        msg.info.origin.position.y = gm.miny
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # gmap[ix, iy] -> data[iy * width + ix]
        # Transpose so that x is the column axis (width), y is the row axis (height)
        scaled = (gm.gmap.T.flatten() * 100.0).clip(0, 100).astype(np.int8)
        msg.data = scaled.tolist()

        self._grid_pub.publish(msg)

    def _publish_grid_raw(self, stamp):
        gm = self._grid_map
        if gm.gmap is None:
            return

        msg = Float32MultiArray()
        # Header metadata: [minx, miny, reso, cells]
        # Followed by flattened gmap in row-major (C) order
        meta = [float(gm.minx), float(gm.miny), float(gm.reso), float(gm.cells)]
        msg.data = meta + gm.gmap.flatten(order='C').astype(np.float32).tolist()
        self._raw_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AStarNode()
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
