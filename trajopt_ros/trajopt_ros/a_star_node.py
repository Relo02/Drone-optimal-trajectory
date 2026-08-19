#!/usr/bin/env python3
"""
Global layer: occupancy grid + rolling-horizon A*, platform-independent.

    sub   pose          geometry_msgs/PoseStamped   robot pose, world frame
    sub   scan          sensor_msgs/PointCloud2     LiDAR hits, world frame
    sub   global_goal   geometry_msgs/PoseStamped   runtime goal override
    pub   path          nav_msgs/Path               local A* waypoints
    pub   local_goal    geometry_msgs/PoseStamped   last waypoint of the segment
    pub   occupancy_grid nav_msgs/OccupancyGrid     the Gaussian grid, for viewers

All names are relative and remapped per platform in the launch file.

The grid is rebuilt from scratch on every tick and carries no memory across
cycles: the robot navigates without a prior map, which is the case this stack
exists to handle.  Note that the occupancy probability P = 1 - Phi(d/sigma)
saturates at 0.5, so any obstacle threshold at or above 0.5 never fires.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from trajopt_core.mapping import FixedGaussianGridMap
from trajopt_core.planning import AStarPlanner
from trajopt_ros.common import cloud_to_xy, sensor_qos, setup_from_parameters


class AStarNode(Node):

    def __init__(self, **kwargs):
        # **kwargs forwards `parameter_overrides`, so the node can be built
        # in-process by tests without a launch file or a parameter server.
        super().__init__("a_star_node", **kwargs)
        self.setup = setup_from_parameters(self)

        grid_cfg = self.setup.grid
        astar_cfg = self.setup.astar

        self.declare_parameter("frame_id", "world")
        self.frame_id = self.get_parameter("frame_id").value

        self._grid = FixedGaussianGridMap(
            reso=float(grid_cfg.get("reso", 0.25)),
            half_width=float(grid_cfg.get("half_width", 5.0)),
            std=float(grid_cfg.get("std", 0.7)),
        )
        self._planner = AStarPlanner(
            obstacle_threshold=float(astar_cfg.get("obstacle_threshold", 0.1)),
            obstacle_cost_weight=float(astar_cfg.get("obstacle_cost_weight", 15.0)),
        )
        self._max_range = float(grid_cfg.get("max_lidar_range", 6.0))
        self._planning_height = float(astar_cfg.get("planning_height", 0.0))
        self._goal_radius = float(astar_cfg.get("goal_reached_radius", 0.3))

        goal = np.asarray(self.setup.goal, dtype=float)
        self._goal = goal[:2] if goal.size >= 2 else np.zeros(2)

        self._pose: PoseStamped | None = None
        self._points: np.ndarray | None = None
        self._goal_announced = False

        self.create_subscription(PoseStamped, "pose", self._on_pose, 10)
        self.create_subscription(PoseStamped, "global_goal", self._on_goal, 10)
        self.create_subscription(PointCloud2, "scan", self._on_scan, sensor_qos())

        self._path_pub = self.create_publisher(Path, "path", 10)
        self._goal_pub = self.create_publisher(PoseStamped, "local_goal", 10)
        self._grid_pub = self.create_publisher(OccupancyGrid, "occupancy_grid", 1)

        rate = float(astar_cfg.get("replan_rate_hz", 5.0))
        self.create_timer(1.0 / rate, self._replan)
        self.get_logger().info(
            f"A* ready | goal=({self._goal[0]:.1f}, {self._goal[1]:.1f}) "
            f"| grid {2 * self._grid.half_width:.0f}x{2 * self._grid.half_width:.0f} m "
            f"@ {self._grid.reso} m/cell | replan {rate} Hz"
        )

    # -- callbacks ---------------------------------------------------------
    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self._goal_announced = False
        self.get_logger().info(
            f"global goal updated: ({self._goal[0]:.2f}, {self._goal[1]:.2f})"
        )

    def _on_scan(self, msg: PointCloud2) -> None:
        robot_xy = None
        if self._pose is not None:
            robot_xy = np.array(
                [self._pose.pose.position.x, self._pose.pose.position.y]
            )
        self._points = cloud_to_xy(msg, robot_xy, self._max_range)

    # -- planning tick -----------------------------------------------------
    def _replan(self) -> None:
        if self._pose is None:
            self.get_logger().warn("waiting for pose ...", throttle_duration_sec=5.0)
            return

        robot_xy = np.array([self._pose.pose.position.x, self._pose.pose.position.y])

        if float(np.linalg.norm(robot_xy - self._goal)) <= self._goal_radius:
            if not self._goal_announced:
                self.get_logger().info("global goal reached")
                self._goal_announced = True
            return
        self._goal_announced = False

        pts3 = (
            np.hstack([self._points, np.zeros((len(self._points), 1))])
            if self._points is not None
            else None
        )
        self._grid.update(pts3, np.append(robot_xy, 0.0))

        stamp = self.get_clock().now().to_msg()
        self._publish_grid(stamp)

        path = self._planner.plan(self._grid, robot_xy, self._goal)
        if not path:
            self.get_logger().warn("no path found", throttle_duration_sec=2.0)
            return

        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        for wx, wy in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.position.z = self._planning_height
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)

        lg = PoseStamped()
        lg.header = msg.header
        lg.pose.position.x = float(path[-1][0])
        lg.pose.position.y = float(path[-1][1])
        lg.pose.position.z = self._planning_height
        lg.pose.orientation.w = 1.0
        self._goal_pub.publish(lg)

        self.get_logger().info(
            f"path={len(path)} wpts  dist_to_goal="
            f"{float(np.linalg.norm(robot_xy - self._goal)):.2f} m",
            throttle_duration_sec=1.0,
        )

    def _publish_grid(self, stamp) -> None:
        g = self._grid
        if not g.is_initialised:
            return
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.info.resolution = g.reso
        msg.info.width = g.cells
        msg.info.height = g.cells
        msg.info.origin.position.x = g.minx
        msg.info.origin.position.y = g.miny
        msg.info.origin.orientation.w = 1.0
        # gmap[ix, iy] -> data[iy * width + ix]; scaled to the int8 [0, 100] range
        msg.data = (g.gmap.T.ravel() * 100.0).clip(0, 100).astype(np.int8).tolist()
        self._grid_pub.publish(msg)


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


if __name__ == "__main__":
    main()
