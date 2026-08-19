#!/usr/bin/env python3
"""
Local layer: the receding-horizon optimal control problem, platform-independent.

    sub   pose            geometry_msgs/PoseStamped    robot pose, world frame
    sub   scan            sensor_msgs/PointCloud2      LiDAR hits, world frame
    sub   path            nav_msgs/Path                A* waypoints
    pub   next_setpoint   geometry_msgs/PoseStamped    lookahead setpoint
    pub   predicted_path  nav_msgs/Path                N-step prediction
    pub   diagnostics     std_msgs/Float64MultiArray   [ok, cost, ms, iters, fails]

All names are relative and remapped per platform in the launch file: on the
aerial platform `next_setpoint` goes to the cascaded PID, on the legged one to the
node that turns it into /cmd_vel.

The node publishes a SETPOINT, not an input.  The optimiser is a reference
generator for a faster platform-specific inner loop, and that separation is not
an implementation detail: measured on the same scenario, applying the first
optimal input directly leaves the robot 8-11 m from the goal after a fixed number
of cycles where the lookahead setpoint leaves it 1.9-2.9 m.

Velocity is obtained by differencing consecutive poses, unfiltered.  Models whose
state carries no velocity (relative degree 1) ignore it entirely; for the others
this is the weakest link in the chain and the natural place for a proper
estimator.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray

from trajopt_core.mapping import FixedGaussianGridMap
from trajopt_core.mpc import TrajectoryOCP, select_lookahead
from trajopt_ros.common import (
    cloud_to_xy,
    quat_to_yaw,
    sensor_qos,
    setup_from_parameters,
    yaw_to_quat,
)


class MPCNode(Node):

    def __init__(self, **kwargs):
        # **kwargs forwards `parameter_overrides`, so the node can be built
        # in-process by tests without a launch file or a parameter server.
        super().__init__("mpc_node", **kwargs)
        self.setup = setup_from_parameters(self)

        self.declare_parameter("frame_id", "world")
        self.declare_parameter("rate_hz", 20.0)
        self.frame_id = self.get_parameter("frame_id").value

        self.model = self.setup.model
        self.cfg = self.setup.cfg
        self._ocp = TrajectoryOCP(self.model, self.cfg, self.setup.obstacle_terms)

        grid_cfg = self.setup.grid
        self._grid = FixedGaussianGridMap(
            reso=float(grid_cfg.get("reso", 0.25)),
            half_width=float(grid_cfg.get("half_width", 5.0)),
            std=float(grid_cfg.get("std", 0.7)),
        )
        self._max_range = float(grid_cfg.get("max_lidar_range", 6.0))
        self._needs_grid = any(
            getattr(t, "name", "").startswith("grid") for t in self.setup.obstacle_terms
        )

        self._pose: PoseStamped | None = None
        self._vel = np.zeros(3)
        self._yaw = 0.0
        self._prev_pos: np.ndarray | None = None
        self._prev_t: float | None = None
        self._path: list | None = None
        self._points: np.ndarray | None = None
        self._n_solves = 0
        self._n_fail = 0

        self.create_subscription(PoseStamped, "pose", self._on_pose, 10)
        self.create_subscription(Path, "path", self._on_path, 10)
        self.create_subscription(PointCloud2, "scan", self._on_scan, sensor_qos())

        self._sp_pub = self.create_publisher(PoseStamped, "next_setpoint", 10)
        self._pred_pub = self.create_publisher(Path, "predicted_path", 10)
        self._diag_pub = self.create_publisher(Float64MultiArray, "diagnostics", 10)

        rate = float(self.get_parameter("rate_hz").value)
        self._deadline_ms = 1e3 / rate
        self.create_timer(1.0 / rate, self._solve)

        s = self._ocp.structure()
        self.get_logger().info(
            f"MPC ready | {rate} Hz (deadline {self._deadline_ms:.0f} ms) | "
            f"{s['n_variables']} vars, {s['n_constraints']} constraints, "
            f"jac density {s['jac_density']:.4f} | "
            f"{'parametric (build once)' if s['is_parametric'] else 'rebuilt each cycle'}"
        )

    # -- callbacks ---------------------------------------------------------
    def _on_pose(self, msg: PoseStamped) -> None:
        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self._prev_pos is not None and self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 1e-3:
                self._vel = (pos - self._prev_pos) / dt

        self._prev_pos, self._prev_t = pos, t
        o = msg.pose.orientation
        self._yaw = quat_to_yaw(o.x, o.y, o.z, o.w)
        self._pose = msg

    def _on_path(self, msg: Path) -> None:
        self._path = (
            [
                (p.pose.position.x, p.pose.position.y, p.pose.position.z)
                for p in msg.poses
            ]
            if msg.poses
            else None
        )

    def _on_scan(self, msg: PointCloud2) -> None:
        robot_xy = None
        if self._pose is not None:
            robot_xy = np.array(
                [self._pose.pose.position.x, self._pose.pose.position.y]
            )
        self._points = cloud_to_xy(msg, robot_xy, self._max_range)

    # -- state assembly ----------------------------------------------------
    def _state(self) -> np.ndarray:
        """Build the model's state vector from the latest pose estimate."""
        p = self._pose.pose.position
        if self.model.PLANS_ALTITUDE:
            return self.model.make_state((p.x, p.y, p.z), self._vel, self._yaw)
        return self.model.make_state((p.x, p.y), self._yaw)

    # -- control tick ------------------------------------------------------
    def _solve(self) -> None:
        if self._pose is None:
            self.get_logger().warn("waiting for pose ...", throttle_duration_sec=5.0)
            return
        if not self._path:
            self.get_logger().warn("waiting for path ...", throttle_duration_sec=5.0)
            return

        state = self._state()
        robot_xy = np.array([self._pose.pose.position.x, self._pose.pose.position.y])

        grid = None
        if self._needs_grid:
            pts3 = (
                np.hstack([self._points, np.zeros((len(self._points), 1))])
                if self._points is not None
                else None
            )
            self._grid.update(pts3, np.append(robot_xy, 0.0))
            grid = self._grid

        try:
            res = self._ocp.solve(state, self._path, points_xy=self._points, grid=grid)
        except Exception as exc:
            self._n_fail += 1
            self.get_logger().error(f"solve raised: {exc}")
            return

        self._n_solves += 1
        if not res.success:
            self._n_fail += 1

        look = select_lookahead(
            self.model,
            res.x_pred,
            robot_xy,
            self.cfg.lookahead_dist,
            fallback_waypoint=self._path[-1],
            z_ref=self.cfg.z_ref,
        )

        stamp = self.get_clock().now().to_msg()
        self._publish_setpoint(stamp, look)
        self._publish_prediction(stamp, res)

        diag = Float64MultiArray()
        diag.data = [
            float(res.success),
            float(res.cost),
            float(res.solve_ms),
            float(res.iterations),
            float(self._n_fail),
        ]
        self._diag_pub.publish(diag)

        if res.solve_ms > self._deadline_ms:
            self.get_logger().warn(
                f"solve {res.solve_ms:.1f} ms exceeded the {self._deadline_ms:.0f} ms "
                f"deadline ({res.iterations} iters, {res.status})",
                throttle_duration_sec=2.0,
            )

        self.get_logger().info(
            f"#{self._n_solves:05d} ok={res.success} cost={res.cost:9.1f} "
            f"solve={res.solve_ms:6.1f} ms iters={res.iterations:3d} "
            f"fails={self._n_fail} "
            f"sp=[{look.position[0]:.2f}, {look.position[1]:.2f}] "
            f"{'(lookahead)' if look.found else '(near goal)'}",
            throttle_duration_sec=0.5,
        )

    # -- publishers --------------------------------------------------------
    def _publish_setpoint(self, stamp, look) -> None:
        sp = PoseStamped()
        sp.header.stamp = stamp
        sp.header.frame_id = self.frame_id
        sp.pose.position.x = float(look.position[0])
        sp.pose.position.y = float(look.position[1])
        sp.pose.position.z = (
            float(look.position[2]) if len(look.position) > 2 else self.cfg.z_ref
        )
        qx, qy, qz, qw = yaw_to_quat(look.yaw)
        sp.pose.orientation.x = qx
        sp.pose.orientation.y = qy
        sp.pose.orientation.z = qz
        sp.pose.orientation.w = qw
        self._sp_pub.publish(sp)

    def _publish_prediction(self, stamp, res) -> None:
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        for k in range(res.x_pred.shape[0]):
            row = res.x_pred[k]
            pos = self.model.position(row)
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(pos[0])
            ps.pose.position.y = float(pos[1])
            ps.pose.position.z = float(pos[2]) if pos.size > 2 else self.cfg.z_ref
            qx, qy, qz, qw = yaw_to_quat(self.model.heading(row))
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            msg.poses.append(ps)
        self._pred_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
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
