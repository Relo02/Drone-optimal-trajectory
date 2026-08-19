"""
The Go2's inner loop: MPC lookahead setpoint -> body-frame /cmd_vel.

    sub   /go2/pose            geometry_msgs/PoseStamped
    sub   /mpc/next_setpoint   geometry_msgs/PoseStamped
    pub   /cmd_vel             geometry_msgs/Twist

This is the same proportional tracker used by the other platforms: the MPC is
a reference generator, and what closes the loop around it is a fast, dumb
tracker.  Keeping that split identical across the three platforms is what makes
their solver metrics comparable.

The Go2 can reverse, so the default command box is symmetric in forward motion
and yaw control stays enabled.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


class SetpointToCmdVelNode(Node):

    def __init__(self, **kwargs):
        super().__init__("setpoint_to_cmd_vel", **kwargs)

        self.declare_parameter("cmd_rate_hz", 20.0)
        self.declare_parameter("cmd_kp_xy", 1.0)
        self.declare_parameter("cmd_kp_yaw", 1.5)
        # Defaults match the planner's own limits in legged_overrides.yaml.
        self.declare_parameter("cmd_max_vx", 1.0)
        self.declare_parameter("cmd_max_vy", 0.5)
        self.declare_parameter("cmd_max_omega", 1.5)
        self.declare_parameter("cmd_stop_radius", 0.2)
        self.declare_parameter("setpoint_timeout_sec", 1.0)
        self.declare_parameter("enable_yaw_control", True)
        self.declare_parameter("allow_reverse", True)
        # Beyond this heading error the robot turns before it walks.
        self.declare_parameter("turn_in_place_angle", 1.2)   # rad, ~69 deg

        self._rate_hz = float(self.get_parameter("cmd_rate_hz").value)
        self._kp_xy = float(self.get_parameter("cmd_kp_xy").value)
        self._kp_yaw = float(self.get_parameter("cmd_kp_yaw").value)
        self._max_vx = float(self.get_parameter("cmd_max_vx").value)
        self._max_vy = float(self.get_parameter("cmd_max_vy").value)
        self._max_omega = float(self.get_parameter("cmd_max_omega").value)
        self._stop_radius = float(self.get_parameter("cmd_stop_radius").value)
        self._timeout = float(self.get_parameter("setpoint_timeout_sec").value)
        self._yaw_control = bool(self.get_parameter("enable_yaw_control").value)
        self._allow_reverse = bool(self.get_parameter("allow_reverse").value)
        self._turn_angle = float(self.get_parameter("turn_in_place_angle").value)

        self._pose: PoseStamped | None = None
        self._yaw = 0.0
        self._setpoint: PoseStamped | None = None
        self._setpoint_time = None

        self.create_subscription(PoseStamped, "/go2/pose", self._on_pose, 10)
        self.create_subscription(PoseStamped, "/mpc/next_setpoint", self._on_setpoint, 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / self._rate_hz, self._control_tick)

        self.get_logger().info(
            "setpoint_to_cmd_vel ready: /mpc/next_setpoint + /go2/pose -> /cmd_vel "
            f"(reverse {'allowed' if self._allow_reverse else 'forbidden'}, "
            f"yaw control {'on' if self._yaw_control else 'off'})"
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg
        q = msg.pose.orientation
        self._yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

    def _on_setpoint(self, msg: PoseStamped) -> None:
        self._setpoint = msg
        self._setpoint_time = self.get_clock().now()

    def _publish_zero(self) -> None:
        self._cmd_pub.publish(Twist())

    def _control_tick(self) -> None:
        if self._pose is None or self._setpoint is None or self._setpoint_time is None:
            self._publish_zero()
            return

        age = (self.get_clock().now() - self._setpoint_time).nanoseconds * 1e-9
        if age > self._timeout:
            self.get_logger().warn(
                f"setpoint stale ({age:.2f}s > {self._timeout:.2f}s), zeroing /cmd_vel",
                throttle_duration_sec=2.0,
            )
            self._publish_zero()
            return

        dx = float(self._setpoint.pose.position.x) - float(self._pose.pose.position.x)
        dy = float(self._setpoint.pose.position.y) - float(self._pose.pose.position.y)
        dist = math.hypot(dx, dy)
        if dist <= self._stop_radius:
            self._publish_zero()
            return

        # World -> body frame (x forward, y left)
        ex = math.cos(self._yaw) * dx + math.sin(self._yaw) * dy
        ey = -math.sin(self._yaw) * dx + math.cos(self._yaw) * dy

        cmd = Twist()
        heading_err = _wrap_to_pi(math.atan2(dy, dx) - self._yaw)

        if self._yaw_control:
            cmd.angular.z = _clamp(self._kp_yaw * heading_err,
                                   -self._max_omega, self._max_omega)

        if not self._allow_reverse and abs(heading_err) > self._turn_angle:
            # Facing away: turn first.  Translating now would only be lateral
            # crabbing, which is slower than turning and reads as circling.
            self._cmd_pub.publish(cmd)
            return

        vx_min = -self._max_vx if self._allow_reverse else 0.0
        cmd.linear.x = _clamp(self._kp_xy * ex, vx_min, self._max_vx)
        cmd.linear.y = _clamp(self._kp_xy * ey, -self._max_vy, self._max_vy)
        self._cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SetpointToCmdVelNode()
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
