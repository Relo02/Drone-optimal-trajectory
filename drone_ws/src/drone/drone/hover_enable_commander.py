#!/usr/bin/env python3
# Hover takeoff commander: holds position until altitude is stable, then enables MPC.

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import Int32
from px4_msgs.msg import (
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleOdometry,
    VehicleCommand,
    VehicleStatus,
    VehicleCommandAck,
)


class HoverEnableCommander(Node):
    def __init__(self):
        super().__init__("hover_enable_commander")

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Parameters
        self.takeoff_height = float(self.declare_parameter("takeoff_height", 0.5).value)
        self.setpoint_rate_hz = float(self.declare_parameter("setpoint_rate_hz", 20.0).value)
        self.offboard_setpoint_min = int(self.declare_parameter("offboard_setpoint_min", 10).value)
        self.hover_xy_tolerance = float(self.declare_parameter("hover_xy_tolerance", 0.3).value)
        self.hover_z_tolerance = float(self.declare_parameter("hover_z_tolerance", 0.2).value)
        self.hover_speed_tolerance = float(self.declare_parameter("hover_speed_tolerance", 0.2).value)
        self.hover_hold_s = float(self.declare_parameter("hover_hold_s", 1.0).value)
        self.handover_delay_s = float(self.declare_parameter("handover_delay_s", 0.5).value)
        self.enable_topic = self.declare_parameter("enable_topic", "/mpc/enable").value

        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10
        )
        self.enable_pub = self.create_publisher(Int32, self.enable_topic, 10)

        # Subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self.status_cb, qos
        )
        self.odom_sub = self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_cb, qos
        )
        self.ack_sub = self.create_subscription(
            VehicleCommandAck, "/fmu/out/vehicle_command_ack", self.ack_cb, qos
        )

        # State
        self.current_odom_enu = None
        self.current_vel_enu = None
        self.current_yaw = 0.0
        self.hover_setpoint = None
        self.hover_reached_since = None
        self.mpc_enabled = False
        self.enable_time = None
        self.handover_complete = False

        self.armed = False
        self.offboard = False
        self.preflight_ok = False
        self.failsafe = False

        self.setpoint_counter = 0
        self.last_cmd_time = 0.0
        self.last_state_log = 0.0
        self.last_setpoint_log = 0.0

        # Timers
        self.publish_timer = self.create_timer(
            1.0 / self.setpoint_rate_hz, self.publish_hover
        )
        self.command_timer = self.create_timer(1.0, self.command_loop)

        self.get_logger().info("=== HOVER ENABLE COMMANDER ===")
        self.get_logger().info("Holding hover until altitude stable, then enabling MPC")

    def _now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def enu_to_ned_position(self, pos_enu):
        return [pos_enu[1], pos_enu[0], -pos_enu[2]]

    def enu_to_ned_yaw(self, yaw_enu):
        yaw_ned = (math.pi / 2.0) - yaw_enu
        return math.atan2(math.sin(yaw_ned), math.cos(yaw_ned))

    def ned_to_enu_position(self, pos_ned):
        return [pos_ned[1], pos_ned[0], -pos_ned[2]]

    def ned_to_enu_yaw(self, yaw_ned):
        yaw_enu = (math.pi / 2.0) - yaw_ned
        return math.atan2(math.sin(yaw_enu), math.cos(yaw_enu))

    # -----------------------------
    # Callbacks
    # -----------------------------
    def odom_cb(self, msg: VehicleOdometry):
        pos_ned = [msg.position[0], msg.position[1], msg.position[2]]
        vel_ned = [msg.velocity[0], msg.velocity[1], msg.velocity[2]]
        self.current_odom_enu = self.ned_to_enu_position(pos_ned)
        self.current_vel_enu = self.ned_to_enu_position(vel_ned)

        q = msg.q
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_ned = math.atan2(siny_cosp, cosy_cosp)
        self.current_yaw = self.ned_to_enu_yaw(yaw_ned)

    def status_cb(self, msg: VehicleStatus):
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.offboard = msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.preflight_ok = msg.pre_flight_checks_pass
        self.failsafe = msg.failsafe

        now = time.time()
        if now - self.last_state_log > 2.0:
            self.get_logger().info(
                f"Preflight={self.preflight_ok} Failsafe={self.failsafe} "
                f"armed={self.armed} offboard={self.offboard}"
            )
            self.last_state_log = now

    def ack_cb(self, msg: VehicleCommandAck):
        if msg.command not in (
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
        ):
            return

        result_map = {
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED: "ACCEPTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED: "TEMP_REJECT",
            VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED: "DENIED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_UNSUPPORTED: "UNSUPPORTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_FAILED: "FAILED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS: "IN_PROGRESS",
            VehicleCommandAck.VEHICLE_CMD_RESULT_CANCELLED: "CANCELLED",
        }
        result_text = result_map.get(msg.result, f"UNKNOWN({msg.result})")
        cmd_name = (
            "ARM"
            if msg.command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
            else "OFFBOARD"
        )
        self.get_logger().info(f"ACK {cmd_name}: {result_text} (reason={msg.result_param1})")

    # -----------------------------
    # Hover logic
    # -----------------------------
    def _init_hover_setpoint(self):
        if self.current_odom_enu is None:
            return
        z = max(self.current_odom_enu[2], self.takeoff_height)
        self.hover_setpoint = [
            self.current_odom_enu[0],
            self.current_odom_enu[1],
            z,
            self.current_yaw,
        ]
        self.hover_reached_since = None
        self.get_logger().info(
            f"Hover target ENU=({self.hover_setpoint[0]:.2f}, "
            f"{self.hover_setpoint[1]:.2f}, {self.hover_setpoint[2]:.2f})"
        )

    def _hover_conditions_met(self):
        if self.hover_setpoint is None or self.current_odom_enu is None:
            return False

        dx = self.current_odom_enu[0] - self.hover_setpoint[0]
        dy = self.current_odom_enu[1] - self.hover_setpoint[1]
        dz = self.current_odom_enu[2] - self.hover_setpoint[2]
        xy_dist = math.hypot(dx, dy)
        z_err = abs(dz)

        speed_ok = True
        if self.current_vel_enu is not None:
            speed = math.sqrt(
                (self.current_vel_enu[0] ** 2)
                + (self.current_vel_enu[1] ** 2)
                + (self.current_vel_enu[2] ** 2)
            )
            speed_ok = speed <= self.hover_speed_tolerance

        return (
            xy_dist <= self.hover_xy_tolerance
            and z_err <= self.hover_z_tolerance
            and speed_ok
        )

    def _update_hover_state(self, now):
        if self.mpc_enabled:
            return
        if self.hover_setpoint is None:
            self._init_hover_setpoint()
            return

        if not self._hover_conditions_met():
            self.hover_reached_since = None
            return

        if self.hover_reached_since is None:
            self.hover_reached_since = now
            return

        if (now - self.hover_reached_since) >= self.hover_hold_s:
            self.mpc_enabled = True
            self.enable_time = now
            self.get_logger().info("Hover stabilized; enabling MPC")

    # -----------------------------
    # Timers
    # -----------------------------
    def publish_hover(self):
        now = time.time()
        self._update_hover_state(now)

        enable_msg = Int32()
        enable_msg.data = 1 if self.mpc_enabled else 0
        self.enable_pub.publish(enable_msg)

        if self.handover_complete:
            return

        if self.mpc_enabled and self.enable_time is not None:
            if (now - self.enable_time) >= self.handover_delay_s:
                self.handover_complete = True
                return

        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = self._now_us()
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.offboard_pub.publish(offboard_msg)

        if self.hover_setpoint is None:
            self._init_hover_setpoint()
        if self.hover_setpoint is None:
            return

        pos_ned = self.enu_to_ned_position(self.hover_setpoint[:3])
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = self._now_us()
        traj_msg.position = [float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])]
        traj_msg.yaw = float(self.enu_to_ned_yaw(self.hover_setpoint[3]))
        self.traj_pub.publish(traj_msg)

        self.setpoint_counter += 1

        if now - self.last_setpoint_log > 2.0:
            self.get_logger().info(
                f"Hover setpoint ENU=({self.hover_setpoint[0]:.2f}, "
                f"{self.hover_setpoint[1]:.2f}, {self.hover_setpoint[2]:.2f}) "
                f"enabled={self.mpc_enabled}"
            )
            self.last_setpoint_log = now

    def command_loop(self):
        if self.handover_complete or not self.preflight_ok:
            return

        now = time.time()
        if now - self.last_cmd_time < 1.0:
            return

        if not self.armed:
            cmd = VehicleCommand()
            cmd.timestamp = self._now_us()
            cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
            cmd.param1 = 1.0
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.source_system = 1
            cmd.source_component = 1
            cmd.from_external = True
            self.cmd_pub.publish(cmd)
            self.last_cmd_time = now
            return

        if not self.offboard and self.setpoint_counter >= self.offboard_setpoint_min:
            cmd = VehicleCommand()
            cmd.timestamp = self._now_us()
            cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
            cmd.param1 = 1.0
            cmd.param2 = 6.0
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.source_system = 1
            cmd.source_component = 1
            cmd.from_external = True
            self.cmd_pub.publish(cmd)
            self.last_cmd_time = now
            self.get_logger().info("Sent OFFBOARD mode command")


def main(args=None):
    rclpy.init(args=args)
    node = HoverEnableCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Hover commander stopped by user")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
