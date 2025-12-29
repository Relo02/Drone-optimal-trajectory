#!/usr/bin/env python3
# Simple offboard commander: follows MPC path or holds position.

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Path
from std_msgs.msg import Int32

from px4_msgs.msg import (
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleOdometry,
    VehicleCommand,
    VehicleStatus,
    VehicleCommandAck,
)


class MPCMissionCommander(Node):
    def __init__(self):
        super().__init__("mpc_mission_commander")

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Publishers
        self.cmd_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
        )
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10
        )

        self.require_mpc_enable = bool(
            self.declare_parameter("require_mpc_enable", False).value
        )
        self.enable_topic = self.declare_parameter("enable_topic", "/mpc/enable").value
        self.mpc_enabled = not self.require_mpc_enable

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
        self.path_sub = self.create_subscription(
            Path, "/mpc/optimal_trajectory", self.path_callback, qos
        )
        self.enable_sub = self.create_subscription(
            Int32, self.enable_topic, self.enable_callback, 10
        )

        # State
        self.current_odom_enu = None
        self.current_yaw = 0.0
        self.latest_path = []
        self.last_path_time = None

        self.armed = False
        self.offboard = False
        self.preflight_ok = False
        self.failsafe = False

        self.last_state_log = 0.0
        self.last_setpoint_log = 0.0
        self.last_cmd_time = 0.0
        self.setpoint_counter = 0
        self.last_target_idx = None
        self.last_target_point = None
        self.last_publish_time = None

        # Tuning
        self.lookahead_idx = 3
        self.path_timeout_s = 1.0
        self.takeoff_height = 0.5
        self.setpoint_rate_hz = 20.0
        self.offboard_setpoint_min = 10

        # Timers
        self.publish_timer = self.create_timer(
            1.0 / self.setpoint_rate_hz, self.publish_setpoints
        )
        self.command_timer = self.create_timer(1.0, self.command_loop)

        self.get_logger().info("=== MPC MISSION COMMANDER ===")
        self.get_logger().info("Waiting for MPC Path and publishing setpoints...")

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
        self.current_odom_enu = self.ned_to_enu_position(pos_ned)

        q = msg.q
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_ned = math.atan2(siny_cosp, cosy_cosp)
        self.current_yaw = self.ned_to_enu_yaw(yaw_ned)

    def path_callback(self, msg: Path):
        if not msg.poses:
            return

        waypoints = []
        for pose_stamped in msg.poses:
            pose = pose_stamped.pose
            q = pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            waypoints.append([pose.position.x, pose.position.y, pose.position.z, yaw])

        self.latest_path = waypoints
        self.last_path_time = time.time()

    def status_cb(self, msg: VehicleStatus):
        old_armed = self.armed
        old_offboard = self.offboard

        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.offboard = msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.preflight_ok = msg.pre_flight_checks_pass
        self.failsafe = msg.failsafe

        if old_armed != self.armed:
            self.get_logger().info("ARMED" if self.armed else "DISARMED")
        if old_offboard != self.offboard:
            self.get_logger().info(
                "OFFBOARD enabled" if self.offboard else "OFFBOARD disabled"
            )

        now = time.time()
        if now - self.last_state_log > 2.0:
            self.get_logger().info(
                f"Preflight={self.preflight_ok} Failsafe={self.failsafe} "
                f"mpc_enabled={self.mpc_enabled}"
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

    def enable_callback(self, msg: Int32):
        enabled = msg.data == 1
        if enabled != self.mpc_enabled:
            self.mpc_enabled = enabled
            self.get_logger().info(f"MPC enable set to {int(self.mpc_enabled)}")

    # -----------------------------
    # Setpoint helpers
    # -----------------------------
    def path_is_fresh(self):
        if not self.latest_path or self.last_path_time is None:
            return False
        return (time.time() - self.last_path_time) <= self.path_timeout_s

    def select_target(self):
        if not self.path_is_fresh():
            self.last_target_idx = None
            return None, None
        if self.current_odom_enu is None:
            idx = min(self.lookahead_idx, len(self.latest_path) - 1)
            return self.latest_path[idx], idx

        nearest_idx = 0
        nearest_dist = None
        for i, wp in enumerate(self.latest_path):
            dx = self.current_odom_enu[0] - wp[0]
            dy = self.current_odom_enu[1] - wp[1]
            dz = self.current_odom_enu[2] - wp[2]
            dist = (dx * dx) + (dy * dy) + (dz * dz)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = i

        target_idx = min(nearest_idx + self.lookahead_idx, len(self.latest_path) - 1)
        if self.last_target_idx is not None and target_idx < self.last_target_idx:
            target_idx = min(self.last_target_idx, len(self.latest_path) - 1)

        return self.latest_path[target_idx], target_idx

    def fallback_setpoint(self):
        if self.current_odom_enu is None:
            return [0.0, 0.0, self.takeoff_height, 0.0]

        z = max(self.current_odom_enu[2], self.takeoff_height)
        return [self.current_odom_enu[0], self.current_odom_enu[1], z, self.current_yaw]

    def publish_setpoints(self):
        if self.require_mpc_enable and not self.mpc_enabled:
            return
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = self._now_us()
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.offboard_pub.publish(offboard_msg)

        target, target_idx = self.select_target()
        target_reason = "path"
        if target is None:
            target = self.fallback_setpoint()
            target_reason = "fallback"
        else:
            self.last_target_idx = target_idx
            self.last_target_point = target[:3]

        pos_ned = self.enu_to_ned_position(target[:3])
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = self._now_us()
        traj_msg.position = [float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])]
        traj_msg.yaw = float(self.enu_to_ned_yaw(target[3]))
        self.traj_pub.publish(traj_msg)

        self.setpoint_counter += 1

        now = time.time()
        publish_rate = None
        if self.last_publish_time is not None:
            dt = now - self.last_publish_time
            if dt > 1e-6:
                publish_rate = 1.0 / dt
        self.last_publish_time = now

        if now - self.last_setpoint_log > 2.0:
            goal_dist_text = "n/a"
            if self.latest_path and self.current_odom_enu is not None:
                goal = self.latest_path[-1]
                dx = self.current_odom_enu[0] - goal[0]
                dy = self.current_odom_enu[1] - goal[1]
                dz = self.current_odom_enu[2] - goal[2]
                goal_dist_text = f"{math.sqrt(dx * dx + dy * dy + dz * dz):.2f}"

            idx_text = "n/a" if target_idx is None else f"{target_idx}/{len(self.latest_path) - 1}"
            rate_text = "n/a" if publish_rate is None else f"{publish_rate:.1f}Hz"
            self.get_logger().info(
                f"Setpoint ({target_reason}) ENU=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) "
                f"yaw={target[3]:.2f} idx={idx_text} goal_dist={goal_dist_text} rate={rate_text} "
                f"armed={self.armed} offboard={self.offboard} fresh={self.path_is_fresh()}"
            )
            self.last_setpoint_log = now

    def command_loop(self):
        if self.require_mpc_enable and not self.mpc_enabled:
            return
        if not self.preflight_ok:
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
            cmd.param1 = 1.0  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            cmd.param2 = 6.0  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
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
    node = MPCMissionCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Mission stopped by user")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
