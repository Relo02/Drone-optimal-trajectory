#!/usr/bin/env python3
# mission_commander.py
# Offboard rectangle coverage mission using position setpoints.

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from px4_msgs.msg import (
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleOdometry,
    VehicleCommand,
    VehicleStatus,
    VehicleCommandAck,
)


class MissionCommander(Node):
    def __init__(self):
        super().__init__('mission_commander')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Publishers
        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)

        # Subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.status_cb, qos)
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos)
        self.ack_sub = self.create_subscription(
            VehicleCommandAck, '/fmu/out/vehicle_command_ack', self.ack_cb, qos)

        # Coverage area (NED frame: x=North, y=East, z=Down)
        self.area_length_m = 10.0
        self.area_width_m = 6.0
        self.lane_spacing_m = 2.0
        self.altitude_m = 2.0
        self.home_setpoint = [0.0, 0.0, -self.altitude_m, 0.0]
        self.waypoints = self.build_coverage_waypoints()

        self.idx = 0
        self.current_pos = [0.0, 0.0, 0.0]
        self.armed = False
        self.offboard = False
        self.preflight_ok = False
        self.safety_off = False
        self.failsafe = False
        self.last_status_log = 0.0

        # Start publishing immediately (PX4 requires a steady stream of setpoints)
        self.publish_timer = self.create_timer(0.02, self.publish_setpoints)  # 50 Hz

        # Sequencing timers
        self.arm_timer = self.create_timer(2.0, self.arm_sequence)
        self.offboard_timer = self.create_timer(2.0, self.offboard_sequence)
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.get_logger().info('=== MISSION COMMANDER ===')
        self.get_logger().info('Publishing setpoints and waiting to arm...')

    def build_coverage_waypoints(self):
        """Create a lawnmower-style rectangle coverage pattern."""
        length = float(self.area_length_m)
        width = float(self.area_width_m)
        spacing = float(self.lane_spacing_m)
        altitude = -float(self.altitude_m)

        points = [(0.0, 0.0)]
        y = 0.0
        direction = 1

        while True:
            target_x = length if direction == 1 else 0.0
            if points[-1] != (target_x, y):
                points.append((target_x, y))

            y_next = y + spacing
            if y_next > width + 1e-6:
                break

            points.append((target_x, y_next))
            y = y_next
            direction *= -1

        if points[-1] != (0.0, 0.0):
            points.append((0.0, 0.0))

        waypoints = []
        for i, (x, y) in enumerate(points):
            if i + 1 < len(points):
                next_x, next_y = points[i + 1]
                yaw = math.atan2(next_y - y, next_x - x)
            elif i > 0:
                prev_x, prev_y = points[i - 1]
                yaw = math.atan2(y - prev_y, x - prev_x)
            else:
                yaw = 0.0
            waypoints.append([x, y, altitude, yaw])

        return waypoints

    def odom_cb(self, msg):
        try:
            self.current_pos = [msg.position[0], msg.position[1], msg.position[2]]
        except Exception:
            pass

    def status_cb(self, msg):
        old_armed = self.armed
        old_offboard = self.offboard

        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.offboard = msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.preflight_ok = msg.pre_flight_checks_pass
        self.safety_off = msg.safety_off
        self.failsafe = msg.failsafe

        if old_armed != self.armed:
            if self.armed:
                self.get_logger().info('ARMED')
            else:
                self.get_logger().warn('DISARMED')

        if old_offboard != self.offboard:
            if self.offboard:
                self.get_logger().info('OFFBOARD mode enabled')
                self.get_logger().info('Starting mission...')
                self.idx = 0
            else:
                self.get_logger().warn('Left OFFBOARD mode')

    def ack_cb(self, msg):
        if msg.command not in (
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
        ):
            return

        result_map = {
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED: 'ACCEPTED',
            VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED: 'TEMP_REJECT',
            VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED: 'DENIED',
            VehicleCommandAck.VEHICLE_CMD_RESULT_UNSUPPORTED: 'UNSUPPORTED',
            VehicleCommandAck.VEHICLE_CMD_RESULT_FAILED: 'FAILED',
            VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS: 'IN_PROGRESS',
            VehicleCommandAck.VEHICLE_CMD_RESULT_CANCELLED: 'CANCELLED',
        }
        result_text = result_map.get(msg.result, f'UNKNOWN({msg.result})')
        cmd_name = 'ARM' if msg.command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM else 'OFFBOARD'
        self.get_logger().info(
            f'ACK {cmd_name}: {result_text} (reason={msg.result_param1})')

    def publish_setpoints(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(time.time() * 1e6)
        offboard_msg.position = True
        self.offboard_pub.publish(offboard_msg)

        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(time.time() * 1e6)

        if self.offboard and self.idx < len(self.waypoints):
            wp = self.waypoints[self.idx]
            traj_msg.position = [float(wp[0]), float(wp[1]), float(wp[2])]
            traj_msg.yaw = float(wp[3])
        else:
            traj_msg.position = [
                self.home_setpoint[0],
                self.home_setpoint[1],
                self.home_setpoint[2],
            ]
            traj_msg.yaw = self.home_setpoint[3]

        self.traj_pub.publish(traj_msg)

    def arm_sequence(self):
        if self.armed:
            return

        now = time.time()
        if now - self.last_status_log > 2.0:
            self.get_logger().info(
                f'Preflight={self.preflight_ok} SafetyOff={self.safety_off} Failsafe={self.failsafe}')
            self.last_status_log = now

        cmd = VehicleCommand()
        cmd.timestamp = int(time.time() * 1e6)
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = 1.0
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.cmd_pub.publish(cmd)
        self.get_logger().info('Sent ARM command')

    def offboard_sequence(self):
        if not self.armed or self.offboard:
            return

        cmd = VehicleCommand()
        cmd.timestamp = int(time.time() * 1e6)
        cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        cmd.param1 = 1.0  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        cmd.param2 = 6.0  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.cmd_pub.publish(cmd)
        self.get_logger().info('Sent OFFBOARD mode command')

    def control_loop(self):
        if not self.armed or not self.offboard:
            return

        if self.idx >= len(self.waypoints):
            return

        wp = self.waypoints[self.idx]
        dx = self.current_pos[0] - wp[0]
        dy = self.current_pos[1] - wp[1]
        dz = self.current_pos[2] - wp[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 0.6:
            self.get_logger().info(f'Reached waypoint {self.idx}: {wp}')
            self.idx += 1
            if self.idx < len(self.waypoints):
                next_wp = self.waypoints[self.idx]
                self.get_logger().info(f'Next waypoint {self.idx}: {next_wp}')
            else:
                self.get_logger().info('Mission complete')


def main(args=None):
    rclpy.init(args=args)
    node = MissionCommander()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Mission stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
