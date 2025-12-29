#!/usr/bin/env python3
# mission_commander.py
# Offboard rectangle coverage mission using position setpoints.

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Path

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

        self.path_sub = self.create_subscription(
            Path,
            '/mpc/optimal_trajectory',
            self.path_callback,
            qos,
        )

        self.idx = 0
        self.current_pos = [0.0, 0.0, 0.0]
        self.current_odom_enu = [0.0, 0.0, 0.0]
        self.armed = False
        self.offboard = False
        self.preflight_ok = False
        self.safety_off = False
        self.failsafe = False
        self.hover_reached = False
        self.last_status_log = 0.0
        self.current_yaw = 0.0
        self.lookahead_idx = 3
        self.mission_complete = False

        # Coverage area (ENU frame: x=East, y=North, z=Up)
        # Hover the drone at 3 meters altitude before starting navigating
        self.hover_setpoint = [0.0, 0.0, 3.0, 0.0]  # x, y, z, yaw
        self.waypoints = []
        self.latest_path = []
        self.waypoints_ready = False

        # if self.hover_reached:
        #     self.get_logger().info('Hover setpoint reached, building waypoints...')
        #     self.waypoints = self.build_coverage_waypoints()
        #     self.waypoints_ready = True
        # else:
        #     self.get_logger().info('Waiting to reach hover setpoint before building waypoints...')

        # Start publishing immediately (PX4 requires a steady stream of setpoints)
        self.publish_timer = self.create_timer(1.0, self.publish_setpoints)  # 50 Hz
        self.hover_timer = self.create_timer(1.0, self.check_hover_reached)

        # Sequencing timers
        self.arm_timer = self.create_timer(2.0, self.arm_sequence)
        self.offboard_timer = self.create_timer(2.0, self.offboard_sequence)
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.get_logger().info('=== MISSION COMMANDER ===')
        self.get_logger().info('Publishing setpoints and waiting to arm...')

    def odom_cb(self, msg):
        pos_ned = [msg.position[0], msg.position[1], msg.position[2]]
        self.current_odom_enu = self.ned_to_enu_position(pos_ned)

    def check_hover_reached(self):
        """Check if the drone has reached the hover setpoint."""
        dx = self.current_odom_enu[0] - self.hover_setpoint[0]
        dy = self.current_odom_enu[1] - self.hover_setpoint[1]
        dz = self.current_odom_enu[2] - self.hover_setpoint[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 0.5:
            self.get_logger().info('Hover setpoint reached.')
            self.hover_reached = True
            if (not self.waypoints_ready) and self.latest_path:
                self.get_logger().info('Building waypoints from latest path...')
                self.waypoints = self.latest_path
                self.waypoints_ready = True
                self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from path.')
        else:
            self.get_logger().info('Waiting to reach hover setpoint...')
            self.hover_reached = False

    def build_coverage_waypoints(self):
        """Get the waypoints from the Path planning mpc."""
        # Get the path from the latest Path message received
        if self.latest_path:
            return self.latest_path

        waypoints = self.current_pos.copy()
        waypoints.append(self.current_yaw)
        return [waypoints]

    def enu_to_ned_position(self, pos_enu):
        """Convert ROS ENU (x=East, y=North, z=Up) to PX4 NED (x=North, y=East, z=Down)."""
        return [pos_enu[1], pos_enu[0], -pos_enu[2]]

    def enu_to_ned_yaw(self, yaw_enu):
        """Convert ENU yaw (about +Z Up) to NED yaw (about +Z Down)."""
        yaw_ned = (math.pi / 2.0) - yaw_enu
        return math.atan2(math.sin(yaw_ned), math.cos(yaw_ned))

    def ned_to_enu_position(self, pos_ned):
        """Convert PX4 NED (x=North, y=East, z=Down) to ROS ENU (x=East, y=North, z=Up)."""
        return [pos_ned[1], pos_ned[0], -pos_ned[2]]

    def path_callback(self, msg):
        try:
            if not msg.poses:
                return

            waypoints = []
            for pose_stamped in msg.poses:
                pose = pose_stamped.pose
                pos = [pose.position.x, pose.position.y, pose.position.z]
                q = pose.orientation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                )
                waypoints.append([pos[0], pos[1], pos[2], yaw])

            self.latest_path = waypoints
            first = waypoints[0]
            self.current_pos = [first[0], first[1], first[2]]
            self.current_yaw = first[3]
            self.get_logger().info(f'Path update - {len(waypoints)} poses, first: {first}')

            if self.hover_reached:
                self.waypoints = self.latest_path
                if not self.waypoints_ready:
                    self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from path.')
                self.waypoints_ready = True
        except Exception as exc:
            self.get_logger().warn(f'Path parsing failed: {exc}')

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
                self.mission_complete = False
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
        """Publish position setpoints in ENU frame."""
        if self.hover_reached:
            self.get_logger().info('Hover setpoint reached, building waypoints...')
            self.waypoints = self.build_coverage_waypoints()
            self.waypoints_ready = True
        else:
            self.get_logger().info('Waiting to reach hover setpoint before building waypoints...')

        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(time.time() * 1e6)
        offboard_msg.position = True
        self.offboard_pub.publish(offboard_msg)

        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(time.time() * 1e6)

        if self.offboard and self.waypoints:
            wp_index = min(self.lookahead_idx, len(self.waypoints) - 1)
            wp = self.waypoints[wp_index]
            pos_ned = self.enu_to_ned_position([wp[0], wp[1], wp[2]])
            traj_msg.position = [float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])]
            traj_msg.yaw = float(self.enu_to_ned_yaw(wp[3]))
        else:
            pos_ned = self.enu_to_ned_position(self.hover_setpoint[:3])
            traj_msg.position = [float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])]
            traj_msg.yaw = float(self.enu_to_ned_yaw(self.hover_setpoint[3]))

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

        if not self.waypoints:
            return

        wp = self.waypoints[-1]
        dx = self.current_odom_enu[0] - wp[0]
        dy = self.current_odom_enu[1] - wp[1]
        dz = self.current_odom_enu[2] - wp[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 0.6 and not self.mission_complete:
            self.get_logger().info('Mission complete')
            self.mission_complete = True


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
