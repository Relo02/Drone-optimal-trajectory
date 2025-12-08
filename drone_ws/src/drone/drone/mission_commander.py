#!/usr/bin/env python3
# fly_now.py
# Complete solution - arms, switches to offboard, and flies mission

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math
import time
import threading

from px4_msgs.msg import (
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleOdometry,
    VehicleCommand,
    VehicleStatus
)

class FlyNow(Node):
    def __init__(self):
        super().__init__('fly_now')
        
        # Mission waypoints
        self.waypoints = [
            [0.0, 0.0, -2.0, 0.0],      # Takeoff to 2m
            [5.0, 0.0, -2.0, 0.0],      # Move 5m North
            [5.0, 5.0, -2.0, 1.57],     # Move 5m East, yaw 90°
            [0.0, 5.0, -2.0, 3.14],     # Move 5m West, yaw 180°
            [0.0, 0.0, -2.0, 0.0],      # Return home
        ]
        
        self.idx = 0
        self.current_pos = [0.0, 0.0, 0.0]
        self.armed = False
        self.offboard = False
        self.publishing = False
        self.start_time = time.time()
        
        # QoS for PX4
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        
        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        
        # Subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_cb, qos)
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos)
        
        # Start publishing immediately
        self.publishing = True
        self.publish_timer = self.create_timer(0.02, self.publish_setpoints)  # 50Hz
        
        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz
        
        self.get_logger().info('=== FLY NOW ===')
        self.get_logger().info('Starting setpoint publishing...')
        
        # Start arming sequence after 2 seconds
        self.create_timer(2.0, self.start_arming_sequence)

    def odom_cb(self, msg):
        """Get current position"""
        try:
            self.current_pos = [msg.position[0], msg.position[1], msg.position[2]]
        except:
            pass

    def status_cb(self, msg):
        """Get vehicle status"""
        old_armed = self.armed
        old_offboard = self.offboard
        
        self.armed = msg.arming_state == 2
        self.offboard = msg.nav_state == 7
        
        if old_armed != self.armed:
            if self.armed:
                self.get_logger().info('✓ ARMED!')
            else:
                self.get_logger().warn('DISARMED!')
        
        if old_offboard != self.offboard:
            if self.offboard:
                self.get_logger().info('✓ OFFBOARD mode!')
                self.get_logger().info('Starting mission in 1 second...')
                # Start mission 1 second after offboard mode
                self.create_timer(1.0, self.start_mission)
            else:
                self.get_logger().warn('LEFT OFFBOARD mode')

    def publish_setpoints(self):
        """Continuously publish setpoints (required by PX4)"""
        if not self.publishing:
            return
            
        # Always publish offboard control mode
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(time.time() * 1e6)
        offboard_msg.position = True
        self.offboard_pub.publish(offboard_msg)
        
        # Publish trajectory setpoint
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(time.time() * 1e6)
        
        if self.offboard and self.idx < len(self.waypoints):
            # During mission, publish target waypoint
            wp = self.waypoints[self.idx]
            traj_msg.position = [float(wp[0]), float(wp[1]), float(wp[2])]
            traj_msg.yaw = float(wp[3])
        else:
            # Before mission or after completion, hold position
            traj_msg.position = [0.0, 0.0, -2.0]
            traj_msg.yaw = 0.0
            
        self.traj_pub.publish(traj_msg)

    def start_arming_sequence(self):
        """Start the arming sequence"""
        self.get_logger().info('Starting arming sequence...')
        self.arm_vehicle()

    def arm_vehicle(self):
        """Send arm command"""
        cmd = VehicleCommand()
        cmd.timestamp = int(time.time() * 1e6)
        cmd.command = 400  # MAV_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = 1.0   # 1 = ARM
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.from_external = True
        self.cmd_pub.publish(cmd)
        self.get_logger().info('Sent ARM command')
        
        # Try to switch to offboard 1 second after arming
        self.create_timer(1.0, self.switch_to_offboard)

    def switch_to_offboard(self):
        """Switch to OFFBOARD mode"""
        if self.armed and not self.offboard:
            cmd = VehicleCommand()
            cmd.timestamp = int(time.time() * 1e6)
            cmd.command = 176  # MAV_CMD_DO_SET_MODE
            cmd.param1 = 1.0   # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            cmd.param7 = 6.0   # PX4_CUSTOM_MAIN_MODE_OFFBOARD
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.from_external = True
            self.cmd_pub.publish(cmd)
            self.get_logger().info('Sent OFFBOARD mode command')

    def start_mission(self):
        """Start the mission"""
        self.get_logger().info('=== MISSION STARTING ===')
        self.get_logger().info(f'First waypoint: {self.waypoints[0]}')

    def control_loop(self):
        """Control mission execution"""
        # Only execute mission if armed and in offboard mode
        if not self.armed or not self.offboard:
            return
            
        # Check if mission complete
        if self.idx >= len(self.waypoints):
            return
            
        # Get current waypoint
        wp = self.waypoints[self.idx]
        
        # Check if reached waypoint
        if len(self.current_pos) == 3:
            dx = self.current_pos[0] - wp[0]
            dy = self.current_pos[1] - wp[1]
            dz = self.current_pos[2] - wp[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist < 0.8:  # 0.8m tolerance
                self.get_logger().info(f'✓ Reached waypoint {self.idx}: {wp}')
                self.idx += 1
                
                if self.idx < len(self.waypoints):
                    next_wp = self.waypoints[self.idx]
                    self.get_logger().info(f'Next: Waypoint {self.idx} -> {next_wp}')
                else:
                    self.get_logger().info('✓ MISSION COMPLETE!')

def main(args=None):
    rclpy.init(args=args)
    flyer = FlyNow()
    
    try:
        rclpy.spin(flyer)
    except KeyboardInterrupt:
        flyer.get_logger().info('Mission stopped by user')
    finally:
        flyer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()