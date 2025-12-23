#!/usr/bin/env python3
# tf_drone_broadcaster.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

from px4_msgs.msg import VehicleOdometry

class DroneTFBroadcaster(Node):
    def __init__(self):
        super().__init__('drone_tf_broadcaster')

        # QoS for PX4 - BEST_EFFORT is correct for PX4 topics
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        
        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribe to PX4 VehicleOdometry
        self.subscription = self.create_subscription(
            VehicleOdometry, 
            '/fmu/out/vehicle_odometry', 
            self.odometry_callback, 
            qos
        )

        # Store drone pose
        self.drone_position = [0.0, 0.0, 0.0]
        self.drone_orientation = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
        
        # Timer to publish TF even if no new messages
        self.timer = self.create_timer(0.01, self.publish_tf)  # 100 Hz
        
        # Counter for logging
        self.callback_count = 0
        
        self.get_logger().info('PX4 Drone TF Broadcaster ready')
    
    def odometry_callback(self, msg):
        # Extract position from PX4 odometry
        # PX4 odometry position is in FRD (Forward, Right, Down) frame
        # We need to convert to ENU (East, North, Up) for ROS
        
        # IMPORTANT: Convert numpy.float32 to regular float
        position_frd = msg.position  # This is a list of numpy.float32
        
        # Convert FRD to ENU and convert to regular float
        self.drone_position = [
            float(position_frd[1]),   # y_frd -> x_enu (East)
            float(position_frd[0]),   # x_frd -> y_enu (North)
            float(-position_frd[2])   # -z_frd -> z_enu (Up) (see if flipping z is needed)
        ]
        
        # Extract quaternion from PX4 odometry
        q_frd = msg.q  # This is a list of numpy.float32 [qx, qy, qz, qw] in FRD
        
        # Convert quaternion from FRD to ENU
        # Using simplified conversion that often works well
        self.drone_orientation = [
            float(q_frd[1]),  # qy_frd -> qx_enu
            float(q_frd[2]),  # qz_frd -> qy_enu
            float(q_frd[0]),  # qx_frd -> qz_enu
            float(q_frd[3])   # qw_frd -> qw_enu
        ]
        
        # Log first few messages for debugging
        self.callback_count += 1
        if self.callback_count <= 3:
            self.get_logger().info(f'Odometry received - Position ENU: {self.drone_position}')
            self.get_logger().info(f'Quaternion ENU: {self.drone_orientation}')
    
    def publish_tf(self):
        # Create transform from world/map to base_link
        t = TransformStamped()
        
        # Use current time for TF stamp
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'  # PX4's global frame
        t.child_frame_id = 'base_link'
        
        # Set translation from drone position (already converted to ENU and float)
        t.transform.translation.x = self.drone_position[0]  # East
        t.transform.translation.y = self.drone_position[1]  # North
        t.transform.translation.z = self.drone_position[2]  # Up
        
        # Set rotation from drone orientation (already converted to ENU and float)
        t.transform.rotation.x = self.drone_orientation[0]
        t.transform.rotation.y = self.drone_orientation[1]
        t.transform.rotation.z = self.drone_orientation[2]
        t.transform.rotation.w = self.drone_orientation[3]
        
        # Publish the transform
        self.tf_broadcaster.sendTransform(t)
        
        # Log periodically (less verbose)
        if not hasattr(self, 'last_log_time'):
            self.last_log_time = self.get_clock().now().nanoseconds / 1e9
        else:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self.last_log_time > 5.0:  # Log every 5 seconds
                self.get_logger().info(f'TF published: Position ({self.drone_position[0]:.3f}, '
                                      f'{self.drone_position[1]:.3f}, {self.drone_position[2]:.3f})')
                self.last_log_time = current_time

def main():
    rclpy.init()
    node = DroneTFBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()