#!/usr/bin/env python3
# laser_bridge_fixed.py
import importlib
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import sys


def _import_gz_module(candidates):
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        f"Unable to import Gazebo module; tried: {', '.join(candidates)}"
    )


# Support versioned Gazebo Python bindings (e.g., gz.transport13, gz.msgs10).
gz_transport = _import_gz_module(['gz.transport', 'gz.transport13', 'gz.transport14'])
gz_laserscan_pb2 = _import_gz_module(
    ['gz.msgs.laserscan_pb2', 'gz.msgs10.laserscan_pb2', 'gz.msgs11.laserscan_pb2']
)


def _gz_subscribe(node, topic, msg_type, callback):
    module_name = getattr(gz_transport, '__name__', '')
    if module_name.startswith('gz.transport') and module_name != 'gz.transport':
        return node.subscribe(msg_type, topic, callback)
    try:
        return node.subscribe(msg_type, topic, callback)
    except (TypeError, AttributeError):
        return node.subscribe(topic, callback, msg_type)

class FixedLaserBridge(Node):
    def __init__(self):
        super().__init__('fixed_laser_bridge')
        
        # Publisher for ROS 2
        self.publisher = self.create_publisher(LaserScan, '/lidar/scan', 10)
        
        # Initialize Gazebo transport
        self.gz_node = gz_transport.Node()
        
        # Subscribe to Gazebo topic
        topic_name = '/world/default/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan'
        _gz_subscribe(
            self.gz_node,
            topic_name,
            gz_laserscan_pb2.LaserScan,
            self.gz_callback,
        )
        
        self.get_logger().info(f'Subscribed to Gazebo topic: {topic_name}')
        self.get_logger().info('Fixed laser bridge ready')
    
    def gz_callback(self, gz_msg):
        try:
            # Create ROS message
            ros_msg = LaserScan()   
            
            # Set timestamp
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Extract frame_id (try multiple sources)
            frame_id = 'lidar_link'
            if hasattr(gz_msg, 'frame') and gz_msg.frame:
                frame_id = gz_msg.frame
            elif hasattr(gz_msg, 'header') and hasattr(gz_msg.header, 'data'):
                for data in gz_msg.header.data:
                    if data.key == 'frame_id' and data.value:
                        frame_id = data.value
                        break
            
            ros_msg.header.frame_id = frame_id
            
            # Map the basic fields (these exist in both)
            ros_msg.angle_min = gz_msg.angle_min
            ros_msg.angle_max = gz_msg.angle_max
            ros_msg.range_min = gz_msg.range_min
            ros_msg.range_max = gz_msg.range_max
            
            # Gazebo uses angle_step, ROS uses angle_increment
            if hasattr(gz_msg, 'angle_step'):
                ros_msg.angle_increment = gz_msg.angle_step
            else:
                # Calculate from min/max and count
                ros_msg.angle_increment = (gz_msg.angle_max - gz_msg.angle_min) / (len(gz_msg.ranges) - 1)
            
            # Handle scan timing
            if hasattr(gz_msg, 'count'):
                # Assuming 10Hz scan rate
                scan_rate = 10.0  # Hz
                ros_msg.scan_time = 1.0 / scan_rate
                ros_msg.time_increment = ros_msg.scan_time / gz_msg.count
            
            # Convert ranges
            ranges = []
            for r in gz_msg.ranges:
                # Handle 'inf' values
                if str(r).lower() == 'inf' or r > 1e6:  # Large number indicates inf
                    ranges.append(float('inf'))
                else:
                    ranges.append(float(r))
            
            ros_msg.ranges = ranges
            
            # Handle intensities if available
            if hasattr(gz_msg, 'intensities') and gz_msg.intensities:
                ros_msg.intensities = [float(i) for i in gz_msg.intensities]
            else:
                ros_msg.intensities = []
            
            # Publish the message
            self.publisher.publish(ros_msg)
            
            # Log first successful message
            if not hasattr(self, 'first_msg_logged'):
                self.get_logger().info(f'First message published: {len(ranges)} ranges')
                self.get_logger().info(f'Frame: {frame_id}, Angle: [{gz_msg.angle_min:.3f}, {gz_msg.angle_max:.3f}] rad')
                self.first_msg_logged = True
                
        except Exception as e:
            self.get_logger().error(f'Error in callback: {str(e)}', throttle_duration_sec=1)

def main():
    rclpy.init(args=sys.argv)
    
    node = FixedLaserBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
