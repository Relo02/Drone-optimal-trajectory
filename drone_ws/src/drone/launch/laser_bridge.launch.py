import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('ros_gz_bridge'),
        'config',
        'default_laserscan.yaml'  # Check if this exists
    )
    
    # If default doesn't exist, use our custom command
    return LaunchDescription([
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='laser_bridge',
            arguments=[
                '/world/default/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan'
                '@sensor_msgs/msg/LaserScan'
                '[gz.msgs.LaserScan'
            ],
            output='screen',
            emulate_tty=True,
            parameters=[{'lazy': False}]
        )
    ])