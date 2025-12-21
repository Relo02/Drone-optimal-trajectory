#!/usr/bin/env python3
# launch_laser_simple.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    # Get the absolute path to the config file
    config_path = os.path.join(os.path.dirname(__file__), 'laser_view.rviz')
    
    return LaunchDescription([
        # Laser bridge
        Node(
            package='drone',
            executable='bridge',
            name='laser_bridge',
            output='screen',
        ),
        
        # TF: map -> base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_map_base',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
        ),
        
        # TF: base_link -> link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_base_link',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'link'],
        ),
        
        # RViz - simpler launch without complex parameters
        ExecuteProcess(
            cmd=['ros2', 'run', 'rviz2', 'rviz2', '-d', config_path],
            output='screen'
        ),
    ])