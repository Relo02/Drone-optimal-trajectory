"""
Launch file for the Skydio X2 MuJoCo simulation node.

Usage:
    ros2 launch new_mujoco skydio_sim.launch.py
    ros2 launch new_mujoco skydio_sim.launch.py ref_x:=10.0 ref_z:=2.0
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('new_mujoco')
    params_file = os.path.join(pkg_share, 'config', 'skydio_params.yaml')

    return LaunchDescription([
        # Allow overriding the reference position from the command line
        DeclareLaunchArgument('ref_x',   default_value='20.0',  description='Reference X [m]'),
        DeclareLaunchArgument('ref_y',   default_value='1.0',  description='Reference Y [m]'),
        DeclareLaunchArgument('ref_z',   default_value='1.5',  description='Reference Z [m]'),
        DeclareLaunchArgument('ref_yaw', default_value='0.0',  description='Reference yaw [rad]'),

        Node(
            package='new_mujoco',
            executable='skydio_sim_node',
            name='skydio_sim',
            output='screen',
            parameters=[
                params_file,
                {
                    'ref_x':   LaunchConfiguration('ref_x'),
                    'ref_y':   LaunchConfiguration('ref_y'),
                    'ref_z':   LaunchConfiguration('ref_z'),
                    'ref_yaw': LaunchConfiguration('ref_yaw'),
                },
            ],
        ),

        # A* node
        Node(
            package='new_mujoco',
            executable='a_star_node',
            name='a_star_node',
            output='screen',
            parameters=[params_file],
        ),

        # MPC node
        Node(
            package='new_mujoco',
            executable='mpc_node',
            name='mpc_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
