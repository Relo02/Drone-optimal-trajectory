#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    """Launch `mission_commander` after a short delay to allow MAVROS/PX4 to initialize.

    Usage:
        ros2 launch drone start_mission_commander.launch.py

    The timer delay is set to 8s by default to match the working `drone_control` launch.
    """

    mission_commander = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='drone',
                executable='mission_commander',
                name='mission_commander',
                output='screen',
                parameters=[{
                    'vehicle_name': 'x500',
                    'auto_arm': True,
                    'auto_takeoff': True,
                    'takeoff_altitude': 2.0,
                }]
            )
        ]
    )

    return LaunchDescription([
        mission_commander
    ])
