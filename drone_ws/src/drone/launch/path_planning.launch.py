#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mpc_node = Node(
        package='drone',
        executable='mpc_obstacle_avoidance',
        name='mpc_obstacle_avoidance',
        output='screen',
        parameters=[
            {'require_mpc_enable': True},
            {'enable_topic': '/mpc/enable'},
            {'use_global_path': True},
            {'global_path_topic': '/plan'},
        ],
    )

    mission_node = Node(
        package='drone',
        executable='mpc_mission_commander',
        name='mpc_mission_commander',
        output='screen',
        parameters=[
            {'require_mpc_enable': True},
            {'enable_topic': '/mpc/enable'},
        ],
    )

    hover_node = Node(
        package='drone',
        executable='hover_enable_commander',
        name='hover_enable_commander',
        output='screen',
        parameters=[
            {'enable_topic': '/mpc/enable'},
        ],
    )

    return LaunchDescription([mpc_node, mission_node, hover_node])
