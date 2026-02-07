#!/usr/bin/env python3
"""
Launch file for MuJoCo simulation with A* path planner.

Launches:
- MuJoCo simulation node (publishes drone pose, lidar; subscribes to planned path)
- A* path planner node (subscribes to pose, lidar; publishes planned path)

Data flow:
  mujoco_sim_node --[/drone/pose, /lidar/points, /goal_pose]--> a_star_planner_node
  a_star_planner_node --[/planned_path]--> mujoco_sim_node
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    goal_x_arg = DeclareLaunchArgument(
        'goal_x',
        default_value='5.0',
        description='Goal X position'
    )

    goal_y_arg = DeclareLaunchArgument(
        'goal_y',
        default_value='5.0',
        description='Goal Y position'
    )

    goal_z_arg = DeclareLaunchArgument(
        'goal_z',
        default_value='1.5',
        description='Goal Z position (flight height)'
    )

    grid_resolution_arg = DeclareLaunchArgument(
        'grid_resolution',
        default_value='0.2',
        description='Grid resolution in meters'
    )

    gaussian_std_arg = DeclareLaunchArgument(
        'gaussian_std',
        default_value='0.3',
        description='Standard deviation for Gaussian obstacle representation'
    )

    obstacle_threshold_arg = DeclareLaunchArgument(
        'obstacle_threshold',
        default_value='0.5',
        description='Probability threshold above which a cell is blocked'
    )

    obstacle_cost_weight_arg = DeclareLaunchArgument(
        'obstacle_cost_weight',
        default_value='10.0',
        description='Weight for obstacle probability in path cost'
    )

    replan_rate_arg = DeclareLaunchArgument(
        'replan_rate',
        default_value='2.0',
        description='Replanning frequency in Hz'
    )

    # A* Planner Node
    a_star_planner_node = Node(
        package='mujoco_sim',
        executable='a_star_planner_node',
        name='a_star_planner',
        output='screen',
        parameters=[{
            'grid_resolution': LaunchConfiguration('grid_resolution'),
            'gaussian_std': LaunchConfiguration('gaussian_std'),
            'obstacle_threshold': LaunchConfiguration('obstacle_threshold'),
            'obstacle_cost_weight': LaunchConfiguration('obstacle_cost_weight'),
            'planning_height': LaunchConfiguration('goal_z'),
            'replan_rate': LaunchConfiguration('replan_rate'),
        }],
    )

    # MuJoCo Simulation Node
    mujoco_sim_node = Node(
        package='mujoco_sim',
        executable='mujoco_sim_node',
        name='mujoco_sim',
        output='screen',
        parameters=[{
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'goal_z': LaunchConfiguration('goal_z'),
            'publish_rate': 50.0,
        }],
    )

    return LaunchDescription([
        # Launch arguments
        goal_x_arg,
        goal_y_arg,
        goal_z_arg,
        grid_resolution_arg,
        gaussian_std_arg,
        obstacle_threshold_arg,
        obstacle_cost_weight_arg,
        replan_rate_arg,
        # Nodes
        a_star_planner_node,
        mujoco_sim_node,
    ])
