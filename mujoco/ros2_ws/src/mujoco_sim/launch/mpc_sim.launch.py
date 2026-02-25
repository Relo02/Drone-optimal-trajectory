#!/usr/bin/env python3
"""
Launch file for MuJoCo simulation with MPC local planner.

Launches:
- MuJoCo simulation + MPC node (integrated: sim publishes drone pose & lidar,
  runs CasADi MPC internally, publishes predicted path & diagnostics)
- MPC real-time visualizer (matplotlib plot of Gaussian grid + trajectory)

Data flow (all inside one node):
  LiDAR + DroneState → MPC.plan() → predicted trajectory → TrajectoryTracker → PD controller

Published topics:
  /drone/pose           - PoseStamped   (current drone pose)
  /lidar/points         - PointCloud2   (lidar point cloud)
  /mpc/predicted_path   - Path          (MPC predicted trajectory)
  /mpc/diagnostics      - Float64MultiArray [cost, solve_time_ms, success]
  /mpc/grid_data        - Float32MultiArray (Gaussian grid for visualizer)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # ── Goal position ──
    goal_x_arg = DeclareLaunchArgument(
        'goal_x',
        default_value='10.0',
        description='Goal X position'
    )

    goal_y_arg = DeclareLaunchArgument(
        'goal_y',
        default_value='1.0',
        description='Goal Y position'
    )

    goal_z_arg = DeclareLaunchArgument(
        'goal_z',
        default_value='1.5',
        description='Goal Z position (flight height)'
    )

    # ── MPC + Simulation Node ──
    mpc_sim_node = Node(
        package='mujoco_sim',
        executable='mpc_sim_node',
        name='mpc_sim',
        output='screen',
        parameters=[{
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'goal_z': LaunchConfiguration('goal_z'),
            'publish_rate': 50.0,
        }],
    )

    # ── Real-time MPC Visualizer ──
    mpc_viz_node = Node(
        package='mujoco_sim',
        executable='mpc_viz_node',
        name='mpc_visualizer',
        output='screen',
        parameters=[{
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'update_rate_hz': 5.0,
        }],
    )

    return LaunchDescription([
        # Launch arguments
        goal_x_arg,
        goal_y_arg,
        goal_z_arg,
        # Nodes
        mpc_sim_node,
        mpc_viz_node,
    ])
