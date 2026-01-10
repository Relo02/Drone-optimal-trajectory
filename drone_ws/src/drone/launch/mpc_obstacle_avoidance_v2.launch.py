"""
Launch file for MPC Obstacle Avoidance v2

This launch file starts the improved MPC obstacle avoidance node
with configurable parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    dt_arg = DeclareLaunchArgument(
        'dt',
        default_value='0.1',
        description='MPC time step in seconds'
    )
    
    horizon_arg = DeclareLaunchArgument(
        'horizon',
        default_value='20',
        description='MPC prediction horizon steps'
    )
    
    safety_radius_arg = DeclareLaunchArgument(
        'safety_radius',
        default_value='0.6',
        description='Minimum distance from obstacles in meters'
    )
    
    gap_safety_margin_arg = DeclareLaunchArgument(
        'gap_safety_margin',
        default_value='0.2',
        description='Extra clearance beyond safety_radius for gap goals'
    )
    
    goal_arg = DeclareLaunchArgument(
        'goal',
        default_value='[10.0, 10.0, 1.5]',
        description='Goal position [x, y, z]'
    )
    
    max_velocity_arg = DeclareLaunchArgument(
        'max_velocity',
        default_value='2.0',
        description='Maximum velocity in m/s'
    )
    
    gap_navigation_arg = DeclareLaunchArgument(
        'gap_navigation_enabled',
        default_value='true',
        description='Enable gap-based navigation'
    )
    
    require_enable_arg = DeclareLaunchArgument(
        'require_enable',
        default_value='false',
        description='Require explicit enable signal'
    )
    
    direct_px4_control_arg = DeclareLaunchArgument(
        'direct_px4_control',
        default_value='true',
        description='Send commands directly to PX4 (no need for mpc_mission_commander)'
    )
    
    auto_arm_arg = DeclareLaunchArgument(
        'auto_arm',
        default_value='true',
        description='Automatically arm and switch to offboard mode'
    )
    
    # MPC Obstacle Avoidance Node
    mpc_node = Node(
        package='drone',
        executable='mpc_obstacle_avoidance_v2',
        name='mpc_obstacle_avoidance',
        output='screen',
        parameters=[{
            'dt': LaunchConfiguration('dt'),
            'horizon': LaunchConfiguration('horizon'),
            'safety_radius': LaunchConfiguration('safety_radius'),
            'emergency_radius': 0.3,
            'max_velocity': LaunchConfiguration('max_velocity'),
            'max_acceleration': 2.5,
            'max_yaw_rate': 1.0,
            'goal': LaunchConfiguration('goal'),
            'goal_threshold': 0.5,
            'max_obstacle_range': 5.0,
            'cluster_size': 0.3,
            'gap_navigation_enabled': LaunchConfiguration('gap_navigation_enabled'),
            'min_gap_width': 1.2,
            'direct_path_threshold': 3.0,
            'gap_safety_margin': LaunchConfiguration('gap_safety_margin'),
            'require_enable': LaunchConfiguration('require_enable'),
            'log_interval': 5,
            'trajectory_frame': 'map',
            # PX4 direct control
            'direct_px4_control': LaunchConfiguration('direct_px4_control'),
            'auto_arm': LaunchConfiguration('auto_arm'),
            'lookahead_index': 3,
            'takeoff_height': 1.0,
        }],
        remappings=[
            ('/fmu/out/vehicle_odometry', '/fmu/out/vehicle_odometry'),
            ('/fmu/out/vehicle_status_v1', '/fmu/out/vehicle_status_v1'),
            ('/fmu/in/vehicle_command', '/fmu/in/vehicle_command'),
            ('/fmu/in/offboard_control_mode', '/fmu/in/offboard_control_mode'),
            ('/fmu/in/trajectory_setpoint', '/fmu/in/trajectory_setpoint'),
            ('/scan', '/scan'),
        ]
    )
    
    return LaunchDescription([
        dt_arg,
        horizon_arg,
        safety_radius_arg,
        gap_safety_margin_arg,
        goal_arg,
        max_velocity_arg,
        gap_navigation_arg,
        require_enable_arg,
        direct_px4_control_arg,
        auto_arm_arg,
        mpc_node,
    ])
