from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # URDF
    urdf_file = os.path.join(
        get_package_share_directory('drone'),
        'urdf',
        'x500_depth.urdf'
    )

    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_description
        }]
    )

    # RViz config (you create this once from RViz and save as .rviz)
    rviz_config = os.path.join(
        get_package_share_directory('drone'),
        'config',
        'x500_depth_viz.rviz'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([rsp, rviz])
