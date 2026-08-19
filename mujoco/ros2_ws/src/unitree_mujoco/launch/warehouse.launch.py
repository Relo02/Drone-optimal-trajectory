#!/usr/bin/env python3
"""
The Go2 navigating the warehouse, end to end.

    ros2 launch go2_mujoco warehouse.launch.py
    ros2 launch go2_mujoco warehouse.launch.py goal_x:=12.5 goal_y:=0.0
    ros2 launch go2_mujoco warehouse.launch.py planner:=false   # sim only

Three nodes, and the middle one is the point: it is not in this package.

    go2_sim_node             plant   MuJoCo: Go2 + warehouse, /go2/pose + /lidar/points_filtered
    trajopt_ros (profile legged) planner A* + MPC, shared with the drone and the Go2
    setpoint_to_cmd_vel      inner   setpoint -> /cmd_vel, this platform's loop

The planner is launched from `trajopt_ros` with `profile:=legged`, unmodified —
the same executables the aerial and Go2 profiles run.  This launch file
contains no planning parameter at all; those live in
`trajopt_core/config/legged_overrides.yaml`.

The goal is published once, a few seconds in, on /global_goal: the A* node
latches the last goal it received, and publishing before it has subscribed would
lose it.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    planner = LaunchConfiguration("planner")

    sim = Node(
        package="go2_mujoco",
        executable="go2_sim_node",
        name="go2_sim",
        output="screen",
        parameters=[{
            "spawn_x": LaunchConfiguration("spawn_x"),
            "spawn_y": LaunchConfiguration("spawn_y"),
            "spawn_yaw": LaunchConfiguration("spawn_yaw"),
            "base_height": LaunchConfiguration("base_height"),
            "viewer": LaunchConfiguration("viewer"),
            "frame_id": "odom",
        }],
    )

    inner_loop = Node(
        package="go2_mujoco",
        executable="setpoint_to_cmd_vel_node",
        name="setpoint_to_cmd_vel",
        output="screen",
        parameters=[{
            "cmd_max_vx": 1.0,
            "cmd_max_vy": 0.5,
            "cmd_max_omega": 1.5,
            "allow_reverse": True,
            "enable_yaw_control": True,
        }],
        condition=IfCondition(planner),
    )

    planner_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("trajopt_ros"), "launch", "planner.launch.py"]
        )),
        launch_arguments={"profile": "legged"}.items(),
        condition=IfCondition(planner),
    )

    send_goal = TimerAction(
        period=4.0,
        actions=[ExecuteProcess(
            cmd=["ros2", "topic", "pub", "--once", "/global_goal",
                 "geometry_msgs/PoseStamped",
                 ["{header: {frame_id: 'map'}, pose: {position: {x: ", goal_x,
                  ", y: ", goal_y, ", z: 0.0}, orientation: {w: 1.0}}}"]],
            output="screen",
        )],
        condition=IfCondition(planner),
    )

    return LaunchDescription([
        DeclareLaunchArgument("spawn_x", default_value="-12.0"),
        DeclareLaunchArgument("spawn_y", default_value="0.0"),
        DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
        DeclareLaunchArgument("goal_x", default_value="12.5",
                      description="goal in the warehouse, odom frame"),
        DeclareLaunchArgument("goal_y", default_value="0.0"),
        DeclareLaunchArgument("base_height", default_value="0.27"),
        DeclareLaunchArgument("viewer", default_value="true",
                              description="MuJoCo passive viewer (needs a display)"),
        DeclareLaunchArgument("planner", default_value="true",
                              description="also start the shared A*/MPC stack"),
        sim,
        planner_stack,
        inner_loop,
        send_goal,
    ])
