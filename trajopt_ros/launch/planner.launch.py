#!/usr/bin/env python3
"""
The planner, for either robot.

    ros2 launch trajopt_ros planner.launch.py profile:=aerial
    ros2 launch trajopt_ros planner.launch.py profile:=legged
    ros2 launch trajopt_ros planner.launch.py profile:=g1

This file is the ONLY place in the running system that knows what robot is being
driven, and it knows it in exactly two ways:

  * which configuration deltas to merge on top of the shared base
    (`config/legged_overrides.yaml`, `config/g1_overrides.yaml`, each ~25-70
    lines);
  * which platform topics the relative names map onto.

The nodes themselves contain no platform knowledge at all — swapping the profile
swaps the model, the limits and the wiring, and nothing is recompiled.  The G1
profile in particular reuses the SAME node code, the SAME kinematic model and
the SAME sigmoid obstacle barrier as the legged (Go2) profile — see
`config/g1_overrides.yaml` for the (small) numeric deltas and for what was
deliberately left out to keep the three platforms comparable on the same
architecture.

Topic map
---------
                     aerial                     legged                     g1
    pose             /skydio/pose               /go2/pose                  /robot_pose
    scan             /skydio/scan3d             /lidar/points_filtered     /scan
    next_setpoint    /goal_pose                 /mpc/next_setpoint         /mpc/next_setpoint
                     (cascaded PID)             (setpoint_to_cmd_vel_node) (setpoint_to_cmd_vel_node)

`scan` MUST be a PointCloud2 already expressed in the planning frame, for every
profile alike — the aerial platform gets this from `skydio_sim_node`, the Go2
from `cloud_self_filter`.  On the G1 the upstream LiDAR pipeline publishes a
LaserScan in a sensor frame; producing a world-frame PointCloud2 from it is
platform-side bridging (analogous to `cloud_self_filter`) that belongs to the
G1's own bringup, not to this package — see `trajopt_ros/README.md`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
from pathlib import Path


#: Everything platform-specific, in one table.
PROFILES = {
    "aerial": {
        "overrides": "",                       # the base file IS the aerial profile
        "frame_id": "world",
        "mpc_rate_hz": 20.0,
        "remap": {
            "pose": "/skydio/pose",
            "scan": "/skydio/scan3d",
            "global_goal": "/global_goal",
            "path": "/a_star/path",
            "local_goal": "/a_star/local_goal",
            "occupancy_grid": "/a_star/occupancy_grid",
            "predicted_path": "/mpc/predicted_path",
            "next_setpoint": "/goal_pose",     # drives the cascaded PID directly
            "diagnostics": "/mpc/diagnostics",
        },
    },
    "legged": {
        "overrides": "legged_overrides.yaml",
        "frame_id": "odom",
        "mpc_rate_hz": 10.0,
        "remap": {
            "pose": "/go2/pose",
            "scan": "/lidar/points_filtered",
            "global_goal": "/global_goal",
            "path": "/a_star/path",
            "local_goal": "/a_star/local_goal",
            "occupancy_grid": "/a_star/occupancy_grid",
            "predicted_path": "/mpc/predicted_path",
            "next_setpoint": "/mpc/next_setpoint",   # consumed by the cmd_vel node
            "diagnostics": "/mpc/diagnostics",
        },
    },
    "g1": {
        "overrides": "g1_overrides.yaml",
        "frame_id": "map",
        "mpc_rate_hz": 10.0,
        "remap": {
            "pose": "/robot_pose",
            # Must already be a world/map-frame PointCloud2 — see the module
            # docstring.  Name matches the Unitree-G1 project's own topic so an
            # existing bringup needs no republishing, only this remap.
            "scan": "/scan",
            "global_goal": "/global_goal",
            "path": "/a_star/path",
            "local_goal": "/a_star/local_goal",
            "occupancy_grid": "/a_star/occupancy_grid",
            "predicted_path": "/mpc/predicted_path",
            "next_setpoint": "/mpc/next_setpoint",   # consumed by the cmd_vel node
            "diagnostics": "/mpc/diagnostics",
        },
    },
}


def _launch_setup(context, *args, **kwargs):
    profile = LaunchConfiguration("profile").perform(context)
    if profile not in PROFILES:
        raise RuntimeError(
            f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}"
        )
    spec = PROFILES[profile]

    config_dir = Path(get_package_share_directory("trajopt_core")) / "config"
    config_file = LaunchConfiguration("config_file").perform(context)
    config_file = config_file or str(config_dir / "planner_params.yaml")

    overrides = LaunchConfiguration("overrides_file").perform(context)
    if not overrides and spec["overrides"]:
        overrides = str(config_dir / spec["overrides"])

    common = {
        "config_file": config_file,
        "overrides_file": overrides,
        "frame_id": spec["frame_id"],
    }
    remap = list(spec["remap"].items())

    return [
        Node(
            package="trajopt_ros",
            executable="a_star_node",
            name="a_star_node",
            output="screen",
            parameters=[common],
            remappings=remap,
        ),
        Node(
            package="trajopt_ros",
            executable="mpc_node",
            name="mpc_node",
            output="screen",
            parameters=[dict(common, rate_hz=spec["mpc_rate_hz"])],
            remappings=remap,
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile",
                default_value="aerial",
                description="aerial | legged | g1 — selects model, limits and topic wiring",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value="",
                description="override the shared base configuration file",
            ),
            DeclareLaunchArgument(
                "overrides_file",
                default_value="",
                description="override the profile delta file",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
