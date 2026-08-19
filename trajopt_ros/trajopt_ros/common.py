"""
Shared plumbing for the trajopt_ros nodes.

Both nodes take a single `config_file` (plus an optional `overrides_file`) and
build their entire configuration through `trajopt_core.config_io.load_planner`.
Nothing about the planner is configured through individual ROS parameters, which
is what stops the two platform workspaces from drifting apart in their tuning
schema: there is one schema, and it lives in the core.

Topic names are RELATIVE (`pose`, `scan`, `path`, ...).  Every platform-specific
name — `/skydio/pose` or `/go2/pose`, `/skydio/scan3d` or
`/lidar/points_filtered` — is applied by remapping in the launch file.  That is
the ROS-idiomatic way to express "the same node, two robots", and it keeps the
node source free of any mention of either platform.
"""

from __future__ import annotations

import math

import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs_py import point_cloud2

from trajopt_core.config_io import load_planner


def sensor_qos(depth: int = 1) -> QoSProfile:
    """Best-effort, keep-last: sensor streams may drop, and stale data is useless."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Return (x, y, z, w) for a rotation about z."""
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def setup_from_parameters(node) -> object:
    """
    Declare and read the two configuration parameters, then build the planner.

    Raises if `config_file` is empty rather than silently falling back to
    defaults: a planner running on unintended settings is worse than one that
    refuses to start.
    """
    node.declare_parameter("config_file", "")
    node.declare_parameter("overrides_file", "")

    config = node.get_parameter("config_file").value
    overrides = node.get_parameter("overrides_file").value or None

    if not config:
        raise RuntimeError(
            "parameter 'config_file' is required (path to planner_params.yaml)"
        )

    setup = load_planner(config, overrides)
    node.get_logger().info(
        f"config: {config}" + (f" + {overrides}" if overrides else "")
    )
    node.get_logger().info(
        f"model: {setup.model.name}  nx={setup.model.NX} nu={setup.model.NU} "
        f"relative_degree={setup.model.RELATIVE_DEGREE}  "
        f"N={setup.cfg.N} dt={setup.cfg.dt}s"
    )
    return setup


def cloud_to_xy(msg, robot_xy: np.ndarray | None, max_range: float) -> np.ndarray | None:
    """
    PointCloud2 (world frame) -> (M, 2) array of planar hits within `max_range`.

    Everything downstream is planar, so the height is dropped here rather than
    being carried through the stack.  Returns None when nothing is in range, which
    the obstacle terms treat as "no obstacles" rather than as an error.
    """
    pts = [
        [p[0], p[1]]
        for p in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
    ]
    if not pts:
        return None

    arr = np.asarray(pts, dtype=float)
    if robot_xy is not None and max_range > 0.0:
        d = np.hypot(arr[:, 0] - robot_xy[0], arr[:, 1] - robot_xy[1])
        arr = arr[d <= max_range]
    return arr if len(arr) else None
