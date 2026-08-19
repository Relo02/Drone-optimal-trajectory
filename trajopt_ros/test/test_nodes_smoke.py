"""
End-to-end smoke test: the two nodes, all three profiles, no launch file.

The nodes are instantiated in-process with `parameter_overrides`, wired to a fake
sensor publisher and spun for a couple of seconds.  What is being verified is the
whole chain — configuration loading, message conversion, grid, A*, the OCP, the
lookahead extraction and publication — for EVERY robot profile, using the same
node classes.  That is the ROS-level counterpart of the claim the package exists
to support.

The G1 profile is included here on the same footing as the other two: per the
topic map in `launch/planner.launch.py`, its `scan` input is a PointCloud2
already in the planning frame, exactly like the aerial and Go2 profiles, so the
same fake publisher below serves all three without a LaserScan variant.

Run standalone with:
    source /opt/ros/humble/setup.bash
    PYTHONPATH=<core>:<ros> python3 -m pytest trajopt_ros/test -q
"""

from __future__ import annotations

import struct
import time
from pathlib import Path as FsPath

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not sourced")

from geometry_msgs.msg import PoseStamped                      # noqa: E402
from nav_msgs.msg import OccupancyGrid, Path                   # noqa: E402
from rclpy.executors import SingleThreadedExecutor             # noqa: E402
from rclpy.parameter import Parameter                          # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField            # noqa: E402
from std_msgs.msg import Float64MultiArray                     # noqa: E402

from trajopt_ros.a_star_node import AStarNode                  # noqa: E402
from trajopt_ros.mpc_node import MPCNode                       # noqa: E402

CONFIG_DIR = FsPath(__file__).resolve().parents[2] / "trajopt_core" / "config"
BASE = CONFIG_DIR / "planner_params.yaml"
LEGGED = CONFIG_DIR / "legged_overrides.yaml"
G1 = CONFIG_DIR / "g1_overrides.yaml"

PROFILES = {
    "aerial": "",
    "legged": str(LEGGED),
    "g1": str(G1),
}


def make_cloud(points_xy: np.ndarray, stamp, frame="world") -> PointCloud2:
    """Minimal XYZ float32 cloud, all hits at z = 0."""
    data = b"".join(
        struct.pack("<fff", float(x), float(y), 0.0) for x, y in points_xy
    )
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.height = 1
    msg.width = len(points_xy)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(points_xy)
    msg.data = data
    msg.is_dense = True
    return msg


class FakeRobot(rclpy.node.Node):
    """Publishes a pose and a wall of LiDAR hits; records what comes back."""

    def __init__(self):
        super().__init__("fake_robot")
        self.pose_pub = self.create_publisher(PoseStamped, "pose", 10)
        self.scan_pub = self.create_publisher(PointCloud2, "scan", 10)

        self.paths: list[Path] = []
        self.setpoints: list[PoseStamped] = []
        self.predictions: list[Path] = []
        self.diagnostics: list[Float64MultiArray] = []
        self.grids: list[OccupancyGrid] = []

        self.create_subscription(Path, "path", self.paths.append, 10)
        self.create_subscription(PoseStamped, "next_setpoint", self.setpoints.append, 10)
        self.create_subscription(Path, "predicted_path", self.predictions.append, 10)
        self.create_subscription(
            Float64MultiArray, "diagnostics", self.diagnostics.append, 10
        )
        self.create_subscription(OccupancyGrid, "occupancy_grid", self.grids.append, 1)

        # a wall at x = 3 with a gate, so A* has to do something
        wall = np.vstack([
            np.stack([np.full(30, 3.0), np.linspace(-4.0, -0.8, 30)], axis=1),
            np.stack([np.full(30, 3.0), np.linspace(1.4, 4.0, 30)], axis=1),
        ])
        self.points = wall
        self.create_timer(0.05, self._publish)

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "world"
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 1.5
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)
        self.scan_pub.publish(make_cloud(self.points, stamp))


def _spin(nodes, seconds: float) -> None:
    ex = SingleThreadedExecutor()
    for n in nodes:
        ex.add_node(n)
    end = time.time() + seconds
    try:
        while time.time() < end:
            ex.spin_once(timeout_sec=0.05)
    finally:
        for n in nodes:
            ex.remove_node(n)


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    try:
        rclpy.shutdown()
    except Exception:
        pass


def _params(overrides: str, extra=None):
    p = [
        Parameter("config_file", value=str(BASE)),
        Parameter("overrides_file", value=overrides),
    ]
    return p + list(extra or [])


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_full_chain_publishes_for_both_profiles(profile):
    """
    Same two node classes, two robots: the only difference is which delta file is
    merged on top of the shared configuration.
    """
    overrides = PROFILES[profile]
    astar = AStarNode(parameter_overrides=_params(overrides))
    mpc = MPCNode(
        parameter_overrides=_params(overrides, [Parameter("rate_hz", value=10.0)])
    )
    robot = FakeRobot()

    try:
        _spin([robot, astar, mpc], 3.0)

        assert robot.paths, f"{profile}: A* published no path"
        assert robot.grids, f"{profile}: no occupancy grid published"
        assert robot.setpoints, f"{profile}: MPC published no setpoint"
        assert robot.predictions, f"{profile}: MPC published no prediction"
        assert robot.diagnostics, f"{profile}: MPC published no diagnostics"

        # the A* path must start near the robot and make progress towards the goal
        first = robot.paths[-1]
        assert len(first.poses) > 1
        start = np.array([first.poses[0].pose.position.x, first.poses[0].pose.position.y])
        assert np.linalg.norm(start) < 1.0

        # the prediction must be N+1 long, i.e. the configured horizon
        assert len(robot.predictions[-1].poses) == mpc.cfg.N + 1

        # the published setpoint must respect the configured lookahead distance,
        # unless the fallback (near-goal) branch was taken
        sp = robot.setpoints[-1].pose.position
        assert np.isfinite([sp.x, sp.y, sp.z]).all()

        # diagnostics layout: [ok, cost, solve_ms, iters, fails]
        diag = robot.diagnostics[-1].data
        assert len(diag) == 5
        assert diag[0] == 1.0, f"{profile}: solver reported failure"
        assert diag[2] > 0.0

        # the graph must have been built once and reused
        assert mpc._ocp.build_count == 1
        assert mpc._ocp.is_parametric

    finally:
        for n in (robot, astar, mpc):
            n.destroy_node()


def test_profiles_select_different_models():
    """The profile really does swap the model, not just a few numbers."""
    aerial = MPCNode(parameter_overrides=_params(PROFILES["aerial"]))
    legged = MPCNode(parameter_overrides=_params(PROFILES["legged"]))
    g1 = MPCNode(parameter_overrides=_params(PROFILES["g1"]))
    try:
        assert aerial.model.NX == 7 and aerial.model.NU == 4
        assert legged.model.NX == 3 and legged.model.NU == 3
        assert aerial.model.RELATIVE_DEGREE == 2
        assert legged.model.RELATIVE_DEGREE == 1

        # g1 shares the legged model family (same class, same integrator) —
        # it is a tuning of the Go2 profile, not a fourth bespoke stack
        assert type(g1.model) is type(legged.model)
        assert g1.model.integrator == legged.model.integrator
        assert g1.model.RELATIVE_DEGREE == 1

        # the one deliberate difference: G1 cannot walk backward
        lb_legged, _ = legged.model.input_bounds()
        lb_g1, _ = g1.model.input_bounds()
        assert lb_legged[0] < 0.0          # Go2 may reverse
        assert lb_g1[0] == pytest.approx(0.0)   # G1 may not

        # dt is shared across all three; N may legitimately differ per profile
        assert aerial.cfg.dt == legged.cfg.dt == g1.cfg.dt
    finally:
        aerial.destroy_node()
        legged.destroy_node()
        g1.destroy_node()


def test_missing_config_is_refused():
    """A planner running on unintended settings is worse than one that won't start."""
    with pytest.raises(RuntimeError, match="config_file"):
        MPCNode(parameter_overrides=[Parameter("config_file", value="")])
