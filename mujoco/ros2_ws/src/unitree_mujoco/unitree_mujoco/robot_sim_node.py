"""
Go2 warehouse simulation — the plant the `legged` planner profile drives.

This is to the humanoid what `skydio_sim_node` is to the drone: physics-side
only, no planning of any kind.  It publishes exactly the two inputs the
platform-agnostic nodes consume, under the names the `legged` profile already
remaps to, so `trajopt_ros` needs no change at all:

    pub   /go2/pose     geometry_msgs/PoseStamped   base pose, planning frame
    pub   /lidar/points_filtered sensor_msgs/PointCloud2 LiDAR hits, PLANNING FRAME
    sub   /cmd_vel      geometry_msgs/Twist         body-frame velocity command

Two decisions worth stating, because they are what make this comparable with
the other two platforms rather than a third architecture:

**The cloud is published in the planning frame, not the sensor frame.**  Every
profile in `trajopt_ros` requires that (the drone gets it from its own sim, the
Go2 from `cloud_self_filter`), and it is what lets the planner nodes stay free
of TF.  The rays are cast from a fixed site mounted on the Go2 base and the
hits are returned in world coordinates, so the transform is not an extra step
here — it is the absence of one.

**The base moves kinematically, integrating the commanded body velocity.**  The
Go2's own walking policy is a separate, heavyweight piece of software; it
decides how the feet realise a velocity command, not where the robot should go.
Leaving it out makes the plant

    x_dot = [v_x cos(psi) - v_y sin(psi),  v_x sin(psi) + v_y cos(psi),  omega]

which is *exactly* `KinematicSE2`, the model the `legged` profile optimises over —
relative degree 1, by construction.  So this simulation tests the planner
against its own model, which is the honest scope of a planning experiment, and
the same scope as the Go2's.  A gait that cannot track the commanded velocity is
a locomotion question, and it is not answered here.
"""

from __future__ import annotations

import math
import os
import pathlib

import mujoco
import mujoco.viewer
import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from unitree_mujoco.warehouse_world import build_model, lidar_directions


def default_go2_xml() -> str:
    """
    Locate `mujoco/model/go2/go2.xml` from this file's position in
    the repository, with GO2_MJCF as an override.  No path is hardcoded: one
    developer's home directory is not a portable default.
    """
    override = os.environ.get("GO2_MJCF")
    if override:
        return override
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / "mujoco" / "model" / "go2" / "go2.xml"
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        "go2.xml not found; set GO2_MJCF to its full path "
        "(expected under <repo>/mujoco/model/go2/)"
    )


class Go2SimNode(Node):

    def __init__(self, **kwargs):
        super().__init__("go2_sim", **kwargs)

        self.declare_parameter("go2_xml", "")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("spawn_x", -12.0)
        self.declare_parameter("spawn_y", 0.0)
        self.declare_parameter("spawn_yaw", 0.0)
        self.declare_parameter("base_height", 0.27)
        self.declare_parameter("sim_rate_hz", 100.0)
        self.declare_parameter("scan_rate_hz", 10.0)
        # LiDAR approximation.  8640 rays at 10 Hz is what the lab simulator
        # uses; the planner subsamples anyway, so this only costs ray-casting.
        self.declare_parameter("lidar_h_samples", 360)
        self.declare_parameter("lidar_v_samples", 24)
        self.declare_parameter("lidar_v_min", -0.60)     # rad, ~ -34 deg
        self.declare_parameter("lidar_v_max", 0.9076)    # rad, ~ +52 deg
        self.declare_parameter("lidar_range_min", 0.20)
        self.declare_parameter("lidar_range_max", 40.0)
        self.declare_parameter("lidar_noise_std", 0.01)
        # Admissible command set of the PLANT.  Deliberately wider than the
        # planner's own limits (config/legged_overrides.yaml): the plant must not be
        # the thing that enforces them, or a bug in the optimiser would be
        # invisible here.
        self.declare_parameter("vx_limit", 1.5)
        self.declare_parameter("vy_limit", 1.0)
        self.declare_parameter("wz_limit", 2.0)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("viewer", True)

        xml = self.get_parameter("go2_xml").value or default_go2_xml()
        self.model, self.info = build_model(xml)
        self.data = mujoco.MjData(self.model)

        self.frame_id = self.get_parameter("frame_id").value
        self.x = float(self.get_parameter("spawn_x").value)
        self.y = float(self.get_parameter("spawn_y").value)
        self.yaw = float(self.get_parameter("spawn_yaw").value)
        self.base_z = float(self.get_parameter("base_height").value)
        self.vx = self.vy = self.wz = 0.0

        self.vx_lim = float(self.get_parameter("vx_limit").value)
        self.vy_lim = float(self.get_parameter("vy_limit").value)
        self.wz_lim = float(self.get_parameter("wz_limit").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.range_min = float(self.get_parameter("lidar_range_min").value)
        self.range_max = float(self.get_parameter("lidar_range_max").value)
        self.noise_std = float(self.get_parameter("lidar_noise_std").value)

        self._apply_pose()
        mujoco.mj_forward(self.model, self.data)

        self._dirs = lidar_directions(
            int(self.get_parameter("lidar_h_samples").value),
            int(self.get_parameter("lidar_v_samples").value),
            float(self.get_parameter("lidar_v_min").value),
            float(self.get_parameter("lidar_v_max").value),
        )
        self._nray = self._dirs.shape[0]
        self._groups = np.zeros(6, dtype=np.uint8)
        self._groups[self.info["lidar_group"]] = 1
        self._geomid = np.full(self._nray, -1, dtype=np.int32)
        self._dist = np.zeros(self._nray, dtype=np.float64)

        self.sim_dt = 1.0 / float(self.get_parameter("sim_rate_hz").value)
        self._last_cmd_t = -1e9

        self.pose_pub = self.create_publisher(PoseStamped, "/go2/pose", 10)
        self.scan_pub = self.create_publisher(PointCloud2, "/lidar/points_filtered", 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)

        self.create_timer(self.sim_dt, self._sim_tick)
        self.create_timer(1.0 / float(self.get_parameter("scan_rate_hz").value),
                          self._scan_tick)

        self.viewer = None
        if bool(self.get_parameter("viewer").value):
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # The warehouse lives in the LiDAR geom group, which MuJoCo hides
                # by default — without this the viewer shows an empty floor.
                with self.viewer.lock():
                    self.viewer.opt.geomgroup[self.info["lidar_group"]] = 1
                self.viewer.sync()
            except Exception as exc:                       # no display, headless CI
                self.get_logger().warn(f"viewer unavailable: {exc}")

        self.get_logger().info(
            f"Go2 warehouse sim: kinematic SE(2) base, {self._nray} LiDAR rays -> /lidar/points_filtered "
            f"in '{self.frame_id}', spawn=({self.x:.1f}, {self.y:.1f}, "
            f"{self.yaw:.2f} rad)"
        )

    # ------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _apply_pose(self) -> None:
        a = self.info["free_qpos_adr"]
        self.data.qpos[a:a + 3] = [self.x, self.y, self.base_z]
        self.data.qpos[a + 3:a + 7] = [math.cos(self.yaw / 2.0), 0.0, 0.0,
                                       math.sin(self.yaw / 2.0)]

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.vx = float(np.clip(msg.linear.x, -self.vx_lim, self.vx_lim))
        self.vy = float(np.clip(msg.linear.y, -self.vy_lim, self.vy_lim))
        self.wz = float(np.clip(msg.angular.z, -self.wz_lim, self.wz_lim))
        self._last_cmd_t = self._now()

    def _sim_tick(self) -> None:
        # A command that stopped arriving means the controller died; a robot that
        # keeps walking on the last command it heard is the wrong failure mode.
        if self._now() - self._last_cmd_t > self.cmd_timeout:
            self.vx = self.vy = self.wz = 0.0

        if self.vx or self.vy or self.wz:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.x += (c * self.vx - s * self.vy) * self.sim_dt
            self.y += (s * self.vx + c * self.vy) * self.sim_dt
            self.yaw = math.atan2(math.sin(self.yaw + self.wz * self.sim_dt),
                                  math.cos(self.yaw + self.wz * self.sim_dt))
            self._apply_pose()
            mujoco.mj_forward(self.model, self.data)

        self._publish_pose()
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def _publish_pose(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.position.z = self.base_z
        msg.pose.orientation.z = math.sin(self.yaw / 2.0)
        msg.pose.orientation.w = math.cos(self.yaw / 2.0)
        self.pose_pub.publish(msg)

    # ------------------------------------------------------------------
    def _scan_tick(self) -> None:
        sid = self.info["site_id"]
        origin = self.data.site_xpos[sid].copy()
        rot = self.data.site_xmat[sid].reshape(3, 3)
        world_dirs = (rot @ self._dirs.T).T

        mujoco.mj_multiRay(self.model, self.data, origin, world_dirs.flatten(),
                           self._groups, 1, -1, self._geomid, self._dist, None,
                           self._nray, self.range_max)

        hit = (self._dist > self.range_min) & (self._dist < self.range_max)
        if not np.any(hit):
            return
        d = self._dist[hit]
        if self.noise_std > 0.0:
            d = d + np.random.normal(0.0, self.noise_std, d.shape)

        # World coordinates: the ray origin plus the hit distance along the
        # world-frame direction.  This is the whole reason no TF is needed
        # downstream — see the module docstring.
        pts = (origin[None, :] + world_dirs[hit] * d[:, None]).astype(np.float32)
        self.scan_pub.publish(self._cloud(pts))

    def _cloud(self, pts: np.ndarray) -> PointCloud2:
        msg = PointCloud2()
        msg.header = Header(stamp=self.get_clock().now().to_msg(),
                            frame_id=self.frame_id)
        msg.height = 1
        msg.width = int(pts.shape[0])
        msg.fields = [
            PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
            for n, o in (("x", 0), ("y", 4), ("z", 8))
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * int(pts.shape[0])
        msg.is_dense = True
        msg.data = pts.tobytes()
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Go2SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.viewer is not None:
            try:
                node.viewer.close()
            except Exception:
                pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
