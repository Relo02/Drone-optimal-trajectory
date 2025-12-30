import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from std_msgs.msg import Float64
from std_msgs.msg import Int32

from px4_msgs.msg import VehicleOdometry

from .mpc_solver import mpc_solve


_NED_TO_ENU = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
_FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


def _quat_to_rot_matrix(qx, qy, qz, qw):
    n = (qx * qx) + (qy * qy) + (qz * qz) + (qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=float)
    s = 2.0 / n
    xx = qx * qx * s
    yy = qy * qy * s
    zz = qz * qz * s
    xy = qx * qy * s
    xz = qx * qz * s
    yz = qy * qz * s
    wx = qw * qx * s
    wy = qw * qy * s
    wz = qw * qz * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def _euler_from_quaternion(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def _ned_to_enu(vec_ned):
    return np.array([vec_ned[1], vec_ned[0], -vec_ned[2]], dtype=float)


class MPCObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__("mpc_obstacle_avoidance")

        # --- Parameters ---
        self.dt = float(self.declare_parameter("dt", 0.2).value)
        self.N = int(self.declare_parameter("N", 20).value)
        self.r_s = float(self.declare_parameter("r_s", 0.5).value)
        self.max_obs_range = float(self.declare_parameter("max_obs_range", 3.5).value)
        self.m_planes = int(self.declare_parameter("m_planes", 3).value)

        self.final_goal = np.array(
            self.declare_parameter("final_goal", [10.0, 2.0, 1.5]).value,
            dtype=float,
        )
        self.goal = np.array(
            self.declare_parameter("initial_goal", [5.0, 1.0, 1.5]).value,
            dtype=float,
        )
        self.goal_reached_threshold = float(
            self.declare_parameter("goal_reached_threshold", 1.5).value
        )
        self.goal_step = float(self.declare_parameter("goal_step", 5.0).value)

        self.use_prev_solution_ref = bool(
            self.declare_parameter("use_prev_solution_ref", False).value
        )
        self.prev_ref_max_start_error = float(
            self.declare_parameter("prev_ref_max_start_error", 0.5).value
        )
        self.prev_ref_max_step_error = float(
            self.declare_parameter("prev_ref_max_step_error", 0.8).value
        )
        self.cluster_size = float(
            self.declare_parameter("obstacle_cluster_size", 0.3).value
        )
        self.cluster_min_points = int(
            self.declare_parameter("obstacle_cluster_min_points", 2).value
        )
        self.require_mpc_enable = bool(
            self.declare_parameter("require_mpc_enable", False).value
        )
        self.enable_topic = self.declare_parameter("enable_topic", "/mpc/enable").value

        self.traj_frame = self.declare_parameter("trajectory_frame", "map").value
        self.mpc_log_every_n = max(
            1, int(self.declare_parameter("mpc_log_every_n", 5).value)
        )

        # --- State ---
        self.p0 = None
        self.v0 = None
        self.R0 = None
        self.yaw0 = None
        self.yaw_dot0 = None
        self.last_scan = None
        self.last_scan_frame = None
        self.last_pose_frame = None
        self.last_velocity_frame = None
        self.prev_X = None
        self.loop_counter = 0
        self._last_frame_warn = 0.0
        self.mpc_enabled = not self.require_mpc_enable
        self._last_enable_log = 0.0

        # --- ROS I/O ---
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.odom_callback,
            qos,
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos,
        )
        self.create_subscription(
            Int32,
            self.enable_topic,
            self.enable_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(Vector3Stamped, "/mpc/acceleration_cmd", 10)
        self.opt_trj_pub = self.create_publisher(Path, "/mpc/optimal_trajectory", 10)
        self.ref_trj_pub = self.create_publisher(Path, "/mpc/reference_trajectory", 10)
        self.cost_pub = self.create_publisher(Float64, "/mpc/cost", 10)

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("MPC Obstacle Avoidance Node started")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def odom_callback(self, msg: VehicleOdometry):
        self.last_pose_frame = int(msg.pose_frame)
        self.last_velocity_frame = int(msg.velocity_frame)

        pos_ned = np.array(msg.position, dtype=float)
        vel_ned = np.array(msg.velocity, dtype=float)
        self.p0 = _ned_to_enu(pos_ned)
        self.v0 = _ned_to_enu(vel_ned)

        q = msg.q
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        R_frd_to_ned = _quat_to_rot_matrix(qx, qy, qz, qw)
        self.R0 = _NED_TO_ENU @ R_frd_to_ned @ _FLU_TO_FRD

        _, _, yaw_ned = _euler_from_quaternion(qx, qy, qz, qw)
        self.yaw0 = (math.pi / 2.0) - yaw_ned # Convert NED yaw to ENU yaw

        try:
            self.yaw_dot0 = -float(msg.angular_velocity[2])
        except Exception:
            self.yaw_dot0 = 0.0

    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_frame = msg.header.frame_id

    def enable_callback(self, msg: Int32):
        enabled = msg.data == 1
        if enabled != self.mpc_enabled:
            self.mpc_enabled = enabled
            self.get_logger().info(f"MPC enable set to {int(self.mpc_enabled)}")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _warn_frame_mismatch(self):
        now = time.time()
        if now - self._last_frame_warn < 5.0:
            return

        warn_msgs = []
        if self.last_pose_frame is not None and self.last_pose_frame != VehicleOdometry.POSE_FRAME_NED:
            warn_msgs.append(f"pose_frame={self.last_pose_frame} (expected NED=1)")
        if self.last_velocity_frame is not None and self.last_velocity_frame != VehicleOdometry.VELOCITY_FRAME_NED:
            warn_msgs.append(f"velocity_frame={self.last_velocity_frame} (expected NED=1)")
        if self.last_scan_frame and self.last_scan_frame not in ("base_link", "link", "lidar", "laser"):
            warn_msgs.append(f"scan_frame='{self.last_scan_frame}' (assumed FLU body frame)")

        if warn_msgs:
            self.get_logger().warn("Frame mismatch warning: " + ", ".join(warn_msgs))
            self._last_frame_warn = now

    def scan_to_world_points(self, scan: LaserScan) -> np.ndarray:
        if self.p0 is None or self.R0 is None:
            return np.zeros((0, 3), dtype=float)

        ranges = np.array(scan.ranges, dtype=float)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        valid = np.isfinite(ranges) & (ranges > scan.range_min)
        ranges = ranges[valid]
        angles = angles[valid]

        mask = ranges < self.max_obs_range
        ranges = ranges[mask]
        angles = angles[mask]

        if ranges.size == 0:
            return np.zeros((0, 3), dtype=float)

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)
        pts_body = np.vstack([x, y, z])

        pts_world = self.p0.reshape(3, 1) + self.R0 @ pts_body
        return pts_world.T

    def cluster_obstacles(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        if self.cluster_size <= 0.0:
            return points

        grid = {}
        for pt in points:
            key = tuple(np.floor(pt / self.cluster_size).astype(int))
            if key not in grid:
                grid[key] = [pt.copy(), 1]
            else:
                grid[key][0] += pt
                grid[key][1] += 1

        centroids = []
        for summed, count in grid.values():
            if count >= self.cluster_min_points:
                centroids.append(summed / count)

        if not centroids:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(centroids)

    def unwrap_yaw_sequence(self, yaw_seq):
        if yaw_seq is None:
            return yaw_seq
        yaw_seq = np.asarray(yaw_seq, dtype=float).copy()
        if yaw_seq.size == 0:
            return yaw_seq

        if self.yaw0 is not None:
            offset = round((self.yaw0 - yaw_seq[0]) / (2.0 * math.pi))
            yaw_seq[0] += offset * 2.0 * math.pi

        for k in range(1, yaw_seq.shape[0]):
            delta = yaw_seq[k] - yaw_seq[k - 1]
            if delta > math.pi:
                yaw_seq[k] -= 2.0 * math.pi
            elif delta < -math.pi:
                yaw_seq[k] += 2.0 * math.pi

        return yaw_seq

    def _maybe_update_goal(self):
        dist_to_goal = np.linalg.norm(self.p0 - self.goal)
        if dist_to_goal >= self.goal_reached_threshold:
            return

        if np.linalg.norm(self.goal - self.final_goal) <= 0.5:
            return

        direction = self.final_goal - self.goal
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return

        step = min(self.goal_step, dist)
        self.goal = self.goal + (direction / dist) * step
        self.prev_X = None
        self.get_logger().info(
            f"Goal reached; new goal: ({self.goal[0]:.1f}, {self.goal[1]:.1f}, {self.goal[2]:.1f})"
        )

    def build_reference_trajectory(self):
        if (
            self.use_prev_solution_ref
            and self.prev_X is not None
            and self.p0 is not None
        ):
            X_prev = np.asarray(self.prev_X, dtype=float)
            if X_prev.shape[0] >= self.N + 1 and X_prev.shape[1] >= 4:
                start_err = np.linalg.norm(X_prev[0, 0:3] - self.p0)
                step_err = np.linalg.norm(X_prev[1, 0:3] - self.p0)
                if (
                    start_err <= self.prev_ref_max_start_error
                    and step_err <= self.prev_ref_max_step_error
                ):
                    p_ref = X_prev[1 : self.N + 1, 0:3].copy()
                    yaw_ref = X_prev[1 : self.N + 1, 3].copy()
                    return p_ref, yaw_ref

        # Straight-line fallback
        start = self.p0.copy()
        direction = self.goal - start
        dist_to_goal = np.linalg.norm(direction)

        if dist_to_goal < 0.1:
            p_ref = np.tile(self.goal.reshape(1, 3), (self.N, 1))
            yaw_ref = np.full(self.N, self.yaw0)
            return p_ref, yaw_ref

        direction_unit = direction / dist_to_goal
        v_desired = 1.5 # m/s, constant desired speed
        v_vec = direction_unit * v_desired

        p_ref = np.zeros((self.N, 3))
        yaw_ref = np.zeros(self.N)
        yaw_goal = np.arctan2(direction[1], direction[0])

        for k in range(self.N):
            t = (k + 1) * self.dt
            pos = start + v_vec * t
            if np.linalg.norm(pos - start) >= dist_to_goal:
                p_ref[k:, :] = self.goal
                yaw_ref[k:] = yaw_goal
                break
            p_ref[k] = pos
            yaw_ref[k] = yaw_goal

        return p_ref, yaw_ref

    def _publish_path(self, pub, pts, yaw_ref=None):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.traj_frame
        poses = []
        for k in range(pts.shape[0]):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(pts[k, 0])
            pose.pose.position.y = float(pts[k, 1])
            pose.pose.position.z = float(pts[k, 2])
            if yaw_ref is not None:
                qx, qy, qz, qw = _quaternion_from_euler(0.0, 0.0, float(yaw_ref[k]))
                pose.pose.orientation.x = qx
                pose.pose.orientation.y = qy
                pose.pose.orientation.z = qz
                pose.pose.orientation.w = qw
            poses.append(pose)
        path_msg.poses = poses
        pub.publish(path_msg)

    # -----------------------------
    # Control loop
    # -----------------------------
    def control_loop(self):
        if not rclpy.ok():
            return

        self.loop_counter += 1
        if self.require_mpc_enable and not self.mpc_enabled:
            now = time.time()
            if now - self._last_enable_log > 2.0:
                self.get_logger().info("Waiting for MPC enable signal")
                self._last_enable_log = now
            return
        if (
            self.p0 is None
            or self.v0 is None
            or self.R0 is None
            or self.yaw0 is None
            or self.yaw_dot0 is None
        ):
            if self.loop_counter % (self.mpc_log_every_n * 5) == 0:
                self.get_logger().info("Waiting for odometry/state to start MPC")
            return

        self._warn_frame_mismatch()
        self._maybe_update_goal()

        x0 = np.hstack([self.p0, self.yaw0, self.v0, self.yaw_dot0])

        if self.last_scan is None:
            obstacles_world = np.zeros((0, 3))
        else:
            raw_obs = self.scan_to_world_points(self.last_scan)
            obstacles_world = self.cluster_obstacles(raw_obs)

        p_ref_traj, yaw_ref_traj = self.build_reference_trajectory()
        yaw_ref_traj = self.unwrap_yaw_sequence(yaw_ref_traj)

        if self.loop_counter % self.mpc_log_every_n == 0:
            ref_start_dist = np.linalg.norm(p_ref_traj[0] - self.p0)
            ref_end_dist = np.linalg.norm(p_ref_traj[-1] - self.goal)
            ref_total_length = np.sum(np.linalg.norm(np.diff(p_ref_traj, axis=0), axis=1))
            self.get_logger().info(
                f"Reference: start_dist={ref_start_dist:.3f}m "
                f"end_dist={ref_end_dist:.3f}m "
                f"total_length={ref_total_length:.3f}m"
            )

        self._publish_path(self.ref_trj_pub, p_ref_traj, yaw_ref_traj)

        try:
            solve_start = time.perf_counter()
            U_opt, X_opt, halfspaces, stats = mpc_solve(
                x0=x0,
                p_goal=self.goal,
                obstacles_world=obstacles_world,
                p_ref_traj=p_ref_traj,
                yaw_ref_traj=yaw_ref_traj,
                dt=self.dt,
                N=self.N,
                r_s=self.r_s,
                m_planes=self.m_planes,
                max_obs_range=self.max_obs_range,
            )
        except Exception as exc:
            self.get_logger().error(f"MPC failed: {exc}")
            return
        finally:
            solve_ms = (time.perf_counter() - solve_start) * 1000.0

        objective = stats.get("objective")
        if objective is not None:
            cost_msg = Float64()
            cost_msg.data = float(objective)
            self.cost_pub.publish(cost_msg)

        if self.loop_counter % self.mpc_log_every_n == 0:
            raw_count = int(raw_obs.shape[0]) if self.last_scan is not None else 0
            obs_count = int(obstacles_world.shape[0])
            plane_count = sum(A_k.shape[0] for A_k, _ in halfspaces)
            iter_count = stats.get("iter_count")
            status = stats.get("return_status")
            pos_change = np.linalg.norm(X_opt[1, 0:3] - X_opt[0, 0:3])
            vel_first = np.linalg.norm(X_opt[0, 4:7])
            acc_cmd = np.linalg.norm(U_opt[0, 0:3])
            goal_dist = np.linalg.norm(self.p0 - self.goal)
            self.get_logger().info(
                f"MPC: {solve_ms:.1f}ms iter={iter_count} status={status} "
                f"| obs_raw={raw_count} obs_clustered={obs_count} planes={plane_count} "
                f"| goal_dist={goal_dist:.2f}m pos_change={pos_change:.3f}m "
                f"vel={vel_first:.2f} acc_cmd={acc_cmd:.2f}"
            )

        self.prev_X = X_opt

        acc = U_opt[0, 0:3]
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = float(acc[0])
        msg.vector.y = float(acc[1])
        msg.vector.z = float(acc[2])
        self.cmd_pub.publish(msg)

        self._publish_path(self.opt_trj_pub, X_opt[:, 0:3], X_opt[:, 3])


def main(args=None):
    rclpy.init(args=args)
    node = MPCObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("MPC obstacle avoidance stopped by user")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
