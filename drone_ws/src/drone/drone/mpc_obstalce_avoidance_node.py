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
        self.max_obs_range = float(self.declare_parameter("max_obs_range", 5.0).value)
        self.m_planes = int(self.declare_parameter("m_planes", 8).value)
        self.wall_detection_enabled = bool(
            self.declare_parameter("wall_detection_enabled", False).value
        )
        self.wall_segment_max_gap = float(
            self.declare_parameter("wall_segment_max_gap", 0.5).value
        )
        self.wall_min_points = int(
            self.declare_parameter("wall_min_points", 5).value
        )

        self.final_goal = np.array(
            self.declare_parameter("final_goal", [10.0, 10.0, 1.5]).value,
            dtype=float,
        )
        self.goal = np.array(
            self.declare_parameter("initial_goal", [10.0, 10.0, 1.5]).value,
            dtype=float,
        )
        self.goal_reached_threshold = float(
            self.declare_parameter("goal_reached_threshold", 1.5).value
        )
        self.goal_step = float(self.declare_parameter("goal_step", 5.0).value)
        
        # Gap-based navigation parameters
        self.gap_nav_enabled = bool(
            self.declare_parameter("gap_nav_enabled", False).value
        )
        self.gap_min_width = float(
            self.declare_parameter("gap_min_width", 1.0).value  # Minimum gap width in meters
        )
        self.gap_goal_distance = float(
            self.declare_parameter("gap_goal_distance", 3.0).value  # How far to place intermediate goal
        )
        self.gap_alignment_weight = float(
            self.declare_parameter("gap_alignment_weight", 0.85).value  # Weight for goal alignment vs gap size (higher = prefer goal direction)
        )
        self.direct_path_threshold = float(
            self.declare_parameter("direct_path_threshold", 3.0).value  # Min clear distance to use direct path
        )
        self.gap_hysteresis = float(
            self.declare_parameter("gap_hysteresis", 1.0).value  # Min change in intermediate goal to switch
        )

        self.ref_speed = float(self.declare_parameter("ref_speed", 1.5).value)

        self.use_prev_solution_ref = bool(
            self.declare_parameter("use_prev_solution_ref", True).value
        )
        self.prev_ref_max_start_error = float(
            self.declare_parameter("prev_ref_max_start_error", 0.5).value
        )
        self.prev_ref_max_step_error = float(
            self.declare_parameter("prev_ref_max_step_error", 0.8).value
        )
        self.cluster_size = float(
            self.declare_parameter("obstacle_cluster_size", 0.25).value
        )
        self.cluster_min_points = int(
            self.declare_parameter("obstacle_cluster_min_points", 1).value
        )
        self.require_mpc_enable = bool(
            self.declare_parameter("require_mpc_enable", False).value
        )
        self.enable_topic = self.declare_parameter("enable_topic", "/mpc/enable").value

        self.traj_frame = self.declare_parameter("trajectory_frame", "map").value
        self.mpc_log_every_n = max(
            1, int(self.declare_parameter("mpc_log_every_n", 5).value)
        )
        self.yaw_rate_limit = float(
            self.declare_parameter("yaw_rate_limit", 0.5).value  # Max yaw change per second (rad/s)
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
        self.last_yaw_ref = None  # For yaw smoothing
        self.last_velocity_frame = None
        self.prev_X = None
        self.loop_counter = 0
        self._last_frame_warn = 0.0
        self.mpc_enabled = not self.require_mpc_enable
        self._last_enable_log = 0.0
        self.intermediate_goal = None  # Gap-based intermediate goal
        self._last_gap_log = 0.0

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
        self.yaw0 = math.atan2(math.sin(self.yaw0), math.cos(self.yaw0))  # Wrap to [-pi, pi]

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

    def detect_wall_segments(self, points: np.ndarray) -> list:
        """Detect wall segments from ordered LiDAR points.
        
        Returns list of wall segments, each as (start_point, end_point, normal).
        Walls are detected as sequences of consecutive points that are roughly collinear.
        """
        if points.shape[0] < self.wall_min_points:
            return []

        segments = []
        current_segment = [points[0]]

        for i in range(1, points.shape[0]):
            dist = np.linalg.norm(points[i] - points[i - 1])
            
            if dist <= self.wall_segment_max_gap:
                current_segment.append(points[i])
            else:
                # End current segment and check if it's a wall
                if len(current_segment) >= self.wall_min_points:
                    seg_points = np.array(current_segment)
                    segment_info = self._fit_wall_segment(seg_points)
                    if segment_info is not None:
                        segments.append(segment_info)
                current_segment = [points[i]]

        # Process last segment
        if len(current_segment) >= self.wall_min_points:
            seg_points = np.array(current_segment)
            segment_info = self._fit_wall_segment(seg_points)
            if segment_info is not None:
                segments.append(segment_info)

        return segments

    def _fit_wall_segment(self, points: np.ndarray):
        """Fit a line to points and return segment info if it's wall-like.
        
        Returns (start, end, normal, sample_points) or None if not a valid wall.
        """
        if points.shape[0] < 2:
            return None

        # Use PCA to find principal direction (wall direction)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Only use x,y for 2D wall detection (assuming z is height)
        centered_2d = centered[:, :2]
        
        if centered_2d.shape[0] < 2:
            return None
            
        # Compute covariance and eigenvectors
        cov = np.cov(centered_2d.T)
        if cov.ndim < 2:
            return None
            
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None

        # Check linearity: ratio of eigenvalues
        # A wall should have one dominant direction
        if eigenvalues[1] < 1e-6:
            return None
        linearity = eigenvalues[0] / eigenvalues[1]
        
        # If ratio is small, points are roughly collinear (wall-like)
        if linearity > 0.3:  # Not linear enough to be a wall
            return None

        # Wall direction is the eigenvector with larger eigenvalue
        wall_dir_2d = eigenvectors[:, 1]
        wall_dir = np.array([wall_dir_2d[0], wall_dir_2d[1], 0.0])
        
        # Normal is perpendicular to wall direction (pointing toward drone)
        normal_2d = np.array([-wall_dir_2d[1], wall_dir_2d[0]])
        
        # Ensure normal points toward the origin (drone position)
        if np.dot(normal_2d, -centroid[:2]) < 0:
            normal_2d = -normal_2d
        normal = np.array([normal_2d[0], normal_2d[1], 0.0])

        # Project points onto wall direction to find extent
        projections = centered_2d @ wall_dir_2d
        min_proj, max_proj = np.min(projections), np.max(projections)
        
        start_2d = centroid[:2] + min_proj * wall_dir_2d
        end_2d = centroid[:2] + max_proj * wall_dir_2d
        
        avg_z = np.mean(points[:, 2])
        start = np.array([start_2d[0], start_2d[1], avg_z])
        end = np.array([end_2d[0], end_2d[1], avg_z])

        # Sample points along the wall for constraint generation
        wall_length = np.linalg.norm(end - start)
        n_samples = max(3, int(wall_length / 0.5))  # Sample every 0.5m
        sample_points = []
        for t in np.linspace(0, 1, n_samples):
            sample_points.append((1 - t) * start + t * end)

        return (start, end, normal, sample_points)

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

    def smooth_yaw_reference(self, yaw_ref: np.ndarray) -> np.ndarray:
        """Smooth yaw reference to prevent rapid heading changes.
        
        Limits the rate of change of yaw reference based on yaw_rate_limit.
        """
        if yaw_ref is None or yaw_ref.size == 0:
            return yaw_ref
        
        yaw_ref = yaw_ref.copy()
        
        # Use current yaw as starting point for smoothing
        if self.yaw0 is not None:
            prev_yaw = self.yaw0
        elif self.last_yaw_ref is not None:
            prev_yaw = self.last_yaw_ref
        else:
            prev_yaw = yaw_ref[0]
        
        max_delta = self.yaw_rate_limit * self.dt  # Max change per step
        
        for k in range(len(yaw_ref)):
            # Compute angle difference (handle wrap-around)
            delta = yaw_ref[k] - prev_yaw
            # Normalize to [-pi, pi]
            while delta > math.pi:
                delta -= 2 * math.pi
            while delta < -math.pi:
                delta += 2 * math.pi
            
            # Limit the rate of change
            if abs(delta) > max_delta:
                delta = math.copysign(max_delta, delta)
            
            yaw_ref[k] = prev_yaw + delta
            # Wrap to [-pi, pi]
            yaw_ref[k] = math.atan2(math.sin(yaw_ref[k]), math.cos(yaw_ref[k]))
            prev_yaw = yaw_ref[k]
        
        # Store last reference for next iteration
        self.last_yaw_ref = yaw_ref[-1]
        
        return yaw_ref

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

    def find_gaps_in_scan(self, scan: LaserScan) -> list:
        """Find gaps (free corridors) in the LiDAR scan.
        
        Returns list of gaps, each as (start_angle, end_angle, center_angle, min_range, gap_width_angle).
        Gaps are regions where ranges exceed max_obs_range or are invalid (free space).
        """
        ranges = np.array(scan.ranges, dtype=float)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        
        # Mark free directions: range > threshold or invalid (no obstacle)
        free_threshold = self.max_obs_range * 0.9
        is_free = (~np.isfinite(ranges)) | (ranges > free_threshold) | (ranges < scan.range_min)
        
        gaps = []
        in_gap = False
        gap_start_idx = 0
        
        for i in range(len(is_free)):
            if is_free[i] and not in_gap:
                # Start of a gap
                in_gap = True
                gap_start_idx = i
            elif not is_free[i] and in_gap:
                # End of a gap
                in_gap = False
                gap_end_idx = i - 1
                self._process_gap(gaps, angles, ranges, gap_start_idx, gap_end_idx, scan)
        
        # Handle gap that extends to end of scan
        if in_gap:
            gap_end_idx = len(is_free) - 1
            self._process_gap(gaps, angles, ranges, gap_start_idx, gap_end_idx, scan)
        
        # Also check for wrap-around gap (end connects to beginning)
        if len(gaps) >= 2:
            first_gap = gaps[0]
            last_gap = gaps[-1]
            # If first gap starts at beginning and last gap ends at end, they might be connected
            if abs(angles[0] - first_gap[0]) < scan.angle_increment * 2:
                if abs(angles[-1] - last_gap[1]) < scan.angle_increment * 2:
                    # Merge the gaps
                    merged_start = last_gap[0]
                    merged_end = first_gap[1]
                    merged_center = (merged_start + merged_end + 2 * np.pi) / 2
                    if merged_center > np.pi:
                        merged_center -= 2 * np.pi
                    merged_width = (first_gap[1] - first_gap[0]) + (last_gap[1] - last_gap[0])
                    merged_range = min(first_gap[3], last_gap[3])
                    gaps = gaps[1:-1]  # Remove first and last
                    gaps.append((merged_start, merged_end, merged_center, merged_range, merged_width))
        
        return gaps

    def _process_gap(self, gaps, angles, ranges, start_idx, end_idx, scan):
        """Process a gap and add it to the list if it's wide enough."""
        if end_idx <= start_idx:
            return
            
        gap_start_angle = angles[start_idx]
        gap_end_angle = angles[end_idx]
        gap_width_angle = gap_end_angle - gap_start_angle
        
        # Estimate gap width in meters at max range
        gap_width_meters = gap_width_angle * self.max_obs_range
        
        if gap_width_meters >= self.gap_min_width:
            center_angle = (gap_start_angle + gap_end_angle) / 2
            
            # Find the minimum range within the gap (how far we can go)
            gap_ranges = ranges[start_idx:end_idx+1]
            valid_ranges = gap_ranges[np.isfinite(gap_ranges) & (gap_ranges > scan.range_min)]
            if len(valid_ranges) > 0:
                min_range = np.min(valid_ranges)
            else:
                min_range = self.max_obs_range  # All free
            
            gaps.append((gap_start_angle, gap_end_angle, center_angle, min_range, gap_width_angle))

    def compute_intermediate_goal(self, scan: LaserScan, final_goal: np.ndarray) -> np.ndarray:
        """Compute intermediate goal through the best gap toward final goal.
        
        Returns intermediate goal position in world frame, or final_goal if direct path is clear.
        """
        if self.p0 is None or self.R0 is None:
            return final_goal
        
        # Direction to final goal in world frame
        goal_dir_world = final_goal[:2] - self.p0[:2]
        goal_dist = np.linalg.norm(goal_dir_world)
        if goal_dist < 0.1:
            return final_goal
        goal_dir_world = goal_dir_world / goal_dist
        
        # Convert goal direction to body frame angle
        # Body x-axis in world frame is R0[:2, 0]
        body_x_world = self.R0[:2, 0]
        body_y_world = self.R0[:2, 1]
        goal_body_x = np.dot(goal_dir_world, body_x_world)
        goal_body_y = np.dot(goal_dir_world, body_y_world)
        goal_angle_body = np.arctan2(goal_body_y, goal_body_x)
        
        # Check if direct path to goal is clear
        ranges = np.array(scan.ranges, dtype=float)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        
        # Find range in goal direction
        goal_idx = np.argmin(np.abs(angles - goal_angle_body))
        direct_range = ranges[goal_idx] if np.isfinite(ranges[goal_idx]) else self.max_obs_range
        
        # Check a cone around goal direction
        cone_half_width = 0.3  # radians (~17 degrees)
        cone_mask = np.abs(angles - goal_angle_body) < cone_half_width
        cone_ranges = ranges[cone_mask]
        cone_ranges = cone_ranges[np.isfinite(cone_ranges)]
        
        if len(cone_ranges) > 0:
            min_cone_range = np.min(cone_ranges)
        else:
            min_cone_range = self.max_obs_range
        
        # If direct path is clear enough, go directly toward goal
        if min_cone_range >= self.direct_path_threshold or min_cone_range >= goal_dist:
            return final_goal
        
        # Find gaps and select best one
        gaps = self.find_gaps_in_scan(scan)
        
        if not gaps:
            # No gaps found, try to go toward goal anyway
            return final_goal
        
        # Score each gap: balance between alignment with goal and gap quality
        best_gap = None
        best_score = -np.inf
        
        for gap in gaps:
            start_angle, end_angle, center_angle, min_range, gap_width = gap
            
            # Alignment score: how well gap center aligns with goal direction
            # Normalize angle difference to [-pi, pi]
            angle_diff = center_angle - goal_angle_body
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # Alignment: 1.0 if perfectly aligned, 0.0 if opposite direction
            alignment = (np.pi - abs(angle_diff)) / np.pi
            
            # Gap quality: wider and farther is better
            gap_quality = (gap_width / np.pi) * (min_range / self.max_obs_range)
            
            # Combined score with strong alignment preference
            score = (self.gap_alignment_weight * alignment + 
                     (1 - self.gap_alignment_weight) * gap_quality)
            
            # Strongly penalize gaps that point away from goal
            # > 60 degrees off: reduce by 50%
            # > 90 degrees off: reduce by 80%
            if abs(angle_diff) > np.pi / 2:  # > 90 degrees
                score *= 0.2
            elif abs(angle_diff) > np.pi / 3:  # > 60 degrees
                score *= 0.5
            
            # Bonus for gaps very well aligned with goal (< 30 degrees)
            if abs(angle_diff) < np.pi / 6:
                score *= 1.3
            
            if score > best_score:
                best_score = score
                best_gap = gap
        
        if best_gap is None:
            return final_goal
        
        # Compute intermediate goal position
        _, _, center_angle, min_range, _ = best_gap
        
        # Distance to place intermediate goal
        goal_dist_intermediate = min(self.gap_goal_distance, min_range * 0.8, goal_dist)
        
        # Convert from body frame angle to world position
        # Direction in body frame
        dir_body = np.array([np.cos(center_angle), np.sin(center_angle), 0.0])
        
        # Transform to world frame
        dir_world = self.R0 @ dir_body
        
        # Intermediate goal position
        intermediate = self.p0 + dir_world * goal_dist_intermediate
        intermediate[2] = final_goal[2]  # Keep target altitude
        
        return intermediate

    def get_active_goal(self, scan: LaserScan) -> np.ndarray:
        """Get the active goal for MPC - either intermediate (gap-based) or final."""
        if not self.gap_nav_enabled or scan is None:
            return self.goal.copy()
        
        # Check if we're close enough to final goal to go direct
        dist_to_final = np.linalg.norm(self.p0[:2] - self.final_goal[:2]) if self.p0 is not None else np.inf
        if dist_to_final < self.gap_goal_distance:
            self.intermediate_goal = None
            return self.goal.copy()
        
        # SPINNING DETECTION: If drone is rotating fast but moving slow, 
        # lock the intermediate goal to prevent oscillation
        if self.yaw_dot0 is not None and self.v0 is not None:
            yaw_rate = abs(self.yaw_dot0)
            speed = np.linalg.norm(self.v0[:2])
            
            # If rotating fast (>0.3 rad/s) and moving slow (<0.5 m/s), don't change goal
            if yaw_rate > 0.3 and speed < 0.5:
                if self.intermediate_goal is not None:
                    # Keep current intermediate goal while spinning
                    return self.intermediate_goal.copy()
                else:
                    # No intermediate goal, go direct to final
                    return self.goal.copy()
        
        # Compute intermediate goal through best gap
        intermediate = self.compute_intermediate_goal(scan, self.goal)
        
        # Check if intermediate is same as current goal (direct path clear)
        if np.linalg.norm(intermediate - self.goal) < 0.1:
            self.intermediate_goal = None
            return self.goal.copy()
        
        # Apply hysteresis: only switch intermediate goal if change is significant
        # This prevents oscillation between nearby gaps
        if self.intermediate_goal is not None:
            change = np.linalg.norm(intermediate[:2] - self.intermediate_goal[:2])
            if change < self.gap_hysteresis:
                # Keep the old intermediate goal
                return self.intermediate_goal.copy()
        
        self.intermediate_goal = intermediate
        return intermediate

    def build_reference_trajectory(self, goal: np.ndarray):
        if (
            self.use_prev_solution_ref
            and self.prev_X is not None
            and self.p0 is not None
        ):
            X_prev = np.asarray(self.prev_X, dtype=float)
            if X_prev.shape[0] >= self.N + 1 and X_prev.shape[1] >= 8:
                start_err = np.linalg.norm(X_prev[0, 0:3] - self.p0)
                step_err = np.linalg.norm(X_prev[1, 0:3] - self.p0)
                if (
                    start_err <= self.prev_ref_max_start_error
                    and step_err <= self.prev_ref_max_step_error
                ):
                    # Shift trajectory forward by 1 step
                    p_ref = X_prev[1 : self.N + 1, 0:3].copy()
                    yaw_ref = X_prev[1 : self.N + 1, 3].copy()
                    
                    # Blend last part of trajectory toward goal
                    blend_steps = min(5, max(1, self.N // 4))  # Blend last 25% toward goal
                    goal_yaw = np.arctan2(goal[1] - self.p0[1], goal[0] - self.p0[0])
                    
                    for i in range(blend_steps):
                        k = self.N - blend_steps + i
                        alpha = (i + 1) / blend_steps  # Linear blend: 0 -> 1
                        p_ref[k] = (1.0 - alpha) * p_ref[k] + alpha * goal
                        yaw_ref[k] = (1.0 - alpha) * yaw_ref[k] + alpha * goal_yaw # Fix the yaw ref computation

                        # Wrap yaw_ref[k] to [-pi, pi] -> to be tested  
                        yaw_ref[k] = math.atan2(math.sin(yaw_ref[k]), math.cos(yaw_ref[k]))
                    
                    if self.loop_counter % (self.mpc_log_every_n * 2) == 0:
                        ref_start_err = np.linalg.norm(p_ref[0] - self.p0)
                        ref_end_err = np.linalg.norm(p_ref[-1] - goal)
                        self.get_logger().info(
                            f"Using warm-start ref: start_err={ref_start_err:.3f}m "
                            f"end_err={ref_end_err:.3f}m (blended last {blend_steps} steps)"
                        )
                    
                    return p_ref, yaw_ref

        # Straight-line fallback
        if self.loop_counter % (self.mpc_log_every_n * 2) == 0:
            self.get_logger().info("Using straight-line reference trajectory")
        start = self.p0.copy()
        direction = goal - start
        dist_to_goal = np.linalg.norm(direction)

        if dist_to_goal < 0.1:
            p_ref = np.tile(goal.reshape(1, 3), (self.N, 1))
            yaw_ref = np.full(self.N, self.yaw0)
            return p_ref, yaw_ref

        direction_unit = direction / dist_to_goal
        v_desired = max(abs(self.ref_speed), 0.1) # m/s, constant desired speed
        v_vec = direction_unit * v_desired

        p_ref = np.zeros((self.N, 3))
        yaw_ref = np.zeros(self.N)
        yaw_goal = np.arctan2(direction[1], direction[0])

        for k in range(self.N):
            t = (k + 1) * self.dt
            pos = start + v_vec * t
            if np.linalg.norm(pos - start) >= dist_to_goal:
                p_ref[k:, :] = goal
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
        
        # Use gap-based navigation to find intermediate goal through free corridors
        active_goal = self.get_active_goal(self.last_scan)
        
        # Log gap navigation status periodically
        if self.loop_counter % self.mpc_log_every_n == 0:
            now = time.time()
            if self.intermediate_goal is not None and now - self._last_gap_log > 1.0:
                int_dist = np.linalg.norm(self.intermediate_goal - self.p0)
                goal_dist = np.linalg.norm(self.goal - self.p0)
                self.get_logger().info(
                    f"Gap-nav: intermediate goal at {int_dist:.2f}m "
                    f"(final goal at {goal_dist:.2f}m)"
                )
                self._last_gap_log = now

        if self.last_scan is None:
            obstacles_world = np.zeros((0, 3))
            wall_segments = []
            raw_obs = np.zeros((0, 3))
        else:
            raw_obs = self.scan_to_world_points(self.last_scan)
            obstacles_world = self.cluster_obstacles(raw_obs)
            
            # Detect walls for better constraint coverage
            if self.wall_detection_enabled and raw_obs.shape[0] >= self.wall_min_points:
                wall_segments = self.detect_wall_segments(raw_obs)
                # Add wall sample points to obstacles for comprehensive coverage
                for _, _, _, sample_pts in wall_segments:
                    wall_pts = np.array(sample_pts)
                    obstacles_world = np.vstack([obstacles_world, wall_pts]) if obstacles_world.size > 0 else wall_pts
            else:
                wall_segments = []

        p_ref_traj, yaw_ref_traj = self.build_reference_trajectory(active_goal)
        yaw_ref_traj = self.unwrap_yaw_sequence(yaw_ref_traj)
        yaw_ref_traj = self.smooth_yaw_reference(yaw_ref_traj)  # Prevent rapid yaw changes

        if self.loop_counter % self.mpc_log_every_n == 0:
            ref_start_dist = np.linalg.norm(p_ref_traj[0] - self.p0)
            ref_end_dist = np.linalg.norm(p_ref_traj[-1] - active_goal)
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
                p_goal=active_goal,
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
            raw_count = int(raw_obs.shape[0])
            obs_count = int(obstacles_world.shape[0])
            wall_count = len(wall_segments)
            plane_count = sum(A_k.shape[0] for A_k, _ in halfspaces)
            iter_count = stats.get("iter_count")
            status = stats.get("return_status")
            pos_change = np.linalg.norm(X_opt[1, 0:3] - X_opt[0, 0:3])
            vel_first = np.linalg.norm(X_opt[0, 4:7])
            acc_cmd = np.linalg.norm(U_opt[0, 0:3])
            goal_dist = np.linalg.norm(self.p0 - active_goal)
            self.get_logger().info(
                f"MPC: {solve_ms:.1f}ms iter={iter_count} status={status} "
                f"| obs_raw={raw_count} obs_clustered={obs_count} walls={wall_count} planes={plane_count} "
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
