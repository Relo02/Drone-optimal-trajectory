"""
MPC Obstacle Avoidance Node - ROS2 Node for Drone Path Planning

Pure MPC implementation for obstacle avoidance with:
- Proper sensor processing and coordinate frame handling
- Direct goal navigation with MPC optimization
- Obstacle avoidance through cost functions and constraints
- Comprehensive logging and diagnostics

Author: Refactored implementation
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from std_msgs.msg import Float64, Int32

from px4_msgs.msg import (
    VehicleOdometry,
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand,
    VehicleStatus,
)

from .mpc_core import MPCConfig, MPCState, MPCSolver, ObstacleSet, MPCResult


# ============================================================================
# Coordinate Frame Utilities
# ============================================================================

def ned_to_enu(vec_ned: np.ndarray) -> np.ndarray:
    """Convert NED coordinates to ENU."""
    return np.array([vec_ned[1], vec_ned[0], -vec_ned[2]], dtype=float)


def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-12:
        return np.eye(3)
    
    s = 2.0 / n
    xx, yy, zz = qx*qx*s, qy*qy*s, qz*qz*s
    xy, xz, yz = qx*qy*s, qx*qz*s, qy*qz*s
    wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s
    
    return np.array([
        [1-(yy+zz), xy-wz, xz+wy],
        [xy+wz, 1-(xx+zz), yz-wx],
        [xz-wy, yz+wx, 1-(xx+yy)]
    ], dtype=float)


def euler_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Extract Euler angles (roll, pitch, yaw) from quaternion."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(np.clip(sinp, -1.0, 1.0))
    
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Create quaternion from Euler angles."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return qx, qy, qz, qw


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# Frame conversion matrices
_NED_TO_ENU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
_FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


# ============================================================================
# LiDAR Processing
# ============================================================================

@dataclass
class LidarProcessor:
    """Process LiDAR scans into obstacle points."""
    
    max_range: float = 5.0
    cluster_size: float = 0.3
    min_cluster_points: int = 2
    
    def scan_to_world_points(
        self,
        scan: LaserScan,
        position: np.ndarray,
        rotation: np.ndarray
    ) -> np.ndarray:
        """
        Convert LiDAR scan to world-frame obstacle points.
        
        Args:
            scan: LaserScan message
            position: Drone position in world frame
            rotation: 3x3 rotation matrix (body to world)
            
        Returns:
            (N, 3) array of obstacle positions in world frame
        """
        ranges = np.array(scan.ranges, dtype=float)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        
        # Filter valid readings
        valid = (np.isfinite(ranges) & 
                 (ranges > scan.range_min) & 
                 (ranges < min(scan.range_max, self.max_range)))
        
        ranges = ranges[valid]
        angles = angles[valid]
        
        if ranges.size == 0:
            return np.zeros((0, 3))
        
        # Convert to body frame (assuming FLU convention)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)  # 2D LiDAR assumption
        
        pts_body = np.vstack([x, y, z])
        
        # Transform to world frame
        pts_world = position.reshape(3, 1) + rotation @ pts_body
        
        return pts_world.T
    
    def cluster_points(self, points: np.ndarray) -> np.ndarray:
        """
        Cluster nearby points to reduce obstacle count.
        
        Uses grid-based clustering for efficiency.
        """
        if points.size == 0 or self.cluster_size <= 0:
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
            if count >= self.min_cluster_points:
                centroids.append(summed / count)
        
        if not centroids:
            return np.zeros((0, 3))
        
        return np.vstack(centroids)


# ============================================================================
# Main ROS2 Node
# ============================================================================

class MPCObstacleAvoidanceNode(Node):
    """ROS2 node for MPC-based obstacle avoidance."""
    
    def __init__(self):
        super().__init__('mpc_obstacle_avoidance_v2')
        
        # ---- Declare Parameters ----
        self._declare_parameters()
        
        # ---- Initialize Components ----
        self._init_components()
        
        # ---- State Variables ----
        self._init_state()
        
        # ---- ROS Setup ----
        self._setup_ros()
        
        self.get_logger().info('MPC Obstacle Avoidance Node v2 started')
    
    def _declare_parameters(self):
        """Declare all ROS parameters."""
        # MPC parameters (pure MPC mode)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('horizon', 20)  # 2 seconds
        self.declare_parameter('safety_radius', 1.5)
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('max_acceleration', 3.0)
        self.declare_parameter('max_yaw_rate', 1.5)
        
        # MPC cost weights
        self.declare_parameter('Q_pos', 15.0)        # Position tracking (follow reference)
        self.declare_parameter('Q_goal', 80.0)       # Goal attraction
        self.declare_parameter('Q_vel', 1.0)         # Velocity regularization
        self.declare_parameter('R_acc', 0.3)         # Acceleration penalty
        self.declare_parameter('Q_terminal', 150.0)  # Terminal cost
        self.declare_parameter('Q_vel_toward_obs', 50.0)  # Velocity toward obstacles
        
        # Potential field parameters (backup to gap navigation)
        self.declare_parameter('Q_obstacle_repulsion', 300.0)
        self.declare_parameter('potential_influence_dist', 3.0)
        self.declare_parameter('potential_steepness', 2.0)
        
        # Goal parameters
        self.declare_parameter('goal', [10.0, 10.0, 1.5])
        self.declare_parameter('goal_threshold', 0.5)
        
        # Obstacle processing
        self.declare_parameter('max_obstacle_range', 6.0)  # Match obstacle_range
        self.declare_parameter('cluster_size', 0.3)

        # Control
        self.declare_parameter('require_enable', False)
        self.declare_parameter('log_interval', 5)
        
        # Frame
        self.declare_parameter('trajectory_frame', 'world')
        
        # PX4 direct control
        self.declare_parameter('direct_px4_control', True)
        self.declare_parameter('auto_arm', True)
        self.declare_parameter('lookahead_index', 5)  # Look further ahead for smoother flight
        self.declare_parameter('velocity_feedforward', True)  # Send velocity with position
        self.declare_parameter('takeoff_height', 1.0)
    
    def _init_components(self):
        """Initialize MPC and processing components."""
        # Build MPC config from parameters
        self.mpc_config = MPCConfig(
            dt=self.get_parameter('dt').value,
            N=self.get_parameter('horizon').value,
            safety_radius=self.get_parameter('safety_radius').value,
            v_max=self.get_parameter('max_velocity').value,
            a_max=self.get_parameter('max_acceleration').value,
            yaw_rate_max=self.get_parameter('max_yaw_rate').value,
            obstacle_range=self.get_parameter('max_obstacle_range').value,
            # Cost weights
            Q_pos=self.get_parameter('Q_pos').value,
            Q_goal=self.get_parameter('Q_goal').value,
            Q_vel=self.get_parameter('Q_vel').value,
            R_acc=self.get_parameter('R_acc').value,
            Q_terminal=self.get_parameter('Q_terminal').value,
            Q_vel_toward_obs=self.get_parameter('Q_vel_toward_obs').value,
            # Potential field parameters
            Q_obstacle_repulsion=self.get_parameter('Q_obstacle_repulsion').value,
            potential_influence_dist=self.get_parameter('potential_influence_dist').value,
            potential_steepness=self.get_parameter('potential_steepness').value,
        )
        
        # MPC solver
        self.mpc_solver = MPCSolver(self.mpc_config)
        
        # LiDAR processor
        self.lidar_processor = LidarProcessor(
            max_range=self.get_parameter('max_obstacle_range').value,
            cluster_size=self.get_parameter('cluster_size').value,
        )
    
    def _init_state(self):
        """Initialize state variables."""
        self.state: Optional[MPCState] = None
        self.rotation: Optional[np.ndarray] = None
        self.last_scan: Optional[LaserScan] = None
        
        goal_param = self.get_parameter('goal').value
        self.goal = np.array(goal_param, dtype=float)
        
        self.mpc_enabled = not self.get_parameter('require_enable').value
        self.loop_counter = 0
        self.log_interval = self.get_parameter('log_interval').value
        
        self._last_result: Optional[MPCResult] = None
        
        # PX4 state
        self.armed = False
        self.offboard = False
        self.preflight_ok = False
        self.setpoint_counter = 0
        self.last_cmd_time = 0.0
        self.lookahead_idx = self.get_parameter('lookahead_index').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
    
    def _setup_ros(self):
        """Setup ROS subscriptions and publishers."""
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        
        # Subscriptions
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self._odom_callback,
            qos
        )
        
        self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            qos
        )
        
        self.create_subscription(
            Int32,
            '/mpc/enable',
            self._enable_callback,
            10
        )
        
        # PX4 status subscription
        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self._status_callback,
            qos
        )
        
        # MPC diagnostic publishers
        self.cmd_pub = self.create_publisher(Vector3Stamped, '/mpc/acceleration_cmd', 10)
        self.opt_traj_pub = self.create_publisher(Path, '/mpc/optimal_trajectory', 10)
        self.ref_traj_pub = self.create_publisher(Path, '/mpc/reference_trajectory', 10)
        self.cost_pub = self.create_publisher(Float64, '/mpc/cost', 10)
        
        # Goal publishers for visualization
        self.active_goal_pub = self.create_publisher(PoseStamped, '/mpc/active_goal', 10)
        self.intermediate_goal_pub = self.create_publisher(PoseStamped, '/mpc/intermediate_goal', 10)
        self.final_goal_pub = self.create_publisher(PoseStamped, '/mpc/final_goal', 10)
        
        # PX4 control publishers
        self.px4_cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10
        )
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10
        )
        self.traj_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10
        )
        
        # Control loop timer
        dt = self.get_parameter('dt').value
        self.timer = self.create_timer(dt, self._control_loop)
        
        # Command loop timer (for arming/offboard)
        if self.get_parameter('direct_px4_control').value:
            self.cmd_timer = self.create_timer(1.0, self._command_loop)
    
    # ---- Callbacks ----
    
    def _odom_callback(self, msg: VehicleOdometry):
        """Process odometry message."""
        # Position (NED to ENU)
        pos_ned = np.array(msg.position, dtype=float)
        position = ned_to_enu(pos_ned)
        
        # Velocity (NED to ENU)
        vel_ned = np.array(msg.velocity, dtype=float)
        velocity = ned_to_enu(vel_ned)
        
        # Orientation
        q = msg.q
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        
        # Rotation matrix (body to world)
        R_frd_ned = quat_to_rotation_matrix(qx, qy, qz, qw)
        self.rotation = _NED_TO_ENU @ R_frd_ned @ _FLU_TO_FRD
        
        # Yaw (NED to ENU)
        _, _, yaw_ned = euler_from_quaternion(qx, qy, qz, qw)
        yaw = wrap_angle(math.pi / 2 - yaw_ned)
        
        # Yaw rate
        try:
            yaw_rate = -float(msg.angular_velocity[2])
        except:
            yaw_rate = 0.0
        
        self.state = MPCState(
            position=position,
            velocity=velocity,
            yaw=yaw,
            yaw_rate=yaw_rate
        )
    
    def _scan_callback(self, msg: LaserScan):
        """Store latest LiDAR scan."""
        self.last_scan = msg
    
    def _enable_callback(self, msg: Int32):
        """Handle enable/disable command."""
        enabled = msg.data == 1
        if enabled != self.mpc_enabled:
            self.mpc_enabled = enabled
            self.get_logger().info(f'MPC enabled: {enabled}')
    
    def _status_callback(self, msg: VehicleStatus):
        """Handle PX4 vehicle status updates."""
        old_armed = self.armed
        old_offboard = self.offboard
        
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.offboard = msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.preflight_ok = msg.pre_flight_checks_pass
        
        if old_armed != self.armed:
            self.get_logger().info(f'Armed: {self.armed}')
        if old_offboard != self.offboard:
            self.get_logger().info(f'Offboard: {self.offboard}')
    
    # ---- PX4 Control ----
    
    def _now_us(self) -> int:
        """Get current time in microseconds."""
        return int(self.get_clock().now().nanoseconds / 1000)
    
    def _enu_to_ned_position(self, pos_enu: np.ndarray) -> List[float]:
        """Convert ENU position to NED."""
        return [float(pos_enu[1]), float(pos_enu[0]), float(-pos_enu[2])]
    
    def _enu_to_ned_yaw(self, yaw_enu: float) -> float:
        """Convert ENU yaw to NED yaw."""
        yaw_ned = (math.pi / 2.0) - yaw_enu
        return wrap_angle(yaw_ned)
    
    def _command_loop(self):
        """Send arm and offboard commands to PX4."""
        if not self.get_parameter('direct_px4_control').value:
            return
        
        if not self.get_parameter('auto_arm').value:
            return
            
        if not self.mpc_enabled:
            return
            
        if not self.preflight_ok:
            return
        
        now = time.time()
        if now - self.last_cmd_time < 1.0:
            return
        
        # Arm if not armed
        if not self.armed:
            cmd = VehicleCommand()
            cmd.timestamp = self._now_us()
            cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
            cmd.param1 = 1.0  # Arm
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.source_system = 1
            cmd.source_component = 1
            cmd.from_external = True
            self.px4_cmd_pub.publish(cmd)
            self.last_cmd_time = now
            self.get_logger().info('Sending ARM command')
            return
        
        # Switch to offboard if armed but not in offboard
        if not self.offboard and self.setpoint_counter >= 10:
            cmd = VehicleCommand()
            cmd.timestamp = self._now_us()
            cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
            cmd.param1 = 1.0  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            cmd.param2 = 6.0  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.source_system = 1
            cmd.source_component = 1
            cmd.from_external = True
            self.px4_cmd_pub.publish(cmd)
            self.last_cmd_time = now
            self.get_logger().info('Sending OFFBOARD mode command')
    
    def _publish_px4_setpoint(self, result: MPCResult):
        """Publish trajectory setpoint to PX4 with velocity feedforward."""
        if not self.get_parameter('direct_px4_control').value:
            return
        
        use_velocity = self.get_parameter('velocity_feedforward').value
        
        # Publish offboard control mode (required at high rate)
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = self._now_us()
        offboard_msg.position = True
        offboard_msg.velocity = use_velocity  # Enable velocity control too
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.offboard_pub.publish(offboard_msg)
        
        # Select target from predicted trajectory
        traj = result.predicted_trajectory
        # traj columns: [px, py, pz, yaw, vx, vy, vz, yaw_rate]
        
        if traj.shape[0] > self.lookahead_idx:
            target_idx = self.lookahead_idx
        else:
            target_idx = traj.shape[0] - 1
        
        # Get target position and yaw (in ENU)
        target_pos_enu = traj[target_idx, :3].copy()
        target_yaw_enu = float(traj[target_idx, 3])
        
        # Get target velocity (in ENU) - use current optimal velocity for feedforward
        # Use index 1 velocity (next step) for immediate response
        vel_idx = min(1, traj.shape[0] - 1)
        target_vel_enu = traj[vel_idx, 4:7].copy()
        target_yawrate = float(traj[vel_idx, 7])
        
        # Ensure minimum altitude
        target_pos_enu[2] = max(target_pos_enu[2], self.takeoff_height)
        
        # Convert to NED for PX4
        pos_ned = self._enu_to_ned_position(target_pos_enu)
        vel_ned = self._enu_to_ned_position(target_vel_enu)  # Same conversion for velocity
        yaw_ned = self._enu_to_ned_yaw(target_yaw_enu)
        
        # Publish trajectory setpoint with velocity feedforward
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = self._now_us()
        traj_msg.position = [pos_ned[0], pos_ned[1], pos_ned[2]]
        traj_msg.yaw = yaw_ned
        
        if use_velocity:
            traj_msg.velocity = [vel_ned[0], vel_ned[1], vel_ned[2]]
            traj_msg.yawspeed = -target_yawrate  # NED yaw is opposite
        
        self.traj_setpoint_pub.publish(traj_msg)
        
        self.setpoint_counter += 1
    
    # ---- Control Loop ----
    
    def _control_loop(self):
        """Main MPC control loop."""
        self.loop_counter += 1
        
        # Check preconditions
        if not self.mpc_enabled:
            if self.loop_counter % (self.log_interval * 5) == 0:
                self.get_logger().info('Waiting for MPC enable...')
            return
        
        if self.state is None or self.rotation is None:
            if self.loop_counter % (self.log_interval * 5) == 0:
                self.get_logger().info('Waiting for odometry...')
            return
        
        # Process obstacles
        if self.last_scan is not None:
            raw_points = self.lidar_processor.scan_to_world_points(
                self.last_scan,
                self.state.position,
                self.rotation
            )
            clustered_points = self.lidar_processor.cluster_points(raw_points)
        else:
            raw_points = np.zeros((0, 3))
            clustered_points = np.zeros((0, 3))
        
        obstacles = ObstacleSet(clustered_points, self.mpc_config)

        # Pure MPC: Go directly to goal without gap navigation
        active_goal = self.goal

        # Publish goals for visualization
        self._publish_goals(active_goal)

        # Build simple straight-line reference (no APF preprocessing)
        reference = self._build_simple_reference(active_goal)
        yaw_ref = self._build_yaw_reference(active_goal)
        
        # Solve MPC (potential field + constraints handle obstacle avoidance)
        result = self.mpc_solver.solve(
            state=self.state,
            goal=active_goal,
            obstacles=obstacles,
            reference_trajectory=reference,
            yaw_reference=yaw_ref
        )
        
        self._last_result = result
        
        # Publish command
        self._publish_command(result)
        
        # Publish trajectories
        self._publish_trajectories(result, reference, yaw_ref)
        
        # Publish cost
        cost_msg = Float64()
        cost_msg.data = result.cost
        self.cost_pub.publish(cost_msg)

        # Publish PX4 trajectory setpoint (direct control)
        self._publish_px4_setpoint(result)
        
        # Logging
        if self.loop_counter % self.log_interval == 0:
            self._log_status(result, raw_points, clustered_points)
    
    def _publish_goals(self, active_goal: np.ndarray):
        """Publish goal positions for visualization."""
        stamp = self.get_clock().now().to_msg()
        frame = self.get_parameter('trajectory_frame').value

        # Active goal (same as final goal in pure MPC mode)
        active_msg = PoseStamped()
        active_msg.header.stamp = stamp
        active_msg.header.frame_id = frame
        active_msg.pose.position.x = float(active_goal[0])
        active_msg.pose.position.y = float(active_goal[1])
        active_msg.pose.position.z = float(active_goal[2])
        self.active_goal_pub.publish(active_msg)

        # Final goal
        final_msg = PoseStamped()
        final_msg.header.stamp = stamp
        final_msg.header.frame_id = frame
        final_msg.pose.position.x = float(self.goal[0])
        final_msg.pose.position.y = float(self.goal[1])
        final_msg.pose.position.z = float(self.goal[2])
        self.final_goal_pub.publish(final_msg)
    
    def _build_simple_reference(self, goal: np.ndarray) -> np.ndarray:
        """
        Build simple straight-line reference trajectory.

        Pure MPC approach: generates a straight-line reference from current
        position to goal. The MPC optimizer will handle obstacle avoidance
        through its cost function and constraints.
        """
        N = self.mpc_config.N
        dt = self.mpc_config.dt

        start = self.state.position.copy()
        goal_dist = np.linalg.norm(goal - start)

        # If already at goal, just repeat goal position
        if goal_dist < 0.1:
            return np.tile(goal, (N, 1))

        # Straight-line interpolation with constant velocity
        direction = (goal - start) / goal_dist

        # Compute desired speed (limit to v_max)
        # Try to reach goal within horizon, but respect velocity limits
        time_horizon = N * dt
        desired_speed = min(goal_dist / time_horizon, self.mpc_config.v_max * 0.8)

        # Generate waypoints along straight line
        ref = np.zeros((N, 3))
        for k in range(N):
            # Distance to travel in k steps
            distance = desired_speed * dt * k

            # Don't overshoot the goal
            if distance >= goal_dist:
                ref[k:] = goal
                break

            ref[k] = start + direction * distance

        return ref
    
    def _build_yaw_reference(self, goal: np.ndarray) -> np.ndarray:
        """Build yaw reference trajectory."""
        N = self.mpc_config.N
        
        goal_dir = goal[:2] - self.state.position[:2]
        if np.linalg.norm(goal_dir) > 0.1:
            target_yaw = np.arctan2(goal_dir[1], goal_dir[0])
        else:
            target_yaw = self.state.yaw
        
        # Smooth transition
        current_yaw = self.state.yaw
        yaw_ref = np.zeros(N)
        
        max_yaw_change = self.mpc_config.yaw_rate_max * self.mpc_config.dt
        
        for k in range(N):
            diff = wrap_angle(target_yaw - current_yaw)
            if abs(diff) > max_yaw_change:
                diff = np.sign(diff) * max_yaw_change
            current_yaw = wrap_angle(current_yaw + diff)
            yaw_ref[k] = current_yaw
        
        return yaw_ref
    
    def _publish_command(self, result: MPCResult):
        """Publish acceleration command."""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = float(result.acceleration[0])
        msg.vector.y = float(result.acceleration[1])
        msg.vector.z = float(result.acceleration[2])
        self.cmd_pub.publish(msg)
    
    def _publish_trajectories(
        self,
        result: MPCResult,
        reference: np.ndarray,
        yaw_ref: np.ndarray
    ):
        """Publish optimal and reference trajectories."""
        frame = self.get_parameter('trajectory_frame').value
        stamp = self.get_clock().now().to_msg()
        
        # Optimal trajectory
        opt_path = Path()
        opt_path.header.stamp = stamp
        opt_path.header.frame_id = frame
        
        for k in range(result.predicted_trajectory.shape[0]):
            pose = PoseStamped()
            pose.header = opt_path.header
            pose.pose.position.x = float(result.predicted_trajectory[k, 0])
            pose.pose.position.y = float(result.predicted_trajectory[k, 1])
            pose.pose.position.z = float(result.predicted_trajectory[k, 2])
            
            yaw = float(result.predicted_trajectory[k, 3])
            qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            
            opt_path.poses.append(pose)
        
        self.opt_traj_pub.publish(opt_path)
        
        # Reference trajectory
        ref_path = Path()
        ref_path.header.stamp = stamp
        ref_path.header.frame_id = frame
        
        for k in range(reference.shape[0]):
            pose = PoseStamped()
            pose.header = ref_path.header
            pose.pose.position.x = float(reference[k, 0])
            pose.pose.position.y = float(reference[k, 1])
            pose.pose.position.z = float(reference[k, 2])
            
            qx, qy, qz, qw = quaternion_from_euler(0, 0, float(yaw_ref[k]))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            
            ref_path.poses.append(pose)
        
        self.ref_traj_pub.publish(ref_path)
    
    def _log_status(
        self,
        result: MPCResult,
        raw_points: np.ndarray,
        clustered_points: np.ndarray
    ):
        """Log MPC status with detailed debugging info."""
        goal_dist = np.linalg.norm(self.state.position - self.goal)

        # Basic status
        self.get_logger().info(
            f"MPC [PURE]: {result.solve_time_ms:.1f}ms | "
            f"status={result.status} | "
            f"obs={raw_points.shape[0]}->{clustered_points.shape[0]} | "
            f"goal_dist={goal_dist:.2f}m | "
            f"cost={result.cost:.1f}"
        )

        # Position and velocity info
        vel = self.state.velocity
        speed = np.linalg.norm(vel)
        self.get_logger().info(
            f"  Position: ({self.state.position[0]:.2f}, {self.state.position[1]:.2f}, {self.state.position[2]:.2f}) | "
            f"Velocity: ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) | "
            f"Speed: {speed:.2f} m/s | "
            f"Yaw: {np.degrees(self.state.yaw):.1f}°"
        )

        # Goal info
        self.get_logger().info(
            f"  Goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f})"
        )

        # MPC trajectory info
        if result.predicted_trajectory.shape[0] > 0:
            traj_end = result.predicted_trajectory[-1, :3]
            traj_dist = np.linalg.norm(traj_end - self.state.position)
            self.get_logger().info(
                f"  MPC traj end: ({traj_end[0]:.2f}, {traj_end[1]:.2f}, {traj_end[2]:.2f}) | "
                f"traj_length={traj_dist:.2f}m"
            )


def main(args=None):
    rclpy.init(args=args)
    node = MPCObstacleAvoidanceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
