"""
MPC Wall-Aware Obstacle Avoidance Node

Hierarchical architecture for handling complex obstacles including long walls:
- Global layer: Generates waypoints around walls using occupancy grid + A*
- Local layer: MPC for smooth trajectory optimization between waypoints

This approach solves the local minima problem that pure MPC faces with walls.

Author: Wall-aware implementation
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from std_msgs.msg import Float64, Int32

from px4_msgs.msg import (
    VehicleOdometry,
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand,
    VehicleStatus,
)

# Import shared utilities from the v2 node
import sys
import os
sys.path.append(os.path.dirname(__file__))

from mpc_core import MPCConfig, MPCState, MPCSolver, ObstacleSet, MPCResult
from mpc_obstacle_avoidance_node_v2 import (
    ned_to_enu, quat_to_rotation_matrix, euler_from_quaternion,
    quaternion_from_euler, wrap_angle, LidarProcessor,
    _NED_TO_ENU, _FLU_TO_FRD
)
from flight_logger import FlightLogger


# ============================================================================
# Occupancy Grid Mapping
# ============================================================================

@dataclass
class OccupancyGridMapper:
    """
    Maintains a local occupancy grid from LiDAR scans.
    Used for detecting walls and planning paths around them.
    """

    resolution: float = 0.2  # meters per cell
    width: int = 100  # cells (20m x 20m grid)
    height: int = 100
    origin_x: float = -10.0  # grid origin in world frame
    origin_y: float = -10.0

    def __post_init__(self):
        """Initialize the grid."""
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.decay_rate = 0.95  # Decay old observations

    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((pos[0] - self.origin_x) / self.resolution)
        gy = int((pos[1] - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> np.ndarray:
        """Convert grid indices to world coordinates."""
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return np.array([x, y, 0.0])

    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are valid."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def update(self, obstacles: np.ndarray):
        """
        Update occupancy grid with new obstacle observations.

        Args:
            obstacles: (N, 3) array of obstacle positions in world frame
        """
        # Decay old observations
        self.grid *= self.decay_rate

        # Mark obstacle cells
        for obs in obstacles:
            gx, gy = self.world_to_grid(obs)
            if self.is_valid(gx, gy):
                self.grid[gy, gx] = min(1.0, self.grid[gy, gx] + 0.5)

        # Inflate obstacles for safety (simple dilation)
        self._inflate_obstacles()

    def _inflate_obstacles(self, inflation_cells: int = 2):
        """Inflate obstacles by dilation for safety margin."""
        from scipy.ndimage import binary_dilation

        occupied = self.grid > 0.5
        inflated = binary_dilation(occupied, iterations=inflation_cells)
        self.grid = np.where(inflated, np.maximum(self.grid, 0.7), self.grid)

    def is_free(self, pos: np.ndarray, threshold: float = 0.3) -> bool:
        """Check if a position is free of obstacles."""
        gx, gy = self.world_to_grid(pos)
        if not self.is_valid(gx, gy):
            return False
        return self.grid[gy, gx] < threshold


# ============================================================================
# A* Path Planning
# ============================================================================

class AStarPlanner:
    """
    A* path planner on occupancy grid.
    Finds optimal path around walls from start to goal.
    """

    def __init__(self, grid_mapper: OccupancyGridMapper):
        self.grid = grid_mapper

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        occupied_threshold: float = 0.5
    ) -> Optional[List[np.ndarray]]:
        """
        Plan path from start to goal using A*.

        Returns:
            List of waypoints in world coordinates, or None if no path found
        """
        start_gx, start_gy = self.grid.world_to_grid(start)
        goal_gx, goal_gy = self.grid.world_to_grid(goal)

        if not self.grid.is_valid(start_gx, start_gy) or not self.grid.is_valid(goal_gx, goal_gy):
            return None

        # A* implementation
        from heapq import heappush, heappop

        open_set = []
        heappush(open_set, (0.0, start_gx, start_gy))

        came_from = {}
        g_score = {(start_gx, start_gy): 0.0}
        f_score = {(start_gx, start_gy): self._heuristic(start_gx, start_gy, goal_gx, goal_gy)}

        closed_set = set()

        # 8-connected grid
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        max_iterations = 5000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1

            _, current_x, current_y = heappop(open_set)

            if (current_x, current_y) in closed_set:
                continue

            closed_set.add((current_x, current_y))

            # Check if reached goal
            if abs(current_x - goal_gx) <= 1 and abs(current_y - goal_gy) <= 1:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current_x, current_y)
                # Simplify path (remove redundant waypoints)
                path = self._simplify_path(path, occupied_threshold)
                return path

            # Explore neighbors
            for dx, dy in neighbors:
                neighbor_x = current_x + dx
                neighbor_y = current_y + dy

                if not self.grid.is_valid(neighbor_x, neighbor_y):
                    continue

                if (neighbor_x, neighbor_y) in closed_set:
                    continue

                # Check if occupied
                if self.grid.grid[neighbor_y, neighbor_x] > occupied_threshold:
                    continue

                # Cost (diagonal moves cost more)
                move_cost = math.sqrt(dx*dx + dy*dy)
                tentative_g = g_score[(current_x, current_y)] + move_cost

                if (neighbor_x, neighbor_y) not in g_score or tentative_g < g_score[(neighbor_x, neighbor_y)]:
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_score[(neighbor_x, neighbor_y)] = tentative_g
                    h = self._heuristic(neighbor_x, neighbor_y, goal_gx, goal_gy)
                    f_score[(neighbor_x, neighbor_y)] = tentative_g + h
                    heappush(open_set, (f_score[(neighbor_x, neighbor_y)], neighbor_x, neighbor_y))

        # No path found
        return None

    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _reconstruct_path(self, came_from: dict, current_x: int, current_y: int) -> List[np.ndarray]:
        """Reconstruct path from A* came_from dict."""
        path = []
        current = (current_x, current_y)

        while current in came_from:
            pos = self.grid.grid_to_world(current[0], current[1])
            path.append(pos)
            current = came_from[current]

        # Add start position
        pos = self.grid.grid_to_world(current[0], current[1])
        path.append(pos)

        path.reverse()
        return path

    def _simplify_path(self, path: List[np.ndarray], threshold: float) -> List[np.ndarray]:
        """
        Simplify path by removing waypoints that don't add value.
        Uses line-of-sight check.
        """
        if len(path) <= 2:
            return path

        simplified = [path[0]]

        i = 0
        while i < len(path) - 1:
            # Find farthest visible point
            j = len(path) - 1
            while j > i + 1:
                if self._has_line_of_sight(path[i], path[j], threshold):
                    break
                j -= 1

            simplified.append(path[j])
            i = j

        return simplified

    def _has_line_of_sight(self, start: np.ndarray, end: np.ndarray, threshold: float) -> bool:
        """Check if there's a clear line of sight between two points."""
        start_gx, start_gy = self.grid.world_to_grid(start)
        end_gx, end_gy = self.grid.world_to_grid(end)

        # Bresenham's line algorithm
        dx = abs(end_gx - start_gx)
        dy = abs(end_gy - start_gy)
        sx = 1 if end_gx > start_gx else -1
        sy = 1 if end_gy > start_gy else -1
        err = dx - dy

        x, y = start_gx, start_gy

        while True:
            if not self.grid.is_valid(x, y):
                return False

            if self.grid.grid[y, x] > threshold:
                return False

            if x == end_gx and y == end_gy:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True


# ============================================================================
# Waypoint Manager
# ============================================================================

@dataclass
class WaypointManager:
    """
    Manages waypoint queue and switching logic.
    """

    waypoint_radius: float = 1.0  # Switch to next waypoint when within this distance

    def __post_init__(self):
        self.waypoints: deque = deque()
        self.current_waypoint: Optional[np.ndarray] = None

    def set_path(self, waypoints: List[np.ndarray]):
        """Set new path (list of waypoints)."""
        self.waypoints = deque(waypoints)
        self.current_waypoint = self.waypoints.popleft() if self.waypoints else None

    def update(self, drone_pos: np.ndarray, final_goal: np.ndarray) -> np.ndarray:
        """
        Update and return current target waypoint.

        Returns:
            Current target position (waypoint or final goal)
        """
        if self.current_waypoint is None:
            return final_goal

        # Check if reached current waypoint
        dist = np.linalg.norm(drone_pos[:2] - self.current_waypoint[:2])

        if dist < self.waypoint_radius:
            # Move to next waypoint
            if self.waypoints:
                self.current_waypoint = self.waypoints.popleft()
            else:
                # Reached all waypoints, target final goal
                self.current_waypoint = None
                return final_goal

        return self.current_waypoint

    def clear(self):
        """Clear all waypoints."""
        self.waypoints.clear()
        self.current_waypoint = None


# ============================================================================
# Main Wall-Aware Node
# ============================================================================

class MPCWallAwareNode(Node):
    """
    Wall-aware MPC node with hierarchical planning.
    """

    def __init__(self):
        super().__init__('mpc_wall_aware')

        # Declare parameters
        self._declare_parameters()

        # Initialize components
        self._init_components()

        # State variables
        self._init_state()

        # ROS setup
        self._setup_ros()

        self.get_logger().info('MPC Wall-Aware Node started')

    def _declare_parameters(self):
        """Declare ROS parameters."""
        # MPC parameters
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('horizon', 20)
        self.declare_parameter('safety_radius', 1.5)
        self.declare_parameter('max_velocity', 1.5)
        self.declare_parameter('max_acceleration', 3.0)
        self.declare_parameter('max_yaw_rate', 1.5)

        # MPC costs
        self.declare_parameter('Q_pos', 15.0)
        self.declare_parameter('Q_goal', 80.0)
        self.declare_parameter('Q_vel', 1.0)
        self.declare_parameter('R_acc', 0.3)
        self.declare_parameter('Q_terminal', 150.0)
        self.declare_parameter('Q_vel_toward_obs', 50.0)
        self.declare_parameter('Q_obstacle_repulsion', 300.0)
        self.declare_parameter('potential_influence_dist', 3.0)
        self.declare_parameter('potential_steepness', 2.0)

        # Goal
        self.declare_parameter('goal', [10.0, 10.0, 1.5])
        self.declare_parameter('goal_threshold', 0.5)

        # Obstacle processing
        self.declare_parameter('max_obstacle_range', 6.0)
        self.declare_parameter('cluster_size', 0.3)

        # Global planner
        self.declare_parameter('grid_resolution', 0.2)
        self.declare_parameter('grid_size', 100)
        self.declare_parameter('replan_interval', 20)  # Replan every N iterations
        self.declare_parameter('waypoint_radius', 1.0)

        # Control
        self.declare_parameter('require_enable', False)
        self.declare_parameter('log_interval', 5)
        self.declare_parameter('trajectory_frame', 'world')

        # PX4
        self.declare_parameter('direct_px4_control', True)
        self.declare_parameter('auto_arm', True)
        self.declare_parameter('lookahead_index', 5)
        self.declare_parameter('velocity_feedforward', True)
        self.declare_parameter('takeoff_height', 1.0)

        # Logging
        self.declare_parameter('enable_logging', True)
        self.declare_parameter('log_directory', '/tmp/drone_logs')
        self.declare_parameter('grid_snapshot_interval', 10)  # Save grid every N cycles

    def _init_components(self):
        """Initialize components."""
        # MPC config
        self.mpc_config = MPCConfig(
            dt=self.get_parameter('dt').value,
            N=self.get_parameter('horizon').value,
            safety_radius=self.get_parameter('safety_radius').value,
            v_max=self.get_parameter('max_velocity').value,
            a_max=self.get_parameter('max_acceleration').value,
            yaw_rate_max=self.get_parameter('max_yaw_rate').value,
            obstacle_range=self.get_parameter('max_obstacle_range').value,
            Q_pos=self.get_parameter('Q_pos').value,
            Q_goal=self.get_parameter('Q_goal').value,
            Q_vel=self.get_parameter('Q_vel').value,
            R_acc=self.get_parameter('R_acc').value,
            Q_terminal=self.get_parameter('Q_terminal').value,
            Q_vel_toward_obs=self.get_parameter('Q_vel_toward_obs').value,
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

        # Occupancy grid
        self.grid_mapper = OccupancyGridMapper(
            resolution=self.get_parameter('grid_resolution').value,
            width=self.get_parameter('grid_size').value,
            height=self.get_parameter('grid_size').value,
        )

        # A* planner
        self.planner = AStarPlanner(self.grid_mapper)

        # Waypoint manager
        self.waypoint_manager = WaypointManager(
            waypoint_radius=self.get_parameter('waypoint_radius').value
        )

    def _init_state(self):
        """Initialize state."""
        self.state: Optional[MPCState] = None
        self.rotation: Optional[np.ndarray] = None
        self.last_scan: Optional[LaserScan] = None

        goal_param = self.get_parameter('goal').value
        self.goal = np.array(goal_param, dtype=float)

        self.mpc_enabled = not self.get_parameter('require_enable').value
        self.loop_counter = 0
        self.replan_counter = 0
        self.log_interval = self.get_parameter('log_interval').value
        self.replan_interval = self.get_parameter('replan_interval').value

        self._last_result: Optional[MPCResult] = None

        # PX4 state
        self.armed = False
        self.offboard = False
        self.preflight_ok = False
        self.setpoint_counter = 0
        self.last_cmd_time = 0.0
        self.lookahead_idx = self.get_parameter('lookahead_index').value
        self.takeoff_height = self.get_parameter('takeoff_height').value

        # Flight logger
        self.enable_logging = self.get_parameter('enable_logging').value
        self.grid_snapshot_interval = self.get_parameter('grid_snapshot_interval').value

        if self.enable_logging:
            log_dir = self.get_parameter('log_directory').value
            self.flight_logger = FlightLogger(log_dir)
            self.get_logger().info(f'Flight logging enabled: {self.flight_logger.session_dir}')
        else:
            self.flight_logger = None
            self.get_logger().info('Flight logging disabled')

    def _setup_ros(self):
        """Setup ROS."""
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Subscriptions
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_callback, qos)
        self.create_subscription(LaserScan, '/scan', self._scan_callback, qos)
        self.create_subscription(Int32, '/mpc/enable', self._enable_callback, 10)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self._status_callback, qos)

        # Publishers
        self.cmd_pub = self.create_publisher(Vector3Stamped, '/mpc/acceleration_cmd', 10)
        self.opt_traj_pub = self.create_publisher(Path, '/mpc/optimal_trajectory', 10)
        self.ref_traj_pub = self.create_publisher(Path, '/mpc/reference_trajectory', 10)
        self.cost_pub = self.create_publisher(Float64, '/mpc/cost', 10)
        self.active_goal_pub = self.create_publisher(PoseStamped, '/mpc/active_goal', 10)
        self.final_goal_pub = self.create_publisher(PoseStamped, '/mpc/final_goal', 10)
        self.waypoint_path_pub = self.create_publisher(Path, '/mpc/waypoint_path', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/mpc/occupancy_grid', 10)

        # PX4 publishers
        self.px4_cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.traj_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)

        # Timers
        dt = self.get_parameter('dt').value
        self.timer = self.create_timer(dt, self._control_loop)

        if self.get_parameter('direct_px4_control').value:
            self.cmd_timer = self.create_timer(1.0, self._command_loop)

    # Callbacks (similar to v2 node)
    def _odom_callback(self, msg: VehicleOdometry):
        """Process odometry."""
        pos_ned = np.array(msg.position, dtype=float)
        position = ned_to_enu(pos_ned)

        vel_ned = np.array(msg.velocity, dtype=float)
        velocity = ned_to_enu(vel_ned)

        q = msg.q
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        R_frd_ned = quat_to_rotation_matrix(qx, qy, qz, qw)
        self.rotation = _NED_TO_ENU @ R_frd_ned @ _FLU_TO_FRD

        _, _, yaw_ned = euler_from_quaternion(qx, qy, qz, qw)
        yaw = wrap_angle(math.pi / 2 - yaw_ned)

        try:
            yaw_rate = -float(msg.angular_velocity[2])
        except:
            yaw_rate = 0.0

        self.state = MPCState(position=position, velocity=velocity, yaw=yaw, yaw_rate=yaw_rate)

    def _scan_callback(self, msg: LaserScan):
        """Store LiDAR scan."""
        self.last_scan = msg

    def _enable_callback(self, msg: Int32):
        """Handle enable/disable."""
        enabled = msg.data == 1
        if enabled != self.mpc_enabled:
            self.mpc_enabled = enabled
            self.get_logger().info(f'MPC enabled: {enabled}')

    def _status_callback(self, msg: VehicleStatus):
        """Handle PX4 status."""
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.offboard = msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.preflight_ok = msg.pre_flight_checks_pass

    # PX4 control (copy from v2)
    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _enu_to_ned_position(self, pos_enu: np.ndarray) -> List[float]:
        return [float(pos_enu[1]), float(pos_enu[0]), float(-pos_enu[2])]

    def _enu_to_ned_yaw(self, yaw_enu: float) -> float:
        yaw_ned = (math.pi / 2.0) - yaw_enu
        return wrap_angle(yaw_ned)

    def _command_loop(self):
        """Send arm/offboard commands."""
        if not self.get_parameter('direct_px4_control').value:
            return
        if not self.get_parameter('auto_arm').value:
            return
        if not self.mpc_enabled or not self.preflight_ok:
            return

        now = time.time()
        if now - self.last_cmd_time < 1.0:
            return

        if not self.armed:
            cmd = VehicleCommand()
            cmd.timestamp = self._now_us()
            cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
            cmd.param1 = 1.0
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.source_system = 1
            cmd.source_component = 1
            cmd.from_external = True
            self.px4_cmd_pub.publish(cmd)
            self.last_cmd_time = now
            self.get_logger().info('Sending ARM command')
            return

        if not self.offboard and self.setpoint_counter >= 10:
            cmd = VehicleCommand()
            cmd.timestamp = self._now_us()
            cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
            cmd.param1 = 1.0
            cmd.param2 = 6.0
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.source_system = 1
            cmd.source_component = 1
            cmd.from_external = True
            self.px4_cmd_pub.publish(cmd)
            self.last_cmd_time = now
            self.get_logger().info('Sending OFFBOARD mode command')

    def _publish_px4_setpoint(self, result: MPCResult):
        """Publish PX4 setpoint."""
        if not self.get_parameter('direct_px4_control').value:
            return

        use_velocity = self.get_parameter('velocity_feedforward').value

        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = self._now_us()
        offboard_msg.position = True
        offboard_msg.velocity = use_velocity
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.offboard_pub.publish(offboard_msg)

        traj = result.predicted_trajectory
        target_idx = min(self.lookahead_idx, traj.shape[0] - 1)

        target_pos_enu = traj[target_idx, :3].copy()
        target_yaw_enu = float(traj[target_idx, 3])

        vel_idx = min(1, traj.shape[0] - 1)
        target_vel_enu = traj[vel_idx, 4:7].copy()
        target_yawrate = float(traj[vel_idx, 7])

        target_pos_enu[2] = max(target_pos_enu[2], self.takeoff_height)

        pos_ned = self._enu_to_ned_position(target_pos_enu)
        vel_ned = self._enu_to_ned_position(target_vel_enu)
        yaw_ned = self._enu_to_ned_yaw(target_yaw_enu)

        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = self._now_us()
        traj_msg.position = [pos_ned[0], pos_ned[1], pos_ned[2]]
        traj_msg.yaw = yaw_ned

        if use_velocity:
            traj_msg.velocity = [vel_ned[0], vel_ned[1], vel_ned[2]]
            traj_msg.yawspeed = -target_yawrate

        self.traj_setpoint_pub.publish(traj_msg)
        self.setpoint_counter += 1

    # Main control loop
    def _control_loop(self):
        """Main control loop with hierarchical planning."""
        self.loop_counter += 1

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
                self.last_scan, self.state.position, self.rotation
            )
            clustered_points = self.lidar_processor.cluster_points(raw_points)
        else:
            raw_points = np.zeros((0, 3))
            clustered_points = np.zeros((0, 3))

        # Update occupancy grid
        self.grid_mapper.update(clustered_points)

        # Global planning (replan periodically)
        self.replan_counter += 1
        if self.replan_counter >= self.replan_interval:
            self.replan_counter = 0
            self._replan_global_path()

        # Get current target waypoint
        active_goal = self.waypoint_manager.update(self.state.position, self.goal)

        # Build reference to active goal
        reference = self._build_straight_reference(active_goal)
        yaw_ref = self._build_yaw_reference(active_goal)

        # Solve MPC
        obstacles = ObstacleSet(clustered_points, self.mpc_config)
        result = self.mpc_solver.solve(
            state=self.state,
            goal=active_goal,
            obstacles=obstacles,
            reference_trajectory=reference,
            yaw_reference=yaw_ref
        )

        self._last_result = result

        # Publish
        self._publish_command(result)
        self._publish_trajectories(result, reference, yaw_ref)
        self._publish_goals(active_goal)
        self._publish_waypoint_path()
        self._publish_grid()

        cost_msg = Float64()
        cost_msg.data = result.cost
        self.cost_pub.publish(cost_msg)

        self._publish_px4_setpoint(result)

        # Logging
        if self.loop_counter % self.log_interval == 0:
            self._log_status(result, raw_points, clustered_points, active_goal)

        # Flight data logging
        if self.flight_logger is not None:
            self._log_flight_data(
                result, raw_points, clustered_points, active_goal
            )

            # Grid snapshot logging (less frequent)
            if self.loop_counter % self.grid_snapshot_interval == 0:
                self._log_grid_snapshot()

    def _replan_global_path(self):
        """Replan global path using A*."""
        if self.state is None:
            return

        path = self.planner.plan(self.state.position, self.goal)

        if path is not None and len(path) > 1:
            # Remove first waypoint (too close to current position)
            if len(path) > 2:
                path = path[1:]

            self.waypoint_manager.set_path(path)
            self.get_logger().info(f'Global path planned: {len(path)} waypoints')
        else:
            # No path found, go direct
            self.waypoint_manager.clear()
            self.get_logger().warn('No global path found, going direct to goal')

    def _build_straight_reference(self, goal: np.ndarray) -> np.ndarray:
        """Build straight-line reference."""
        N = self.mpc_config.N
        dt = self.mpc_config.dt

        start = self.state.position.copy()
        goal_dist = np.linalg.norm(goal - start)

        if goal_dist < 0.1:
            return np.tile(goal, (N, 1))

        direction = (goal - start) / goal_dist
        time_horizon = N * dt
        desired_speed = min(goal_dist / time_horizon, self.mpc_config.v_max * 0.8)

        ref = np.zeros((N, 3))
        for k in range(N):
            distance = desired_speed * dt * k
            if distance >= goal_dist:
                ref[k:] = goal
                break
            ref[k] = start + direction * distance

        return ref

    def _build_yaw_reference(self, goal: np.ndarray) -> np.ndarray:
        """Build yaw reference."""
        N = self.mpc_config.N
        goal_dir = goal[:2] - self.state.position[:2]

        if np.linalg.norm(goal_dir) > 0.1:
            target_yaw = np.arctan2(goal_dir[1], goal_dir[0])
        else:
            target_yaw = self.state.yaw

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
        """Publish command."""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = float(result.acceleration[0])
        msg.vector.y = float(result.acceleration[1])
        msg.vector.z = float(result.acceleration[2])
        self.cmd_pub.publish(msg)

    def _publish_trajectories(self, result: MPCResult, reference: np.ndarray, yaw_ref: np.ndarray):
        """Publish trajectories."""
        frame = self.get_parameter('trajectory_frame').value
        stamp = self.get_clock().now().to_msg()

        # Optimal
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

        # Reference
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

    def _publish_goals(self, active_goal: np.ndarray):
        """Publish goals."""
        stamp = self.get_clock().now().to_msg()
        frame = self.get_parameter('trajectory_frame').value

        active_msg = PoseStamped()
        active_msg.header.stamp = stamp
        active_msg.header.frame_id = frame
        active_msg.pose.position.x = float(active_goal[0])
        active_msg.pose.position.y = float(active_goal[1])
        active_msg.pose.position.z = float(active_goal[2])
        self.active_goal_pub.publish(active_msg)

        final_msg = PoseStamped()
        final_msg.header.stamp = stamp
        final_msg.header.frame_id = frame
        final_msg.pose.position.x = float(self.goal[0])
        final_msg.pose.position.y = float(self.goal[1])
        final_msg.pose.position.z = float(self.goal[2])
        self.final_goal_pub.publish(final_msg)

    def _publish_waypoint_path(self):
        """Publish waypoint path for visualization."""
        if not self.waypoint_manager.waypoints and self.waypoint_manager.current_waypoint is None:
            return

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.get_parameter('trajectory_frame').value

        # Add current waypoint
        if self.waypoint_manager.current_waypoint is not None:
            pose = PoseStamped()
            pose.header = path.header
            wp = self.waypoint_manager.current_waypoint
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = float(wp[2])
            path.poses.append(pose)

        # Add remaining waypoints
        for wp in self.waypoint_manager.waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = float(wp[2])
            path.poses.append(pose)

        self.waypoint_path_pub.publish(path)

    def _publish_grid(self):
        """Publish occupancy grid for visualization."""
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.get_parameter('trajectory_frame').value

        grid_msg.info.resolution = self.grid_mapper.resolution
        grid_msg.info.width = self.grid_mapper.width
        grid_msg.info.height = self.grid_mapper.height
        grid_msg.info.origin.position.x = self.grid_mapper.origin_x
        grid_msg.info.origin.position.y = self.grid_mapper.origin_y
        grid_msg.info.origin.position.z = 0.0

        # Convert to int8 (0-100)
        grid_data = (self.grid_mapper.grid * 100).astype(np.int8)
        grid_msg.data = grid_data.flatten().tolist()

        self.grid_pub.publish(grid_msg)

    def _log_status(self, result: MPCResult, raw_points: np.ndarray,
                    clustered_points: np.ndarray, active_goal: np.ndarray):
        """Log status."""
        goal_dist = np.linalg.norm(self.state.position - self.goal)
        active_dist = np.linalg.norm(self.state.position - active_goal)

        num_waypoints = len(self.waypoint_manager.waypoints)
        if self.waypoint_manager.current_waypoint is not None:
            num_waypoints += 1

        self.get_logger().info(
            f"MPC [WALL-AWARE]: {result.solve_time_ms:.1f}ms | "
            f"status={result.status} | "
            f"obs={raw_points.shape[0]}->{clustered_points.shape[0]} | "
            f"waypoints={num_waypoints} | "
            f"active_dist={active_dist:.2f}m | "
            f"goal_dist={goal_dist:.2f}m | "
            f"cost={result.cost:.1f}"
        )

        vel = self.state.velocity
        speed = np.linalg.norm(vel)
        self.get_logger().info(
            f"  Position: ({self.state.position[0]:.2f}, {self.state.position[1]:.2f}, {self.state.position[2]:.2f}) | "
            f"Speed: {speed:.2f} m/s"
        )

    def _log_flight_data(self, result: MPCResult, raw_points: np.ndarray,
                         clustered_points: np.ndarray, active_goal: np.ndarray):
        """Log detailed flight data for post-processing."""
        if self.flight_logger is None or self.state is None:
            return

        num_waypoints = len(self.waypoint_manager.waypoints)
        if self.waypoint_manager.current_waypoint is not None:
            num_waypoints += 1

        goal_dist = np.linalg.norm(self.state.position - self.goal)
        active_dist = np.linalg.norm(self.state.position - active_goal)

        self.flight_logger.log_flight_data(
            timestamp=self.get_clock().now().nanoseconds / 1e9,
            position=self.state.position,
            velocity=self.state.velocity,
            yaw=self.state.yaw,
            yaw_rate=self.state.yaw_rate,
            active_goal=active_goal,
            final_goal=self.goal,
            num_waypoints=num_waypoints,
            current_waypoint=self.waypoint_manager.current_waypoint,
            mpc_solve_time_ms=result.solve_time_ms,
            mpc_cost=result.cost,
            mpc_status=result.status,
            num_raw_obstacles=raw_points.shape[0],
            num_clustered_obstacles=clustered_points.shape[0],
            distance_to_active_goal=active_dist,
            distance_to_final_goal=goal_dist,
            acceleration_cmd=result.acceleration
        )

    def _log_grid_snapshot(self):
        """Log occupancy grid snapshot."""
        if self.flight_logger is None or self.state is None:
            return

        self.flight_logger.log_grid_snapshot(
            timestamp=self.get_clock().now().nanoseconds / 1e9,
            grid=self.grid_mapper.grid,
            drone_position=self.state.position,
            resolution=self.grid_mapper.resolution,
            origin_x=self.grid_mapper.origin_x,
            origin_y=self.grid_mapper.origin_y
        )

    def cleanup(self):
        """Cleanup and save logged data."""
        if self.flight_logger is not None:
            self.get_logger().info('Saving flight logs...')
            session_dir = self.flight_logger.save()
            self.get_logger().info(f'Flight logs saved to: {session_dir}')
            self.get_logger().info('To visualize: python visualize_flight.py ' + str(session_dir))


def main(args=None):
    rclpy.init(args=args)
    node = MPCWallAwareNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        # Save flight logs before shutdown
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
