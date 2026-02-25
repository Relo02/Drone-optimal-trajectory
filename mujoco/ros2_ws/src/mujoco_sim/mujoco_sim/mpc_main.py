"""
MuJoCo Drone Simulation with ROS2 Integration + MPC Local Planner

Runs the MuJoCo drone simulation with:
- CasADi MPC local planner for obstacle-aware trajectory generation
- LiDAR-based Gaussian occupancy grid + RANSAC wall extraction
- Publishes drone pose to /drone/pose
- Publishes lidar points to /lidar/points
- Publishes MPC predicted path to /mpc/predicted_path
- Publishes MPC diagnostics to /mpc/diagnostics

author: Lorenzo Ortolani
"""

import mujoco
import mujoco.viewer
import numpy as np
import signal
import atexit
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header, Float64MultiArray, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from mujoco_sim.controllers import DroneController, TrajectoryTracker
from mujoco_sim.flight_logger import FlightLogger
from mujoco_sim.MPC import MPCLocalPlanner, MPCConfig, DroneState, MPCResult


# Lidar configuration
NUM_LIDAR_RAYS = 360
LIDAR_SENSOR_START = 6


def get_lidar_points(model, data):
    """Convert lidar readings to 3D points in world frame."""
    points = []
    drone_pos = data.qpos[:3]
    drone_quat = data.qpos[3:7]

    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, drone_quat)
    rot_mat = rot_mat.reshape(3, 3)

    for i in range(NUM_LIDAR_RAYS):
        distance = data.sensordata[LIDAR_SENSOR_START + i]
        if distance > 0:
            angle = np.radians(i)
            ray_dir_body = np.array([np.cos(angle), np.sin(angle), 0])
            ray_dir_world = rot_mat @ ray_dir_body
            hit_point = drone_pos + distance * ray_dir_world
            points.append(hit_point)

    return np.array(points) if points else np.empty((0, 3))


def add_lidar_visualization(viewer, model, data):
    """Add lidar points to the viewer."""
    points = get_lidar_points(model, data)
    viewer.user_scn.ngeom = 0

    if len(points) == 0:
        return

    for point in points:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.03, 0, 0]),
            point,
            np.eye(3).flatten(),
            np.array([0.0, 1.0, 0.0, 0.8])
        )
        viewer.user_scn.ngeom += 1


def add_target_visualization(viewer, target_pos):
    """Add target position marker."""
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.1, 0, 0]),
        target_pos,
        np.eye(3).flatten(),
        np.array([1.0, 0.0, 0.0, 0.5])  # Red, semi-transparent
    )
    viewer.user_scn.ngeom += 1


class MujocoSimNode(Node):
    """ROS2 node for MuJoCo simulation interface."""

    def __init__(self):
        super().__init__('mujoco_sim')

        # Declare parameters
        self.declare_parameter('goal_x', 10.0)
        self.declare_parameter('goal_y', 1.0)
        self.declare_parameter('goal_z', 1.5)
        self.declare_parameter('publish_rate', 50.0)

        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.goal_z = self.get_parameter('goal_z').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/drone/pose', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/lidar/points', sensor_qos)
        self.mpc_path_pub = self.create_publisher(Path, '/mpc/predicted_path', 10)
        self.mpc_diag_pub = self.create_publisher(Float64MultiArray, '/mpc/diagnostics', 10)
        self.grid_pub = self.create_publisher(Float32MultiArray, '/mpc/grid_data', 10)

        # MPC Local Planner
        mpc_cfg = MPCConfig(
            N=20, dt=0.1,
            v_max=2.0, a_max=1.5,
            Q_pos=5.0, Q_terminal=15.0,
            W_obs_grid=40.0, W_obs_line=200.0,
            W_goal=8.0, W_goal_terminal=40.0,
            d_safe=0.6,
            subgoal_lookahead=3.0, subgoal_lateral=1.5,
            grid_reso=0.25, grid_std=0.6,
            print_level=0,
        )
        self.mpc_planner = MPCLocalPlanner(mpc_cfg)

        # State
        self.trajectory_tracker = None
        self.last_mpc_result: MPCResult = None

        # ── Takeoff / hover gate ──
        self.hover_altitude = self.goal_z          # target altitude for hover & flight
        self.hover_threshold = 0.15                # m – how close to altitude before MPC starts
        self.hover_vel_threshold = 0.15            # m/s – how slow before we consider it stable
        self.takeoff_complete = False               # flipped once, never goes back

        # ── Async MPC threading ──
        self._mpc_thread: threading.Thread = None
        self._mpc_lock = threading.Lock()           # protects shared data
        self._mpc_busy = False                      # True while solver is running
        self._pending_waypoints = None               # latest result from bg thread
        self._pending_result: MPCResult = None
        self.mpc_replan_interval = 50                # sim steps between MPC triggers
        self.mpc_step_counter = 0

        self.get_logger().info('MuJoCo Sim Node initialized (MPC planner, async)')
        self.get_logger().info(f'Goal position: ({self.goal_x}, {self.goal_y}, {self.goal_z})')

    def set_trajectory_tracker(self, tracker):
        """Set the trajectory tracker reference."""
        self.trajectory_tracker = tracker

    # ─────────────────────────────────────────
    # Async MPC: launch solver in background
    # ─────────────────────────────────────────
    def run_mpc(self, model, data):
        """
        Trigger MPC solve in a background thread (non-blocking).
        If the solver is still running from a previous call, skip.
        After the solver finishes, apply its result on the next call.

        IMPORTANT: MPC only activates after the drone has completed takeoff
        (reached hover altitude with low vertical velocity).  Before that,
        the initial hover trajectory drives the low-level controller.
        """
        # ── 0. Takeoff gate — don't touch the trajectory until hover is stable ──
        if not self.takeoff_complete:
            pos = data.qpos[:3]
            vel = data.qvel[:3]
            z_err = abs(pos[2] - self.hover_altitude)
            vz = abs(vel[2])
            if z_err < self.hover_threshold and vz < self.hover_vel_threshold:
                self.takeoff_complete = True
                self.get_logger().info(
                    f'Takeoff complete — altitude {pos[2]:.2f}m, '
                    f'Vz {vel[2]:.3f} m/s.  MPC planner activated.'
                )
            else:
                return  # still climbing — don't plan yet

        # ── 1. Apply any finished result from the background thread ──
        with self._mpc_lock:
            if self._pending_waypoints is not None:
                if self.trajectory_tracker is not None:
                    self.trajectory_tracker.update_waypoints(self._pending_waypoints)
                self.last_mpc_result = self._pending_result
                self._publish_mpc_path(self._pending_waypoints)
                self._publish_mpc_diagnostics(self._pending_result)
                self._pending_waypoints = None
                self._pending_result = None

        # ── 2. Decide whether to launch a new solve ──
        self.mpc_step_counter += 1
        if self.mpc_step_counter % self.mpc_replan_interval != 0:
            return

        # Don't launch if previous solve is still running
        if self._mpc_busy:
            return

        # ── 3. Snapshot current state (read from MuJoCo — must be on main thread) ──
        pos = data.qpos[:3].copy()
        vel = data.qvel[:3].copy()
        quat = data.qpos[3:7].copy()
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        drone_state = DroneState(
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
            vx=float(vel[0]), vy=float(vel[1]), yaw=float(yaw),
        )
        goal = np.array([self.goal_x, self.goal_y])
        lidar_points = get_lidar_points(model, data)

        # ── 4. Launch background thread ──
        self._mpc_busy = True
        self._mpc_thread = threading.Thread(
            target=self._solve_mpc_background,
            args=(drone_state, goal, lidar_points),
            daemon=True,
        )
        self._mpc_thread.start()

    def _solve_mpc_background(self, drone_state, goal, lidar_points):
        """Run MPC solve on background thread — never touches MuJoCo data."""
        try:
            result = self.mpc_planner.plan(
                drone_state=drone_state,
                goal=goal,
                lidar_points=lidar_points if len(lidar_points) > 0 else None,
            )

            # Build 3-D waypoints (skip index 0 = current pos)
            # Z = hover_altitude (goal_z) so the drone stays at the
            # commanded flight level, NOT whatever Z it had at solve time.
            # Also skip waypoints that are too close to the drone —
            # the first few MPC steps are within the tracker threshold
            # and would be instantly "reached", causing waypoint spam.
            target_z = self.hover_altitude
            min_dist_sq = 0.25 ** 2   # skip waypoints within 25 cm of drone
            drone_xy = np.array([drone_state.x, drone_state.y])
            waypoints = []
            for k in range(1, result.x_pred.shape[0]):
                wp_xy = result.x_pred[k, :2]
                if np.sum((wp_xy - drone_xy)**2) >= min_dist_sq or len(waypoints) > 0:
                    waypoints.append([
                        float(wp_xy[0]),
                        float(wp_xy[1]),
                        target_z,
                    ])

            # Fallback: if all waypoints were too close, use the last one
            if len(waypoints) == 0:
                last = result.x_pred[-1, :2]
                waypoints.append([float(last[0]), float(last[1]), target_z])

            # Store result for the main thread to pick up
            with self._mpc_lock:
                self._pending_waypoints = waypoints
                self._pending_result = result

            # Publish grid data for the visualization node
            self._publish_grid_data()
        except Exception as e:
            # Log but don't crash the simulation
            print(f"[MPC background] solver error: {e}")
        finally:
            self._mpc_busy = False

    def _publish_mpc_path(self, waypoints):
        """Publish the MPC predicted trajectory as a nav_msgs/Path."""
        msg = Path()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        for wp in waypoints:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(wp[0])
            ps.pose.position.y = float(wp[1])
            ps.pose.position.z = float(wp[2])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self.mpc_path_pub.publish(msg)

    def _publish_mpc_diagnostics(self, result: MPCResult):
        """Publish MPC diagnostics [cost, solve_time_ms, success] on /mpc/diagnostics."""
        msg = Float64MultiArray()
        msg.data = [
            float(result.cost),
            float(result.solve_time_ms),
            1.0 if result.success else 0.0,
        ]
        self.mpc_diag_pub.publish(msg)

    def _publish_grid_data(self):
        """
        Publish the Gaussian grid map as a flat Float32MultiArray.

        Layout: [minx, miny, reso, xw, yw, data...]
        where data is the flattened grid map in row-major order.
        """
        gm = self.mpc_planner.grid_map
        if gm.gmap is None:
            return

        msg = Float32MultiArray()
        header = [
            float(gm.minx),
            float(gm.miny),
            float(gm.xyreso),
            float(gm.xw),
            float(gm.yw),
        ]
        msg.data = header + gm.gmap.ravel(order='C').astype(np.float32).tolist()
        self.grid_pub.publish(msg)

    def publish_pose(self, position, orientation):
        """Publish drone pose."""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        # MuJoCo quaternion is [w, x, y, z], ROS uses [x, y, z, w]
        msg.pose.orientation.x = float(orientation[1])
        msg.pose.orientation.y = float(orientation[2])
        msg.pose.orientation.z = float(orientation[3])
        msg.pose.orientation.w = float(orientation[0])

        self.pose_pub.publish(msg)

    def publish_lidar(self, points):
        """Publish lidar point cloud."""
        if len(points) == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'world'

        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg = point_cloud2.create_cloud(header, fields, points.astype(np.float32))
        self.lidar_pub.publish(msg)


def main():
    # Initialize ROS2
    rclpy.init()
    ros_node = MujocoSimNode()

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path('/home/lorenzo/Drone-optimal-trajectory/mujoco/model/drone_world.xml')
    data = mujoco.MjData(model)

    # Initialize flight logger
    flight_logger = FlightLogger()

    # Cleanup handler
    _cleanup_done = False

    def cleanup_handler():
        nonlocal _cleanup_done
        if not _cleanup_done:
            _cleanup_done = True
            print("\nSaving flight logs...")
            flight_logger.finalize(trajectory_completed=False)
            print(f"Logs saved to: {flight_logger.log_dir}")
            ros_node.destroy_node()
            rclpy.shutdown()

    def signal_handler(sig, frame):
        cleanup_handler()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_handler)

    # Initialize controller
    controller = DroneController(model, data)
    tracker = TrajectoryTracker(controller)

    # Connect ROS node to trajectory tracker
    ros_node.set_trajectory_tracker(tracker)

    # Initial hover trajectory — drone climbs to goal altitude before MPC activates
    initial_trajectory = [
        [0, 0, ros_node.hover_altitude],  # Hover at goal altitude
    ]
    tracker.set_trajectory(initial_trajectory)

    # Log trajectory to flight logger
    flight_logger.set_trajectory(initial_trajectory)
    flight_logger.log_event("Simulation started - MPC local planner active")

    # Print info
    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"Drone mass: 0.027 kg")
    print(f"Lidar rays: {NUM_LIDAR_RAYS}")
    print()
    print("Controller: Cascaded PD (Position -> Attitude -> Moments)")
    print("Planner:    CasADi MPC (Gaussian grid + RANSAC walls)")
    print(f"Goal position: ({ros_node.goal_x}, {ros_node.goal_y}, {ros_node.goal_z})")
    print()
    print("ROS2 Topics:")
    print("  Publishers:")
    print("    /drone/pose           - Current drone pose")
    print("    /lidar/points         - Lidar point cloud")
    print("    /mpc/predicted_path   - MPC predicted trajectory")
    print("    /mpc/diagnostics      - MPC cost, solve time, success [Float64MultiArray]")
    print()
    print("Controls:")
    print("  - Green spheres: Lidar hit points")
    print("  - Red sphere: Current target waypoint")
    print("  - Space: Pause simulation")
    print()
    print("Launching viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        prev_waypoint_idx = 0
        log_interval = 10  # Log every N steps
        ros_publish_interval = int(500 / ros_node.publish_rate)  # Publish at configured rate

        while viewer.is_running():
            # Process ROS2 callbacks (non-blocking)
            rclpy.spin_once(ros_node, timeout_sec=0)

            # ── Run MPC planner (replans at configured interval) ──
            ros_node.run_mpc(model, data)

            # Track previous waypoint for detecting transitions
            prev_waypoint_idx = tracker.current_waypoint_idx

            # Update controller (follows MPC-planned waypoints)
            state = tracker.update()

            # Step simulation
            mujoco.mj_step(model, data)

            # Get current simulation time
            sim_time = data.time

            # Publish to ROS2 topics periodically
            if step % ros_publish_interval == 0:
                position = data.qpos[:3]
                orientation = data.qpos[3:7]
                ros_node.publish_pose(position, orientation)

                lidar_points = get_lidar_points(model, data)
                ros_node.publish_lidar(lidar_points)

            # Log state periodically
            if step % log_interval == 0:
                error = controller.get_position_error()
                control = controller.ctrl_filtered.copy()

                flight_logger.log_state(
                    time=sim_time,
                    state=state,
                    control=control,
                    target_pos=controller.target_pos,
                    error=error
                )

                # Log MPC diagnostics alongside flight state
                if ros_node.last_mpc_result is not None:
                    r = ros_node.last_mpc_result
                    flight_logger.log_mpc_state(
                        time=sim_time,
                        cost=r.cost,
                        solve_time_ms=r.solve_time_ms,
                        success=r.success,
                    )

            # Detect waypoint reached
            if tracker.current_waypoint_idx > prev_waypoint_idx and prev_waypoint_idx < len(tracker.waypoints):
                flight_logger.log_waypoint_reached(
                    waypoint_idx=prev_waypoint_idx,
                    time=sim_time,
                    position=state['position']
                )

            # Print status periodically
            if step % 500 == 0:
                pos = state['position']
                euler = np.degrees(state['euler'])
                error = controller.get_position_error()
                progress = tracker.get_progress() * 100

                mpc_info = ""
                if ros_node.last_mpc_result is not None:
                    r = ros_node.last_mpc_result
                    mpc_info = f"MPC cost={r.cost:.1f} t={r.solve_time_ms:.0f}ms ok={r.success}"
                else:
                    mpc_info = "MPC: not yet solved"
                print(f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                      f"Euler: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] | "
                      f"Error: {error:.3f}m | Progress: {progress:.0f}% | {mpc_info}")

            viewer.sync()
            step += 1

        # Finalize logging
        _cleanup_done = True
        flight_logger.finalize(trajectory_completed=tracker.is_complete())

        print("\nSimulation ended.")
        if tracker.is_complete():
            print("Trajectory completed successfully!")
        else:
            print(f"Trajectory progress: {tracker.get_progress()*100:.0f}%")

        print(f"\nFlight logs saved to: {flight_logger.log_dir}")

        # Cleanup ROS2
        ros_node.destroy_node()
        rclpy.shutdown()

        # Show flight plots
        try:
            from mujoco_sim.plot_flight import create_summary_figure, load_flight_data
            print("\nGenerating flight plots...")
            plot_data = load_flight_data(flight_logger.log_dir, flight_logger.session_id)
            save_path = flight_logger.log_dir / f"flight_plot_{flight_logger.session_id}.png"
            create_summary_figure(plot_data, save_path)
            import matplotlib.pyplot as plt
            plt.show()
        except ImportError:
            print("Install matplotlib to see flight plots: pip install matplotlib")
        except Exception as e:
            print(f"Could not generate plots: {e}")


if __name__ == '__main__':
    main()
