"""
MuJoCo Drone Simulation with ROS2 Integration

Runs the MuJoCo drone simulation with:
- Publishes drone pose to /drone/pose
- Publishes lidar points to /lidar/points
- Subscribes to /planned_path from A* planner
- Publishes goal pose to /goal_pose

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

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from mujoco_sim.controllers import DroneController, TrajectoryTracker
from mujoco_sim.flight_logger import FlightLogger


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
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Subscriber for planned path
        self.path_sub = self.create_subscription(
            Path,
            '/planned_path',
            self.path_callback,
            10
        )

        # State
        self.latest_path = None
        self.path_received = False
        self.trajectory_tracker = None

        self.get_logger().info('MuJoCo Sim Node initialized')
        self.get_logger().info(f'Goal position: ({self.goal_x}, {self.goal_y}, {self.goal_z})')

    def set_trajectory_tracker(self, tracker):
        """Set the trajectory tracker reference."""
        self.trajectory_tracker = tracker

    def path_callback(self, msg: Path):
        """Handle incoming planned path from A* planner."""
        if len(msg.poses) == 0:
            self.get_logger().warn('Received empty path')
            return

        # Convert Path to waypoints list
        waypoints = []
        for pose in msg.poses:
            waypoints.append([
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z
            ])

        # Only set trajectory ONCE - don't update during flight to avoid oscillation
        if not self.path_received:
            self.latest_path = waypoints
            self.path_received = True

            # Update trajectory tracker if available
            if self.trajectory_tracker is not None:
                self.trajectory_tracker.set_trajectory(waypoints)
                self.get_logger().info(f'Trajectory set with {len(waypoints)} waypoints from A* planner')
        # Ignore subsequent path updates to prevent oscillation

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

    def publish_goal(self):
        """Publish goal pose to trigger A* planning."""
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        msg.pose.position.x = self.goal_x
        msg.pose.position.y = self.goal_y
        msg.pose.position.z = self.goal_z
        msg.pose.orientation.w = 1.0

        self.goal_pub.publish(msg)


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

    # Set initial fallback trajectory (will be replaced by A* path)
    initial_trajectory = [
        [0, 0, 1.5],  # Hover at start
    ]
    tracker.set_trajectory(initial_trajectory)

    # Log trajectory to flight logger
    flight_logger.set_trajectory(initial_trajectory)
    flight_logger.log_event("Simulation started - waiting for A* path")

    # Print info
    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"Drone mass: 0.027 kg")
    print(f"Lidar rays: {NUM_LIDAR_RAYS}")
    print()
    print("Controller: Cascaded PD (Position -> Attitude -> Moments)")
    print(f"Goal position: ({ros_node.goal_x}, {ros_node.goal_y}, {ros_node.goal_z})")
    print()
    print("ROS2 Topics:")
    print("  Publishers:")
    print("    /drone/pose     - Current drone pose")
    print("    /lidar/points   - Lidar point cloud")
    print("    /goal_pose      - Goal for A* planner")
    print("  Subscribers:")
    print("    /planned_path   - Path from A* planner")
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

        # Publish initial goal to trigger A* planning (global goal)
        ros_node.publish_goal()

        while viewer.is_running():
            # Process ROS2 callbacks (non-blocking)
            rclpy.spin_once(ros_node, timeout_sec=0)

            # Track previous waypoint for detecting transitions
            prev_waypoint_idx = tracker.current_waypoint_idx

            # Update controller
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

                # Only publish goal at start until we have a path
                if not ros_node.path_received:
                    if step % (ros_publish_interval * 10) == 0:
                        ros_node.publish_goal()

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

                path_status = "A* path" if ros_node.path_received else "waiting for A*"
                print(f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                      f"Euler: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] | "
                      f"Error: {error:.3f}m | Progress: {progress:.0f}% | {path_status}")

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
