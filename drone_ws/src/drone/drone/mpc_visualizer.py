#!/usr/bin/env python3
"""
Real-time 3D MPC Visualization

This script provides a live 3D visualization of:
- Drone position and trajectory
- Obstacles (raw and clustered)
- MPC predicted trajectory
- Reference trajectory
- Goal and intermediate goal positions
- Gap directions

Run alongside the MPC node for debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import PointCloud2, LaserScan
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray

from px4_msgs.msg import VehicleOdometry


class MPCVisualizer(Node):
    def __init__(self):
        super().__init__('mpc_visualizer')
        
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        
        # Data storage
        self.drone_position = np.zeros(3)
        self.drone_yaw = 0.0
        self.drone_history = deque(maxlen=500)  # Position history
        self.optimal_trajectory = np.zeros((0, 3))
        self.reference_trajectory = np.zeros((0, 3))
        self.obstacles = np.zeros((0, 3))
        self.goal = np.array([10.0, 10.0, 1.5])
        self.intermediate_goal = None
        self.cost = 0.0
        self.last_update = time.time()
        
        # Thread lock for data access
        self.lock = threading.Lock()
        
        # Subscribers
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self._odom_callback,
            qos
        )
        
        self.create_subscription(
            Path,
            '/mpc/optimal_trajectory',
            self._opt_traj_callback,
            qos
        )
        
        self.create_subscription(
            Path,
            '/mpc/reference_trajectory',
            self._ref_traj_callback,
            qos
        )
        
        self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            qos
        )
        
        self.create_subscription(
            Float64,
            '/mpc/cost',
            self._cost_callback,
            10
        )
        
        # Goal subscriber (you may need to create this publisher in main node)
        self.create_subscription(
            PoseStamped,
            '/mpc/active_goal',
            self._goal_callback,
            10
        )
        
        self.create_subscription(
            PoseStamped,
            '/mpc/intermediate_goal',
            self._intermediate_goal_callback,
            10
        )
        
        self.get_logger().info('MPC Visualizer started')
    
    def _ned_to_enu(self, pos_ned):
        return np.array([pos_ned[1], pos_ned[0], -pos_ned[2]])
    
    def _odom_callback(self, msg: VehicleOdometry):
        with self.lock:
            pos_ned = np.array(msg.position)
            self.drone_position = self._ned_to_enu(pos_ned)
            self.drone_history.append(self.drone_position.copy())
            
            # Extract yaw
            q = msg.q
            qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw_ned = np.arctan2(siny_cosp, cosy_cosp)
            self.drone_yaw = (np.pi / 2) - yaw_ned
    
    def _opt_traj_callback(self, msg: Path):
        with self.lock:
            if msg.poses:
                self.optimal_trajectory = np.array([
                    [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                    for p in msg.poses
                ])
                self.last_update = time.time()
    
    def _ref_traj_callback(self, msg: Path):
        with self.lock:
            if msg.poses:
                self.reference_trajectory = np.array([
                    [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                    for p in msg.poses
                ])
    
    def _scan_callback(self, msg: LaserScan):
        with self.lock:
            # Convert scan to world points (simplified - assumes drone at origin facing +X)
            ranges = np.array(msg.ranges)
            angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
            
            valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
            ranges = ranges[valid]
            angles = angles[valid]
            
            if len(ranges) > 0:
                # Body frame points
                x = ranges * np.cos(angles)
                y = ranges * np.sin(angles)
                z = np.zeros_like(x)
                
                # Rotate by yaw and translate
                cos_yaw = np.cos(self.drone_yaw)
                sin_yaw = np.sin(self.drone_yaw)
                
                x_world = self.drone_position[0] + x * cos_yaw - y * sin_yaw
                y_world = self.drone_position[1] + x * sin_yaw + y * cos_yaw
                z_world = self.drone_position[2] + z
                
                self.obstacles = np.column_stack([x_world, y_world, z_world])
    
    def _cost_callback(self, msg: Float64):
        with self.lock:
            self.cost = msg.data
    
    def _goal_callback(self, msg: PoseStamped):
        with self.lock:
            self.goal = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
    
    def _intermediate_goal_callback(self, msg: PoseStamped):
        with self.lock:
            self.intermediate_goal = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
    
    def get_visualization_data(self):
        """Thread-safe getter for visualization data."""
        with self.lock:
            return {
                'drone_pos': self.drone_position.copy(),
                'drone_yaw': self.drone_yaw,
                'drone_history': np.array(list(self.drone_history)) if self.drone_history else np.zeros((0, 3)),
                'opt_traj': self.optimal_trajectory.copy(),
                'ref_traj': self.reference_trajectory.copy(),
                'obstacles': self.obstacles.copy(),
                'goal': self.goal.copy(),
                'intermediate_goal': self.intermediate_goal.copy() if self.intermediate_goal is not None else None,
                'cost': self.cost,
                'last_update': self.last_update,
            }


def run_visualization(node: MPCVisualizer):
    """Run the matplotlib visualization in the main thread."""
    
    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    
    # 3D plot
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Top-down view (XY)
    ax_xy = fig.add_subplot(2, 2, 2)
    
    # Side view (XZ)
    ax_xz = fig.add_subplot(2, 2, 3)
    
    # Info panel
    ax_info = fig.add_subplot(2, 2, 4)
    ax_info.axis('off')
    
    plt.tight_layout()
    
    def update(frame):
        data = node.get_visualization_data()
        
        # Clear axes
        ax3d.cla()
        ax_xy.cla()
        ax_xz.cla()
        ax_info.cla()
        ax_info.axis('off')
        
        drone_pos = data['drone_pos']
        goal = data['goal']
        
        # ========== 3D Plot ==========
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title('3D View')
        
        # Drone position
        ax3d.scatter(*drone_pos, c='blue', s=100, marker='o', label='Drone')
        
        # Drone heading arrow
        yaw = data['drone_yaw']
        arrow_len = 0.5
        ax3d.quiver(drone_pos[0], drone_pos[1], drone_pos[2],
                   arrow_len * np.cos(yaw), arrow_len * np.sin(yaw), 0,
                   color='blue', arrow_length_ratio=0.3)
        
        # Drone history
        history = data['drone_history']
        if history.shape[0] > 1:
            ax3d.plot(history[:, 0], history[:, 1], history[:, 2], 
                     'b-', alpha=0.3, linewidth=1, label='Path history')
        
        # Obstacles
        obs = data['obstacles']
        if obs.shape[0] > 0:
            # Subsample for performance
            if obs.shape[0] > 200:
                idx = np.random.choice(obs.shape[0], 200, replace=False)
                obs = obs[idx]
            ax3d.scatter(obs[:, 0], obs[:, 1], obs[:, 2], 
                        c='red', s=10, alpha=0.5, label=f'Obstacles ({data["obstacles"].shape[0]})')
        
        # Optimal trajectory
        opt = data['opt_traj']
        if opt.shape[0] > 0:
            ax3d.plot(opt[:, 0], opt[:, 1], opt[:, 2], 
                     'g-', linewidth=2, label='MPC trajectory')
            ax3d.scatter(opt[-1, 0], opt[-1, 1], opt[-1, 2], 
                        c='green', s=50, marker='x')
        
        # Reference trajectory
        ref = data['ref_traj']
        if ref.shape[0] > 0:
            ax3d.plot(ref[:, 0], ref[:, 1], ref[:, 2], 
                     'c--', linewidth=1, alpha=0.7, label='Reference')
        
        # Goal
        ax3d.scatter(*goal, c='gold', s=200, marker='*', label='Goal')
        
        # Intermediate goal
        int_goal = data['intermediate_goal']
        if int_goal is not None:
            ax3d.scatter(*int_goal, c='orange', s=150, marker='^', label='Intermediate')
            # Line from drone to intermediate goal
            ax3d.plot([drone_pos[0], int_goal[0]], 
                     [drone_pos[1], int_goal[1]], 
                     [drone_pos[2], int_goal[2]], 
                     'orange', linestyle=':', linewidth=1)
        
        # Set axis limits centered on drone
        margin = 8
        ax3d.set_xlim(drone_pos[0] - margin, drone_pos[0] + margin)
        ax3d.set_ylim(drone_pos[1] - margin, drone_pos[1] + margin)
        ax3d.set_zlim(0, 5)
        ax3d.legend(loc='upper left', fontsize=8)
        
        # ========== XY Plot (Top-down) ==========
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        ax_xy.set_title('Top-Down View (XY)')
        ax_xy.set_aspect('equal')
        ax_xy.grid(True, alpha=0.3)
        
        # Obstacles
        if obs.shape[0] > 0:
            ax_xy.scatter(obs[:, 0], obs[:, 1], c='red', s=5, alpha=0.5)
        
        # History
        if history.shape[0] > 1:
            ax_xy.plot(history[:, 0], history[:, 1], 'b-', alpha=0.3, linewidth=1)
        
        # Optimal trajectory
        if opt.shape[0] > 0:
            ax_xy.plot(opt[:, 0], opt[:, 1], 'g-', linewidth=2)
        
        # Reference
        if ref.shape[0] > 0:
            ax_xy.plot(ref[:, 0], ref[:, 1], 'c--', linewidth=1, alpha=0.7)
        
        # Drone
        ax_xy.scatter(drone_pos[0], drone_pos[1], c='blue', s=100, zorder=10)
        ax_xy.arrow(drone_pos[0], drone_pos[1], 
                   0.8 * np.cos(yaw), 0.8 * np.sin(yaw),
                   head_width=0.2, head_length=0.1, fc='blue', ec='blue')
        
        # Goals
        ax_xy.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=10)
        if int_goal is not None:
            ax_xy.scatter(int_goal[0], int_goal[1], c='orange', s=150, marker='^', zorder=10)
        
        ax_xy.set_xlim(drone_pos[0] - margin, drone_pos[0] + margin)
        ax_xy.set_ylim(drone_pos[1] - margin, drone_pos[1] + margin)
        
        # ========== XZ Plot (Side view) ==========
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        ax_xz.set_title('Side View (XZ)')
        ax_xz.grid(True, alpha=0.3)
        
        # Obstacles
        if obs.shape[0] > 0:
            ax_xz.scatter(obs[:, 0], obs[:, 2], c='red', s=5, alpha=0.5)
        
        # History
        if history.shape[0] > 1:
            ax_xz.plot(history[:, 0], history[:, 2], 'b-', alpha=0.3, linewidth=1)
        
        # Optimal trajectory
        if opt.shape[0] > 0:
            ax_xz.plot(opt[:, 0], opt[:, 2], 'g-', linewidth=2)
        
        # Drone
        ax_xz.scatter(drone_pos[0], drone_pos[2], c='blue', s=100, zorder=10)
        
        # Goal
        ax_xz.scatter(goal[0], goal[2], c='gold', s=200, marker='*', zorder=10)
        
        ax_xz.set_xlim(drone_pos[0] - margin, drone_pos[0] + margin)
        ax_xz.set_ylim(0, 5)
        
        # ========== Info Panel ==========
        goal_dist = np.linalg.norm(drone_pos - goal)
        int_goal_dist = np.linalg.norm(drone_pos - int_goal) if int_goal is not None else 0
        
        time_since_update = time.time() - data['last_update']
        status_color = 'green' if time_since_update < 0.5 else 'orange' if time_since_update < 2.0 else 'red'
        
        info_text = f"""
        === MPC STATUS ===
        
        Drone Position:
          X: {drone_pos[0]:.2f} m
          Y: {drone_pos[1]:.2f} m
          Z: {drone_pos[2]:.2f} m
          Yaw: {np.degrees(yaw):.1f}°
        
        Goal Position:
          X: {goal[0]:.2f} m
          Y: {goal[1]:.2f} m
          Z: {goal[2]:.2f} m
        
        Distances:
          To Goal: {goal_dist:.2f} m
          To Intermediate: {int_goal_dist:.2f} m
        
        MPC:
          Cost: {data['cost']:.1f}
          Obstacles: {data['obstacles'].shape[0]}
          Trajectory pts: {opt.shape[0]}
        
        Update: {time_since_update:.2f}s ago
        """
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Status indicator
        ax_info.scatter([0.9], [0.95], c=status_color, s=200, transform=ax_info.transAxes)
        
        return []
    
    ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    plt.show(block=True)


def main(args=None):
    rclpy.init(args=args)
    node = MPCVisualizer()
    
    # Spin ROS in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        # Run visualization in main thread (required for matplotlib)
        run_visualization(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
