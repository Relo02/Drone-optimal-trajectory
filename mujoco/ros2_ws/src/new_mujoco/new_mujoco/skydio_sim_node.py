#!/usr/bin/env python3
"""
Skydio X2 MuJoCo simulation — ROS2 node.

Control architecture (all gains loaded from YAML parameter file):
  Outer position loop  : xyz position error → desired xyz velocity
  Middle velocity loop : velocity error → desired roll/pitch angle setpoints
                         + altitude thrust adjustment
  Inner attitude loop  : roll/pitch/yaw error → torque commands

3D lidar (VLP-16 style, 16 elevation × 36 azimuth = 576 rays):
  Published as sensor_msgs/PointCloud2 in world frame.

Topics published:
  /skydio/pose           — geometry_msgs/PoseStamped  (drone pose, world frame)
  /skydio/scan3d         — sensor_msgs/PointCloud2    (3-D lidar hits, world frame)
  /skydio/imu            — sensor_msgs/Imu            (body frame)
  /skydio/reference_pose — geometry_msgs/PoseStamped  (active reference, world frame)

Topics subscribed:
  /goal_pose — geometry_msgs/PoseStamped  (runtime reference override)

author: Lorenzo Ortolani
"""

import math
import os
import signal
import time

import numpy as np

import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField, Imu

# ---------------------------------------------------------------------------
# Default model path — overridable via the 'model_path' ROS2 parameter
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.expanduser('~'),
    'Drone-optimal-trajectory', 'mujoco', 'MuJoCo',
    'mujoco_menagerie-main', 'skydio_x2', 'skydio_world.xml',
)

# ---------------------------------------------------------------------------
# 3-D lidar constants  (must match generate_x2_lidar.py)
# ---------------------------------------------------------------------------
NUM_LIDAR_ELEVATIONS = 16     # −15° … +15° in 2° steps
NUM_LIDAR_AZIMUTHS   = 36     # 0° … 350° in 10° steps
NUM_LIDAR_RAYS       = NUM_LIDAR_ELEVATIONS * NUM_LIDAR_AZIMUTHS   # 576
LIDAR_SENSOR_START   = 10     # sensor-data offset (gyro=0-2, accel=3-5, framequat=6-9)
LIDAR_CUTOFF_M       = 50.0   # metres — matches generate_x2_lidar.py


# ---------------------------------------------------------------------------
# Generic discrete PID controller
# ---------------------------------------------------------------------------
class PIDController:
    """
    Discrete PID with forward-Euler integration and output clamping.

    Anti-windup: integral is conditionally frozen when output is saturated
    and the error would further wind up the integrator (back-calculation
    via conditional integration).
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float | None = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

        self._integral   = 0.0
        self._prev_error = 0.0
        self._first_call = True

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._first_call = True

    def update(self, error: float, dt: float) -> float:
        if self._first_call:
            self._prev_error = error
            self._first_call = False

        derivative = (error - self._prev_error) / dt if dt > 1e-9 else 0.0

        # Conditional integration (anti-windup)
        saturated = (self.output_limit is not None) and (
            abs(self._integral * self.ki) >= self.output_limit
        )
        if not saturated or (error * self._integral < 0.0):
            self._integral += error * dt

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        self._prev_error = error

        if self.output_limit is not None:
            output = float(np.clip(output, -self.output_limit, self.output_limit))

        return output


# ---------------------------------------------------------------------------
# Quaternion → ZYX Euler angles
# ---------------------------------------------------------------------------
def quat_to_euler_zyx(qw: float, qx: float, qy: float, qz: float):
    """Return (roll, pitch, yaw) in radians — ZYX convention."""
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Cascaded PID controller
# ---------------------------------------------------------------------------
class CascadedPositionController:
    """
    Three-layer cascaded PID for Skydio X2 position and attitude control.

    Layer 1 — Position (slow):
        position error  →  desired velocity  (x, y, z)

    Layer 2 — Velocity / altitude (medium):
        velocity error  →  desired roll/pitch setpoints
        altitude error  →  thrust correction

    Layer 3 — Attitude (fast):
        roll/pitch/yaw error  →  torque commands (roll_cmd, pitch_cmd, yaw_cmd)

    All gains and limits are injected via the constructor so they can come
    from ROS2 parameters (YAML file) — nothing is hardcoded here.
    """

    def __init__(
        self,
        data: mujoco.MjData,
        # Position PIDs
        pos_kp: float, pos_ki: float, pos_kd: float, pos_vel_limit: float,
        # Altitude PID
        alt_kp: float, alt_ki: float, alt_kd: float, alt_thrust_limit: float,
        # Velocity → angle setpoint PIDs
        vel_kp: float, vel_ki: float, vel_kd: float, vel_angle_limit: float,
        # Attitude PIDs
        roll_kp:  float, roll_ki:  float, roll_kd:  float, roll_limit:  float,
        pitch_kp: float, pitch_ki: float, pitch_kd: float, pitch_limit: float,
        yaw_kp:   float, yaw_ki:   float, yaw_kd:   float, yaw_limit:   float,
        # Feed-forward
        hover_thrust: float,
        # Kinematic limits
        max_speed_xy: float,
        max_speed_z: float,
        max_accel_xy: float,
        max_accel_z: float,
        # Loop rate dividers  (vs sim step)
        pos_divider: int,
        vel_divider: int,
    ):
        self._d = data

        # Target state
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0

        # Feed-forward hover thrust per motor
        self.hover_thrust = hover_thrust

        # Kinematic limits
        self._max_speed_xy = max_speed_xy
        self._max_speed_z  = max_speed_z
        self._max_accel_xy = max_accel_xy
        self._max_accel_z  = max_accel_z

        # Loop dividers
        self._pos_div = pos_divider
        self._vel_div = vel_divider
        self._step    = 0

        # ── Layer 1: position → velocity ──────────────────────────────────
        self._pos_pid_x = PIDController(pos_kp, pos_ki, pos_kd, pos_vel_limit)
        self._pos_pid_y = PIDController(pos_kp, pos_ki, pos_kd, pos_vel_limit)

        # ── Layer 2a: velocity (XY) → angle setpoints ─────────────────────
        self._vel_pid_x = PIDController(vel_kp, vel_ki, vel_kd, vel_angle_limit)
        self._vel_pid_y = PIDController(vel_kp, vel_ki, vel_kd, vel_angle_limit)

        # ── Layer 2b: altitude → thrust correction ─────────────────────────
        self._alt_pid = PIDController(alt_kp, alt_ki, alt_kd, alt_thrust_limit)

        # ── Layer 3: attitude → torques ────────────────────────────────────
        self._roll_pid  = PIDController(roll_kp,  roll_ki,  roll_kd,  roll_limit)
        self._pitch_pid = PIDController(pitch_kp, pitch_ki, pitch_kd, pitch_limit)
        self._yaw_pid   = PIDController(yaw_kp,   yaw_ki,   yaw_kd,   yaw_limit)

        # Desired attitude setpoints (updated by middle layer)
        self._roll_des  = 0.0
        self._pitch_des = 0.0

    def set_target(self, x: float, y: float, z: float, yaw: float = 0.0):
        self.target_pos[:] = [x, y, z]
        self.target_yaw    = yaw

    def step(self, dt: float):
        """Advance one simulation step. `dt` is the MuJoCo timestep."""
        pos = self._d.qpos[:3]
        vel = self._d.qvel[:3]
        qw, qx, qy, qz = (
            self._d.qpos[3], self._d.qpos[4],
            self._d.qpos[5], self._d.qpos[6],
        )
        roll, pitch, yaw = quat_to_euler_zyx(qw, qx, qy, qz)

        # ── Layer 1: position → desired velocity (slow loop) ──────────────
        if self._step % self._pos_div == 0:
            effective_dt = dt * self._pos_div
            g = 9.81

            # Layer 1: position error → desired velocity (clamped)
            des_vel_x = self._pos_pid_x.update(self.target_pos[0] - pos[0], effective_dt)
            des_vel_y = self._pos_pid_y.update(self.target_pos[1] - pos[1], effective_dt)
            des_vel_x = float(np.clip(des_vel_x, -self._max_speed_xy, self._max_speed_xy))
            des_vel_y = float(np.clip(des_vel_y, -self._max_speed_xy, self._max_speed_xy))

            # Layer 2a: velocity error → desired acceleration (clamped) → desired angles
            # pitch_des = +accel_x / g  (positive pitch → forward acceleration)
            # roll_des  = -accel_y / g  (positive roll  → rightward acceleration)
            accel_x = float(np.clip(
                self._vel_pid_x.update(des_vel_x - vel[0], effective_dt),
                -self._max_accel_xy, self._max_accel_xy,
            ))
            accel_y = float(np.clip(
                self._vel_pid_y.update(des_vel_y - vel[1], effective_dt),
                -self._max_accel_xy, self._max_accel_xy,
            ))

            self._pitch_des = float(np.clip(accel_x / g, -0.5, 0.5))
            self._roll_des  = float(np.clip(-accel_y / g, -0.5, 0.5))

        # ── Layer 2b: altitude control (medium loop) ───────────────────────
        # Clamp vertical velocity so altitude PID sees bounded feedback
        vel_z_clamped = float(np.clip(vel[2], -self._max_speed_z, self._max_speed_z))
        alt_error = self.target_pos[2] - pos[2]
        if self._step % self._vel_div == 0:
            alt_correction = self._alt_pid.update(alt_error, dt * self._vel_div)
        else:
            alt_correction = self._alt_pid.update(alt_error, dt)
        alt_correction = float(np.clip(
            alt_correction - vel_z_clamped * 0.5,   # soft damping on vertical speed
            -self._max_accel_z, self._max_accel_z,
        ))

        # ── Layer 3: attitude control (fast loop, every step) ─────────────
        roll_cmd  = self._roll_pid.update(self._roll_des  - roll,  dt)
        pitch_cmd = self._pitch_pid.update(self._pitch_des - pitch, dt)
        yaw_cmd   = self._yaw_pid.update(self.target_yaw   - yaw,  dt)

        # ── Motor mixing ───────────────────────────────────────────────────
        # Thrust per motor = hover_thrust + altitude_correction
        thrust = self.hover_thrust + alt_correction

        #   Motor layout (x2.xml, top view, drone front = +X, +Y = LEFT):
        #     ctrl[0] = thrust1  rear-right  (−x, −y)  gear yaw +0.0201 (CW)
        #     ctrl[1] = thrust2  rear-left   (−x, +y)  gear yaw −0.0201 (CCW)
        #     ctrl[2] = thrust3  front-left  (+x, +y)  gear yaw +0.0201 (CW)
        #     ctrl[3] = thrust4  front-right (+x, −y)  gear yaw −0.0201 (CCW)
        #
        #   Roll torque  = 0.18 * (ctrl[1]+ctrl[2] − ctrl[0]−ctrl[3])
        #                  → +roll_cmd increases left  (y=+0.18) motors
        #   Pitch torque = 0.14 * (ctrl[0]+ctrl[1] − ctrl[2]−ctrl[3])
        #                  → +pitch_cmd increases rear (x=−0.14) motors
        #   Yaw torque   = 0.0201 * (ctrl[0]−ctrl[1]+ctrl[2]−ctrl[3])
        #                  → +yaw_cmd increases CW motors (ctrl[0], ctrl[2])
        self._d.ctrl[0] = thrust - roll_cmd + pitch_cmd + yaw_cmd
        self._d.ctrl[1] = thrust + roll_cmd + pitch_cmd - yaw_cmd
        self._d.ctrl[2] = thrust + roll_cmd - pitch_cmd + yaw_cmd
        self._d.ctrl[3] = thrust - roll_cmd - pitch_cmd - yaw_cmd

        self._step += 1


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------
class SkydioSimNode(Node):
    """Skydio X2 MuJoCo simulation with cascaded PID and 3-D lidar."""

    def __init__(self):
        super().__init__('skydio_sim')

        # ── MuJoCo model path ─────────────────────────────────────────────
        self.declare_parameter('model_path', _DEFAULT_MODEL_PATH)

        # ── Reference position (from YAML / override topic) ────────────────
        self.declare_parameter('ref_x', 0.0)
        self.declare_parameter('ref_y', 0.0)
        self.declare_parameter('ref_z', 1.5)
        self.declare_parameter('ref_yaw', 0.0)

        # ── Publish rate ───────────────────────────────────────────────────
        self.declare_parameter('publish_rate_hz', 20.0)

        # ── Lidar ground removal ───────────────────────────────────────────
        # Points with world-frame z below this threshold are discarded.
        # Set to a value slightly above 0 to strip ground-plane returns.
        self.declare_parameter('ground_removal_z', 0.0)

        # ── Controller loop dividers (relative to sim steps) ───────────────
        self.declare_parameter('pos_loop_divider', 20)    # position loop rate
        self.declare_parameter('vel_loop_divider', 5)     # velocity / altitude loop rate

        # ── Kinematic limits ──────────────────────────────────────────────
        self.declare_parameter('max_speed_xy',  3.0)   # m/s horizontal
        self.declare_parameter('max_speed_z',   1.5)   # m/s vertical
        self.declare_parameter('max_accel_xy',  4.0)   # m/s² horizontal
        self.declare_parameter('max_accel_z',   3.0)   # m/s² vertical

        # ── Hover feed-forward (N per motor) ──────────────────────────────
        self.declare_parameter('hover_thrust', 3.2495625)

        # ── Position PID ──────────────────────────────────────────────────
        self.declare_parameter('pos_kp', 1.5)
        self.declare_parameter('pos_ki', 0.0)
        self.declare_parameter('pos_kd', 0.3)
        self.declare_parameter('pos_vel_limit', 2.0)

        # ── Altitude PID ──────────────────────────────────────────────────
        self.declare_parameter('alt_kp', 5.5)
        self.declare_parameter('alt_ki', 0.5)
        self.declare_parameter('alt_kd', 1.2)
        self.declare_parameter('alt_thrust_limit', 3.0)

        # ── Velocity → angle setpoint PIDs ────────────────────────────────
        self.declare_parameter('vel_kp', 0.1)
        self.declare_parameter('vel_ki', 0.003)
        self.declare_parameter('vel_kd', 0.02)
        self.declare_parameter('vel_angle_limit', 0.15)

        # ── Attitude PIDs ─────────────────────────────────────────────────
        self.declare_parameter('roll_kp', 2.5)
        self.declare_parameter('roll_ki', 0.5)
        self.declare_parameter('roll_kd', 1.2)
        self.declare_parameter('roll_limit', 1.0)

        self.declare_parameter('pitch_kp', 2.5)
        self.declare_parameter('pitch_ki', 0.5)
        self.declare_parameter('pitch_kd', 1.2)
        self.declare_parameter('pitch_limit', 1.0)

        self.declare_parameter('yaw_kp', 0.5)
        self.declare_parameter('yaw_ki', 0.0)
        self.declare_parameter('yaw_kd', 5.0)
        self.declare_parameter('yaw_limit', 3.0)

        # ── Wind and aerodynamic drag ──────────────────────────────────────
        self.declare_parameter('wind_x', 0.0)
        self.declare_parameter('wind_y', 0.0)
        self.declare_parameter('wind_z', 0.0)
        self.declare_parameter('wind_turbulence_std', 0.3)
        self.declare_parameter('wind_turbulence_tau', 2.0)
        self.declare_parameter('drag_coeff', 0.25)

        # ── Read all parameters ────────────────────────────────────────────
        p = self.get_parameters([
            'model_path',
            'ref_x', 'ref_y', 'ref_z', 'ref_yaw',
            'publish_rate_hz',
            'ground_removal_z',
            'wind_x', 'wind_y', 'wind_z',
            'wind_turbulence_std', 'wind_turbulence_tau',
            'drag_coeff',
            'pos_loop_divider', 'vel_loop_divider',
            'max_speed_xy', 'max_speed_z', 'max_accel_xy', 'max_accel_z',
            'hover_thrust',
            'pos_kp', 'pos_ki', 'pos_kd', 'pos_vel_limit',
            'alt_kp', 'alt_ki', 'alt_kd', 'alt_thrust_limit',
            'vel_kp', 'vel_ki', 'vel_kd', 'vel_angle_limit',
            'roll_kp',  'roll_ki',  'roll_kd',  'roll_limit',
            'pitch_kp', 'pitch_ki', 'pitch_kd', 'pitch_limit',
            'yaw_kp',   'yaw_ki',   'yaw_kd',   'yaw_limit',
        ])
        self._params = {param.name: param.value for param in p}

        # ── QoS ───────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publishers ─────────────────────────────────────────────────────
        self.pose_pub = self.create_publisher(PoseStamped, '/skydio/pose',           10)
        self.scan_pub = self.create_publisher(PointCloud2, '/skydio/scan3d',  sensor_qos)
        self.imu_pub  = self.create_publisher(Imu,         '/skydio/imu',     sensor_qos)
        self.ref_pub  = self.create_publisher(PoseStamped, '/skydio/reference_pose', 10)

        # Publish reference at steady rate (2 Hz)
        self.create_timer(0.5, self._publish_reference_pose)

        # ── Subscriber — runtime goal override ────────────────────────────
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)
        self._goal_updated = False

        self.get_logger().info(
            f"SkydioSimNode ready — reference = "
            f"({self._params['ref_x']}, {self._params['ref_y']}, {self._params['ref_z']})"
        )

    # ── Accessors ──────────────────────────────────────────────────────────

    def get_reference(self):
        return (
            self._params['ref_x'],
            self._params['ref_y'],
            self._params['ref_z'],
            self._params['ref_yaw'],
        )

    def build_controller(self, data: mujoco.MjData) -> CascadedPositionController:
        """Construct controller from current ROS2 parameters."""
        p = self._params
        return CascadedPositionController(
            data=data,
            pos_kp=p['pos_kp'],   pos_ki=p['pos_ki'],   pos_kd=p['pos_kd'],
            pos_vel_limit=p['pos_vel_limit'],
            alt_kp=p['alt_kp'],   alt_ki=p['alt_ki'],   alt_kd=p['alt_kd'],
            alt_thrust_limit=p['alt_thrust_limit'],
            vel_kp=p['vel_kp'],   vel_ki=p['vel_ki'],   vel_kd=p['vel_kd'],
            vel_angle_limit=p['vel_angle_limit'],
            roll_kp=p['roll_kp'],   roll_ki=p['roll_ki'],   roll_kd=p['roll_kd'],
            roll_limit=p['roll_limit'],
            pitch_kp=p['pitch_kp'], pitch_ki=p['pitch_ki'], pitch_kd=p['pitch_kd'],
            pitch_limit=p['pitch_limit'],
            yaw_kp=p['yaw_kp'],   yaw_ki=p['yaw_ki'],   yaw_kd=p['yaw_kd'],
            yaw_limit=p['yaw_limit'],
            hover_thrust=p['hover_thrust'],
            max_speed_xy=p['max_speed_xy'],
            max_speed_z=p['max_speed_z'],
            max_accel_xy=p['max_accel_xy'],
            max_accel_z=p['max_accel_z'],
            pos_divider=int(p['pos_loop_divider']),
            vel_divider=int(p['vel_loop_divider']),
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _goal_cb(self, msg: PoseStamped):
        self._params['ref_x'] = msg.pose.position.x
        self._params['ref_y'] = msg.pose.position.y
        self._params['ref_z'] = msg.pose.position.z
        self._goal_updated = True
        self.get_logger().info(
            f"Reference updated via /goal_pose: "
            f"({self._params['ref_x']:.2f}, "
            f"{self._params['ref_y']:.2f}, "
            f"{self._params['ref_z']:.2f})"
        )

    def _publish_reference_pose(self):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(self._params['ref_x'])
        msg.pose.position.y = float(self._params['ref_y'])
        msg.pose.position.z = float(self._params['ref_z'])
        msg.pose.orientation.w = 1.0
        self.ref_pub.publish(msg)

    # ── Sensor publishers ──────────────────────────────────────────────────

    def publish_pose(self, pos, quat_mj):
        """PoseStamped — MuJoCo quat [w,x,y,z] → ROS [x,y,z,w]."""
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat_mj[1])
        msg.pose.orientation.y = float(quat_mj[2])
        msg.pose.orientation.z = float(quat_mj[3])
        msg.pose.orientation.w = float(quat_mj[0])
        self.pose_pub.publish(msg)

    def publish_imu(self, sensordata):
        """
        Imu — sensor layout:
          [0:3]  gyro (angular velocity, body frame)
          [3:6]  accelerometer (linear acceleration, body frame)
          [6:10] framequat (orientation, MuJoCo [w,x,y,z])
        """
        msg = Imu()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'drone_base_link'
        msg.angular_velocity.x    = float(sensordata[0])
        msg.angular_velocity.y    = float(sensordata[1])
        msg.angular_velocity.z    = float(sensordata[2])
        msg.linear_acceleration.x = float(sensordata[3])
        msg.linear_acceleration.y = float(sensordata[4])
        msg.linear_acceleration.z = float(sensordata[5])
        # MuJoCo framequat [w,x,y,z] → ROS [x,y,z,w]
        msg.orientation.x = float(sensordata[7])
        msg.orientation.y = float(sensordata[8])
        msg.orientation.z = float(sensordata[9])
        msg.orientation.w = float(sensordata[6])
        self.imu_pub.publish(msg)

    def publish_scan3d(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Compute 3-D lidar point cloud in world frame.

        Ray direction = +Z axis of each rangefinder site = column 2 of
        site_xmat (row-major 3×3 → indices 2, 5, 8).

        Ground removal: hits with world-frame z < ground_removal_z are
        discarded to strip flat-ground returns before publishing.
        """
        hits = []
        sd   = data.sensordata
        ground_z: float = self._params['ground_removal_z']

        for i in range(NUM_LIDAR_RAYS):
            dist = sd[LIDAR_SENSOR_START + i]
            if dist <= 0.0 or dist >= LIDAR_CUTOFF_M:
                continue
            sid     = model.site(f'lidar3d_{i}').id
            xmat    = data.site_xmat[sid]
            ray_dir = np.array([xmat[2], xmat[5], xmat[8]])   # Z column
            hit     = data.site_xpos[sid] + dist * ray_dir
            # ── Ground removal ─────────────────────────────────────────────
            if hit[2] < ground_z:
                continue
            hits.append(hit)

        if not hits:
            return

        arr = np.array(hits, dtype=np.float32)
        hdr = Header()
        hdr.stamp    = self.get_clock().now().to_msg()
        hdr.frame_id = 'world'

        cloud = PointCloud2(
            header=hdr,
            height=1,
            width=len(arr),
            fields=[
                PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            ],
            is_bigendian=False,
            point_step=12,
            row_step=12 * len(arr),
            data=arr.tobytes(),
            is_dense=True,
        )
        self.scan_pub.publish(cloud)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = SkydioSimNode()

    model_path = node._params['model_path']

    if not os.path.isfile(model_path):
        node.get_logger().error(f'Model not found:\n  {model_path}')
        node.destroy_node()
        rclpy.shutdown()
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    node.get_logger().info(
        f'Model: {model.nq} qpos, {model.nu} actuators, '
        f'{model.nsensordata} sensor values'
    )
    node.get_logger().info(
        f'Lidar: {NUM_LIDAR_RAYS} rays '
        f'(sensor indices {LIDAR_SENSOR_START}–{LIDAR_SENSOR_START + NUM_LIDAR_RAYS - 1})'
    )

    ctrl = node.build_controller(data)
    ref  = node.get_reference()
    ctrl.set_target(*ref)

    dt_sim    = model.opt.timestep
    pub_every = max(1, int(1.0 / (dt_sim * node._params['publish_rate_hz'])))

    # ── Aerodynamics setup ───────────────────────────────────────────────
    drone_body_id = model.body('x2').id
    # Ornstein-Uhlenbeck turbulence state (world-frame velocity perturbation)
    _turb         = np.zeros(3)
    _turb_tau     = float(node._params['wind_turbulence_tau'])
    _turb_std     = float(node._params['wind_turbulence_std'])
    _wind_mean    = np.array([
        node._params['wind_x'],
        node._params['wind_y'],
        node._params['wind_z'],
    ], dtype=float)
    _drag_coeff   = float(node._params['drag_coeff'])

    print()
    print('=== Skydio X2 MuJoCo simulation ===')
    print(f'  Model     : {model_path}')
    print(f'  dt        : {dt_sim * 1000:.1f} ms')
    print(f'  Lidar     : {NUM_LIDAR_RAYS} rays '
          f'({NUM_LIDAR_ELEVATIONS} el × {NUM_LIDAR_AZIMUTHS} az), '
          f'cutoff {LIDAR_CUTOFF_M} m')
    print(f'  Reference : {ref[:3]}')
    print(f'  Wind      : mean={_wind_mean}, turb_std={_turb_std} m/s, tau={_turb_tau} s')
    print(f'  Drag coeff: {_drag_coeff} kg/m')
    print()
    print('ROS2 topics:')
    print('  pub  /skydio/pose           PoseStamped (drone pose)')
    print('  pub  /skydio/scan3d         PointCloud2 (3-D lidar hits)')
    print('  pub  /skydio/imu            Imu')
    print('  pub  /skydio/reference_pose PoseStamped (active reference @ 2 Hz)')
    print('  sub  /goal_pose             PoseStamped (runtime reference override)')
    print()
    print('Launching viewer…')

    _shutdown = [False]
    signal.signal(signal.SIGINT,  lambda *_: _shutdown.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: _shutdown.__setitem__(0, True))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running() and not _shutdown[0]:
            step_start = time.perf_counter()

            # ── ROS2 callbacks ─────────────────────────────────────────────
            rclpy.spin_once(node, timeout_sec=0)

            if node._goal_updated:
                ctrl.set_target(*node.get_reference())
                node._goal_updated = False

            # ── PID → actuators ────────────────────────────────────────────
            ctrl.step(dt_sim)
            # ── Wind + aerodynamic drag ────────────────────────────────────────
            # Ornstein-Uhlenbeck turbulence update
            if _turb_std > 0.0:
                alpha = dt_sim / _turb_tau
                _turb += (
                    -alpha * _turb
                    + _turb_std * math.sqrt(2.0 * alpha) * np.random.randn(3)
                )
            wind_vel = _wind_mean + _turb
            # Relative velocity of drone w.r.t. air (world frame)
            v_rel = data.qvel[:3] - wind_vel
            # Drag: F = -k * |v_rel| * v_rel  (quadratic, opposes relative motion)
            drag_force = -_drag_coeff * np.linalg.norm(v_rel) * v_rel
            data.xfrc_applied[drone_body_id, :3] = drag_force
            # ── Physics ────────────────────────────────────────────────────
            mujoco.mj_step(model, data)

            # ── Publish ────────────────────────────────────────────────────
            if step % pub_every == 0:
                node.publish_pose(data.qpos[:3], data.qpos[3:7])
                node.publish_imu(data.sensordata)
                node.publish_scan3d(model, data)

            # ── Status ─────────────────────────────────────────────────────
            if step % 200 == 0:
                pos  = data.qpos[:3]
                dist = float(np.linalg.norm(pos - ctrl.target_pos))
                sd   = data.sensordata
                hits = int(np.sum(
                    (sd[LIDAR_SENSOR_START: LIDAR_SENSOR_START + NUM_LIDAR_RAYS] > 0) &
                    (sd[LIDAR_SENSOR_START: LIDAR_SENSOR_START + NUM_LIDAR_RAYS] < LIDAR_CUTOFF_M)
                ))
                roll, pitch, yaw = quat_to_euler_zyx(
                    data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
                )
                print(
                    f't={data.time:6.2f}s | '
                    f'pos=[{pos[0]:5.2f} {pos[1]:5.2f} {pos[2]:5.2f}] | '
                    f'rpy=[{math.degrees(roll):5.1f}° {math.degrees(pitch):5.1f}° '
                    f'{math.degrees(yaw):5.1f}°] | '
                    f'dist={dist:.2f}m | hits={hits}'
                )

            # ── Real-time sync ─────────────────────────────────────────────
            # Throttle to wall-clock speed so motion looks natural in viewer.
            elapsed = time.perf_counter() - step_start
            remaining = dt_sim - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

            viewer.sync()
            step += 1

    node.get_logger().info('Simulation ended.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
