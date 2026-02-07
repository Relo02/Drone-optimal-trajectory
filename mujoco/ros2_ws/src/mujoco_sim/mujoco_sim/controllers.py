"""
Low-level controllers for CF2 quadrotor drone.

Cascaded control structure:
    Position Controller → Attitude Controller → Motor Commands

Usage:
    controller = DroneController(model, data)
    controller.set_target_position([x, y, z])

    while running:
        controller.update()
        mujoco.mj_step(model, data)
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class AttitudeController:
    """
    PD attitude controller.
    Converts desired attitude (roll, pitch, yaw) to body moments.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

        # PD gains for attitude control
        self.Kp_roll = 0.003
        self.Kp_pitch = 0.003
        self.Kp_yaw = 0.001

        self.Kd_roll = 0.0006
        self.Kd_pitch = 0.0006
        self.Kd_yaw = 0.0003

    def quaternion_to_euler(self, quat):
        """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]."""
        w, x, y, z = quat

        # # Roll (x-axis rotation)
        # sinr_cosp = 2 * (w * x + y * z)
        # cosr_cosp = 1 - 2 * (x * x + y * y)
        # roll = np.arctan2(sinr_cosp, cosr_cosp)

        # # Pitch (y-axis rotation)
        # sinp = 2 * (w * y - z * x)
        # if abs(sinp) >= 1:
        #     pitch = np.copysign(np.pi / 2, sinp)
        # else:
        #     pitch = np.arcsin(sinp)

        # # Yaw (z-axis rotation)
        # siny_cosp = 2 * (w * z + x * y)
        # cosy_cosp = 1 - 2 * (y * y + z * z)
        # yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Convert from MuJoCo [w, x, y, z] to SciPy [x, y, z, w]
        scipy_quat = [quat[1], quat[2], quat[3], quat[0]]
        euler_angles = R.from_quat(scipy_quat).as_euler('xyz', degrees=False)

        return np.array(euler_angles)

    def compute(self, desired_roll, desired_pitch, desired_yaw):
        """
        Compute body moments to achieve desired attitude.

        Args:
            desired_roll: Desired roll angle (rad)
            desired_pitch: Desired pitch angle (rad)
            desired_yaw: Desired yaw angle (rad)

        Returns:
            moments: [mx, my, mz] body moments
        """
        # Get current attitude
        quat = self.data.qpos[3:7]
        current_euler = self.quaternion_to_euler(quat)

        # Get angular velocity (body frame)
        omega = self.data.qvel[3:6]

        # Attitude errors
        roll_err = desired_roll - current_euler[0]
        pitch_err = desired_pitch - current_euler[1]
        yaw_err = desired_yaw - current_euler[2]

        # Wrap yaw error to [-pi, pi]
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))

        # PD control
        mx = self.Kp_roll * roll_err - self.Kd_roll * omega[0]
        my = self.Kp_pitch * pitch_err - self.Kd_pitch * omega[1]
        mz = self.Kp_yaw * yaw_err - self.Kd_yaw * omega[2]

        return np.array([mx, my, mz])


class PositionController:
    """
    PD position controller.
    Converts desired position to desired thrust and attitude.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Drone parameters
        self.mass = 0.027  # kg
        self.g = 9.81  # m/s^2

        # PD gains for position control
        self.Kp_xy = 1.0
        self.Kp_z = 2.0

        self.Kd_xy = 1.5
        self.Kd_z = 2.0

        # Limits
        self.max_tilt = np.radians(40)  # Max roll/pitch angle
        self.max_thrust = 1.5  # N (from actuator limits)
        self.max_vel_xy = 0.3  # m/s
        self.max_vel_z = 0.3   # m/s

    def compute(self, desired_pos, desired_yaw=0.0):
        """max_thrust
        Compute thrust and desired attitude to reach target position.

        Args:
            desired_pos: [x, y, z] target position
            desired_yaw: Desired yaw angle (rad)

        Returns:
            thrust: Total thrust (N)
            desired_roll: Desired roll angle (rad)
            desired_pitch: Desired pitch angle (rad)
            desired_yaw: Desired yaw angle (rad)
        """
        # Get current state
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]

        # Position error
        pos_err = np.array(desired_pos) - pos

        # Velocity command from position error, clamped to speed limits
        vel_cmd = np.array([
            np.clip(self.Kp_xy * pos_err[0], -self.max_vel_xy, self.max_vel_xy),
            np.clip(self.Kp_xy * pos_err[1], -self.max_vel_xy, self.max_vel_xy),
            np.clip(self.Kp_z * pos_err[2], -self.max_vel_z, self.max_vel_z),
        ])

        # Desired acceleration (velocity tracking)
        acc_des = np.zeros(3)
        acc_des[0] = self.Kd_xy * (vel_cmd[0] - vel[0])
        acc_des[1] = self.Kd_xy * (vel_cmd[1] - vel[1])
        acc_des[2] = self.Kd_z * (vel_cmd[2] - vel[2]) + self.g

        # Compute thrust magnitude
        thrust = self.mass * acc_des[2]           #np.linalg.norm(acc_des)
        thrust = np.clip(thrust, 0, self.max_thrust)

        # Compute desired attitude from desired acceleration
        # Using small angle approximation for roll/pitch
        if acc_des[2] > 0.1:
            # Desired roll and pitch to achieve horizontal acceleration
            # desired_roll = (acc_des[1] * np.cos(desired_yaw) - acc_des[0] * np.sin(desired_yaw)) / acc_des[2]
            # desired_pitch = (acc_des[0] * np.cos(desired_yaw) + acc_des[1] * np.sin(desired_yaw)) / acc_des[2]
            desired_pitch = acc_des[0] / self.g
            desired_roll = -acc_des[1] / self.g
        else:
            desired_roll = 0
            desired_pitch = 0

        # Clamp angles
        desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)

        return thrust, desired_roll, desired_pitch, desired_yaw


class DroneController:
    """
    Complete cascaded drone controller.
    Combines position and attitude control.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.position_ctrl = PositionController(model, data)
        self.attitude_ctrl = AttitudeController(model, data)

        # Target state
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0

        # Control mode
        self.position_control_enabled = True

        # Aerodynamic drag coefficients
        self.k_drag_linear = 0.01     # N·s/m
        self.k_drag_angular = 0.0002  # N·m·s/rad

        # Motor dynamics (first-order lag)
        self.motor_tau = 0.02  # 20ms time constant
        self.ctrl_filtered = np.zeros(4)

        # Propeller visual spinning
        self.hover_thrust = self.position_ctrl.mass * self.position_ctrl.g
        self.prop_speed_hover = 1500.0  # rad/s (~14,000 RPM)

    def set_target_position(self, pos, yaw=None):
        """Set target position [x, y, z] and optionally yaw."""
        self.target_pos = np.array(pos)
        if yaw is not None:
            self.target_yaw = yaw

    def set_target_velocity(self, vel, dt=0.02):
        """Set target velocity by moving target position."""
        self.target_pos += np.array(vel) * dt

    def update(self):
        """
        Run one control cycle.
        Call this before mujoco.mj_step().
        """
        if self.position_control_enabled:
            # Position controller → thrust + desired attitude
            thrust, des_roll, des_pitch, des_yaw = self.position_ctrl.compute(
                self.target_pos, self.target_yaw
            )

            # Attitude controller → moments
            moments = self.attitude_ctrl.compute(des_roll, des_pitch, des_yaw)

            # Motor lag filter (first-order)
            dt = self.model.opt.timestep
            alpha = dt / (self.motor_tau + dt)
            ctrl_raw = np.array([thrust, moments[0], moments[1], moments[2]])
            self.ctrl_filtered += alpha * (ctrl_raw - self.ctrl_filtered)

            # Apply filtered control outputs
            self.data.ctrl[0] = self.ctrl_filtered[0]
            self.data.ctrl[1] = self.ctrl_filtered[1]
            self.data.ctrl[2] = self.ctrl_filtered[2]
            self.data.ctrl[3] = self.ctrl_filtered[3]

            # Drive propeller spin proportional to thrust
            thrust_ratio = max(self.ctrl_filtered[0], 0) / self.hover_thrust
            prop_speed = self.prop_speed_hover * np.sqrt(max(thrust_ratio, 0))
            self.data.ctrl[4] = -prop_speed   # FR (CW)
            self.data.ctrl[5] = prop_speed    # FL (CCW)
            self.data.ctrl[6] = -prop_speed   # BL (CW)
            self.data.ctrl[7] = prop_speed    # BR (CCW)

        # Aerodynamic drag (applied every step regardless of control mode)
        vel = self.data.qvel[:3]
        omega = self.data.qvel[3:6]
        self.data.qfrc_applied[:3] = -self.k_drag_linear * vel
        self.data.qfrc_applied[3:6] = -self.k_drag_angular * omega

        return self.get_state()

    def get_state(self):
        """Get current drone state."""
        return {
            'position': self.data.qpos[:3].copy(),
            'orientation': self.data.qpos[3:7].copy(),
            'velocity': self.data.qvel[:3].copy(),
            'angular_velocity': self.data.qvel[3:6].copy(),
            'euler': self.attitude_ctrl.quaternion_to_euler(self.data.qpos[3:7]),
        }

    def get_position_error(self):
        """Get position error magnitude."""
        return np.linalg.norm(self.target_pos - self.data.qpos[:3])


class TrajectoryTracker:
    """
    Trajectory tracking controller for later on MPC integration.
    Accepts a sequence of waypoints and tracks them.
    """

    def __init__(self, controller):
        self.controller = controller
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_threshold = 0.2  # meters - increased for smoother transitions

    def set_trajectory(self, waypoints):
        """
        Set trajectory as list of waypoints.

        Args:
            waypoints: List of [x, y, z] positions or [x, y, z, yaw]
        """
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.current_waypoint_idx = 0

        if len(self.waypoints) > 0:
            wp = self.waypoints[0]
            yaw = wp[3] if len(wp) > 3 else 0.0
            self.controller.set_target_position(wp[:3], yaw)

    def update(self):
        """Update trajectory tracking."""
        if len(self.waypoints) == 0:
            return self.controller.update()

        # Check if reached current waypoint (only if not already complete)
        if self.current_waypoint_idx < len(self.waypoints):
            error = self.controller.get_position_error()

            if error < self.waypoint_threshold:
                # Move to next waypoint
                self.current_waypoint_idx += 1

                if self.current_waypoint_idx < len(self.waypoints):
                    wp = self.waypoints[self.current_waypoint_idx]
                    yaw = wp[3] if len(wp) > 3 else self.controller.target_yaw
                    self.controller.set_target_position(wp[:3], yaw)

        return self.controller.update()

    def is_complete(self):
        """Check if trajectory is complete."""
        return self.current_waypoint_idx >= len(self.waypoints)

    def get_progress(self):
        """Get trajectory progress (0.0 to 1.0)."""
        if len(self.waypoints) == 0:
            return 1.0
        return self.current_waypoint_idx / len(self.waypoints)
