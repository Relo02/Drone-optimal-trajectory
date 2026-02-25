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

import time
import threading
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

try:
    from mujoco_sim.MPC import MPCLocalPlanner, DroneState, MPCConfig
    _MPC_AVAILABLE = True
except ImportError:
    _MPC_AVAILABLE = False

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
        self.max_vel_xy = 0.1  # m/s
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

class MPCDroneController:
    """
    Cascaded drone controller driven by the MPC local planner (MPC.py).

    The MPC operates on the 2-D horizontal plane and outputs at every step:
        first_control = [ax, ay, yaw_rate]   (horizontal accelerations + yaw rate)

    This controller:
      - Feeds ax, ay directly to the attitude conversion (no position-PD cascade)
      - Retains a simple PD loop for altitude (z), which the 2-D MPC does not handle
      - Integrates yaw_rate to obtain desired_yaw for the inner AttitudeController
      - Leaves the motor lag filter, drag model and propeller visuals unchanged

    Typical usage::

        ctrl = MPCDroneController(model, data)
        ctrl.set_target_altitude(1.5)
        while running:
            state = ctrl.update(goal_xy=[gx, gy], lidar_points=scan)
            mujoco.mj_step(model, data)
    """

    def __init__(self, model, data, mpc_config=None):
        if not _MPC_AVAILABLE:
            raise ImportError(
                "CasADi / MPC.py not available. "
                "Install casadi and ensure mujoco_sim.MPC is on the path."
            )

        self.model = model
        self.data = data

        # ── Inner-loop attitude controller (unchanged) ──
        self.attitude_ctrl = AttitudeController(model, data)

        # ── MPC local planner (2-D: x, y, yaw) ──
        self.mpc_planner = MPCLocalPlanner(mpc_config)

        # ── Physical parameters ──
        self.mass = 0.027        # kg
        self.g = 9.81            # m/s²

        # ── Altitude PD gains (z not handled by 2-D MPC) ──
        self.Kp_z = 2.0
        self.Kd_z = 2.0
        self.max_vel_z = 0.3     # m/s
        self.target_z = 1.5      # m  — desired altitude

        # ── Limits ──
        self.max_tilt   = np.radians(40)
        self.max_thrust = 1.5    # N

        # ── Aerodynamic drag ──
        self.k_drag_linear  = 0.01      # N·s/m
        self.k_drag_angular = 0.0002    # N·m·s/rad

        # ── Motor dynamics (first-order lag) ──
        self.motor_tau    = 0.02        # 20 ms time constant
        self.ctrl_filtered = np.zeros(4)

        # ── Propeller visual spin ──
        self.hover_thrust    = self.mass * self.g
        self.prop_speed_hover = 1500.0  # rad/s

        # ── Async MPC thread ──
        # The MPC solver (IPOPT) is slow (~10-100 ms wall-clock) and runs at
        # cfg.dt = 0.1 s intervals.  MuJoCo steps at 0.002 s (500 Hz).
        # We decouple them: a background thread runs the MPC continuously at
        # its own rate while the main loop only reads the latest cached command.
        #
        #   _cmd_lock   — protects cached [ax, ay, yaw_rate], goal, lidar
        #   _state_lock — protects the state snapshot written by the main thread
        self._cmd_lock   = threading.Lock()
        self._state_lock = threading.Lock()

        # Cached MPC horizontal command (written by MPC thread, read by main)
        self._cached_ax       = 0.0
        self._cached_ay       = 0.0
        self._cached_yaw_rate = 0.0
        self._desired_yaw     = 0.0   # yaw setpoint (rad), integrated in MPC thread

        # State snapshot (written by main thread, read by MPC thread)
        self._snap_pos  = np.zeros(3)
        self._snap_vel  = np.zeros(3)
        self._snap_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Goal + lidar (written by main thread, read by MPC thread)
        self._goal_xy   = np.zeros(2)
        self._lidar_pts = None

        # Thread control
        self._stop_event  = threading.Event()
        self._mpc_thread  = None

        # ── Diagnostics ──
        self.last_mpc_result = None

    # ─────────────────────────────────────────
    # Background MPC thread
    # ─────────────────────────────────────────
    def start(self):
        """Start the background MPC solver thread."""
        self._stop_event.clear()
        self._mpc_thread = threading.Thread(
            target=self._mpc_loop, daemon=True, name="mpc_solver"
        )
        self._mpc_thread.start()

    def stop(self):
        """Signal the MPC thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._mpc_thread is not None:
            self._mpc_thread.join(timeout=5.0)
            self._mpc_thread = None

    def _mpc_loop(self):
        """
        Background loop: snapshot state → solve IPOPT → update cached cmd.

        Yaw is integrated HERE with the actual wall-clock elapsed time so it
        is never applied for longer than the true inter-solve period, fixing
        the 2× over-rotation that occurs when IPOPT overruns cfg.dt.
        """
        mpc_dt = self.mpc_planner.cfg.dt
        a_max  = self.mpc_planner.cfg.a_max
        _fail_count = 0
        t_prev = time.perf_counter()

        while not self._stop_event.is_set():
            t_now = time.perf_counter()
            # Actual elapsed time since the previous solve started (capped at 1 s
            # to avoid a large spike on the very first iteration).
            actual_dt = min(t_now - t_prev, 1.0)
            t_prev = t_now

            # 0. Integrate the *previous* yaw_rate over the actual elapsed period.
            #    This is the only place yaw is accumulated, so it is always
            #    proportional to real time, not to sim-step count.
            with self._cmd_lock:
                self._desired_yaw += self._cached_yaw_rate * actual_dt
                self._desired_yaw = float(np.arctan2(
                    np.sin(self._desired_yaw), np.cos(self._desired_yaw)
                ))

            # 1. Snapshot current state and goal (fast, locked)
            with self._state_lock:
                pos  = self._snap_pos.copy()
                vel  = self._snap_vel.copy()
                quat = self._snap_quat.copy()
            with self._cmd_lock:
                goal_xy   = self._goal_xy.copy()
                lidar_pts = self._lidar_pts   # already copied by main thread

            # 2. Build DroneState from snapshot (no lock needed)
            scipy_quat = [quat[1], quat[2], quat[3], quat[0]]
            euler = R.from_quat(scipy_quat).as_euler('xyz', degrees=False)
            drone_state = DroneState(
                x=float(pos[0]),  y=float(pos[1]),  z=float(pos[2]),
                vx=float(vel[0]), vy=float(vel[1]),
                yaw=float(euler[2]),
            )

            # 3. Solve MPC — slow IPOPT call; NO lock held during solve
            try:
                result = self.mpc_planner.plan(drone_state, goal_xy, lidar_pts)
                self.last_mpc_result = result
                if result.success:
                    ax, ay, yr = result.first_control
                    _fail_count = 0
                else:
                    # IPOPT returned best iterate but did not converge:
                    # fall back to velocity damping to stop uncontrolled drift.
                    _fail_count += 1
                    ax = float(np.clip(-2.0 * vel[0], -a_max, a_max))
                    ay = float(np.clip(-2.0 * vel[1], -a_max, a_max))
                    yr = 0.0
            except Exception:
                _fail_count += 1
                ax = float(np.clip(-2.0 * vel[0], -a_max, a_max))
                ay = float(np.clip(-2.0 * vel[1], -a_max, a_max))
                yr = 0.0

            # Clear warm-start after 3 consecutive failures so IPOPT gets a
            # fresh initial guess instead of starting from a diverged iterate.
            if _fail_count >= 3:
                self.mpc_planner._prev_u_flat = None
                self.mpc_planner._prev_x_flat = None
                _fail_count = 0

            # 4. Update cached command (fast, locked)
            with self._cmd_lock:
                self._cached_ax       = float(ax)
                self._cached_ay       = float(ay)
                self._cached_yaw_rate = float(yr)   # applied next iteration

            # 5. Sleep the remainder of the MPC period
            elapsed = time.perf_counter() - t_now
            sleep_time = mpc_dt - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────
    def _get_drone_state(self) -> "DroneState":
        """Build a DroneState from the current MuJoCo data."""
        pos  = self.data.qpos[:3]
        vel  = self.data.qvel[:3]
        euler = self.attitude_ctrl.quaternion_to_euler(self.data.qpos[3:7])
        return DroneState(
            x=float(pos[0]),  y=float(pos[1]),  z=float(pos[2]),
            vx=float(vel[0]), vy=float(vel[1]),
            yaw=float(euler[2]),
        )

    def set_target_altitude(self, z: float):
        """Set the desired altitude for the z PD loop."""
        self.target_z = z

    # ─────────────────────────────────────────
    # Main update
    # ─────────────────────────────────────────
    def update(self, goal_xy, lidar_points=None):
        """
        Run one MPC + attitude-control cycle.

        Args:
            goal_xy:      [gx, gy] or [gx, gy, g_yaw] 2-D goal for the MPC.
            lidar_points: (M, 3) LiDAR scan in world frame, or None to reuse
                          the last scan.

        Returns:
            state dict (position, orientation, velocity, …)
        """
        dt = self.model.opt.timestep

        # ── 1. Current state ──
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]

        # ── 2a. Push current state snapshot to MPC thread (fast, locked) ──
        with self._state_lock:
            self._snap_pos[:]  = pos
            self._snap_vel[:]  = vel
            self._snap_quat[:] = self.data.qpos[3:7]

        # ── 2b. Push new goal / lidar to MPC thread (fast, locked) ──
        with self._cmd_lock:
            self._goal_xy[:] = goal_xy[:2]
            if lidar_points is not None:
                self._lidar_pts = lidar_points.copy()   # copy so main thread can reuse array

        # ── 2c. Read latest cached MPC command (fast, locked) ──
        with self._cmd_lock:
            ax_mpc      = self._cached_ax
            ay_mpc      = self._cached_ay
            desired_yaw = self._desired_yaw  # integrated with correct timing in MPC thread

        # ── 3. Altitude PD (z is outside the 2-D MPC scope) ──
        z_err     = self.target_z - float(pos[2])
        vel_cmd_z = np.clip(self.Kp_z * z_err, -self.max_vel_z, self.max_vel_z)
        az        = self.Kd_z * (vel_cmd_z - float(vel[2])) + self.g

        # ── 4. Full 3-D desired acceleration ──
        # ax, ay come directly from MPC; az from altitude PD
        acc_des = np.array([ax_mpc, ay_mpc, az])

        # ── 5. Thrust magnitude ──
        thrust = self.mass * acc_des[2]
        thrust = np.clip(thrust, 0.0, self.max_thrust)

        # ── 6. Desired roll / pitch from horizontal MPC acceleration ──
        if acc_des[2] > 0.1:
            desired_pitch =  acc_des[0] / self.g
            desired_roll  = -acc_des[1] / self.g
        else:
            desired_pitch = 0.0
            desired_roll  = 0.0

        desired_roll  = np.clip(desired_roll,  -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)

        # ── 7. desired_yaw is integrated in the MPC thread with wall-clock timing ──

        # ── 8. Attitude controller → body moments ──
        moments = self.attitude_ctrl.compute(
            desired_roll, desired_pitch, desired_yaw
        )

        # ── 9. First-order motor lag filter ──
        alpha    = dt / (self.motor_tau + dt)
        ctrl_raw = np.array([thrust, moments[0], moments[1], moments[2]])
        self.ctrl_filtered += alpha * (ctrl_raw - self.ctrl_filtered)

        self.data.ctrl[0] = self.ctrl_filtered[0]  # thrust
        self.data.ctrl[1] = self.ctrl_filtered[1]  # mx
        self.data.ctrl[2] = self.ctrl_filtered[2]  # my
        self.data.ctrl[3] = self.ctrl_filtered[3]  # mz

        # ── 10. Propeller visual spin ──
        thrust_ratio = max(self.ctrl_filtered[0], 0.0) / self.hover_thrust
        prop_speed   = self.prop_speed_hover * np.sqrt(max(thrust_ratio, 0.0))
        self.data.ctrl[4] = -prop_speed   # FR  (CW)
        self.data.ctrl[5] =  prop_speed   # FL  (CCW)
        self.data.ctrl[6] = -prop_speed   # BL  (CW)
        self.data.ctrl[7] =  prop_speed   # BR  (CCW)

        # ── 11. Aerodynamic drag (world frame) ──
        self.data.qfrc_applied[:3]  = -self.k_drag_linear  * vel
        self.data.qfrc_applied[3:6] = -self.k_drag_angular * self.data.qvel[3:6]

        return self.get_state()

    def get_state(self):
        """Get current drone state plus MPC diagnostics."""
        return {
            'position':         self.data.qpos[:3].copy(),
            'orientation':      self.data.qpos[3:7].copy(),
            'velocity':         self.data.qvel[:3].copy(),
            'angular_velocity': self.data.qvel[3:6].copy(),
            'euler':            self.attitude_ctrl.quaternion_to_euler(
                                    self.data.qpos[3:7]),
            'mpc_success':      self.last_mpc_result.success
                                if self.last_mpc_result else False,
            'mpc_solve_ms':     self.last_mpc_result.solve_time_ms
                                if self.last_mpc_result else 0.0,
            'mpc_first_control': self.last_mpc_result.first_control.copy()
                                if self.last_mpc_result is not None else np.zeros(3),
        }


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
        Set trajectory as list of waypoints (full reset).

        Args:
            waypoints: List of [x, y, z] positions or [x, y, z, yaw]
        """
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.current_waypoint_idx = 0

        if len(self.waypoints) > 0:
            wp = self.waypoints[0]
            yaw = wp[3] if len(wp) > 3 else 0.0
            self.controller.set_target_position(wp[:3], yaw)

    def update_waypoints(self, waypoints):
        """
        Replace the waypoint list for a new MPC replan.

        Because each MPC solve produces a fresh trajectory starting from
        *near* the drone's current position (index 0 = next step), we
        reset the index to 0 so the tracker starts chasing the first
        new waypoint.  This differs from set_trajectory() only in that
        it is a deliberate "replan update" entry point.

        Args:
            waypoints: List of [x, y, z] positions or [x, y, z, yaw]
        """
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.current_waypoint_idx = 0

        if len(self.waypoints) > 0:
            wp = self.waypoints[0]
            yaw = wp[3] if len(wp) > 3 else self.controller.target_yaw
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
