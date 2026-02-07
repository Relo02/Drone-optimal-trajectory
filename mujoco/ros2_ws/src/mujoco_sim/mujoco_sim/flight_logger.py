"""
Flight data logging for drone simulation.

Records state history, trajectory, and control inputs for analysis and debugging.
"""

import numpy as np
import json
import csv
import logging
from datetime import datetime
from pathlib import Path


class FlightLogger:
    """
    Records flight data during simulation.

    Tracks:
        - Time history
        - Position (x, y, z)
        - Orientation (roll, pitch, yaw)
        - Linear velocity (xdot, ydot, zdot)
        - Angular velocity (p, q, r)
        - Control inputs (thrust, moments)
        - Trajectory waypoints
    """

    def __init__(self, log_dir: str = None):
        """
        Initialize flight logger.

        Args:
            log_dir: Directory for log files. Defaults to ./logs/
        """
        # Setup log directory
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Session timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup Python logging
        self._setup_logging()

        # State history arrays
        self.time_history = []
        self.position_history = []      # [x, y, z]
        self.euler_history = []         # [roll, pitch, yaw]
        self.velocity_history = []      # [xdot, ydot, zdot]
        self.angular_vel_history = []   # [p, q, r]
        self.control_history = []       # [thrust, mx, my, mz]
        self.target_history = []        # [x, y, z] target position
        self.error_history = []         # position error magnitude

        # Trajectory waypoints
        self.trajectory_waypoints = []
        self.waypoint_times = []        # Time when each waypoint was reached

        # Metadata
        self.metadata = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_steps': 0,
            'trajectory_completed': False,
        }

        self.logger.info(f"Flight logger initialized. Session: {self.session_id}")

    def _setup_logging(self):
        """Configure Python logging."""
        self.logger = logging.getLogger(f"FlightLogger_{self.session_id}")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        log_file = self.log_dir / f"flight_{self.session_id}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def set_trajectory(self, waypoints):
        """
        Record the planned trajectory waypoints.

        Args:
            waypoints: List of [x, y, z] or [x, y, z, yaw] waypoints
        """
        self.trajectory_waypoints = [list(wp) for wp in waypoints]
        self.logger.info(f"Trajectory set: {len(waypoints)} waypoints")
        for i, wp in enumerate(waypoints):
            self.logger.debug(f"  WP{i}: {wp}")

    def log_state(self, time: float, state: dict, control: np.ndarray = None,
                  target_pos: np.ndarray = None, error: float = None):
        """
        Record current state.

        Args:
            time: Simulation time (seconds)
            state: Dict with 'position', 'euler', 'velocity', 'angular_velocity'
            control: Control input [thrust, mx, my, mz]
            target_pos: Current target position [x, y, z]
            error: Position error magnitude
        """
        self.time_history.append(time)
        self.position_history.append(state['position'].tolist())
        self.euler_history.append(state['euler'].tolist())
        self.velocity_history.append(state['velocity'].tolist())
        self.angular_vel_history.append(state['angular_velocity'].tolist())

        if control is not None:
            self.control_history.append(control.tolist())
        else:
            self.control_history.append([0, 0, 0, 0])

        if target_pos is not None:
            self.target_history.append(target_pos.tolist())
        else:
            self.target_history.append([0, 0, 0])

        if error is not None:
            self.error_history.append(error)
        else:
            self.error_history.append(0)

        self.metadata['total_steps'] = len(self.time_history)

    def log_waypoint_reached(self, waypoint_idx: int, time: float, position: np.ndarray):
        """Record when a waypoint is reached."""
        self.waypoint_times.append({
            'index': waypoint_idx,
            'time': time,
            'position': position.tolist()
        })
        self.logger.info(f"Waypoint {waypoint_idx} reached at t={time:.2f}s, "
                        f"pos=[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")

    def log_event(self, event: str, level: str = "info"):
        """Log a custom event."""
        getattr(self.logger, level)(event)

    def finalize(self, trajectory_completed: bool = False):
        """
        Finalize logging and save data.

        Args:
            trajectory_completed: Whether the full trajectory was completed
        """
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['trajectory_completed'] = trajectory_completed

        # Calculate statistics
        if len(self.position_history) > 0:
            positions = np.array(self.position_history)
            self.metadata['stats'] = {
                'duration_s': self.time_history[-1] if self.time_history else 0,
                'max_altitude': float(np.max(positions[:, 2])),
                'total_distance': float(self._calculate_path_length(positions)),
                'mean_error': float(np.mean(self.error_history)) if self.error_history else 0,
                'max_error': float(np.max(self.error_history)) if self.error_history else 0,
            }

            if len(self.velocity_history) > 0:
                velocities = np.array(self.velocity_history)
                speeds = np.linalg.norm(velocities, axis=1)
                self.metadata['stats']['max_speed'] = float(np.max(speeds))
                self.metadata['stats']['mean_speed'] = float(np.mean(speeds))

        # Save files
        self._save_csv()
        self._save_json()

        self.logger.info(f"Flight data saved. Duration: {self.metadata.get('stats', {}).get('duration_s', 0):.2f}s")
        self.logger.info(f"Log files: {self.log_dir}")

    def _calculate_path_length(self, positions: np.ndarray) -> float:
        """Calculate total path length."""
        if len(positions) < 2:
            return 0.0
        diffs = np.diff(positions, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def _save_csv(self):
        """Save state history to CSV."""
        csv_file = self.log_dir / f"state_history_{self.session_id}.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
                'xdot', 'ydot', 'zdot', 'p', 'q', 'r',
                'thrust', 'mx', 'my', 'mz',
                'target_x', 'target_y', 'target_z', 'error'
            ])

            # Data rows
            for i in range(len(self.time_history)):
                pos = self.position_history[i]
                euler = self.euler_history[i]
                vel = self.velocity_history[i]
                ang_vel = self.angular_vel_history[i]
                ctrl = self.control_history[i]
                target = self.target_history[i]

                writer.writerow([
                    f"{self.time_history[i]:.4f}",
                    f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                    f"{euler[0]:.6f}", f"{euler[1]:.6f}", f"{euler[2]:.6f}",
                    f"{vel[0]:.6f}", f"{vel[1]:.6f}", f"{vel[2]:.6f}",
                    f"{ang_vel[0]:.6f}", f"{ang_vel[1]:.6f}", f"{ang_vel[2]:.6f}",
                    f"{ctrl[0]:.6f}", f"{ctrl[1]:.6f}", f"{ctrl[2]:.6f}", f"{ctrl[3]:.6f}",
                    f"{target[0]:.6f}", f"{target[1]:.6f}", f"{target[2]:.6f}",
                    f"{self.error_history[i]:.6f}"
                ])

        self.logger.debug(f"CSV saved: {csv_file}")

    def _save_json(self):
        """Save full flight data to JSON."""
        json_file = self.log_dir / f"flight_data_{self.session_id}.json"

        data = {
            'metadata': self.metadata,
            'trajectory': {
                'waypoints': self.trajectory_waypoints,
                'waypoint_times': self.waypoint_times,
            },
            'state_history': {
                'time': self.time_history,
                'position': self.position_history,
                'euler': self.euler_history,
                'velocity': self.velocity_history,
                'angular_velocity': self.angular_vel_history,
                'control': self.control_history,
                'target': self.target_history,
                'error': self.error_history,
            }
        }

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.debug(f"JSON saved: {json_file}")

    def get_trajectory_array(self) -> np.ndarray:
        """Get recorded trajectory as numpy array."""
        return np.array(self.position_history)

    def get_state_dataframe(self):
        """
        Get state history as pandas DataFrame (if pandas available).

        Returns:
            DataFrame or None if pandas not available
        """
        try:
            import pandas as pd

            data = {
                'time': self.time_history,
                'x': [p[0] for p in self.position_history],
                'y': [p[1] for p in self.position_history],
                'z': [p[2] for p in self.position_history],
                'roll': [e[0] for e in self.euler_history],
                'pitch': [e[1] for e in self.euler_history],
                'yaw': [e[2] for e in self.euler_history],
                'xdot': [v[0] for v in self.velocity_history],
                'ydot': [v[1] for v in self.velocity_history],
                'zdot': [v[2] for v in self.velocity_history],
                'error': self.error_history,
            }
            return pd.DataFrame(data)
        except ImportError:
            return None
