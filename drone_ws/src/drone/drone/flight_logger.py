"""
Flight Data Logger and Visualization System

Logs flight data during execution and provides post-flight analysis
with detailed plots showing:
- Occupancy grid evolution
- MPC performance metrics
- Trajectory tracking
- Waypoint following

Author: Flight Analysis Team
"""

import numpy as np
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import time


@dataclass
class FlightDataPoint:
    """Single timestep of flight data."""
    timestamp: float

    # State
    position: List[float]
    velocity: List[float]
    yaw: float
    yaw_rate: float

    # Goals
    active_goal: List[float]
    final_goal: List[float]

    # Waypoints
    num_waypoints: int
    current_waypoint: Optional[List[float]]

    # MPC metrics
    mpc_solve_time_ms: float
    mpc_cost: float
    mpc_status: str

    # Obstacles
    num_raw_obstacles: int
    num_clustered_obstacles: int

    # Distances
    distance_to_active_goal: float
    distance_to_final_goal: float

    # Control
    acceleration_cmd: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert None to null for JSON
        if self.current_waypoint is None:
            data['current_waypoint'] = None
        return data


@dataclass
class GridSnapshot:
    """Snapshot of occupancy grid at a timestep."""
    timestamp: float
    grid: np.ndarray
    drone_position: List[float]
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int


class FlightLogger:
    """
    Comprehensive flight data logger.

    Records all relevant flight data for post-flight analysis and visualization.
    """

    def __init__(self, log_dir: str = "/tmp/drone_logs"):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)

        # Data storage
        self.flight_data: List[FlightDataPoint] = []
        self.grid_snapshots: List[GridSnapshot] = []

        # Metadata
        self.metadata = {
            'session_id': self.session_id,
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
        }

        print(f"FlightLogger initialized. Logs will be saved to: {self.session_dir}")

    def log_flight_data(
        self,
        timestamp: float,
        position: np.ndarray,
        velocity: np.ndarray,
        yaw: float,
        yaw_rate: float,
        active_goal: np.ndarray,
        final_goal: np.ndarray,
        num_waypoints: int,
        current_waypoint: Optional[np.ndarray],
        mpc_solve_time_ms: float,
        mpc_cost: float,
        mpc_status: str,
        num_raw_obstacles: int,
        num_clustered_obstacles: int,
        distance_to_active_goal: float,
        distance_to_final_goal: float,
        acceleration_cmd: np.ndarray
    ):
        """Log a single timestep of flight data."""

        data_point = FlightDataPoint(
            timestamp=timestamp,
            position=position.tolist(),
            velocity=velocity.tolist(),
            yaw=float(yaw),
            yaw_rate=float(yaw_rate),
            active_goal=active_goal.tolist(),
            final_goal=final_goal.tolist(),
            num_waypoints=num_waypoints,
            current_waypoint=current_waypoint.tolist() if current_waypoint is not None else None,
            mpc_solve_time_ms=float(mpc_solve_time_ms),
            mpc_cost=float(mpc_cost),
            mpc_status=mpc_status,
            num_raw_obstacles=num_raw_obstacles,
            num_clustered_obstacles=num_clustered_obstacles,
            distance_to_active_goal=float(distance_to_active_goal),
            distance_to_final_goal=float(distance_to_final_goal),
            acceleration_cmd=acceleration_cmd.tolist()
        )

        self.flight_data.append(data_point)

    def log_grid_snapshot(
        self,
        timestamp: float,
        grid: np.ndarray,
        drone_position: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float
    ):
        """Log occupancy grid snapshot."""

        snapshot = GridSnapshot(
            timestamp=timestamp,
            grid=grid.copy(),
            drone_position=drone_position.tolist(),
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            width=grid.shape[1],
            height=grid.shape[0]
        )

        self.grid_snapshots.append(snapshot)

    def save(self):
        """Save all logged data to files."""

        # Update metadata
        self.metadata['end_time'] = time.time()
        self.metadata['duration'] = self.metadata['end_time'] - self.metadata['start_time']
        self.metadata['num_data_points'] = len(self.flight_data)
        self.metadata['num_grid_snapshots'] = len(self.grid_snapshots)

        # Save metadata
        metadata_file = self.session_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save flight data as JSON
        flight_data_file = self.session_dir / "flight_data.json"
        with open(flight_data_file, 'w') as f:
            data_dicts = [dp.to_dict() for dp in self.flight_data]
            json.dump(data_dicts, f, indent=2)

        # Save grid snapshots as pickle (binary for efficiency)
        grid_file = self.session_dir / "grid_snapshots.pkl"
        with open(grid_file, 'wb') as f:
            pickle.dump(self.grid_snapshots, f)

        print(f"Flight data saved:")
        print(f"  - Metadata: {metadata_file}")
        print(f"  - Flight data: {flight_data_file}")
        print(f"  - Grid snapshots: {grid_file}")
        print(f"  - Total duration: {self.metadata['duration']:.2f}s")
        print(f"  - Data points: {len(self.flight_data)}")
        print(f"  - Grid snapshots: {len(self.grid_snapshots)}")

        return self.session_dir


def load_flight_data(session_dir: str):
    """
    Load flight data from a previous session.

    Args:
        session_dir: Path to session directory

    Returns:
        (flight_data, grid_snapshots, metadata)
    """
    session_path = Path(session_dir)

    # Load metadata
    with open(session_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load flight data
    with open(session_path / "flight_data.json", 'r') as f:
        flight_data_dicts = json.load(f)
        flight_data = [FlightDataPoint(**d) for d in flight_data_dicts]

    # Load grid snapshots
    with open(session_path / "grid_snapshots.pkl", 'rb') as f:
        grid_snapshots = pickle.load(f)

    return flight_data, grid_snapshots, metadata


if __name__ == '__main__':
    # Example usage
    logger = FlightLogger()

    # Simulate logging
    for i in range(10):
        logger.log_flight_data(
            timestamp=i * 0.1,
            position=np.array([i, i, 1.0]),
            velocity=np.array([1.0, 1.0, 0.0]),
            yaw=0.0,
            yaw_rate=0.0,
            active_goal=np.array([10.0, 10.0, 1.0]),
            final_goal=np.array([10.0, 10.0, 1.0]),
            num_waypoints=3,
            current_waypoint=np.array([5.0, 5.0, 1.0]),
            mpc_solve_time_ms=5.0,
            mpc_cost=100.0,
            mpc_status="solved",
            num_raw_obstacles=50,
            num_clustered_obstacles=10,
            distance_to_active_goal=5.0,
            distance_to_final_goal=10.0,
            acceleration_cmd=np.array([0.1, 0.1, 0.0])
        )

    session_dir = logger.save()
    print(f"\nTest data saved to: {session_dir}")
