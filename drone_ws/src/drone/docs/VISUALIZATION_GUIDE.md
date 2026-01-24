# Flight Data Visualization Guide

## Overview

The wall-aware MPC node includes comprehensive flight data logging and visualization capabilities. This guide explains how to enable logging, run flights, and generate visualizations.

---

## Quick Start

### 1. Enable Logging (Default)

Logging is enabled by default. The node will automatically record flight data to `/tmp/drone_logs/`.

```bash
ros2 run drone mpc_wall_aware
```

### 2. Run Your Flight

Execute your mission. The node logs data at 10Hz and occupancy grid snapshots periodically.

### 3. Generate Visualizations

After the flight, find your session directory:

```bash
ls /tmp/drone_logs/
# Output: session_20250124_153045/
```

Generate all plots:

```bash
cd ~/drone_ws/src/drone/drone
python visualize_flight.py /tmp/drone_logs/session_20250124_153045
```

---

## Logging Configuration

### Parameters

```yaml
# Enable/disable logging
enable_logging: true

# Directory for log files
log_directory: '/tmp/drone_logs'

# Grid snapshot frequency (every N control cycles)
grid_snapshot_interval: 10  # 10 cycles = 1 second at 10Hz
```

### Example: Custom Log Directory

```bash
ros2 run drone mpc_wall_aware \
  --ros-args \
  -p log_directory:=/home/user/my_drone_logs \
  -p grid_snapshot_interval:=5
```

### Disable Logging

```bash
ros2 run drone mpc_wall_aware \
  --ros-args \
  -p enable_logging:=false
```

---

## Logged Data

Each flight session creates a directory with three files:

```
/tmp/drone_logs/session_20250124_153045/
├── metadata.json          # Session information
├── flight_data.json       # Time-series flight data
└── grid_snapshots.pkl     # Occupancy grid evolution
```

### metadata.json

```json
{
  "session_id": "20250124_153045",
  "start_time": 1706106645.123,
  "end_time": 1706106705.456,
  "duration": 60.333,
  "num_data_points": 603,
  "num_grid_snapshots": 61
}
```

### flight_data.json

Contains time-series data for each control cycle:

```json
[
  {
    "timestamp": 0.1,
    "position": [0.0, 0.0, 1.0],
    "velocity": [0.0, 0.0, 0.0],
    "yaw": 0.0,
    "yaw_rate": 0.0,
    "active_goal": [5.0, 5.0, 1.0],
    "final_goal": [10.0, 10.0, 1.0],
    "num_waypoints": 3,
    "current_waypoint": [5.0, 5.0, 1.0],
    "mpc_solve_time_ms": 5.2,
    "mpc_cost": 123.4,
    "mpc_status": "solved",
    "num_raw_obstacles": 145,
    "num_clustered_obstacles": 12,
    "distance_to_active_goal": 7.07,
    "distance_to_final_goal": 14.14,
    "acceleration_cmd": [0.5, 0.5, 0.0]
  },
  ...
]
```

### grid_snapshots.pkl

Binary file containing occupancy grid arrays at different timestamps.

---

## Generated Visualizations

The visualization tool generates 6 comprehensive plots:

### 1. **trajectory_2d.png**

**Description**: Top-down view of the flight path

**Features:**
- Actual trajectory (blue line)
- Start position (green circle)
- End position (red triangle)
- Final goal (red star)
- Waypoint transitions (yellow dashed lines)
- Intermediate waypoints (yellow circles)

**Use Case**: Overall path assessment and waypoint following analysis

---

### 2. **trajectory_3d.png**

**Description**: 3D perspective of flight path

**Features:**
- Full 3D trajectory
- Altitude changes
- Start, end, and goal markers

**Use Case**: Verify altitude tracking and 3D path smoothness

---

### 3. **mpc_performance.png**

**Description**: 6-panel dashboard of MPC metrics

**Panels:**

1. **MPC Solver Time**
   - Solve time per cycle (ms)
   - Mean solve time
   - 100ms real-time limit indicator

2. **MPC Cost Function Evolution**
   - Total cost over time
   - Shows optimization convergence

3. **Distance to Goals**
   - Distance to final goal (blue)
   - Distance to active waypoint (orange)
   - Shows waypoint switching behavior

4. **Velocity Profile**
   - Speed over time
   - v_max limit indicator
   - Acceleration/deceleration phases

5. **Remaining Waypoints**
   - Number of waypoints in queue
   - Shows progress through path

6. **Acceleration Command Magnitude**
   - Control effort
   - a_max limit indicator
   - Smoothness assessment

**Use Case**: Detailed performance analysis and tuning

---

### 4. **occupancy_grid_evolution.png**

**Description**: 3x3 grid showing occupancy map at different times

**Features:**
- 9 snapshots evenly distributed over flight
- Inflated obstacles visualization
- Drone position (red circle)
- Goal position (green star)
- Grid coordinates

**Use Case**:
- Understand obstacle inflation
- See how map builds over time
- Verify wall detection

**Key Insights:**
- **White areas**: Free space
- **Black areas**: Occupied (obstacles)
- **Gray areas**: Inflated safety margin
- Grid updates based on LiDAR observations

---

### 5. **occupancy_grid_with_trajectory.png**

**Description**: Final occupancy grid with complete trajectory overlay

**Features:**
- Final state of inflated occupancy grid
- Complete trajectory path
- Start/end/goal markers
- Combined view of planning space and execution

**Use Case:**
- Verify path went through free space
- Check safety margins maintained
- Analyze obstacle avoidance behavior

**Critical Analysis:**
- Trajectory should stay in white/light gray areas
- Should NOT pass through black (occupied) areas
- Shows effectiveness of inflation for safety

---

### 6. **statistics_summary.png**

**Description**: Statistical performance summary

**Features:**

**Summary Panel:**
- Total flight duration
- Distance traveled vs. straight-line distance
- Path efficiency percentage
- Final goal accuracy
- Average/max speeds
- Average/max MPC solve times
- Obstacle statistics

**Speed Distribution Histogram:**
- Frequency distribution of velocities
- Mean speed indicator

**Solve Time Distribution Histogram:**
- MPC computation time distribution
- Real-time constraint compliance

**Use Case**:
- Quick performance assessment
- Compare different flights
- Benchmark tuning changes

**Key Metrics:**
- **Path Efficiency**: Higher is better (100% = straight line)
- **Final Distance**: Should be < 0.5m for success
- **Solve Time**: Should be < 100ms for real-time operation

---

## Visualization Options

### Generate Specific Plots

Edit `visualize_flight.py` to generate only desired plots:

```python
# In generate_all_plots()
visualizer.plot_trajectory_2d()
visualizer.plot_mpc_performance()
# Comment out others
```

### Custom Output Directory

```bash
python visualize_flight.py /tmp/drone_logs/session_20250124_153045 \
  --output-dir ~/my_plots
```

---

## Analysis Workflow

### 1. Initial Assessment

**Check:** `statistics_summary.png`

**Questions:**
- Did we reach the goal? (Final distance < 0.5m?)
- Was path efficient? (>80% is good)
- Were solve times acceptable? (All <100ms?)

### 2. Trajectory Analysis

**Check:** `trajectory_2d.png` and `occupancy_grid_with_trajectory.png`

**Questions:**
- Did drone follow waypoints correctly?
- Did path avoid obstacles?
- Were safety margins maintained?

### 3. Performance Deep Dive

**Check:** `mpc_performance.png`

**Questions:**
- Any solve time spikes? (indicates complex scenarios)
- Cost function converging? (should decrease over time)
- Smooth velocity profile? (no jerky movements)
- Waypoint progress consistent?

### 4. Obstacle Handling

**Check:** `occupancy_grid_evolution.png`

**Questions:**
- Did grid build correctly from LiDAR data?
- Was inflation appropriate? (not too conservative)
- Did walls get detected early enough?

---

## Troubleshooting

### Issue: No grid snapshots

**Symptom:** `occupancy_grid_evolution.png` not generated

**Solution:** Grid snapshots might be empty. Check:
```bash
# Verify grid_snapshots.pkl exists and has data
python -c "import pickle; print(len(pickle.load(open('/tmp/drone_logs/session_*/grid_snapshots.pkl', 'rb'))))"
```

Increase snapshot frequency:
```bash
ros2 run drone mpc_wall_aware --ros-args -p grid_snapshot_interval:=5
```

### Issue: Plots look empty

**Symptom:** Trajectories show minimal movement

**Possible Causes:**
- Flight was very short
- Drone didn't move (MPC not enabled)
- Logging started too late

**Solution:** Ensure `mpc_enabled` and drone is in OFFBOARD mode.

### Issue: Memory error during visualization

**Symptom:** Python crashes with MemoryError

**Solution:** Process has too many grid snapshots. Reduce:
```bash
python visualize_flight.py session_dir --num-snapshots 4
```

Or modify `visualize_flight.py`:
```python
visualizer.plot_occupancy_grid_evolution(num_snapshots=4)
```

---

## Advanced Usage

### Programmatic Analysis

Load and analyze data in your own scripts:

```python
from flight_logger import load_flight_data
import numpy as np

# Load data
flight_data, grid_snapshots, metadata = load_flight_data('/tmp/drone_logs/session_20250124_153045')

# Extract positions
positions = np.array([d.position for d in flight_data])

# Compute total distance
distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
total_distance = np.sum(distances)

print(f"Total distance: {total_distance:.2f}m")

# Find maximum speed
speeds = np.array([np.linalg.norm(d.velocity) for d in flight_data])
max_speed = np.max(speeds)
max_speed_time = flight_data[np.argmax(speeds)].timestamp

print(f"Max speed: {max_speed:.2f} m/s at t={max_speed_time:.2f}s")
```

### Export to CSV

```python
import pandas as pd
from flight_logger import load_flight_data

flight_data, _, _ = load_flight_data('/tmp/drone_logs/session_20250124_153045')

# Convert to DataFrame
df = pd.DataFrame([d.to_dict() for d in flight_data])

# Save to CSV
df.to_csv('flight_data.csv', index=False)
```

### Compare Multiple Flights

```python
sessions = [
    '/tmp/drone_logs/session_20250124_153045',
    '/tmp/drone_logs/session_20250124_154500',
]

for session in sessions:
    flight_data, _, metadata = load_flight_data(session)
    positions = np.array([d.position for d in flight_data])
    total_dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

    print(f"Session {metadata['session_id']}:")
    print(f"  Duration: {metadata['duration']:.2f}s")
    print(f"  Distance: {total_dist:.2f}m")
    print()
```

---

## Tips for Better Visualizations

### 1. Consistent Goal Positions

Use the same goal position for comparable flights:
```bash
ros2 run drone mpc_wall_aware --ros-args -p goal:=[10.0,10.0,1.5]
```

### 2. Longer Flights

More data = better visualization. Aim for >30 seconds of flight.

### 3. Varied Scenarios

Test different obstacle configurations to see grid evolution:
- Open space
- Single long wall
- Corridor
- Maze

### 4. Snapshot Frequency

Adjust based on flight duration:
- Short flights (<30s): `grid_snapshot_interval:=5`
- Long flights (>2min): `grid_snapshot_interval:=20`

---

## References

- [MPC Wall-Aware Documentation](MPC_WALL_AWARE_DOCUMENTATION.md)
- [Flight Logger API](../drone/flight_logger.py)
- [Visualization Script](../drone/visualize_flight.py)

---

**Author**: Drone Optimal Trajectory Team
**Version**: 1.0.0
**Last Updated**: 2025-01-24