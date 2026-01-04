# Drone-optimal-trajectory

This repository contains a ROS2 + PX4 SITL + Gazebo stack for a quadrotor MPC with LiDAR-based obstacle avoidance. The MPC runs inside ROS2, consumes PX4 odometry and LaserScan data, and outputs acceleration and yaw commands.

## Contents
- `drone_ws/src/drone/drone/mpc_obstalce_avoidance_node.py` - ROS2 node that builds the reference trajectory, converts frames, and runs the MPC loop.
- `drone_ws/src/drone/drone/mpc_solver.py` - CasADi MPC formulation with dynamics, costs, and half-space obstacle constraints.
- `drone_ws/src/drone/drone/mpc_mission_commander.py` - Offboard commander that tracks the MPC trajectory.
- `drone_ws/src/drone/drone/hover_enable_commander.py` - Hover gate that enables MPC once altitude is stable.
- `drone_ws/scripts/plot_topics.py` and `drone_ws/scripts/plot_csv.py` - Plotters for rosbag and CSV data.

---

## MPC overview (current ROS2 implementation)

### State and input
The solver in `mpc_solver.py` uses:

$$
x_k = [p_x, p_y, p_z, \psi, v_x, v_y, v_z, r]^T
$$
$$
u_k = [a_x, a_y, a_z, \dot{r}]^T
$$

with yaw rate $r = \dot{\psi}$.

### Discrete-time dynamics

$$
p_{k+1} = p_k + \Delta t \, v_k
$$
$$
\psi_{k+1} = \psi_k + \Delta t \, r_k
$$
$$
v_{k+1} = v_k + \Delta t \, a_k
$$
$$
r_{k+1} = r_k + \Delta t \, \dot{r}_k
$$

### Reference generation
`mpc_obstalce_avoidance_node.py` builds a reference trajectory in ENU by:
- Reusing the previous MPC solution if it is still close to the current state.
- Otherwise, generating a straight-line segment to the current goal at a constant desired speed.

Yaw reference is aligned with the line from the current position to the goal.

### Obstacle half-spaces
For each step, the solver builds half-space constraints from LiDAR points:

$$
n_i^T p_k \ge n_i^T z_i + r_s - s_{i,k}
$$

where $z_i$ are obstacle points, $n_i$ is the unit normal pointing from the obstacle to the reference position, $r_s$ is the safety radius, and $s_{i,k} \ge 0$ are slack variables. Only the nearest points (up to `m_planes`) within a range are used to build the half-spaces.

### Gap-based navigation

The MPC uses a two-layer approach for obstacle avoidance. While the half-space constraints handle low-level collision avoidance, a **gap-based navigation** layer provides high-level path planning through cluttered environments. This is essential for handling non-convex obstacle configurations (e.g., a cylinder blocking the direct path where the drone must choose to go left or right).

#### Gap detection algorithm

The gap detection analyzes the LiDAR scan to find free corridors:

1. **Identify free directions**: A scan ray is marked as "free" if:
   - Range exceeds 90% of `max_obs_range` (no obstacle detected)
   - Range is invalid/infinite (clear path)
   - Range is below minimum threshold (sensor noise)

2. **Group consecutive free rays into gaps**: The algorithm sweeps through the scan and groups consecutive free directions. Each gap is characterized by:
   - Start/end angles (angular extent)
   - Center angle (middle direction)
   - Minimum range (depth of the gap)
   - Angular width

3. **Filter by minimum width**: Gaps narrower than `gap_min_width` (default 1.0m at max range) are discarded.

4. **Handle wrap-around**: If a gap spans the scan boundaries (e.g., -π to +π), the algorithm merges the first and last gaps.

#### Gap selection and scoring

When the direct path to the goal is blocked, the algorithm scores each gap:

$$
\text{score} = w_{align} \cdot \text{alignment} + (1 - w_{align}) \cdot \text{quality}
$$

where:
- **Alignment** (0 to 1): How well the gap center points toward the goal direction
  $$\text{alignment} = \frac{\pi - |\theta_{gap} - \theta_{goal}|}{\pi}$$
- **Quality**: Combined measure of gap width and depth
  $$\text{quality} = \frac{\text{gap\_width}}{\pi} \cdot \frac{\text{min\_range}}{\text{max\_obs\_range}}$$
- $w_{align}$ is `gap_alignment_weight` (default 0.85, strongly preferring goal direction)

Additional scoring modifiers:
- Gaps >90° off goal direction: 80% penalty
- Gaps >60° off goal direction: 50% penalty  
- Gaps <30° off goal direction: 30% bonus

#### Intermediate goal placement

Once the best gap is selected, an intermediate goal is placed through it:

$$
p_{intermediate} = p_{drone} + d \cdot \hat{n}_{gap}
$$

where $d = \min(\text{gap\_goal\_distance}, 0.8 \cdot \text{min\_range}, \text{dist\_to\_goal})$ and $\hat{n}_{gap}$ is the unit direction toward the gap center in world frame.

#### Stability mechanisms

To prevent oscillation between gaps:
- **Hysteresis**: The intermediate goal only changes if the new position differs by more than `gap_hysteresis` (default 1.0m)
- **Spin detection**: If the drone is rotating fast (>0.3 rad/s) but moving slow (<0.5 m/s), the intermediate goal is locked
- **Direct path check**: A cone of ±17° around the goal direction is checked; if clear beyond `direct_path_threshold`, gap navigation is bypassed

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gap_nav_enabled` | `true` | Enable/disable gap-based navigation |
| `gap_min_width` | `1.0` | Minimum gap width in meters |
| `gap_goal_distance` | `3.0` | Distance to place intermediate goal |
| `gap_alignment_weight` | `0.85` | Weight for goal alignment vs gap quality |
| `direct_path_threshold` | `3.0` | Min clear distance to use direct path |
| `gap_hysteresis` | `1.0` | Min change in goal to switch gaps |

### Cost function

$$
J = \sum_{k=0}^{N-1} \Big(
Q_p^{ref} \lVert p_k - p_k^{ref} \rVert^2 + Q_p^{goal} \lVert p_k - p_{goal} \rVert^2 + Q_\psi \lVert \psi_k - \psi_k^{ref} \rVert^2 + Q_v \lVert v_k \rVert^2 + Q_r \lVert r_k \rVert^2 + R_a \lVert a_k \rVert^2 + R_{\dot{r}} \lVert \dot{r}_k \rVert^2 + \rho \lVert s_k \rVert^2 \Big) + J_{terminal}
$$

The terminal cost applies additional weight to the final position and yaw error.

### Constraints
- Dynamics equality constraints for each step.
- Altitude bounds: $z_{min} \le z_k \le z_{max}$ (including terminal).
- Velocity and yaw rate bounds.
- Acceleration and yaw acceleration bounds.
- Obstacle half-space constraints with non-negative slack.

### Solver
The MPC is solved with CasADi + IPOPT. Decision variables are bounded explicitly (state, input, and slack), and a warm start is initialized from the reference trajectory.

---

## How to run the ROS2 simulation environment (quick guide)

This project contains a separate, full ROS2 simulation environment. It packages PX4 SITL, Gazebo (gz), and helper launch scripts. Below are high-level steps to start the ROS2-based simulation used by the workspace.
ROS2 acts as middleware for trajectory planning.

```bash
# Allow GUI forwarding from containers (on host)
xhost +

cd /path/to/Drone-optimal-trajectory/docker/PX4-dev-env/full_docker_setup

# Build and start the docker-compose service
docker-compose build
docker-compose up -d

# On the host (not in the container), open QGC in a new terminal
cd /path/to/Drone-optimal-trajectory/QGC
./QGroundControl-x86_64.AppImage

# Enter the px4 container in a new terminal
docker-compose exec px4-stack bash

# Inside the container, start the 2D lidar drone simulation
make px4_sitl gz_x500_lidar_2d

# after gazebo is launch, add some obstacles in the world for testing in a simpler environment setting

# In another container terminal, run the micro-ROS agent
cd /Micro-XRCE-DDS-Agent/build
MicroXRCEAgent udp4 -p 8888

# In another container terminal, launch RViz2 with the laser scan bridge
cd /workspace/
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch drone laser_bridge.launch.py

# In another container terminal, run the MPC planner + hover gate + commander
cd /workspace/
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch drone path_planning.launch.py

```

Note: Once you launch the laser_bridge, you should set on Rviz2 the fixed frame to `world` and enable the `LaserScan` display subscribing to the `/scan` topic to visualize the LiDAR data.

The `path_planning.launch.py` launch file starts:
- `hover_enable_commander` (arms, holds hover, then publishes `/mpc/enable`)
- `mpc_obstacle_avoidance` (waits for enable, then optimizes)
- `mpc_mission_commander` (waits for enable, then tracks the MPC path)

## Plotting results (rosbag + CSV)

To generate the csv files from rosbag data, run the following script:

```bash
cd /workspace/scripts
./plot_topics.py  --backend csv --out ./plot_data
```

To plot from CSV data:

```bash
cd /workspace/scripts
./plot_csv.py --dir ./plot_data
```
