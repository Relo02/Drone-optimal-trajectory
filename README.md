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
./plot_topics.py  --backend csv --out ./workspace/scripts/plot_data
```

To plot from CSV data:

```bash
cd /workspace/scripts
./plot_csv.py --dir ./workspace/scripts/plot_data
```
