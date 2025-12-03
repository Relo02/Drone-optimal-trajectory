# Drone-optimal-trajectory

This folder contains a simple Model Predictive Control (MPC) demonstration for a quadrotor-like drone. The implementation is a discrete-time, linear time-invariant (LTI) MPC that plans accelerations (and yaw acceleration) over a finite horizon and solves a quadratic program (QP) with OSQP at each control step.

## Contents
Inside the trajectory optimization folder are available the one step ahead dynamic model and the k-step ahead predictive model. In addition is also present the main code which runs the finite horizon MPC trajectory planner.  
- `model.py` — DroneModel class: builds discrete dynamics (A,B), constructs prediction matrices (Sx0, Su), and assembles the QP matrices (H, g) for a terminal-goal cost.
- `sim_main.py` — Simulation and closed-loop receding-horizon MPC example using OSQP. Plots and animates the planned trajectories and actual closed-loop motion.

---

## MPC overview (implementation details)

### State and input definition

The MPC in this code uses a reduced trajectory state vector (example, n=8):

- States x_k = [x, y, z, vx, vy, vz, yaw, yaw_dot]^T
- Inputs u_k = [a_x, a_y, a_z, yaw_ddot]^T

Note: `model.DroneModel.dynamics()` contains placeholders for building A and B matrices. The code uses these matrices as an LTI linearization of the true drone dynamics.

### Discrete-time LTI dynamics

The discrete-time dynamics are assumed in the standard form:

$$
x_{k+1} = A x_k + B u_k
$$

Prediction matrices are built for a horizon of length $N$ producing stacked forms:

$$
X = S_{x0} x_0 + S_u U
$$

where X = [x_1; x_2; ...; x_N] and U = [u_0; u_1; ...; u_{N-1}]. The implementation builds `Sx0` and `Su` in `model.build_prediction_matrices(A,B,N)`.

### Cost function

The MPC minimizes a quadratic terminal-goal cost of the form:

$$
J(U) = \tfrac{1}{2} U^T H U + g^T U
$$

where

$$
H = S_u^T Q_{bar} S_u + R_{bar}
$$
$$
g = S_u^T Q_{bar} (S_{x0} x_0 - X_{ref})
$$

Details in the code:
- `Q_stage` — per-stage state cost (applied to each stage except last)
- `Qf` — terminal state cost (heavy weight on terminal position)
- `R` — input (control) cost
- `Qbar` and `Rbar` — block-diagonal expanded costs across the horizon

In `model.build_qp_in_u_terminal_goal(Q,R,Qf,Sx0,Su,x0,x_goal)` the reference is encoded only at the terminal stage (X_ref has x_goal at the final block).

### Constraints

This example uses simple box constraints on each element of the stacked control vector U. In `sim_main.py`:

- `a_max` — maximum absolute acceleration (m/s^2)
- lower/upper bound vectors `lb`, `ub` created via Kronecker product to apply the same bound at every horizon step.

The QP formulation used by OSQP is:

minimize 0.5 U^T H U + g^T U
subject to lb <= U <= ub

The code inserts these bounds as linear constraints via `A = I` in OSQP.

### Solver and implementation notes

- The QP is solved with OSQP (sparse QP solver). The code prepares a symmetric (numerically PSD) Hessian `P = (H+H^T)/2 + eps*I` before converting to sparse CSC. A tiny `eps` regularizer ensures numerical stability.
- Warm-starting: the solver is warm-started with the shifted previous solution (useful to speed up convergence).
- The code attempts to only update the linear term `q` each iteration and falls back to updating the Hessian if necessary.

### Typical parameters (from `sim_main.py`)

- time-step dt = 0.05 s
- horizon N = 30
- state dimension n = 8 (example)
- input dimension m = 4 (example)
- Q_stage: small weights on position/velocity in each stage
- Qf: very large terminal weights on final position (to enforce terminal goal)
- R: moderate penalty on control effort (0.5 * I)

---

## How to run the local MPC demo (trajectory_opt)

Prerequisites

- Python 3.8+ (code was tested with Python 3.12 in a venv in this workspace)
- Install required packages (numpy, scipy, matplotlib, osqp)

Create a virtual environment and install requirements (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib osqp
```

Run the simulation

```bash
cd trajectory_opt
python3 sim_main.py
```

What the demo does

- Builds prediction matrices using the provided A,B in `model.py`.
- Constructs the QP (H,g) for a given initial state and terminal goal.
- Sets up OSQP with simple box constraints on inputs and solves the QP in a receding-horizon loop.
- Applies the first control input, simulates the next state with the discrete-time model, and repeats.
- Produces plots showing the final planned input sequence and the actual trajectory; also shows a 3D animation comparing the actual path with the current plan at each step.

---

## Suggested extensions / improvements

- Replace the placeholder `model.DroneModel.dynamics()` with a proper linearization of a full nonlinear model (or export linearized A,B around operating point).
- Add state constraints (e.g., position or velocity limits) and model them as linear constraints in OSQP.
- Include disturbance handling or a soft-constraint terminal set (MPC with constraint tightening).
- Pre-factor or cache the sparse structure of H (if A,B constant) to avoid re-creating sparse matrices every iteration — only update `q`.
- Add logging and unit tests for `build_prediction_matrices` and `build_qp_in_u_terminal_goal`.

---

## How to run the ROS2 simulation environment (quick guide)

This project contains a separate, full ROS2 simulation environment located at `../Ros-2-Environment`. That environment packages PX4 SITL, Gazebo (gz), MAVROS, and helper launch scripts. Below are high-level steps to start the ROS2-based simulation used by the workspace.

1) Use the provided Docker setup (recommended)

```bash
cd /path/to/Ros-2-Environment/drone_control_ws/docker

# Allow GUI forwarding (on host)
xhost +

# Build and run the docker-compose service (named `drone-control` in the repo)
docker-compose build drone-control
docker-compose up -d drone-control
docker-compose exec drone-control bash
```

Inside the container you can use the helper scripts. Common useful scripts (located in `drone_control_ws/docker` or `/usr/local/bin/` inside the container):

- `launch_px4_gz_x500.sh` — launches PX4 SITL with the x500 Gazebo model
- `launch_gazebo_x500_auto.sh` — launch Gazebo only (for controller development)
- `launch_mavros.sh` — start MAVROS and connect to PX4 SITL

2) Alternatively, run locally (native) if you have ROS2 Humble/Foxy and PX4 sources installed.

High-level steps (native):

```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Build the workspace (if you changed packages)
cd /path/to/Ros-2-Environment/drone_control_ws
colcon build
source install/setup.bash

# Launch PX4 SITL with Gazebo (example helper provided)
./drone_control_ws/docker/launch_px4_gz_x500.sh

# Start MAVROS (if needed)
ros2 launch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
```

3) After the whole system is up:

- Verify PX4 console (`pxh>` prompt) is available.
- Check MAVROS connection: `ros2 topic echo /mavros/state` (should show connected: true)
- Run your ROS2 nodes (e.g., `mission_commander` from the workspace) or publish commands to `/mavros/setpoint_position/local`.

For a full, step-by-step guide see `Ros-2-Environment/README.md` which contains container-based quick start and troubleshooting notes.

---

## Small checklist (to reproduce results)

1. Create and activate a Python venv and install `numpy scipy matplotlib osqp`.
2. Run `trajectory_opt/sim_main.py` to see the MPC demo and plots.
3. To test with the full ROS2+PX4 simulation, start the Docker container described in `Ros-2-Environment/drone_control_ws/docker` and follow the Quick Start in that repo's README.

---

If you want, I can:

- Add a `requirements.txt` in `trajectory_opt` and a one-line `run.sh` helper.
- Create a `README.md` copy in `Ros-2-Environment/trajectory_opt/` (if you want the same docs there).
- Implement a small unit test for `build_prediction_matrices` and `build_qp_in_u_terminal_goal`.

Tell me what you'd like next and I will apply it.
