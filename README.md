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

- States $$x_k = [x, y, z, v_x, v_y, v_z, \psi, \dot{\psi}]^T$$
- Inputs $$u_k = [a_x, a_y, a_z, \ddot{\psi}]^T$$

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

where $$X = [x_1; x_2; ...; x_N]$$ and $$U = [u_0; u_1; ...; u_{N-1}]$$. The implementation builds `Sx0` and `Su` in `model.build_prediction_matrices(A,B,N)`.

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

$$min {0.5 U^T H U + g^T U$$}
subject to $$l_b <= U <= u_b$$

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

- Add state constraints (e.g., position or velocity limits) and model them as linear constraints in OSQP.
- Include disturbance handling or a soft-constraint terminal set (MPC with constraint tightening).
- Add obstacle avoidance constraints coming from the VO SLAM.
- Add logging and unit tests for `build_prediction_matrices` and `build_qp_in_u_terminal_goal`.

---

## How to run the ROS2 simulation environment (quick guide)

This project contains a separate, full ROS2 simulation environment. That environment packages PX4 SITL, Gazebo (gz) and helper launch scripts. Below are high-level steps to start the ROS2-based simulation used by the workspace.
Ros2 works as a middleware for handling the trajectory planning and the simulated VO SLAM implemented inside the `trajectory_opt` folder.

1) Use the provided Docker setup (recommended)

```bash
# Allow GUI forwarding (on host)
xhost +

cd /path/to/Drone-optimal-trajectory/docker/PX4-dev-env

# Build and run the docker-compose service
docker-compose build

docker-compose up

# In a new terminal run the px4 service for starting px4 and gazebo
docker-compose run px4-sim bash

# Launch px4 and gazebo in the walls.sdf world
make px4_sitl gz_x500_depth_walls

# Verify if the drone is capable to recive commands from the GCS
# In the px4 terminal run the following command for arming the drone
commander arm -f

# Test if the drone is capable to initiate the takeoff service
commander takeoff

# In a new terminal execute the running container
docker-conpose exec px4-sim bash

# In a new terminal, check if the depth camera topics are available in gazebo
gz topic -l
```

The last two commands that we runned in the px4 terminal will be handled by a state machine implemented in ros2.

---

## TODO steps

- Test with a simple mission commander if the drone is capable of reaching some desired goals in the ros2 environment
- Bridge necessary topics that are available in gazebo to ros2 (e.g. camera point cloud topic) and configure rviz2 for viewing the sensors topcis
- Implement a state machine for checking at which state the drone is at along with the trajectory planner MPC implemented inside the `trajectory_opt` folder example.

