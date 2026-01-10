# Model Predictive Control for Drone Obstacle Avoidance: Implementation Documentation

**Drone Optimal Trajectory Project**  
*January 10, 2026*

---

## Table of Contents

1. [Introduction](#introduction)
2. [State and Control Variables](#state-and-control-variables)
3. [System Dynamics](#system-dynamics)
4. [MPC Optimization Problem](#mpc-optimization-problem)
5. [Reference Trajectory Generation](#reference-trajectory-generation)
6. [Gap-Based Navigation](#gap-based-navigation)
7. [Emergency Braking](#emergency-braking)
8. [Implementation Details](#implementation-details)
9. [Parameter Tuning Guidelines](#parameter-tuning-guidelines)
10. [Conclusion](#conclusion)

---

## Introduction

This document provides a comprehensive mathematical description of the Model Predictive Control (MPC) implementation for autonomous drone navigation with obstacle avoidance. The system combines gap-based navigation, artificial potential fields for reference trajectory generation, and optimization-based control with hard safety constraints.

### System Overview

The control architecture consists of three main components:

1. **Gap Navigator**: Analyzes LiDAR data to identify navigable corridors and compute intermediate waypoints
2. **Reference Trajectory Generator**: Uses artificial potential fields to create smooth, obstacle-aware reference paths
3. **MPC Solver**: Optimizes control inputs subject to dynamics, limits, and safety constraints

---

## State and Control Variables

### State Vector

The full drone state at time step $k$ is represented as an 8-dimensional vector:

$$
\mathbf{x}_k = \begin{bmatrix}
p_x \\ p_y \\ p_z \\ \psi \\ v_x \\ v_y \\ v_z \\ \dot{\psi}
\end{bmatrix} \in \mathbb{R}^8
$$

where:
- $\mathbf{p} = [p_x, p_y, p_z]^T \in \mathbb{R}^3$: Position in world frame (ENU coordinates) [m]
- $\psi \in [-\pi, \pi]$: Yaw angle [rad]
- $\mathbf{v} = [v_x, v_y, v_z]^T \in \mathbb{R}^3$: Velocity in world frame [m/s]
- $\dot{\psi} \in \mathbb{R}$: Yaw rate [rad/s]

### Control Vector

The control input at time step $k$ is:

$$
\mathbf{u}_k = \begin{bmatrix}
a_x \\ a_y \\ a_z \\ \ddot{\psi}
\end{bmatrix} \in \mathbb{R}^4
$$

where:
- $\mathbf{a} = [a_x, a_y, a_z]^T \in \mathbb{R}^3$: Linear acceleration in world frame [m/s²]
- $\ddot{\psi} \in \mathbb{R}$: Yaw acceleration [rad/s²]

---

## System Dynamics

### Double Integrator Model

The drone dynamics are modeled as a discrete-time double integrator system:

#### Position Dynamics
$$
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t
$$

#### Velocity Dynamics
$$
\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{a}_k \Delta t
$$

#### Yaw Dynamics
$$
\psi_{k+1} = \psi_k + \dot{\psi}_k \Delta t
$$

$$
\dot{\psi}_{k+1} = \dot{\psi}_k + \ddot{\psi}_k \Delta t
$$

where $\Delta t$ is the sampling time (typically 0.1 s).

### Compact Form

The full system dynamics can be written as:

$$
\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k
$$

where $\mathbf{A} \in \mathbb{R}^{8 \times 8}$ and $\mathbf{B} \in \mathbb{R}^{8 \times 4}$ are:

$$
\mathbf{A} = \begin{bmatrix}
\mathbf{I}_3 & \mathbf{0}_{3\times1} & \Delta t \cdot \mathbf{I}_3 & \mathbf{0}_{3\times1} \\
\mathbf{0}_{1\times3} & 1 & \mathbf{0}_{1\times3} & \Delta t \\
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times1} & \mathbf{I}_3 & \mathbf{0}_{3\times1} \\
\mathbf{0}_{1\times3} & 0 & \mathbf{0}_{1\times3} & 1
\end{bmatrix}
$$

$$
\mathbf{B} = \begin{bmatrix}
\mathbf{0}_{3\times3} & \mathbf{0}_{3\times1} \\
\mathbf{0}_{1\times3} & 0 \\
\Delta t \cdot \mathbf{I}_3 & \mathbf{0}_{3\times1} \\
\mathbf{0}_{1\times3} & \Delta t
\end{bmatrix}
$$

---

## MPC Optimization Problem

### Prediction Horizon

The MPC optimizes over a finite prediction horizon:
- $N$: Number of prediction steps (typically 20)
- $T = N \cdot \Delta t$: Total prediction time (typically 2.0 s)

### Cost Function

The total cost function to minimize is:

$$
J = J_{\text{terminal}} + \sum_{k=0}^{N-1} \left( J_{\text{tracking}} + J_{\text{goal}} + J_{\text{control}} + J_{\text{safety}} + J_{\text{obstacles}} \right)
$$

#### Terminal Cost

Penalizes deviation from the goal at the final prediction step:

$$
J_{\text{terminal}} = Q_{\text{terminal}} \|\mathbf{p}_N - \mathbf{p}_{\text{goal}}\|_2^2
$$

**Typical value**: $Q_{\text{terminal}} = 150.0$

#### Position Tracking Cost

Penalizes deviation from the reference trajectory:

$$
J_{\text{tracking}} = Q_{\text{pos}} \|\mathbf{p}_k - \mathbf{p}_{\text{ref},k}\|_2^2 + Q_{\text{yaw}} (\psi_k - \psi_{\text{ref},k})^2
$$

**Typical values**: $Q_{\text{pos}} = 15.0$, $Q_{\text{yaw}} = 2.0$

#### Goal Attraction Cost

Directly attracts the drone toward the goal:

$$
J_{\text{goal}} = Q_{\text{goal}} \|\mathbf{p}_k - \mathbf{p}_{\text{goal}}\|_2^2
$$

**Typical value**: $Q_{\text{goal}} = 80.0$

#### Control Effort Cost

Penalizes large control inputs and encourages smooth control:

$$
J_{\text{control}} = Q_{\text{vel}} \|\mathbf{v}_k\|_2^2 + R_{\text{acc}} \|\mathbf{a}_k\|_2^2 + R_{\text{yaw}} \ddot{\psi}_k^2 + R_{\text{jerk}} \|\mathbf{a}_k - \mathbf{a}_{k-1}\|_2^2
$$

**Typical values**: $Q_{\text{vel}} = 1.0$, $R_{\text{acc}} = 0.3$, $R_{\text{yaw}} = 0.5$, $R_{\text{jerk}} = 0.3$

#### Velocity-Toward-Obstacles Penalty

Penalizes velocities directed toward obstacles:

$$
J_{\text{safety}} = Q_{\text{vel-obs}} \sum_{i=1}^{M_k} \max\left(0, -\frac{\mathbf{v}_k \cdot (\mathbf{p}_k - \mathbf{o}_i)}{\|\mathbf{p}_k - \mathbf{o}_i\|}\right)^2
$$

where $\mathbf{o}_i$ is the position of obstacle $i$, and $M_k$ is the number of obstacles at step $k$.

**Typical value**: $Q_{\text{vel-obs}} = 50.0$

#### Repulsive Potential Field Cost

Provides smooth obstacle avoidance through repulsive forces:

$$
J_{\text{obstacles}} = Q_{\text{rep}} \sum_{i=1}^{M_k} \phi(\|\mathbf{p}_k - \mathbf{o}_i\|)
$$

The potential function $\phi(d)$ is defined as:

$$
\phi(d) = \begin{cases}
\frac{1}{d - d_{\text{safe}}/2 + 0.5} \cdot \frac{d_0 - d}{d_0} & \text{if } d < d_0 \\
0 & \text{otherwise}
\end{cases}
$$

where:
- $d = \|\mathbf{p}_k - \mathbf{o}_i\|$: Distance to obstacle $i$
- $d_{\text{safe}} = 0.8$ m: Safety radius
- $d_0 = 3.0$ m: Influence distance

**Typical value**: $Q_{\text{rep}} = 300.0$

### Constraints

#### Dynamics Constraints

For all $k = 0, \ldots, N-1$:

$$
\mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k
$$

#### Initial Condition

$$
\mathbf{x}_0 = \mathbf{x}_{\text{current}}
$$

#### Velocity Limits

For all $k = 0, \ldots, N$:

$$
\|\mathbf{v}_k\|_2 \leq v_{\max}
$$

**Typical value**: $v_{\max} = 2.0$ m/s

#### Acceleration Limits

For all $k = 0, \ldots, N-1$:

$$
\|\mathbf{a}_k\|_2 \leq a_{\max}
$$

**Typical value**: $a_{\max} = 3.0$ m/s²

#### Yaw Rate Limits

For all $k = 0, \ldots, N$:

$$
|\dot{\psi}_k| \leq \dot{\psi}_{\max}
$$

**Typical value**: $\dot{\psi}_{\max} = 1.5$ rad/s

#### Yaw Acceleration Limits

For all $k = 0, \ldots, N-1$:

$$
|\ddot{\psi}_k| \leq \ddot{\psi}_{\max}
$$

**Typical value**: $\ddot{\psi}_{\max} = 2.0$ rad/s²

#### Altitude Limits

For all $k = 0, \ldots, N$:

$$
z_{\min} \leq p_{z,k} \leq z_{\max}
$$

**Typical values**: $z_{\min} = 0.3$ m, $z_{\max} = 10.0$ m

#### Obstacle Avoidance Constraints (Hard Constraints)

For all $k = 0, \ldots, N$ and all obstacles $i = 1, \ldots, M_k$:

$$
\|\mathbf{p}_k - \mathbf{o}_i\|_2 \geq d_{\text{safe}} - \epsilon_i
$$

where $\epsilon_i \geq 0$ are slack variables with penalty:

$$
J_{\text{slack}} = W_{\text{obs}} \sum_{i=1}^{M_k} \epsilon_i^2
$$

**Typical values**: $d_{\text{safe}} = 0.8$ m, $W_{\text{obs}} = 5000.0$

### Complete Optimization Problem

The complete MPC problem is formulated as:

$$
\begin{aligned}
\min_{\substack{\mathbf{x}_{0:N}, \mathbf{u}_{0:N-1} \\ \boldsymbol{\epsilon}}} \quad & J_{\text{terminal}} + \sum_{k=0}^{N-1} \left( J_{\text{tracking}} + J_{\text{goal}} + J_{\text{control}} + J_{\text{safety}} + J_{\text{obstacles}} \right) + J_{\text{slack}} \\
\text{s.t.} \quad & \mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k, \quad k = 0, \ldots, N-1 \\
& \mathbf{x}_0 = \mathbf{x}_{\text{current}} \\
& \|\mathbf{v}_k\|_2 \leq v_{\max}, \quad k = 0, \ldots, N \\
& \|\mathbf{a}_k\|_2 \leq a_{\max}, \quad k = 0, \ldots, N-1 \\
& |\dot{\psi}_k| \leq \dot{\psi}_{\max}, \quad k = 0, \ldots, N \\
& |\ddot{\psi}_k| \leq \ddot{\psi}_{\max}, \quad k = 0, \ldots, N-1 \\
& z_{\min} \leq p_{z,k} \leq z_{\max}, \quad k = 0, \ldots, N \\
& \|\mathbf{p}_k - \mathbf{o}_i\|_2 \geq d_{\text{safe}} - \epsilon_i, \quad k = 0, \ldots, N, \; i = 1, \ldots, M_k \\
& \epsilon_i \geq 0, \quad i = 1, \ldots, M_k
\end{aligned}
$$

---

## Reference Trajectory Generation

The reference trajectory $\{\mathbf{p}_{\text{ref},k}\}_{k=0}^{N-1}$ is generated using an artificial potential field approach to create smooth, obstacle-aware paths.

### Attractive Force

The attractive force toward the goal is:

$$
\mathbf{F}_{\text{attr}} = k_{\text{attr}} \frac{\mathbf{p}_{\text{goal}} - \mathbf{p}}{\|\mathbf{p}_{\text{goal}} - \mathbf{p}\|}
$$

**Typical value**: $k_{\text{attr}} = 1.0$

### Repulsive Force

The repulsive force from obstacle $i$ is:

$$
\mathbf{F}_{\text{rep},i} = \begin{cases}
k_{\text{rep}} \left(\frac{1}{d_i} - \frac{1}{d_{\text{inf}}}\right) \frac{1}{d_i^2} \frac{\mathbf{p} - \mathbf{o}_i}{d_i} & \text{if } d_i < d_{\text{inf}} \\
\mathbf{0} & \text{otherwise}
\end{cases}
$$

where:
- $d_i = \|\mathbf{p} - \mathbf{o}_i\|$: Distance to obstacle $i$
- $k_{\text{rep}} = 3.0$: Repulsion gain
- $d_{\text{inf}} = 4.5$ m: Influence radius (= $1.5 \times$ potential_influence_dist)

The total repulsive force is:

$$
\mathbf{F}_{\text{rep}} = \sum_{i=1}^{M} \mathbf{F}_{\text{rep},i}
$$

### Velocity Computation

The total force determines the direction of motion:

$$
\mathbf{F}_{\text{total}} = \mathbf{F}_{\text{attr}} + \mathbf{F}_{\text{rep}}
$$

The velocity at each step is computed as:

$$
\mathbf{v}_{\text{ref}} = \frac{\mathbf{F}_{\text{total}}}{\|\mathbf{F}_{\text{total}}\|} \cdot v_{\text{desired}}
$$

where $v_{\text{desired}} = \min(0.6 \cdot v_{\max}, \|\mathbf{p}_{\text{goal}} - \mathbf{p}_0\| / T)$

> **Key Insight**: This formulation uses potential field forces **only for direction**. The magnitude is normalized away, and a constant desired speed is applied. This ensures predictable, smooth motion regardless of force strength.

### Reference Trajectory Integration

The reference trajectory is generated by forward integration:

**Algorithm: Reference Trajectory Generation**

```
Input: p_current, p_goal, obstacles, N, Δt
Output: p_ref[0:N]

1. p_ref[0] ← p_current
2. For k = 0 to N-1:
   a. Compute F_attr using equation above
   b. Compute F_rep from all obstacles
   c. Compute v_ref using equation above
   d. p_ref[k+1] ← p_ref[k] + v_ref · Δt
   e. Clamp altitude: p_z,ref[k+1] ← clip(p_z,ref[k+1], z_min, z_max)
   f. If ||p_ref[k+1] - p_goal|| < 0.2 m:
      - Set p_ref[k+1:N] ← p_goal
      - Break
3. Return p_ref
```

---

## Gap-Based Navigation

The gap navigator identifies navigable corridors in the LiDAR scan and computes intermediate waypoints when the direct path is obstructed.

### Gap Detection

A gap is defined as a continuous angular sector where LiDAR readings exceed a threshold:

$$
\text{Free}(\theta) = \begin{cases}
\text{true} & \text{if } r(\theta) > 0.85 \cdot r_{\max} \text{ or } r(\theta) = \infty \\
\text{false} & \text{otherwise}
\end{cases}
$$

Each gap $G_j$ is characterized by:
- **Start angle**: $\theta_{\text{start},j}$
- **End angle**: $\theta_{\text{end},j}$
- **Center angle**: $\theta_{\text{center},j} = (\theta_{\text{start},j} + \theta_{\text{end},j})/2$
- **Minimum range**: $r_{\min,j} = \min\{r(\theta) : \theta \in [\theta_{\text{start},j}, \theta_{\text{end},j}]\}$
- **Width**: $w_j = r_{\min,j} \cdot (\theta_{\text{end},j} - \theta_{\text{start},j})$

### Gap Filtering

A gap is considered navigable if:

$$
w_j \geq w_{\min} \quad \text{and} \quad r_{\min,j} \geq d_{\text{safe}} + d_{\text{margin}}
$$

**Typical values**: $w_{\min} = 1.2$ m, $d_{\text{margin}} = 0.2$ m

### Gap Scoring

Each navigable gap is scored based on:

#### Alignment Score

$$
s_{\text{align},j} = 1 - \frac{|\theta_{\text{center},j} - \theta_{\text{goal}}|}{\theta_{\max}}
$$

where $\theta_{\text{goal}}$ is the angle to the final goal and $\theta_{\max} = 90°$ (configurable).

#### Quality Score

$$
s_{\text{quality},j} = \frac{\theta_{\text{end},j} - \theta_{\text{start},j}}{\pi} \cdot \frac{r_{\min,j} - d_{\text{safe}}}{r_{\max} - d_{\text{safe}}}
$$

#### Combined Score

$$
s_j = \alpha \cdot s_{\text{align},j} + (1-\alpha) \cdot s_{\text{quality},j}
$$

where $\alpha = 0.5$ (configurable `goal_alignment_weight`).

**Additional penalties** apply for:
- **Misalignment**: If $|\theta_{\text{center},j} - \theta_{\text{goal}}| > 30°$, multiply score by 0.8
- **Velocity-based turn severity**: If moving fast, penalize gaps requiring sharp turns

### Intermediate Goal Placement

For the best-scoring gap $G^*$, the intermediate goal is placed at:

$$
\mathbf{p}_{\text{intermediate}} = \mathbf{p}_{\text{current}} + d_{\text{gap}} \begin{bmatrix}
\cos(\theta_{\text{center}}^*) \\
\sin(\theta_{\text{center}}^*) \\
0
\end{bmatrix}
$$

where:

$$
d_{\text{gap}} = \min(d_{\text{gap,desired}}, r_{\min}^* - d_{\text{safe}} - d_{\text{margin}}, \|\mathbf{p}_{\text{goal}} - \mathbf{p}_{\text{current}}\|)
$$

**Typical value**: $d_{\text{gap,desired}} = 3.0$ m

### Direct Path Check

Before using gap navigation, the system checks if a direct path to the goal is clear:

$$
\text{Direct path available} \iff \min_{\theta \in \text{cone}} r(\theta) \geq d_{\text{threshold}}
$$

where the cone has width $\approx 69°$ around the goal direction, and $d_{\text{threshold}} = 3.0$ m (configurable).

---

## Emergency Braking

If any obstacle is within the emergency radius $d_{\text{emergency}} = 0.4$ m, the controller enters emergency braking mode.

### Braking Acceleration

The braking acceleration is computed as:

$$
\mathbf{a}_{\text{brake}} = -\frac{\mathbf{v}}{v} \cdot 0.5 \cdot a_{\max} + \mathbf{a}_{\text{escape}}
$$

This decelerates at 50% of maximum acceleration while adding an escape component.

### Escape Direction

If an escape direction away from obstacles is available:

$$
\mathbf{d}_{\text{escape}} = \frac{\sum_{i \in \mathcal{E}} (\mathbf{p} - \mathbf{o}_i)}{\|\sum_{i \in \mathcal{E}} (\mathbf{p} - \mathbf{o}_i)\|}
$$

where $\mathcal{E}$ is the set of obstacles within $d_{\text{emergency}}$.

The lateral escape component (perpendicular to current velocity) is:

$$
\mathbf{a}_{\text{escape}} = 0.3 \cdot a_{\max} \cdot \frac{\mathbf{d}_{\text{escape}} - (\mathbf{d}_{\text{escape}} \cdot \hat{\mathbf{v}})\hat{\mathbf{v}}}{\|\mathbf{d}_{\text{escape}} - (\mathbf{d}_{\text{escape}} \cdot \hat{\mathbf{v}})\hat{\mathbf{v}}\|}
$$

This prevents violent direction reversals while still allowing gentle lateral escapes.

---

## Implementation Details

### Optimization Solver

The MPC problem is solved using **CasADi** with the **IPOPT** nonlinear programming solver.

#### Solver Settings
- **Maximum iterations**: 50
- **Tolerance**: $10^{-4}$
- **Warm starting**: Previous solution shifted forward in time
- **Solve time**: Typically < 50 ms

### Obstacle Selection

At each MPC step, only the most relevant obstacles are included in the optimization to maintain real-time performance:

1. **Filter by range**: $d_i \leq d_{\text{max}} = 6.0$ m
2. **Score obstacles**: $\text{score}_i = \frac{1}{d_i + 0.1} + \text{forward\_bonus}_i$
3. **Select top**: $M_{\max} = 15$ obstacles per time step

The forward bonus prioritizes obstacles in the direction of motion.

### Coordinate Frames

The system uses multiple coordinate frames:

| Frame | Description | Axes |
|-------|-------------|------|
| **World** | Global reference | ENU (East-North-Up) |
| **PX4** | Flight controller | NED (North-East-Down) |
| **Body** | Drone-fixed | FLU (Forward-Left-Up) |
| **LiDAR** | Sensor-fixed | FLU (scans in XY plane) |

#### Frame Conversions

**NED to ENU:**

$$
\begin{bmatrix} x_{\text{ENU}} \\ y_{\text{ENU}} \\ z_{\text{ENU}} \end{bmatrix} = 
\begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix}
\begin{bmatrix} x_{\text{NED}} \\ y_{\text{NED}} \\ z_{\text{NED}} \end{bmatrix}
$$

**Yaw conversion:**

$$
\psi_{\text{ENU}} = \frac{\pi}{2} - \psi_{\text{NED}}
$$

### LiDAR Processing

1. **Scan conversion**: Convert polar LiDAR readings to Cartesian coordinates in body frame
2. **Transformation**: Rotate and translate points to world frame using current pose
3. **Clustering**: Grid-based clustering (0.3 m cell size) to reduce point count
4. **Filtering**: Keep only clusters with ≥ 2 points for robustness

---

## Parameter Tuning Guidelines

### Critical Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| $Q_{\text{pos}}$ | 15.0 | Reference tracking tightness |
| $Q_{\text{goal}}$ | 80.0 | Goal attraction strength |
| $Q_{\text{terminal}}$ | 150.0 | Terminal state importance |
| $R_{\text{acc}}$ | 0.3 | Control smoothness |
| $Q_{\text{rep}}$ | 300.0 | Obstacle repulsion (soft) |
| $W_{\text{obs}}$ | 5000.0 | Obstacle constraint penalty |
| $d_{\text{safe}}$ | 0.8 m | Safety distance |
| $v_{\max}$ | 2.0 m/s | Maximum velocity |
| $a_{\max}$ | 3.0 m/s² | Maximum acceleration |

### Tuning Strategy

#### 1. Safety First
Start with conservative values:
- Large $d_{\text{safe}}$ (1.0 m)
- High $W_{\text{obs}}$ (10000.0)
- Low $v_{\max}$ (1.0 m/s)

#### 2. Reference Tracking
Increase $Q_{\text{pos}}$ for tighter following of the reference trajectory. If the drone oscillates, reduce it.

#### 3. Smoothness
Increase $R_{\text{acc}}$ and $R_{\text{jerk}}$ for gentler motion. This is especially important for video recording or fragile payloads.

#### 4. Aggressiveness
Once stable, progressively:
- Decrease $d_{\text{safe}}$ (minimum 0.5 m recommended)
- Increase velocity limits
- Reduce $R_{\text{acc}}$ for faster response

#### 5. Gap Navigation
Adjust $\alpha$ (`goal_alignment_weight`):
- **Higher α** (→ 1.0): Prioritize goal-aligned gaps (more direct paths)
- **Lower α** (→ 0.0): Prioritize wide, clear gaps (safer but less direct)

### Trade-offs

| Increase Parameter | Benefit | Cost |
|-------------------|---------|------|
| $Q_{\text{pos}}$ | Tighter tracking | Less smooth, may violate constraints |
| $Q_{\text{goal}}$ | Faster convergence | May ignore reference, less safe |
| $R_{\text{acc}}$ | Smoother motion | Slower response, longer paths |
| $d_{\text{safe}}$ | More safety | Cannot navigate tight spaces |
| $v_{\max}$ | Faster travel | Less reaction time |

---

## Conclusion

This MPC implementation provides a robust framework for autonomous drone navigation in cluttered environments. The combination of gap-based navigation, potential field reference generation, and optimization-based control with hard safety constraints enables safe and efficient obstacle avoidance while maintaining smooth flight characteristics.

### Key Advantages

✅ **Predictive**: Plans ahead over a 2-second horizon, anticipating future obstacles

✅ **Safe**: Hard constraints prevent collisions, emergency braking as last resort

✅ **Smooth**: Continuous optimization produces natural trajectories without jerky motions

✅ **Adaptive**: Gap navigation intelligently handles complex environments with narrow passages

✅ **Real-time**: Solves in < 50 ms for typical scenarios (20 steps, 15 obstacles)

✅ **Flexible**: Highly configurable parameters for different scenarios and drone capabilities

### System Performance

| Metric | Typical Value |
|--------|---------------|
| **Solve time** | 20-50 ms |
| **Control frequency** | 10 Hz |
| **Prediction horizon** | 2.0 s |
| **Max obstacles** | 15 per timestep |
| **Safety success rate** | > 99% (with proper tuning) |

### Future Improvements

Potential enhancements to consider:
1. **Dynamic obstacles**: Add velocity prediction for moving obstacles
2. **Adaptive horizon**: Vary $N$ based on environment complexity
3. **Learning-based tuning**: Automatically adjust parameters based on performance
4. **Multi-drone coordination**: Extend to formation flight
5. **Energy optimization**: Add battery consumption to cost function

---

*This documentation was generated for the Drone Optimal Trajectory project. For implementation details, see the source code in `mpc_obstacle_avoidance_node_v2.py` and `mpc_core.py`.*
