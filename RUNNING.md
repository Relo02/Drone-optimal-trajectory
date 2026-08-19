# Running everything

Every command needed to (1) run the three platforms and (2) produce the numbers
the report quotes. Nothing here is a new capability — it is the operating manual
for what is already in [`trajopt_core`](trajopt_core/README.md) and
[`trajopt_ros`](trajopt_ros/README.md).

Two ways to run the stack, and it is worth being explicit about which is which:

| | what runs | needs |
|---|---|---|
| **headless mission simulator** | the real core (grid → A\* → OCP → lookahead → inner loop), a point-mass plant, no ROS | Python + CasADi only |
| **ROS 2 closed loop** | the same core inside the two nodes, driving a real or simulated robot | ROS 2 Humble + that robot's workspace |

The headless simulator is what produces every metric below: it is deterministic,
it runs on all three platforms with the *same* code path, and it does not need a
robot. The ROS path is what proves the same core survives contact with one.

**MuJoCo covers the drone, the G1 and the Go2.** The drone's scene comes in two
generations, only one of which runs today (read §1 before assuming which); the
G1 walks a 30 x 20 m warehouse and is driven by the shared stack (§3). The Go2
uses the same industrial warehouse in `go2_mujoco`. The three-way comparison is
still the headless simulator, because only it runs the identical code path on
all three.

---

## 0. Setup

```bash
cd ~/Drone-optimal-trajectory
python3 -m pip install --user casadi numpy scipy pyyaml matplotlib
```

For anything ROS:

```bash
source /opt/ros/humble/setup.bash
cd ~/Drone-optimal-trajectory/mujoco/ros2_ws
colcon build --symlink-install --packages-select \
  trajopt_core trajopt_ros go2_mujoco mujoco_sim new_mujoco
source install/setup.bash
```

`--symlink-install` matters: the config files are symlinked, so editing
`trajopt_core/config/*.yaml` takes effect without rebuilding.

---

## 1. Drone in MuJoCo

There are **two** MuJoCo setups in this repository, of different generations.
They are easy to confuse and only one of them runs today.

| | `mujoco_sim` (§1.1) | `new_mujoco` (§1.2) |
|---|---|---|
| airframe | Crazyflie 2, `mujoco/model/drone_world.xml` | Skydio X2, `skydio_x2/skydio_world.xml` |
| sensor | 2-D LiDAR ring (367 sensors in the scene) | 3-D LiDAR, 576 rangefinder sites |
| planner | its own A\*, MPC and Gaussian grid, inside the package | `trajopt_core`, via the `trajopt_ros` nodes |
| assets | **in the repository** — meshes and scene both | **missing** — see §1.2 |
| runs today | yes | no |

So the mapless demo — global A\* + local MPC over a Gaussian occupancy grid,
with path smoothing — is §1.1, and it works. What does *not* run is the newer
Skydio scene, which is the one the unified stack was written against.

### 1.1 Crazyflie mapless navigation — the original demo

```bash
cd ~/Drone-optimal-trajectory/mujoco/ros2_ws
colcon build --symlink-install --packages-select mujoco_sim
source install/setup.bash

# plant + local MPC + Gaussian grid, one node, with the viewer
ros2 launch mujoco_sim mpc_sim.launch.py goal_x:=10.0 goal_y:=1.0 goal_z:=1.5
```

Or the A\*-planner variant, where the plant tracks a smoothed global path:

```bash
ros2 run mujoco_sim mujoco_sim_node        # plant: /drone/pose, /lidar/points
ros2 run mujoco_sim a_star_planner_node    # global path on /planned_path
```

Everything is published in the `world` frame. `mpc_sim.launch.py` also starts
`mpc_viz_node`, a live matplotlib view of the grid and the predicted trajectory.

The scene path used to be hardcoded to `/home/lorenzo/...`, so the simulation
could only start on the machine that wrote it; it is now resolved from the
repository layout, with `MUJOCO_DRONE_WORLD` as an override.

**This generation does not use `trajopt_core`** — it carries its own MPC and its
own A\*. It is the predecessor of the unified stack, kept because it runs.

### 1.2 Skydio X2 + the unified stack — assets missing

This is the setup the `aerial` profile targets (`/skydio/pose`,
`/skydio/scan3d`), and it cannot start as things stand:

```bash
bash mujoco/MuJoCo/download_menagerie.sh
```

fetches the stock MuJoCo Menagerie, which is **not** enough: the scene actually
loaded is `skydio_x2/skydio_world.xml`, a custom file carrying the 576
rangefinder sites that produce `/skydio/scan3d`. It is not in this repository
and not in its git history, and neither is the `generate_x2_lidar.py` that
produced it. Until one of the two is restored, `skydio_sim.launch.py` aborts
with `Model not found`. `new_mujoco/config/skydio_params.yaml` also still
carries a `/home/lorenzo/...` path.

Once the scene is back, four terminals, each with `source install/setup.bash`:

```bash
# 1 — physics, 3-D LiDAR, cascaded PID inner loop
ros2 launch new_mujoco skydio_sim.launch.py

# 2 — the planner: A* + MPC, aerial profile
ros2 launch trajopt_ros planner.launch.py profile:=aerial

# 3 — send a goal (or click "2D Goal Pose" in RViz)
ros2 topic pub --once /global_goal geometry_msgs/PoseStamped \
  '{header: {frame_id: "world"}, pose: {position: {x: 20.0, y: 1.0, z: 1.5}, orientation: {w: 1.0}}}'

# 4 — watch it
rviz2
```

In RViz add: `/a_star/occupancy_grid` (Map), `/a_star/path` and
`/mpc/predicted_path` (Path), `/skydio/scan3d` (PointCloud2), `/skydio/pose`
(Pose). Fixed frame `world`.

Live solver health, one line per cycle — `[success, cost, solve_ms, iterations,
cumulative_failures]`:

```bash
ros2 topic echo /mpc/diagnostics
```

Everything in §4 and §5 (the metrics) runs without any of this.

## 2. Go2 (quadruped)

There is no simulator for the Go2 in this repository — its workspace is a
separate repo. Two things you can do:

**Closed-loop mission, headless, here:**

```bash
cd trajopt_core
PYTHONPATH=. python3 examples/cross_platform_demo.py     # runs all three platforms
```

**On the robot / its own workspace**, once the two packages are available there
(see the wiring options in [`trajopt_ros/README.md`](trajopt_ros/README.md)):

```bash
ros2 launch trajopt_ros planner.launch.py profile:=legged
```

It expects `/go2/pose` (from `odom_to_pose_node`) and
`/lidar/points_filtered` (a world-frame PointCloud2, from `cloud_self_filter`),
and publishes `/mpc/next_setpoint` for `setpoint_to_cmd_vel_node`.

## 3. Go2 (quadruped) — MuJoCo warehouse, one command

```bash
cd ~/Drone-optimal-trajectory/mujoco/ros2_ws
colcon build --symlink-install --packages-select trajopt_core trajopt_ros go2_mujoco
source install/setup.bash

ros2 launch go2_mujoco warehouse.launch.py
ros2 launch go2_mujoco warehouse.launch.py goal_x:=12.5 goal_y:=0.0   # pick a goal
ros2 launch go2_mujoco warehouse.launch.py planner:=false             # plant only
```

That single launch starts three things, and the middle one is the point — it
is not part of the Go2 package:

| node | role | where it lives |
|---|---|---|
| `go2_sim_node` | plant: MuJoCo Go2 in a 30 x 20 m warehouse, simulated lidar (8640 rays, 10 Hz) → `/go2/pose`, `/lidar/points_filtered` | `go2_mujoco` |
| `a_star_node` + `mpc_node` | the planner, `profile:=legged` | **`trajopt_ros`, shared with the drone and the Go2** |
| `setpoint_to_cmd_vel_node` | inner loop: setpoint → `/cmd_vel` | `go2_mujoco` |

The robot spawns at `(-12, 0)`; the default goal is `(12.5, 0)`. Pick goals in
free space: the warehouse is dense, and a goal that lands **inside** an
obstacle — `(6, 2)` is on a pallet, for instance — makes the robot stop short
and hold there, which is the planner refusing to enter an obstacle, not a
failure. `warehouse_geoms()` in `go2_mujoco/warehouse_world.py` lists every
obstacle with its coordinates.

Watch it in the MuJoCo viewer (`viewer:=false` for headless), and the planner
from RViz on `/a_star/path`, `/mpc/predicted_path`, `/a_star/occupancy_grid`,
fixed frame `odom`. `/mpc/diagnostics` carries the same five numbers as on the
other platforms, so §5's plotting works here unchanged.

Two things this simulation deliberately is and is not:

- **The base is kinematic**, integrating the commanded body velocity. That makes
  the plant exactly `KinematicSE2` — the model the `legged` profile optimises
  over, relative degree 1 by construction. It tests the planner against its own
  model, which is the scope of a planning experiment and the same scope as the
  Go2's.
- **The cloud is published in the planning frame**, so no TF is involved. Every
  profile requires that; on the real Go2 it is the bringup's job.

The same profile drives the real robot, from its own workspace:

```bash
ros2 launch trajopt_ros planner.launch.py profile:=legged
```

Headless, without MuJoCo, the Go2 also runs in the cross-platform demo alongside
the other two.

> **If a node dies with `rmw_create_node: failed to create domain`**, that is
> `CYCLONEDDS_URI` in your shell pinning a network interface that is currently
> down (`eth2` here). Point it at an interface that is up:
> ```bash
> export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>eth1</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
> ```


## 4. Recording the metrics

### 4.1 The cross-platform comparison (the core claim)

```bash
cd trajopt_core
PYTHONPATH=. python3 examples/cross_platform_demo.py --out ../results/cross_platform
```

Prints, and with `--out` writes:

| artefact | content |
|---|---|
| stdout instantiation table | the three state/input spaces side by side, generated from the code |
| stdout NLP structure table | variables, constraints, Jacobian/Hessian sparsity per platform |
| `summary.json` | mission metrics: reached, time, path length, min clearance, tracking error |
| `solves_*.csv` | one row per solve: cost, iterations, solve time, status |
| `traj_*.csv` | the executed trajectory, for plotting |

Runtime: about 3 minutes for 3 platforms × 2 scenarios.

### 4.2 The formulation benchmarks (the course topics)

```bash
cd trajopt_core
PYTHONPATH=. python3 examples/bench_formulations.py --out docs
```

That regenerates [`trajopt_core/docs/benchmarks.md`](trajopt_core/docs/benchmarks.md)
in place. Each experiment isolates **one** modelling decision, and each maps onto
a section of the course notes:

| `--only` | question it answers | course notes |
|---|---|---|
| `build` | is a parametric NLP built once actually cheaper than rebuilding? | §7.2.2 |
| `horizon` | how do NLP size and sparsity grow with N? | §7.2.2 (multiple shooting) |
| `barrier` | C¹ hinge² vs C^∞ logistic barrier — smoothness vs conditioning | §4.2.5, §4.4.4 |
| `penalty` | exact L¹ vs quadratic L² slack penalty, ρ sweep | §6.3.3, Thm 6.3.1 |
| `discretise` | forward Euler vs mid-point rule, order and closed-loop effect | §2.1.3 |
| `warmstart` | shifted solution vs reference vs cold start | §7.1.1, §7.2.5 |
| `pathcond` | does conditioning the A\* polyline help the optimiser? | §7.1.1 |

One experiment at a time (the full sweep takes ~25 min):

```bash
PYTHONPATH=. python3 examples/bench_formulations.py --only barrier,penalty
```

Two of these results contradicted the expectation going in and are worth
reporting as such: build-once is *not* uniformly a win (a baked grid term is far
cheaper to evaluate than a parametric one), and the smoother barrier costs
roughly 3× the iterations of the non-smooth one.

### 4.3 From a live ROS run

```bash
mkdir -p results/rosbag && cd results/rosbag
ros2 bag record /mpc/diagnostics /mpc/predicted_path /a_star/path \
                /skydio/pose /mpc/next_setpoint
```

`/mpc/diagnostics` is a `Float64MultiArray` laid out as
`[success, cost, solve_ms, iterations, cumulative_failures]`. To get a CSV of
solve times out of it, echo while the bag replays — terminal 1 first, then
terminal 2:

```bash
ros2 topic echo --csv /mpc/diagnostics > diagnostics.csv   # terminal 1
ros2 bag play results/rosbag                               # terminal 2
```

The headless simulator records the same five quantities per solve, on every
platform, without a bag — prefer it whenever the question is about the
*optimiser* rather than about the *integration*.

---

## 5. Visualizing the metrics

`examples/plot_metrics.py` turns the recorded artefacts into figures. Recording
and plotting are separate steps on purpose: a mission takes minutes, changing an
axis label should not.

```bash
cd trajopt_core
PYTHONPATH=. python3 examples/plot_metrics.py ../results/cross_platform
# -> ../results/cross_platform/figures/*.png
```

| figure | what it shows |
|---|---|
| `trajectories.png` | the executed paths of all three platforms over the obstacles, one panel per scenario, with length and time in the legend |
| `solve_time.png` | box plot + empirical CDF of solve time, **log axes**, with the `dt` deadline drawn on both — the tail is what decides real-time admissibility |
| `per_cycle.png` | IPOPT iterations and optimal value `J*` cycle by cycle: the peaks are the obstacles |
| `mission_summary.png` | the trade-off in four bars: time, path length, min clearance, tracking error |

Colours are fixed per platform (aerial blue, legged red, g1 green) across every
figure, and scenarios are distinguished by line style, so plots from different
runs stay comparable.

From a live ROS run instead, plot the diagnostics CSV directly:

```bash
PYTHONPATH=. python3 examples/plot_metrics.py --diagnostics ../results/rosbag/diagnostics.csv
```

That produces solve time (with its p95), iterations and `J*` per cycle, and
titles the figure with the cycle count, the solved percentage and the failure
count. It tolerates a leading timestamp column, so `ros2 topic echo --csv`
output goes in unmodified.

Both modes accept `--out DIR`, and both can be given at once.

The log axis on `solve_time.png` is not cosmetic: the first solve of every run
carries the CasADi graph construction and lands one to two orders of magnitude
above the steady-state cycle, so on a linear axis that single point flattens
everything else into a flat line.

---

## 6. Verifying the stack

```bash
# core: models, OCP, config, planning, golden-equivalence
PYTHONPATH=trajopt_core python3 -m pytest trajopt_core/tests -q        # 61 tests

# ROS: both nodes, all three profiles, end to end
source /opt/ros/humble/setup.bash
PYTHONPATH=$PWD/trajopt_core:$PWD/trajopt_ros python3 -m pytest trajopt_ros/test -q   # 5 tests
```

The config for a profile can be checked without running anything:

```bash
PYTHONPATH=trajopt_core python3 -c "
from trajopt_core.config_io import load_planner
s = load_planner('trajopt_core/config/planner_params.yaml',
                 'trajopt_core/config/g1_overrides.yaml')
print(s.model.describe()); print(s.model.input_bounds())"
```

---

## 7. Reading the numbers

A few things that will look wrong at first glance and are not:

- **`min_clearance` below `r_safe`.** The barrier is a *penalty*, not a
  constraint — clearance is bought against tracking error, and the G1 profile's
  very high position weight (`q_pos_xy = 200`) deliberately buys less of it than
  the Go2's (30). That is the Pareto trade-off of §7.4, visible in one column.
- **`deadline_miss_rate > 0` with `success_rate = 1`.** The solver converged;
  it just did not converge within one `dt`. The first solve of a run also
  carries the graph construction, which is why `solve_ms_max` is an outlier.
- **The three platforms have different mission times on the same scenario.**
  They have different speed limits, not different algorithms. Compare
  iterations, solve-time distributions and clearance — those are the quantities
  the shared formulation actually controls.
