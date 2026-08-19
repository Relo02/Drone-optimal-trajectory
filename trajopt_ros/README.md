# trajopt_ros

ROS 2 wrappers for [`trajopt_core`](../trajopt_core/README.md): a global A\* node
and a local MPC node that contain **no algorithm and no platform knowledge**.

`trajopt_core` is deliberately ROS-free so it can be unit-tested and benchmarked
without a robot. This package is the adapter: it converts ROS messages into the
core's data structures and back, and nothing else.

```bash
ros2 launch trajopt_ros planner.launch.py profile:=aerial
ros2 launch trajopt_ros planner.launch.py profile:=legged
ros2 launch trajopt_ros planner.launch.py profile:=g1
```

Full runbook — simulators, goal publishing, RViz, metric recording:
[`RUNNING.md`](../RUNNING.md).

---

## Where the platform knowledge lives

In exactly two places, both declarative:

| | file | size |
|---|---|---|
| model, limits, weights that differ (Go2) | `trajopt_core/config/legged_overrides.yaml` | ~25 lines of deltas |
| model, limits, weights that differ (G1) | `trajopt_core/config/g1_overrides.yaml` | ~70 lines, mostly comments |
| which platform topics the relative names map onto | `launch/planner.launch.py`, `PROFILES` | one dict |

The node sources mention neither robot. Switching profile swaps the motion model,
the admissible sets and the wiring, and recompiles nothing. The G1 profile reuses
the same `kinematic_se2` model, the same mid-point integrator and the same
sigmoid obstacle barrier as the Go2 — it is a tuning of the legged stack, not a
fourth bespoke one; see "What was not ported for the G1" below.

## Topic map

Node-side names are **relative**; the launch file remaps them.

| relative | aerial | legged | g1 |
|---|---|---|---|
| `pose` | `/skydio/pose` | `/go2/pose` | `/robot_pose` |
| `scan` | `/skydio/scan3d` | `/lidar/points_filtered` | `/scan` |
| `global_goal` | `/global_goal` | `/global_goal` | `/global_goal` |
| `path` | `/a_star/path` | `/a_star/path` | `/a_star/path` |
| `occupancy_grid` | `/a_star/occupancy_grid` | `/a_star/occupancy_grid` | `/a_star/occupancy_grid` |
| `predicted_path` | `/mpc/predicted_path` | `/mpc/predicted_path` | `/mpc/predicted_path` |
| `next_setpoint` | `/goal_pose` → cascaded PID | `/mpc/next_setpoint` → `setpoint_to_cmd_vel_node` | `/mpc/next_setpoint` → `setpoint_to_cmd_vel_node` |
| `diagnostics` | `/mpc/diagnostics` | `/mpc/diagnostics` | `/mpc/diagnostics` |

`diagnostics` carries `[success, cost, solve_ms, iterations, cumulative_failures]`.

`scan` **must be a PointCloud2 already expressed in the planning frame**, for
every profile alike — this is a hard requirement, not a default. The aerial
platform gets this from `skydio_sim_node`, the G1 in simulation from
`g1_mujoco/g1_sim_node` (which ray-casts and returns the hits already in world
coordinates), the Go2 from `cloud_self_filter`. On the **real** G1 the upstream
LiDAR pipeline publishes a `LaserScan` in a sensor frame; turning that into a
world-frame `PointCloud2` is platform-side bridging (analogous to
`cloud_self_filter`) that belongs to the G1's own bringup, not to
this package — adding `LaserScan`+TF support here would give the G1 profile a
capability the other two don't have.

**The MPC publishes a setpoint, not an input.** The optimiser is a reference
generator for a faster platform-specific inner loop. That is not a detail of
implementation: measured on the same scenario, applying the first optimal input
directly leaves the robot 8–11 m from the goal after a fixed number of cycles,
where the lookahead setpoint leaves it 1.9–2.9 m.

---

## Wiring into a workspace

Both packages are plain `ament_python` and build with `colcon` unmodified.

### Aerial — already wired

`trajopt_core` and `trajopt_ros` live in this repository, so the MuJoCo workspace
reaches them through relative symlinks:

```
mujoco/ros2_ws/src/trajopt_core -> ../../../trajopt_core
mujoco/ros2_ws/src/trajopt_ros  -> ../../../trajopt_ros
```

```bash
cd mujoco/ros2_ws
colcon build --symlink-install --packages-select trajopt_core trajopt_ros
source install/setup.bash
ros2 launch trajopt_ros planner.launch.py profile:=aerial
```

The existing `new_mujoco` package keeps `skydio_sim_node` (physics, 3-D LiDAR,
cascaded PID). Its `a_star_node` and `mpc_node` are superseded by these.

### Legged — a decision to make first

The Go2 workspace is a **separate git repository** on the Windows filesystem
(`/mnt/c/Users/franc/Desktop/Tesi/Go2_navigation`), so it cannot use a relative
symlink. Three options, in order of preference:

1. **Promote `trajopt_core` + `trajopt_ros` to their own repository** and add it
   as a `git submodule` under `src/` of both workspaces. Cleanest, makes "the same
   package runs on both robots" literally true of the deployed system, and is what
   the report describes. Requires creating and pushing a new repository — a
   decision left to you rather than taken here.
2. **`pip install -e`** both packages into the environment the Go2 workspace uses,
   and copy just `launch/planner.launch.py`. Fastest to try; the drawback is that
   the version is no longer pinned by git.
3. **Absolute symlink** into `Go2_navigation/src/`. Works on this machine only —
   fine for a demo, not for anything reproducible.

Whichever is chosen, the Go2 side keeps `odom_to_pose_node` (EKF bridge) and
`setpoint_to_cmd_vel_node` (setpoint → `/cmd_vel`); its `a_star_node` and
`mpc_node` are superseded.

### Go2 — simulated here, same options as on hardware

For the **real robot**, the Go2 workspace is a separate repository, so the same
three options above apply (submodule preferred). Its own bringup keeps whatever
pose source and `LaserScan`→`PointCloud2` bridging it already has, plus a
`setpoint → cmd_vel` consumer of `next_setpoint`; its own planner/tracker nodes
are superseded by these two.

In **simulation** nothing has to be decided: `mujoco/ros2_ws/src/go2_mujoco`
provides a MuJoCo Go2 in an industrial warehouse and satisfies the `legged`
profile's contract directly — `/go2/pose`, and `/lidar/points_filtered` as a
PointCloud2 already in the planning frame — so this package drives it
unmodified:

```bash
ros2 launch go2_mujoco warehouse.launch.py goal_x:=12.5 goal_y:=0.0
```

That plant is kinematic: it integrates the commanded body velocity, which makes
it exactly the `KinematicSE2` model the profile optimises over. See
[`RUNNING.md`](../RUNNING.md) §3.

## What was not ported for the G1

The G1's source project (`mpc_planner.py` / `mpc_tracker.py`) is a more capable
stack than what is ported here: it tracks dynamic obstacles and yields to them,
rate-limits commands, watches per-input freshness, visualises a cost field, and
gets its global path from Nav2 over a SLAM map. None of that exists in the
aerial or Go2 instantiations, so none of it was carried over — adding it only
for the G1 would give it a capability the other two platforms don't have,
which breaks the entire premise of this package: that the three robots solve
*the same* OCP and can be compared on those terms. Everything listed below
remains available, unmodified, in the Unitree-G1 project itself; it was
consulted only as a source of numeric tuning values (see
`trajopt_core/config/g1_overrides.yaml`), never copied as code.

- Dynamic-obstacle prediction and "yield" behaviour
- Command slew-rate limiter
- Per-input freshness watchdogs
- Cost-field RViz visualisation
- Nav2/SLAM as the global planning layer (this package always uses the shared
  A\* node over the rolling occupancy grid, for every profile)

The same reasoning governed what was taken from the CIHR lab's MuJoCo G1
simulator and adapted for the Go2 wrapper. Kept: the warehouse geometry, the
lidar ray-cast, the kinematic base. Left out: the RL walking policy and its DDS
bridge (a locomotion question, and a heavyweight dependency — the planner is
tested against its own model here, as on the Go2), Nav2 and slam_toolbox (this
stack plans without a prior map, on every platform), and the simulator's moving
"people" — for the same reason dynamic obstacles were left out of the planner.

---

## Tests

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=$PWD/trajopt_core:$PWD/trajopt_ros:$PYTHONPATH
python3 -m pytest trajopt_ros/test -q
```

The smoke test instantiates both nodes in-process for **all three profiles**,
wires them to a fake robot publishing a pose and a wall of LiDAR hits (a
PointCloud2, in the planning frame, same as every profile requires), and checks
the whole chain end to end: configuration loading, message conversion, grid,
A\*, the OCP, lookahead extraction, publication — and that the CasADi graph was
built exactly once and reused, for each profile alike.

---

## Known limitations

- **Velocity is differenced from consecutive poses, unfiltered.** Relative-degree-1
  models ignore it; for the aerial platform this is the weakest link in the chain
  and the natural place for a proper estimator (an EKF, or the moving-horizon
  estimator of the course notes §7.3.2).
- **No TF broadcasting.** The nodes assume poses already arrive in the planning
  frame. The previous `new_mujoco/a_star_node` re-broadcast `world → base_link`;
  if a viewer depends on that, keep a small dedicated broadcaster rather than
  putting it back into a platform-independent node.
- **The occupancy grid has no memory across cycles**, by design — this stack
  navigates without a prior map. It also means it cannot escape a trap larger than
  the 10 m window. Adding accumulation would benefit both platforms at once, which
  is the point of having a shared core.
