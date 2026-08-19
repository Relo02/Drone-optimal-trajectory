# trajopt_core

Platform-agnostic local trajectory optimisation: a Gaussian occupancy grid, a
rolling-horizon A\* planner and a CasADi/IPOPT receding-horizon optimal control
problem, written **once** and instantiated on an aerial robot, a quadruped and
a bipedal humanoid.

The package depends on neither ROS nor any simulator, so every claim below is
reproducible on a laptop without a robot in the loop.

```bash
PYTHONPATH=. python3 -m pytest tests/ -q          # 61 tests
PYTHONPATH=. python3 examples/cross_platform_demo.py
PYTHONPATH=. python3 examples/bench_formulations.py --out docs
```

---

## 1. The claim

Every layer of the stack — grid, A\*, obstacle cost, reference generation,
lookahead extraction — depends on the state **only through the planar position of
the centre of mass**, `p = Cx`. The robot therefore enters the optimal control
problem through exactly four objects:

| | meaning | where it lives |
|---|---|---|
| `f_Σ` | discrete-time dynamics | `MotionModel.step` |
| `U_Σ` | input admissible set | `MotionModel.input_bounds` |
| `X_Σ` | state admissible set | `MotionModel.add_state_constraints` |
| `C` | CoM output map | `MotionModel.PLANAR_IDX` |

Everything else in `trajopt_core.mpc` is shared verbatim. This is not a rhetorical
claim about the architecture: it is the class hierarchy, and
`examples/cross_platform_demo.py` runs the same `TrajectoryOCP` instance type with
different models.

**What actually differs is the relative degree** between the planned output and the
commanded input:

| | state `x` | input `u` | rel. deg. | why |
|---|---|---|---|---|
| aerial (drone) | `[px py pz vx vy vz ψ]` ∈ ℝ⁷ | `[ax ay az ψ̇]` ∈ ℝ⁴ | **2** | the inner attitude loop cannot change velocity instantaneously; the tilt limit `a_max ≈ g·tan(φ_max)` is a genuine dynamic constraint |
| legged (Go2) | `[px py ψ]` ∈ ℝ³ | `[vx vy ω]` ∈ ℝ³ | **1** | the gait controller accepts velocity commands and tracks them far faster than the planner |
| humanoid (G1) | `[px py ψ]` ∈ ℝ³ | `[vx vy ω]` ∈ ℝ³ | **1** | same model class and integrator as the Go2 — the whip-crank walking gait accepts velocity commands too |

The G1 is not a fourth bespoke stack: it is `KinematicSE2` again, with the same
mid-point integrator, driven by a set of numeric deltas
(`config/g1_overrides.yaml`) on top of the shared base — the exact same pattern
`config/legged_overrides.yaml` already uses for the Go2. The one genuinely new
piece is a restriction of the admissible set, not a new capability: a biped
should not walk backward, so `ModelLimits.v_min_xy = 0.0` makes `U_Σ` one-sided
in the forward direction while `f_Σ` stays untouched — the same "shrink `U_Σ`,
don't touch the dynamics" pattern already used for a nonholonomic base
(`v_max_lat = 0`).

Two caveats to state explicitly rather than bury:

- **Validity rests on time-scale separation** between the inner loop and the
  planner. The abstraction covers any platform whose CoM output is trackable by a
  faster inner loop — a nonholonomic base fits by setting `v_max_lat = 0` — and
  fails where actuation timing enters planning (high-speed footstep planning,
  manipulation).
- **Both instantiations plan in the horizontal plane.** The aerial model adds a
  dynamically decoupled altitude channel that does not participate in obstacle
  avoidance. This is not 3-D planning.

---

## 2. Layout

```
trajopt_core/
├── mapping/gaussian_grid_map.py    Gaussian occupancy grid (shared, unmodified)
├── planning/a_star_planner.py      rolling-horizon A*      (shared, unmodified)
├── models/                         THE platform abstraction
│   ├── base.py                     MotionModel, CostWeights, ModelLimits
│   ├── double_integrator_z.py      aerial
│   └── kinematic_se2.py            legged
├── mpc/
│   ├── ocp.py                      TrajectoryOCP — parametric, build-once
│   ├── reference.py                arc-length reference + path conditioning
│   ├── obstacles.py                four interchangeable obstacle strategies
│   ├── lookahead.py                setpoint extraction
│   └── config.py                   OCPConfig, SolverOptions
├── bench/                          mission simulator + solver-stats recorder
└── config_io.py                    one YAML schema, both robots
```

`bench/` lives in the core deliberately: the numbers reported for the two robots
are produced by the same code path, which is what makes a cross-platform
comparison admissible at all.

---

## 3. Deduplication is proven, not asserted

The mapping and planning layers were already byte-identical across the two
platform repositories (the diff was an import path, a docstring and a matplotlib
helper). `tests/generate_golden.py` runs a deterministic scenario through
`new_mujoco`, `a_star_mpc_planner` and `trajopt_core` and checks the three agree
exactly; golden vectors are checked in so the test also runs where neither
workspace is available.

```
core == new_mujoco (aerial)          grid (40,40), path 19 wpts
core == a_star_mpc_planner (legged)  grid (40,40), path 19 wpts
```

---

## 4. Course mapping: design choice → section → metric

The project is coursework for *Numerical Optimization for Control* (062047,
PoliMi, Fagiano). Every decision below carries the section that motivates it and
the measurement that settles it. Every table below is reproduced verbatim from
[`docs/benchmarks.md`](docs/benchmarks.md), regenerable with
`python3 examples/bench_formulations.py --out docs`.

Iteration counts are deterministic and repeat exactly across runs; wall-clock
timings were taken on a WSL2 laptop and move with machine load, so read the ratios
rather than the absolute milliseconds.

| # | Design choice | Course | Metric | Result |
|---|---|---|---|---|
| 1 | multiple shooting | §7.2.2 | Jacobian density, scaling in N | 0.4–2 %, linear in N |
| 2 | build-once parametric NLP | §7.2.2 | builds/mission, ms/solve | see 4.1 — **refuted** |
| 3 | exact ZOH vs Euler vs RK2 | §2.1.3 | fitted truncation order | 1.999 / 3.000 |
| 4 | CasADi + AD | §5.2–5.3 | derivative-callback share | `t_proc_nlp_*` in every record |
| 5 | interior point over active set | §6.2.2 | — | several hundred inequalities |
| 6 | C¹ hinge² vs C^∞ barrier | §4.2.5, §4.4.4 | iterations, clearance | see 4.2 — **refuted** |
| 7 | L¹ vs L² slack penalty | §6.3.3, Thm 6.3.1 | residual violation vs ρ | exact vs 1/ρ |
| 8 | warm start | §7.1.1, §7.2.5 | iterations vs baseline | see 4.3 — **refuted** |
| 9 | terminal ingredients | §7.2.5 | recursive feasibility | `terminal_zero_velocity` |
| 10 | LICQ-preserving point selection | §6.1.1 | — | `max_points`, per-sector |

### 4.1 Build-once is not free — hypothesis refuted

The occupancy grid translates with the robot, so a B-spline read off it has moving
knots and forces a rebuild of the CasADi graph every cycle. Putting the knots in
the local frame and passing the cell values as a parameter fixes that — verified
bit-identical — and it is nonetheless the **wrong trade**:

| strategy | builds | ms/solve (p50) | iterations |
|---|---|---|---|
| point barrier, parametric | 1 | 29.3 | 12.6 |
| grid B-spline, baked (rebuild) | 15 | 50.4 | 8.1 |
| grid B-spline, parametric | 1 | **14325** | 8.1 |

Identical iteration counts, and the entire penalty sits in the derivative
callbacks (`f` 2872 ms, `grad_f` 6189 ms, `hess_L` 5023 ms): with symbolic
coefficients CasADi cannot exploit the spline structure and re-evaluates the
tensor-product basis over all 1600 coefficients at each of the N+1 horizon points.

**Eliminating a rebuild only pays when the parametrisation leaves the
per-evaluation cost intact.** It does for the point-based terms; it does not for a
spline whose coefficients become symbolic. The deployed Go2 stack had already
reached the same conclusion empirically by dropping the grid from its MPC cost.

### 4.2 Convexity beats smoothness — hypothesis refuted

The hypothesis was that smoothness dominates, IPOPT being a Newton-type method and
`max(0,·)²` being C¹ but not C². The measurement says otherwise:

| platform / term | iterations | ms/solve | min clearance |
|---|---|---|---|
| aerial / hinge² (C¹) | **5.2** | 23.7 | 0.723 m |
| aerial / logistic (C^∞) | 14.4 | 70.3 | **0.862 m** |
| legged / hinge² (C¹) | **5.4** | 12.9 | 0.720 m |
| legged / logistic (C^∞) | 20.3 | 47.2 | **0.745 m** |

The squared hinge is a hinge of an *affine* function (the SCA half-space), hence
**convex**, and exactly inactive over most of the horizon most of the time. The
logistic barrier is isotropic — no normal, so no linearisation error — but
non-convex and contributing at every predicted point. Convexity plus activation
sparsity beat the extra derivative. The logistic buys clearance, so this is a
genuine cost/safety Pareto choice rather than a dominated option.

### 4.3 Warm starting against which baseline? — hypothesis refuted

| scenario / platform | shifted solution | reference | cold (zeros) |
|---|---|---|---|
| open field / aerial | 5.24 | **4.33** | 6.38 |
| open field / legged | 5.00 | **4.00** | 5.00 |
| corridor / aerial | 14.97 | 13.14 | **13.18** |
| corridor / legged | 18.74 | 17.83 | **16.26** |

Mean IPOPT iterations. The robust finding, 4 cases out of 4, is that the **shifted
previous solution is never the best guess** — the reference trajectory, which is
computed anyway, wins as a default.

A plausible mechanism: IPOPT is an interior-point method whose iterates must stay
strictly inside the feasible set (§6.2.2, the central path), and the previous
optimum sits *on* the boundary with constraints active. Report it as plausible,
not demonstrated: it explains why the shift is uniformly worst but not why the
cold/reference ranking flips between scenarios. Warm starting remains right for
active-set and SQP solvers, where the previous active set is exactly the reused
information — the advice does not transfer across solver families unexamined, and
any quoted speed-up is meaningless until its baseline is named.

### 4.4 Exact penalty — confirmed

| ρ | max slack, L¹ | max slack, L² |
|---|---|---|
| 20 | 1.35e-1 | 1.79e-1 |
| 200 | **0** | 1.07e-1 |
| 2 000 | **0** | 2.5e-2 |
| 20 000 | **0** | 3.0e-3 |

The theorem predicts a *threshold*, and the threshold is visible: L¹ is not exact
at ρ = 20, becomes exactly zero from ρ = 200 onwards, and stays there. The
multiplier |μ\*| of the hard constraint therefore lies between the two. L², by
contrast, leaves a residual of order μ\*/(2ρ) at every weight — decaying, never
vanishing.

Practical consequence: solve once with hard constraints offline, read μ\* from the
dual solution, set ρ above it. The slack weight stops being a tuning knob and
becomes a quantity derived from the problem.

### 4.5 Discretisation — confirmed

| Δt [s] | Euler | mid-point |
|---|---|---|
| 0.200 | 2.0e-2 | 1.3e-4 |
| 0.100 | 5.0e-3 | 1.6e-5 |
| 0.050 | 1.2e-3 | 2.0e-6 |
| 0.025 | 3.1e-4 | 2.5e-7 |
| **fitted order** | **1.999** | **3.000** |

The aerial double integrator under a zero-order-hold input is discretised
*exactly* (< 1e-12 against fine RK4): choosing the model so that discretisation is
exact is a modelling decision, not an accident. The SE(2) model is not, and the
mid-point rule buys an order for the price of one addition.

### 4.6 The architecture outweighs the tuning

Path conditioning of the jagged 8-connected A\* polyline helps modestly (6–17 %
less distance to goal at fixed budget). The **closed-loop mode dominates it**:

| mode | distance left after 150 cycles |
|---|---|
| lookahead setpoint (deployed) | 1.9 – 2.9 m |
| first optimal input applied directly | 8.9 – 11.3 m |

Using the optimiser as a *reference generator* rather than as a direct controller
is worth several times more than any tuning of the reference it generates. Easy to
present as an implementation detail; it is load-bearing.

---

## 5. Two findings that came out of building this

**The initial-condition trap.** State constraints must be imposed from k = 1
onwards. The first column is pinned by `X[:,0] == x0`, so it is not a degree of
freedom: a constraint there cannot influence the solution but *can* make the whole
problem infeasible whenever the measured state momentarily sits outside the
admissible set — routine, since the estimate comes from a real inner loop that
overshoots. Fixing this took solver success from 0.46 to 1.00.

**SCA can contradict itself.** If the linearisation trajectory pierces an obstacle,
the horizon steps beyond it generate opposing normals and the half-spaces demand
`p_x ≤ 0.7` and `p_x ≥ 2.3` simultaneously. The problem is then genuinely
infeasible and the slack saturates — the diagnostic signature being a slack that
no longer responds to its penalty weight. `sca_iterations > 1` resolves it.

---

## 6. Configuration

One YAML file, split into a `common:` block valid for every platform and a
`model:` block that is the only part knowing what the robot is.

```python
from trajopt_core.config_io import load_planner
setup = load_planner("config/planner_params.yaml")
```

Unknown keys raise rather than being silently ignored — a typo in a weight would
otherwise be a debugging nightmare. ROS parameter files wrapped in
`/**: ros__parameters:` are accepted directly.

---

Three profiles: the aerial one is the base file, the legged one is
`config/legged_overrides.yaml` — about 25 lines of deltas deep-merged on top —
and the humanoid one is `config/g1_overrides.yaml`, a comparably small delta
that reuses the legged profile's model and obstacle term (see §1). Each file
is the answer to "how much of the stack is platform-specific?" for that robot.

```python
load_planner("config/planner_params.yaml", "config/legged_overrides.yaml")
load_planner("config/planner_params.yaml", "config/g1_overrides.yaml")
```

---

## 7. Running it on a robot

[`trajopt_ros`](../trajopt_ros/README.md) is the ROS 2 adapter: a generic A\* node
and a generic MPC node whose topic names are relative and remapped per platform.

```bash
ros2 launch trajopt_ros planner.launch.py profile:=aerial   # or legged, or g1
```

The aerial workspace is already wired through relative symlinks in
`mujoco/ros2_ws/src/`. The legged and G1 workspaces are separate repositories
and need a decision about how to consume the two packages — see the options in
the `trajopt_ros` README, which also documents what was deliberately left out
of the G1 port and why.

---

## 8. Status

Done: shared layers with golden-equivalence proof, the `MotionModel` abstraction
with two instantiations (aerial and SE(2), the latter shared by the Go2 and the
G1), four obstacle strategies, the parametric OCP, the benchmark harness and
mission simulator, the YAML schema with per-platform profiles, the ROS adapter
with both nodes exercised end to end on all three profiles. The G1 addition
needed exactly one new admissible-set restriction (`v_min_xy`, no reverse
walking) and no new model, obstacle term or node code — see §1 and
`config/g1_overrides.yaml`. **61 core tests + 5 ROS smoke tests.**
Every command for running the three platforms and reproducing the metrics is in
[`RUNNING.md`](../RUNNING.md).

Remaining: promote the two packages to their own repository so both workspaces can
consume them as a submodule; and, on the planning side, map accumulation — the
grid currently has no memory across cycles, so a trap larger than the 10 m window
cannot be escaped. Adding it in the core would benefit both platforms at once.
