"""
Closed-loop mission simulation, shared by every platform.

This lives in the core, not in either platform layer, on purpose: the numbers
reported for the aerial and the legged robot are then produced by the same code
path, which is what makes a cross-platform comparison admissible in the first
place.

What is and is not simulated
----------------------------
The loop implements the receding-horizon algorithm itself (Algorithm 7.2.1 of the
course notes): sense, map, replan, solve, act, repeat.  Two ways of closing the
loop are provided, because they answer different questions:

  closed_loop="lookahead"  (default, matches deployment)
      The MPC publishes the lookahead setpoint and a platform-specific inner
      loop tracks it — exactly the architecture of both robots, where the
      optimiser is a REFERENCE GENERATOR rather than a direct controller.
      Skipping this distinction is not a detail: a body-frame relative-degree-1
      platform handed its own first optimal input will happily spend the whole
      cycle rotating ("turn first, then translate") and stall, because only the
      rotation part of the plan is ever executed before the next re-solve.

  closed_loop="direct"     (textbook receding horizon)
      The first optimal input is applied to the plant.  Useful when the inner
      loop must be removed as a confounding variable.

The plant is the platform model itself.  That is the right fidelity for
questions about the OPTIMISATION — solve time, iteration counts, feasibility,
discretisation error, penalty exactness.  It is NOT a substitute for the MuJoCo /
Gazebo runs when the question is about the physics: those keep wind, drag, motor
lag and gait dynamics, none of which appear here.

Perception is simulated the way the real stack sees the world: only obstacle
points within `lidar_range` of the robot are visible, so the planner works
without a prior map exactly as it does on the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trajopt_core.bench.recorder import SolveRecorder
from trajopt_core.mapping import FixedGaussianGridMap
from trajopt_core.models.base import InnerLoopGains
from trajopt_core.mpc.lookahead import select_lookahead
from trajopt_core.mpc.ocp import TrajectoryOCP
from trajopt_core.planning import AStarPlanner


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    obstacles: np.ndarray          # (M, 2) world-frame obstacle points
    start_xy: np.ndarray           # (2,)
    start_yaw: float
    goal_xy: np.ndarray            # (2,)
    lidar_range: float = 6.0
    goal_radius: float = 0.4
    max_steps: int = 400

    def visible(self, robot_xy: np.ndarray) -> np.ndarray:
        """LiDAR hits inside the sensing radius — the robot has no prior map."""
        if len(self.obstacles) == 0:
            return np.zeros((0, 2))
        d = np.linalg.norm(self.obstacles - robot_xy, axis=1)
        return self.obstacles[d <= self.lidar_range]


def _segment(p0, p1, n):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return p0[None, :] + t * (p1 - p0)[None, :]


def corridor_with_pillars(spacing: float = 0.12) -> Scenario:
    """
    A 4 m wide corridor obstructed by three staggered pillars.

    Traversable by both platforms without being trivial: the direct line is
    blocked, so A* must commit to a side at each pillar and the MPC must keep
    clearance while doing so.
    """
    walls = np.vstack([
        _segment((-1.0, 2.0), (14.0, 2.0), int(15.0 / spacing)),
        _segment((-1.0, -2.0), (14.0, -2.0), int(15.0 / spacing)),
    ])

    pillars = []
    for cx, cy in ((3.5, -0.5), (6.5, 0.7), (9.5, -0.6)):
        ang = np.linspace(0.0, 2 * np.pi, 26, endpoint=False)
        pillars.append(np.stack([cx + 0.45 * np.cos(ang), cy + 0.45 * np.sin(ang)], axis=1))

    return Scenario(
        name="corridor_with_pillars",
        obstacles=np.vstack([walls] + pillars),
        start_xy=np.array([0.0, 0.0]),
        start_yaw=0.0,
        goal_xy=np.array([12.5, 0.0]),
    )


def open_field() -> Scenario:
    """Baseline with no obstacles: isolates the tracking behaviour."""
    return Scenario(
        name="open_field",
        obstacles=np.zeros((0, 2)),
        start_xy=np.array([0.0, 0.0]),
        start_yaw=0.0,
        goal_xy=np.array([10.0, 0.0]),
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class MissionResult:
    label: str
    platform: str
    scenario: str
    reached: bool
    steps: int
    mission_time_s: float
    path_length_m: float
    min_clearance_m: float
    mean_clearance_m: float
    mean_track_err_m: float
    max_track_err_m: float
    n_replan_failures: int
    trajectory: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))
    recorder: SolveRecorder | None = field(repr=False, default=None)

    def summary(self) -> dict:
        out = {
            "label": self.label,
            "platform": self.platform,
            "scenario": self.scenario,
            "reached": self.reached,
            "steps": self.steps,
            "mission_time_s": self.mission_time_s,
            "path_length_m": self.path_length_m,
            "min_clearance_m": self.min_clearance_m,
            "mean_clearance_m": self.mean_clearance_m,
            "mean_track_err_m": self.mean_track_err_m,
            "max_track_err_m": self.max_track_err_m,
            "n_replan_failures": self.n_replan_failures,
        }
        if self.recorder is not None:
            out.update(self.recorder.summary())
            out["label"] = self.label      # recorder.summary() also carries a label
        return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _point_to_polyline(p: np.ndarray, poly: np.ndarray) -> float:
    """Shortest distance from a point to a polyline (used for tracking error)."""
    if len(poly) == 0:
        return float("nan")
    if len(poly) == 1:
        return float(np.linalg.norm(p - poly[0]))
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom < 1e-12] = 1e-12
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(proj - p, axis=1)))


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def run_mission(
    model,
    cfg,
    obstacle_terms,
    scenario: Scenario,
    *,
    label: str | None = None,
    grid_reso: float = 0.25,
    grid_half_width: float = 5.0,
    grid_std: float = 0.7,
    obstacle_threshold: float = 0.1,
    obstacle_cost_weight: float = 15.0,
    replan_every: int = 2,
    deadline_ms: float | None = None,
    closed_loop: str = "lookahead",
    gains=None,
    seed: int = 0,
) -> MissionResult:
    """
    Run one closed-loop mission and return trajectory-level and solver-level
    metrics.

    `replan_every` sets the ratio between the MPC rate and the A* replanning
    rate, mirroring the deployed configuration (A* is the slower layer).
    `closed_loop` selects how the optimiser output reaches the plant — see the
    module docstring.
    """
    if closed_loop not in ("lookahead", "direct"):
        raise ValueError("closed_loop must be 'lookahead' or 'direct'")
    if gains is None:
        gains = InnerLoopGains()

    rng = np.random.default_rng(seed)      # reserved for future disturbance models
    label = label or f"{model.name}/{scenario.name}"

    grid = FixedGaussianGridMap(reso=grid_reso, half_width=grid_half_width, std=grid_std)
    planner = AStarPlanner(
        obstacle_threshold=obstacle_threshold,
        obstacle_cost_weight=obstacle_cost_weight,
    )
    ocp = TrajectoryOCP(model, cfg, obstacle_terms)
    rec = SolveRecorder(label, deadline_ms=deadline_ms or (1e3 * cfg.dt))

    # --- initial state -----------------------------------------------------
    if model.PLANS_ALTITUDE:
        state = model.make_state(
            (scenario.start_xy[0], scenario.start_xy[1], cfg.z_ref), yaw=scenario.start_yaw
        )
    else:
        state = model.make_state(scenario.start_xy, yaw=scenario.start_yaw)

    i, j = model.PLANAR_IDX
    traj = [np.array([state[i], state[j]])]
    clearances, track_errs = [], []
    path, n_fail, reached = [], 0, False

    for step in range(scenario.max_steps):
        robot_xy = np.array([state[i], state[j]])

        if np.linalg.norm(robot_xy - scenario.goal_xy) <= scenario.goal_radius:
            reached = True
            break

        points = scenario.visible(robot_xy)

        # --- global layer (slower) -----------------------------------------
        if step % replan_every == 0:
            grid.update(
                np.hstack([points, np.zeros((len(points), 1))]) if len(points) else None,
                np.append(robot_xy, 0.0),
            )
            new_path = planner.plan(grid, robot_xy, scenario.goal_xy)
            if new_path:
                path = [(float(x), float(y)) for x, y in new_path]
            else:
                n_fail += 1

        if not path:
            n_fail += 1
            break

        # --- local layer ----------------------------------------------------
        res = ocp.solve(state, path, points_xy=points, grid=grid)
        rec.add(res, step=step)

        # --- metrics --------------------------------------------------------
        if len(scenario.obstacles):
            clearances.append(float(np.min(np.linalg.norm(scenario.obstacles - robot_xy, axis=1))))
        track_errs.append(_point_to_polyline(robot_xy, np.asarray(path, dtype=float)))

        # --- act -------------------------------------------------------------
        if closed_loop == "direct":
            u = res.u0
        else:
            look = select_lookahead(
                model, res.x_pred, robot_xy, cfg.lookahead_dist,
                fallback_waypoint=path[-1], z_ref=cfg.z_ref,
            )
            u = model.tracking_input(state, look.position, look.yaw, gains)

        state = np.array(model.step(state, u, cfg.dt), dtype=float)
        if not np.isfinite(state).all():
            n_fail += 1
            break
        traj.append(np.array([state[i], state[j]]))

    traj_arr = np.asarray(traj)
    seg = np.linalg.norm(np.diff(traj_arr, axis=0), axis=1) if len(traj_arr) > 1 else np.zeros(0)

    return MissionResult(
        label=label,
        platform=model.name,
        scenario=scenario.name,
        reached=reached,
        steps=len(rec),
        mission_time_s=len(rec) * cfg.dt,
        path_length_m=float(seg.sum()),
        min_clearance_m=float(np.min(clearances)) if clearances else float("inf"),
        mean_clearance_m=float(np.mean(clearances)) if clearances else float("inf"),
        mean_track_err_m=float(np.nanmean(track_errs)) if track_errs else float("nan"),
        max_track_err_m=float(np.nanmax(track_errs)) if track_errs else float("nan"),
        n_replan_failures=n_fail,
        trajectory=traj_arr,
        recorder=rec,
    )


def table(results, keys) -> str:
    """Render mission summaries as a markdown table."""
    rows = [r.summary() if isinstance(r, MissionResult) else r for r in results]
    header = "| " + " | ".join(keys) + " |"
    sep = "|" + "---|" * len(keys)
    lines = [header, sep]
    for row in rows:
        cells = []
        for k in keys:
            v = row.get(k, "")
            if isinstance(v, float):
                cells.append("inf" if math.isinf(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
