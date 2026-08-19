"""
Declarative configuration: one schema, two robots.

A single YAML file describes the whole stack.  It is split into a `common:`
block, valid for every platform, and a `model:` block that selects and
parameterises the `MotionModel`.  That split mirrors the architectural claim of
the package: the optimal control problem is platform-agnostic, and the platform
enters only through (f_Sigma, U_Sigma, X_Sigma, C).

The ROS wrappers pass the path of such a file; nothing else about the planner is
configured through ROS parameters, so the two platform workspaces cannot drift
apart in their tuning schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from trajopt_core.models import CostWeights, InnerLoopGains, ModelLimits, build_model
from trajopt_core.mpc.config import OCPConfig, SolverOptions
from trajopt_core.mpc.obstacles import build_obstacle_terms


@dataclass
class PlannerSetup:
    """Everything needed to instantiate the stack, built from one YAML file."""

    model: object
    cfg: OCPConfig
    obstacle_terms: list
    grid: dict = field(default_factory=dict)     # reso / half_width / std / max_lidar_range
    astar: dict = field(default_factory=dict)    # obstacle_threshold / cost weight / rates
    gains: InnerLoopGains = field(default_factory=InnerLoopGains)
    goal: tuple = (0.0, 0.0, 0.0)

    def describe(self) -> dict:
        return {
            "model": self.model.describe(),
            "N": self.cfg.N,
            "dt": self.cfg.dt,
            "horizon_s": self.cfg.horizon_seconds,
            "obstacle_terms": [t.describe() for t in self.obstacle_terms],
            "grid": self.grid,
            "astar": self.astar,
        }


def _subset(d: dict, cls) -> dict:
    """Keep only the keys that `cls` actually declares, so typos surface loudly."""
    known = set(cls.__dataclass_fields__)
    unknown = set(d) - known
    if unknown:
        raise KeyError(
            f"unknown {cls.__name__} field(s): {sorted(unknown)}; known: {sorted(known)}"
        )
    return d


def _load_yaml(source):
    if isinstance(source, dict):
        return source
    import yaml

    with Path(source).open() as fh:
        return yaml.safe_load(fh)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into `base`, returning a new dict.

    Mappings merge key by key; every other type (including lists) is replaced
    wholesale, because a partially merged list of obstacle terms would be far
    more surprising than a replaced one.
    """
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_planner(source, overrides=None) -> PlannerSetup:
    """
    Build a :class:`PlannerSetup` from a YAML path, a file object or a dict.

    Missing sections fall back to the dataclass defaults, so a minimal file that
    only overrides a handful of weights is perfectly valid.

    `overrides` is a second YAML path or dict, deep-merged on top of the first.
    That is how the two robot profiles are expressed: one shared base file plus a
    small delta file per platform, so the claim that the platforms differ by a
    handful of numbers is checkable by reading the delta rather than diffing two
    near-identical configurations.
    """
    data = _load_yaml(source)
    if overrides is not None:
        data = _deep_merge(data, _load_yaml(overrides))

    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")

    # ROS parameter files wrap everything in /**: ros__parameters:
    if "/**" in data:
        data = data["/**"].get("ros__parameters", data["/**"])

    common = dict(data.get("common", {}))
    model_spec = dict(data.get("model", {"type": "kinematic_se2"}))

    limits = ModelLimits(**_subset(dict(common.pop("limits", {})), ModelLimits))
    weights = CostWeights(**_subset(dict(common.pop("weights", {})), CostWeights))
    solver = SolverOptions(**_subset(dict(common.pop("solver", {})), SolverOptions))
    gains = InnerLoopGains(**_subset(dict(common.pop("inner_loop", {})), InnerLoopGains))

    grid = dict(common.pop("grid", {}))
    astar = dict(common.pop("astar", {}))
    obstacles = list(common.pop("obstacles", []))
    goal = tuple(common.pop("goal", (0.0, 0.0, 0.0)))

    cfg = OCPConfig(
        **_subset(common, OCPConfig),
        weights=weights,
        limits=limits,
        solver=solver,
    )

    # the model owns the limits; passing them here keeps a single source of truth
    model_spec.setdefault("limits", limits)
    model = build_model(model_spec)

    return PlannerSetup(
        model=model,
        cfg=cfg,
        obstacle_terms=build_obstacle_terms(obstacles),
        grid=grid,
        astar=astar,
        gains=gains,
        goal=goal,
    )
