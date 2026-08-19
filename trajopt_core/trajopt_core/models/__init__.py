from trajopt_core.models.base import (
    CostWeights,
    InnerLoopGains,
    ModelLimits,
    MotionModel,
)
from trajopt_core.models.double_integrator_z import DoubleIntegratorZ
from trajopt_core.models.kinematic_se2 import KinematicSE2

#: Registry so a YAML string can select the platform model.
MOTION_MODELS = {
    "double_integrator_z": DoubleIntegratorZ,
    "kinematic_se2": KinematicSE2,
}


def build_model(spec):
    """Instantiate a model from {"type": ..., **kwargs}."""
    s = dict(spec)
    kind = s.pop("type")
    if kind not in MOTION_MODELS:
        raise KeyError(f"unknown model {kind!r}; known: {sorted(MOTION_MODELS)}")
    return MOTION_MODELS[kind](**s)


__all__ = [
    "MotionModel", "CostWeights", "ModelLimits", "InnerLoopGains",
    "DoubleIntegratorZ", "KinematicSE2",
    "MOTION_MODELS", "build_model",
]
