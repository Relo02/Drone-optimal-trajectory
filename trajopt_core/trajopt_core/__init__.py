"""
trajopt_core — platform-agnostic local trajectory optimisation.

A Gaussian occupancy grid, a rolling-horizon A* planner and a CasADi/IPOPT
receding-horizon optimal control problem, written once and instantiated on
different robotic platforms through the MotionModel abstraction.

The package deliberately depends on neither ROS nor any simulator, so it can be
unit-tested and benchmarked without a robot in the loop.
"""

__version__ = "0.1.0"

from trajopt_core.mapping import FixedGaussianGridMap
from trajopt_core.models import (
    CostWeights,
    DoubleIntegratorZ,
    KinematicSE2,
    ModelLimits,
    MotionModel,
    build_model,
)
from trajopt_core.mpc import OCPConfig, SolverOptions, TrajectoryOCP, select_lookahead
from trajopt_core.planning import AStarPlanner

__all__ = [
    "FixedGaussianGridMap", "AStarPlanner",
    "MotionModel", "DoubleIntegratorZ", "KinematicSE2",
    "CostWeights", "ModelLimits", "build_model",
    "TrajectoryOCP", "OCPConfig", "SolverOptions", "select_lookahead",
]
