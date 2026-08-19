"""
Lookahead setpoint extraction.

The MPC does not hand its first predicted state to the inner loop.  At cruise
speed the one-step-ahead point sits only v_ref*dt metres away (0.1 m with the
default settings), which is well inside the settling distance of any real inner
controller: tracking it makes the loop chatter.

Instead the predicted trajectory is walked outwards and the first state at least
`lookahead_dist` away is published.  The lookahead distance is therefore not a
free parameter — it encodes the settling distance of the platform-specific inner
loop, and it is the quantitative expression of the time-scale separation the
whole hierarchical architecture rests upon.

When the entire horizon falls inside that radius the robot is near the goal, and
the last path waypoint is published instead so that it homes in precisely rather
than orbiting the horizon endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LookaheadResult:
    index: int                 # index into the predicted trajectory
    position: np.ndarray       # (2 or 3,) setpoint position
    yaw: float                 # setpoint heading [rad]
    found: bool                # False -> fell back to the goal waypoint
    distance: float            # actual distance from the robot


def select_lookahead(
    model,
    x_pred: np.ndarray,
    robot_xy: np.ndarray,
    lookahead_dist: float,
    fallback_waypoint=None,
    z_ref: float | None = None,
) -> LookaheadResult:
    """
    Parameters
    ----------
    x_pred            : (N+1, NX) predicted state trajectory
    robot_xy          : (2,) current planar position
    lookahead_dist    : minimum distance of the published setpoint [m]
    fallback_waypoint : (2,) or (3,) waypoint used when the whole horizon is
                        closer than `lookahead_dist` (typically the last A*
                        waypoint, i.e. the local goal)
    z_ref             : height assigned to the fallback when it is planar
    """
    x_pred = np.asarray(x_pred, dtype=float)
    robot_xy = np.asarray(robot_xy, dtype=float).ravel()[:2]
    i, j = model.PLANAR_IDX

    last = x_pred.shape[0] - 1
    for k in range(1, x_pred.shape[0]):
        d = float(np.hypot(x_pred[k, i] - robot_xy[0], x_pred[k, j] - robot_xy[1]))
        if d >= lookahead_dist:
            return LookaheadResult(
                index=k,
                position=model.position(x_pred[k]),
                yaw=model.heading(x_pred[k]),
                found=True,
                distance=d,
            )

    # Near the goal: steer at the local goal itself.
    if fallback_waypoint is not None:
        wp = np.asarray(fallback_waypoint, dtype=float).ravel()
        if len(model.POS_IDX) == 3:
            z = wp[2] if wp.size > 2 else (z_ref if z_ref is not None else x_pred[last, 2])
            pos = np.array([wp[0], wp[1], float(z)])
        else:
            pos = np.array([wp[0], wp[1]])
        d = float(np.hypot(pos[0] - robot_xy[0], pos[1] - robot_xy[1]))
        return LookaheadResult(
            index=last, position=pos, yaw=model.heading(x_pred[last]),
            found=False, distance=d,
        )

    d = float(np.hypot(x_pred[last, i] - robot_xy[0], x_pred[last, j] - robot_xy[1]))
    return LookaheadResult(
        index=last, position=model.position(x_pred[last]),
        yaw=model.heading(x_pred[last]), found=False, distance=d,
    )
