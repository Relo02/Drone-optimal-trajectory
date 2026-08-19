"""
Reference trajectory generation from the A* path.

Platform-agnostic by construction: the geometry is computed entirely in the
horizontal plane (arc-length parameterisation of the waypoint polyline), and the
result is handed to `MotionModel.lift_reference` which raises it to whatever the
platform's state vector happens to be.

    planar geometry  ->  (p_ref_k, tangent_k)  ->  model  ->  x_ref_k

This is the concrete point where the platform-independent layer meets the
platform-specific one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PathReference:
    x_ref: np.ndarray        # (N+1, NX) full state reference
    p_ref: np.ndarray        # (N+1, 2)  planar position reference
    tangent: np.ndarray      # (N+1, 2)  unit tangent of the path
    arc: np.ndarray          # (N+1,)    arc-length coordinate of each sample
    idx_closest: int         # index of the path waypoint closest to the robot
    total_arc: float
    degenerate: bool = False # True if the path was empty / too short


def _as_xy(path) -> tuple[np.ndarray, np.ndarray | None]:
    """Split a waypoint list into planar coordinates and optional heights."""
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2)), None
    xy = arr[:, :2]
    z = arr[:, 2] if arr.shape[1] >= 3 else None
    return xy, z


# ---------------------------------------------------------------------------
# Path conditioning
# ---------------------------------------------------------------------------
# An 8-connected grid search can only ever emit headings that are multiples of
# 45 degrees, so the raw A* polyline zig-zags at the cell scale and its tangent
# flips by +/-45 deg from one waypoint to the next.  Whether that matters depends
# on the platform:
#
#   relative degree 2, world frame (aerial)  -- the commanded accelerations are
#       expressed in the world frame and the heading channel is dynamically
#       decoupled, so a chattering yaw reference is merely tracked badly.
#
#   relative degree 1, body frame (legged)   -- the input is a BODY-frame
#       velocity, so the heading multiplies the translation.  A yaw reference
#       that flips at every replan makes the optimiser rotate instead of
#       advancing, and the robot stalls.
#
# Conditioning the polyline before differentiating it is therefore not cosmetic:
# it is what makes the shared reference builder usable by a body-frame platform.

def resample_polyline(pts: np.ndarray, ds: float) -> np.ndarray:
    """Uniform arc-length resampling of a polyline at spacing `ds`."""
    if ds <= 0.0 or len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < ds:
        return pts
    s_new = np.append(np.arange(0.0, total, ds), total)
    return np.stack([np.interp(s_new, arc, pts[:, c]) for c in range(pts.shape[1])], axis=1)


def smooth_polyline(pts: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing that leaves the endpoints untouched."""
    if window < 3 or len(pts) < window:
        return pts
    if window % 2 == 0:
        window += 1
    half = window // 2
    kernel = np.ones(window) / window
    padded = np.vstack([np.repeat(pts[:1], half, axis=0), pts,
                        np.repeat(pts[-1:], half, axis=0)])
    out = np.stack(
        [np.convolve(padded[:, c], kernel, mode="valid") for c in range(pts.shape[1])],
        axis=1,
    )
    out[0], out[-1] = pts[0], pts[-1]     # anchor the start and the local goal
    return out


def condition_path(path, resample_ds: float = 0.0, smooth_window: int = 0) -> np.ndarray:
    """Resample then smooth a raw A* polyline.  Returns an (M, 2 or 3) array."""
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return arr
    arr = resample_polyline(arr, resample_ds)
    return smooth_polyline(arr, smooth_window)


def build_path_reference(
    model,
    state: np.ndarray,
    path,
    N: int,
    dt: float,
    v_ref: float,
    z_ref: float,
    anchor_first_to_state: bool = False,
    resample_ds: float = 0.0,
    smooth_window: int = 0,
) -> PathReference:
    """
    Build an (N+1, NX) reference by advancing along the A* path at `v_ref`.

    The advance starts from the waypoint closest to the current robot position,
    so the reference is re-anchored to the actual pose at every solve: this is
    what makes the scheme a genuine receding-horizon reference rather than a
    fixed trajectory played back open loop.

    `resample_ds` / `smooth_window` condition the raw grid polyline first — see
    the note above `resample_polyline` for why a body-frame platform needs it.
    """
    nx = model.NX
    state = np.asarray(state, dtype=float).ravel()
    if resample_ds > 0.0 or smooth_window >= 3:
        path = condition_path(path, resample_ds, smooth_window)
    path_xy, path_z = _as_xy(path)

    # --- degenerate path: hold the current pose --------------------------
    if path_xy.shape[0] < 2:
        x_ref = np.tile(state, (N + 1, 1))
        i, j = model.PLANAR_IDX
        p_ref = np.tile(state[[i, j]], (N + 1, 1))
        tangent = np.tile(np.array([1.0, 0.0]), (N + 1, 1))
        return PathReference(
            x_ref=x_ref, p_ref=p_ref, tangent=tangent,
            arc=np.zeros(N + 1), idx_closest=0, total_arc=0.0, degenerate=True,
        )

    # --- arc-length parameterisation -------------------------------------
    diffs = np.diff(path_xy, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_arc = float(arc[-1])

    i, j = model.PLANAR_IDX
    robot_xy = np.array([state[i], state[j]], dtype=float)
    idx_closest = int(np.argmin(np.linalg.norm(path_xy - robot_xy, axis=1)))
    s0 = float(arc[idx_closest])

    x_ref = np.zeros((N + 1, nx))
    p_ref = np.zeros((N + 1, 2))
    tang = np.zeros((N + 1, 2))
    s_samples = np.zeros(N + 1)

    for k in range(N + 1):
        s_k = min(s0 + v_ref * k * dt, total_arc)
        s_samples[k] = s_k

        idx = int(np.searchsorted(arc, s_k, side="right")) - 1
        idx = int(np.clip(idx, 0, len(path_xy) - 2))

        seg = seg_len[idx]
        t = 0.0 if seg < 1e-9 else np.clip((s_k - arc[idx]) / seg, 0.0, 1.0)

        pos = path_xy[idx] + t * diffs[idx]
        direction = diffs[idx] / (seg + 1e-9)

        z_k = z_ref
        if path_z is not None:
            z_k = float(path_z[idx] + t * (path_z[idx + 1] - path_z[idx]))

        p_ref[k] = pos
        tang[k] = direction
        x_ref[k] = model.lift_reference(pos, direction, v_ref, z_k)

    if anchor_first_to_state:
        x_ref[0] = state
        p_ref[0] = robot_xy

    if not np.isfinite(x_ref).all():
        x_ref = np.tile(state, (N + 1, 1))

    return PathReference(
        x_ref=x_ref, p_ref=p_ref, tangent=tang, arc=s_samples,
        idx_closest=idx_closest, total_arc=total_arc, degenerate=False,
    )
