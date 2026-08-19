"""
The MuJoCo model driven by `go2_sim_node`: a Unitree Go2 in an industrial warehouse.

The Go2 MJCF (`mujoco/model/go2/`) already carries its own meshes, floor plane
and free joint, so this module only augments its spec with the warehouse:
perimeter walls, columns, racking, conveyors, workcells, pallets and a
forklift.

Two details decide whether the simulated LiDAR is usable:

    * every warehouse geom goes into geom group 3, and the ray cast is restricted
        to that group.  The robot's own body is therefore invisible to its own
        sensor — without this the robot maps itself and the planner sees an obstacle
        permanently at zero range;
    * the geoms are visual-only (`contype = conaffinity = 0`).  The base is moved
        kinematically (see `go2_sim_node`), so contact would be resolved against a
        body that does not obey the contact forces anyway.

Geometry is defined here in Python rather than parsed from a scene file, which
keeps the warehouse a few lines of readable numbers and avoids carrying a second
asset format. Sizes follow the two conversions that trip everyone up: a box
`size` is a HALF extent in MuJoCo, and a cylinder is `[radius, half_length]`.

Adapted from the CIHR lab simulator, reduced to what this repository needs: the
dynamic "people" bodies of the original are deliberately absent, because neither
the aerial nor the Go2 instantiation has moving obstacles and adding them for
one platform alone would make the three platforms incomparable.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

#: Geom group ray-cast by the simulated LiDAR — the warehouse, never the robot.
LIDAR_GROUP = 3

#: Go2 lidar mount approximation on the base body.
GO2_LIDAR_POS = (0.0, 0.0, 0.29)


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _box(x, y, z, sx, sy, sz, rgba, yaw=0.0) -> dict:
    """A box given by its FULL extents, as one reads them off a drawing."""
    return dict(shape="box", pos=[x, y, z], size=[sx / 2, sy / 2, sz / 2],
                rgba=rgba, yaw=yaw)


def _cyl(x, y, z, radius, length, rgba) -> dict:
    """A cylinder given by its full length."""
    return dict(shape="cyl", pos=[x, y, z], size=[radius, length / 2],
                rgba=rgba, yaw=0.0)


_WALL = [0.80, 0.80, 0.80, 1]
_COL = [0.40, 0.40, 0.50, 1]
_RACK = [0.60, 0.40, 0.20, 1]
_PALLET = [0.70, 0.55, 0.20, 1]
_BOXC = [0.30, 0.50, 0.70, 1]
_GREEN = [0.35, 0.60, 0.40, 1]
_CONV = [0.30, 0.30, 0.34, 1]
_ARM = [0.90, 0.45, 0.10, 1]
_DARK = [0.20, 0.20, 0.20, 1]
_SHELF = [0.55, 0.40, 0.25, 1]
_FORK = [0.85, 0.70, 0.10, 1]


def warehouse_geoms() -> list[dict]:
    """The obstacle set: a 30 x 20 m hall with aisles wide enough to plan in."""
    g = []
    # Perimeter walls
    g += [_box(0, 10, 1.5, 30, 0.2, 3, _WALL), _box(0, -10, 1.5, 30, 0.2, 3, _WALL),
          _box(15, 0, 1.5, 0.2, 20, 3, _WALL), _box(-15, 0, 1.5, 0.2, 20, 3, _WALL)]
    # Structural columns
    for cx in (-10, 0, 10):
        for cy in (6, -6):
            g.append(_cyl(cx, cy, 1.5, 0.15, 3.0, _COL))
    # Racking
    for (rx, ry) in [(-12, 7.5), (4, 7.5), (-12, -7.5), (4, -7.5)]:
        g.append(_box(rx, ry, 1.25, 6, 0.6, 2.5, _RACK))
    # Pallets and a crate
    g += [_box(6, 2, 0.075, 1.2, 0.8, 0.15, _PALLET),
          _box(6, 2, 0.475, 0.6, 0.4, 0.5, _BOXC, yaw=0.3),
          _box(7, -3, 0.075, 1.2, 0.8, 0.15, _PALLET)]
    # Conveyor lines — the two long obstacles that force a detour
    g += [_box(-1, 4.3, 0.35, 8, 0.7, 0.7, _CONV), _box(1, -4.3, 0.35, 8, 0.7, 0.7, _CONV)]
    # Robotic-arm workcells (base, column, arm)
    for (ax, ay) in [(-1, 3.3), (1, -3.3)]:
        g += [_cyl(ax, ay, 0.25, 0.30, 0.5, _DARK),
              _cyl(ax, ay, 1.05, 0.12, 1.1, _ARM),
              _box(ax, ay, 1.5, 0.9, 0.18, 0.18, _ARM)]
    # East shelving
    g += [_box(13, 4, 1.3, 0.6, 5, 2.6, _SHELF), _box(13, -4, 1.3, 0.6, 5, 2.6, _SHELF)]
    # Pallet staging
    g += [_box(-6, -4, 0.075, 1.2, 0.8, 0.15, _PALLET),
          _box(-6, -4, 0.55, 0.9, 0.7, 0.8, _BOXC),
          _box(-7.4, -4, 0.075, 1.2, 0.8, 0.15, _PALLET),
          _box(-7.4, -4, 0.40, 0.8, 0.6, 0.5, _GREEN)]
    # Loose crates
    g += [_box(8, 5.5, 0.3, 0.6, 0.6, 0.6, _BOXC, yaw=0.4),
          _box(8.7, 6, 0.45, 0.5, 0.5, 0.9, _GREEN),
          _box(-3, -6.8, 0.3, 0.6, 0.6, 0.6, _BOXC, yaw=0.8)]
    # Forklift (body, cabin, mast)
    g += [_box(10, 3.5, 0.35, 1.1, 0.7, 0.7, _FORK, yaw=1.2),
          _box(10, 3.5, 1.05, 0.6, 0.6, 0.7, _FORK, yaw=1.2),
          _box(10, 3.5, 1.0, 0.1, 0.6, 2.0, _DARK, yaw=1.2)]
    return g


def build_model(go2_xml_path: str):
    """
    Compile the combined Go2 + warehouse model.

    Returns `(model, info)`, where `info` carries the handles the simulation
    node needs: the LiDAR site, the address of the free joint in `qpos`, and the
    geom group to ray-cast.
    """
    spec = mujoco.MjSpec.from_file(str(go2_xml_path))
    wb = spec.worldbody

    # The Go2 file already provides the floor; adding a second coplanar plane
    # z-fights into a speckled mess.  Only widen the camera extent, which
    # defaults to robot size and would otherwise clip the warehouse away.
    spec.stat.extent = 18.0
    spec.stat.center = [0.0, 0.0, 1.0]
    try:
        spec.material("groundplane").reflectance = 0.0
    except Exception:
        pass

    # A DIRECTIONAL light: MuJoCo's default spot light puts a bright hotspot over
    # the origin, and specular highlights blow pale surfaces out to white when
    # the warehouse is viewed from above.
    sun = wb.add_light(pos=[0, 0, 15], dir=[-0.3, -0.4, -1.0])
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.diffuse = [0.5, 0.5, 0.5]
    sun.specular = [0.0, 0.0, 0.0]
    sun.castshadow = 1
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    for ge in warehouse_geoms():
        if ge["shape"] == "box":
            gg = wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=ge["size"],
                             pos=ge["pos"], rgba=ge["rgba"], quat=_yaw_quat(ge["yaw"]))
        else:
            gg = wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=ge["size"],
                             pos=ge["pos"], rgba=ge["rgba"])
        gg.group = LIDAR_GROUP
        gg.contype = 0
        gg.conaffinity = 0

    spec.body("base").add_site(name="lidar", pos=list(GO2_LIDAR_POS))

    model = spec.compile()

    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    free_jid = int(model.body_jntadr[base_bid])
    info = dict(
        site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar"),
        free_qpos_adr=int(model.jnt_qposadr[free_jid]),
        lidar_group=LIDAR_GROUP,
    )
    return model, info


def lidar_directions(n_azimuth: int, n_elevation: int,
                     el_min: float, el_max: float) -> np.ndarray:
    """
    Unit ray directions in the sensor frame, as a (n_azimuth * n_elevation, 3)
    array — constant, so the simulation computes them once and rotates them into
    the world at every scan.
    """
    az = np.repeat(np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False), n_elevation)
    el = np.tile(np.linspace(el_min, el_max, n_elevation), n_azimuth)
    return np.column_stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    ).astype(np.float64)
