"""
Fixed-area Gaussian grid map for local obstacle mapping.

Platform-agnostic: this layer only ever sees the *planar position* of the robot
centre of mass and a set of LiDAR hits.  It is shared verbatim between the
aerial and the legged instantiation of the stack.

Key design:
  - The grid always has a fixed spatial extent (2*half_width x 2*half_width metres).
  - It translates rigidly with the robot centre of mass — no growing or shrinking.
  - LiDAR points outside the fixed window are silently ignored.
  - Occupancy probability is computed via a Gaussian CDF of the distance from
    each cell centre to the nearest obstacle point.

Note on the probability range: P = 1 - Phi(d/sigma) saturates at 0.5 for d -> 0,
it does NOT reach 1.  Any obstacle threshold must therefore be < 0.5 to ever
trigger.  The hard-blocking radius implied by a threshold tau is

    d_block = sigma * Phi^-1(1 - tau)

e.g. sigma=0.7, tau=0.1 -> d_block ~ 0.90 m;  sigma=0.05, tau=0.45 -> ~6 mm.

author: Lorenzo Ortolani (original), unified for trajopt_core
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


class FixedGaussianGridMap:
    """
    A 2-D Gaussian occupancy grid with a fixed spatial extent.

    The grid is always centred on the robot position. Its dimensions are:
        cells_per_axis = round(2 * half_width / reso)

    Each call to update() rebuilds the map from scratch at the new robot
    position — there is no map accumulation across steps.

    Parameters
    ----------
    reso       : float  — cell size [m]
    half_width : float  — half-extent of the square grid [m]
    std        : float  — Gaussian spread applied to each obstacle point [m]
    """

    def __init__(self, reso: float = 0.25, half_width: float = 5.0, std: float = 0.5):
        self.reso = float(reso)
        self.half_width = float(half_width)
        self.std = float(std)

        # Number of cells along each axis — fixed for the lifetime of this object
        self.cells = int(round(2.0 * half_width / reso))

        # Occupancy map — shape (cells, cells).  None until first update().
        self.gmap: np.ndarray | None = None

        # World-frame coordinates of the grid origin (bottom-left corner).
        # Updated on every call to update().
        self.minx: float = 0.0
        self.miny: float = 0.0

        # Aliases expected by the A* planner and MPC (mujoco_sim convention)
        self.xw: int = self.cells
        self.yw: int = self.cells
        self.xyreso: float = self.reso

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, lidar_points, robot_pos) -> bool:
        """
        Rebuild the occupancy grid centred on robot_pos.

        Parameters
        ----------
        lidar_points : (N, >=2) float array of LiDAR hits in world frame,
                       or None / empty for an obstacle-free map.
        robot_pos    : array-like [x, y, (z)]

        Returns
        -------
        True if at least one obstacle point was inside the grid; False otherwise.
        """
        dx = float(robot_pos[0])
        dy = float(robot_pos[1])

        # Re-centre grid origin at robot position
        self.minx = dx - self.half_width
        self.miny = dy - self.half_width
        self.xw = self.cells
        self.yw = self.cells

        # Start with an empty (zero-probability) map
        self.gmap = np.zeros((self.cells, self.cells), dtype=np.float32)

        if lidar_points is None or len(lidar_points) == 0:
            return False   # no obstacles, map is empty but valid

        # Project to 2-D and discard points that fall outside the fixed window
        pts = np.asarray(lidar_points, dtype=float)
        ox = pts[:, 0]
        oy = pts[:, 1]

        maxx = self.minx + 2.0 * self.half_width
        maxy = self.miny + 2.0 * self.half_width
        mask = (ox >= self.minx) & (ox < maxx) & (oy >= self.miny) & (oy < maxy)
        ox = ox[mask]
        oy = oy[mask]

        if len(ox) == 0:
            return False   # all points fall outside the fixed window

        # Build grid-centre coordinate arrays
        ix_arr = np.arange(self.cells, dtype=float)
        cx_arr = ix_arr * self.reso + self.minx   # world x of each column
        cy_arr = ix_arr * self.reso + self.miny   # world y of each row

        # Vectorised min-distance from every cell to the nearest obstacle.
        # Broadcast: (cells, 1, 1) and (1, N) -> (cells, cells, N)
        # Peak memory: cells^2 * N * 8 bytes
        cx_grid = cx_arr[:, np.newaxis]            # (cells, 1)
        cy_grid = cy_arr[np.newaxis, :]            # (1, cells)

        dx_obs = cx_grid[:, :, np.newaxis] - ox[np.newaxis, np.newaxis, :]
        dy_obs = cy_grid[:, :, np.newaxis] - oy[np.newaxis, np.newaxis, :]
        min_dists = np.hypot(dx_obs, dy_obs).min(axis=2)   # (cells, cells)

        # Gaussian CDF: P(cell is occupied) increases as distance to obstacles decreases
        self.gmap = (1.0 - norm.cdf(min_dists, 0.0, self.std)).astype(np.float32)
        return True   # at least one point contributed to the map

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_index(self, x: float, y: float):
        """
        Continuous world coordinates -> discretized grid indices.
        Returns (ix, iy) inside [0, cells), or (None, None) if outside.
        """
        ix = int((x - self.minx) / self.reso)
        iy = int((y - self.miny) / self.reso)
        if 0 <= ix < self.cells and 0 <= iy < self.cells:
            return ix, iy
        return None, None

    def index_to_world(self, ix: int, iy: int):
        """Discretized grid indices -> world coordinates at cell centre."""
        return (
            ix * self.reso + self.minx,
            iy * self.reso + self.miny,
        )

    def get_probability(self, x: float, y: float) -> float:
        """Obstacle probability at world (x, y);  0.0 if outside grid or not yet updated."""
        if self.gmap is None:
            return 0.0
        ix, iy = self.world_to_index(x, y)
        if ix is None:
            return 0.0
        return float(self.gmap[ix, iy])

    # ------------------------------------------------------------------
    # Hooks used by the parametric MPC obstacle cost
    # ------------------------------------------------------------------

    def local_axes(self) -> list:
        """
        Grid axes expressed in the LOCAL frame (origin at the grid corner).

        These are constant for the lifetime of the object, which is what allows
        the CasADi B-spline interpolant to be built ONCE with fixed knots while
        the map keeps translating with the robot: the MPC evaluates the spline
        at (p_k - origin) and passes the cell values as a parameter.
        """
        axis = (np.arange(self.cells, dtype=float) * self.reso).tolist()
        return [axis, axis]

    def origin(self) -> np.ndarray:
        """World-frame position of the grid corner, i.e. the local-frame offset."""
        return np.array([self.minx, self.miny], dtype=float)

    def coefficients(self) -> np.ndarray:
        """
        Cell values flattened in the column-major order expected by CasADi.

        For a scalar B-spline interpolant the coefficient vector has exactly
        cells*cells entries, so the raw occupancy values can be handed over
        directly (verified bit-identical against the non-parametric interpolant).
        """
        if self.gmap is None:
            return np.zeros(self.cells * self.cells, dtype=float)
        return self.gmap.ravel(order="F").astype(float)

    # ------------------------------------------------------------------
    # Convenience read-only properties
    # ------------------------------------------------------------------

    @property
    def maxx(self) -> float:
        return self.minx + 2.0 * self.half_width

    @property
    def maxy(self) -> float:
        return self.miny + 2.0 * self.half_width

    @property
    def is_initialised(self) -> bool:
        return self.gmap is not None
