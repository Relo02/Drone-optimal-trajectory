"""
A* local path planner with rolling-horizon local goal selection.

Design
------
- The planner operates entirely within a FixedGaussianGridMap that is
  always centred on the drone.  The grid moves with the drone every cycle.

- Local goal selection (rolling horizon):
    * If the global goal lies inside the current grid, that cell is used
      directly as the A* target.
    * If the global goal is outside the grid, the planner intersects the
      ray (drone -> global_goal) with the grid boundary and uses the
      boundary cell as the local target.  This makes the drone advance
      toward the global goal one grid-width at a time.

- The planner re-runs from scratch every call to plan().  No persistent
  state between calls is required — the caller (e.g. a ROS2 timer) is
  responsible for the replanning frequency.

author: Lorenzo Ortolani
"""

import math
import heapq
import numpy as np

from new_mujoco.gaussian_grid_map import FixedGaussianGridMap


# ---------------------------------------------------------------------------
# A* node
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ('ix', 'iy', 'g', 'parent')

    def __init__(self, ix: int, iy: int, g: float, parent):
        self.ix = ix
        self.iy = iy
        self.g = g          # cost from start
        self.parent = parent  # _Node or None

    def __lt__(self, other: '_Node') -> bool:
        return self.g < other.g


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class AStarPlanner:
    """
    Rolling-horizon A* planner on a FixedGaussianGridMap.

    Usage
    -----
    planner = AStarPlanner(obstacle_threshold=0.5, obstacle_cost_weight=10.0)
    path = planner.plan(grid_map, drone_pos_xy, global_goal_xy)
    # path: list of (x, y) world-frame waypoints from drone to local goal,
    #       or None if A* fails.
    """

    # 8-connected motion: (dx, dy, euclidean_cost)
    _MOTION = [
        ( 1,  0, 1.0),
        ( 0,  1, 1.0),
        (-1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 1,  1, math.sqrt(2)),
        ( 1, -1, math.sqrt(2)),
        (-1,  1, math.sqrt(2)),
        (-1, -1, math.sqrt(2)),
    ]

    def __init__(
        self,
        obstacle_threshold: float = 0.5,
        obstacle_cost_weight: float = 10.0,
    ):
        """
        Parameters
        ----------
        obstacle_threshold   : cells with probability >= this are treated as
                               hard obstacles (infinite cost).
        obstacle_cost_weight : soft cost multiplier for cells below threshold.
                               Higher values push the path further from obstacles.
        """
        self.obstacle_threshold = obstacle_threshold
        self.obstacle_cost_weight = obstacle_cost_weight

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def plan(
        self,
        grid_map: FixedGaussianGridMap,
        drone_pos_xy,
        global_goal_xy,
    ):
        """
        Plan a path from the current drone position to a local goal.

        Parameters
        ----------
        grid_map       : up-to-date FixedGaussianGridMap (already updated
                         with the latest LiDAR scan)
        drone_pos_xy   : (x, y) drone position in world frame
        global_goal_xy : (x, y) final global goal in world frame

        Returns
        -------
        List of (x, y) world-frame waypoints [start ... local_goal],
        or None if the grid is uninitialised or A* finds no path.
        """
        if grid_map.gmap is None:
            return None

        # --- convert start to grid indices ---
        sx = float(drone_pos_xy[0])
        sy = float(drone_pos_xy[1])
        six, siy = grid_map.world_to_index(sx, sy)

        if six is None:
            # Drone is outside its own grid — should not happen in normal use
            return None

        # --- determine local goal ---
        gx = float(global_goal_xy[0])
        gy = float(global_goal_xy[1])
        gix, giy = self._local_goal(grid_map, six, siy, gx, gy)

        if gix is None:
            return None

        # Already at goal cell
        if six == gix and siy == giy:
            wx, wy = grid_map.index_to_world(six, siy)
            return [(wx, wy)]

        # --- A* search ---
        path_grid = self._a_star(grid_map, six, siy, gix, giy)
        if path_grid is None:
            return None

        # Convert grid path to world coordinates
        return [grid_map.index_to_world(ix, iy) for ix, iy in path_grid]

    # ------------------------------------------------------------------
    # Local goal selection
    # ------------------------------------------------------------------

    def _local_goal(
        self,
        grid_map: FixedGaussianGridMap,
        six: int, siy: int,
        gx: float, gy: float,
    ):
        """
        Compute the A* target cell.

        If the global goal is inside the grid, return its cell (or the
        nearest free cell if that cell is occupied).

        If the global goal is outside the grid, find the intersection of
        the ray (drone -> global_goal) with the grid boundary and return
        the last free boundary cell along that ray.
        """
        gix_raw, giy_raw = grid_map.world_to_index(gx, gy)

        if gix_raw is not None:
            # Goal is inside the grid
            if self._is_free(grid_map, gix_raw, giy_raw):
                return gix_raw, giy_raw
            return self._nearest_free(grid_map, gix_raw, giy_raw)

        # Goal is outside — walk the ray from start toward goal and stop
        # at the last cell still inside the grid boundary
        dx = gix_raw if gix_raw is not None else (
            int((gx - grid_map.minx) / grid_map.reso)   # may be negative / OOB
        )
        dy = giy_raw if giy_raw is not None else (
            int((gy - grid_map.miny) / grid_map.reso)
        )

        # Recompute raw (possibly out-of-bounds) indices
        gix_oob = int((gx - grid_map.minx) / grid_map.reso)
        giy_oob = int((gy - grid_map.miny) / grid_map.reso)

        # Parametric boundary intersection:  (six, siy) + t*(dir) hits grid edge
        border_ix, border_iy = self._ray_grid_boundary(
            grid_map, six, siy, gix_oob, giy_oob
        )

        if self._is_free(grid_map, border_ix, border_iy):
            return border_ix, border_iy
        return self._nearest_free(grid_map, border_ix, border_iy)

    def _ray_grid_boundary(
        self,
        grid_map: FixedGaussianGridMap,
        six: int, siy: int,
        gix: int, giy: int,
    ):
        """
        Find the grid cell closest to the global goal along the line
        (six, siy) -> (gix, giy) that still lies inside [0, cells).

        Uses Bresenham-style parametric clipping.
        """
        cells = grid_map.cells
        ddx = gix - six
        ddy = giy - siy

        # t in [0,1] parameterises the segment; find max t still inside grid
        t_max = 0.0

        if ddx > 0:
            t_max = max(t_max, min(1.0, (cells - 1 - six) / ddx))
        elif ddx < 0:
            t_max = max(t_max, min(1.0, -six / ddx))
        else:
            t_max = 1.0  # no x movement; leave as 1 and let y clip

        t_from_y = 1.0
        if ddy > 0:
            t_from_y = min(1.0, (cells - 1 - siy) / ddy)
        elif ddy < 0:
            t_from_y = min(1.0, -siy / ddy)

        t = min(t_max, t_from_y) * 0.97  # pull slightly inward from edge

        bix = int(six + t * ddx)
        biy = int(siy + t * ddy)

        # Hard clamp to valid range
        bix = max(0, min(bix, cells - 1))
        biy = max(0, min(biy, cells - 1))
        return bix, biy

    # ------------------------------------------------------------------
    # A* core
    # ------------------------------------------------------------------

    def _a_star(
        self,
        grid_map: FixedGaussianGridMap,
        six: int, siy: int,
        gix: int, giy: int,
    ):
        """
        Standard A* on the grid.

        Returns list of (ix, iy) from start to goal (inclusive),
        or None if no path exists.
        """
        start = _Node(six, siy, 0.0, None)
        open_heap = []
        heapq.heappush(open_heap, (self._h(six, siy, gix, giy), start))

        # closed: (ix, iy) -> best g seen
        closed: dict[tuple, float] = {}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            key = (current.ix, current.iy)

            if key in closed:
                continue
            closed[key] = current.g

            if current.ix == gix and current.iy == giy:
                return self._extract_path(current)

            for ddx, ddy, move_cost in self._MOTION:
                nix = current.ix + ddx
                niy = current.iy + ddy
                nkey = (nix, niy)

                if not self._is_free(grid_map, nix, niy):
                    continue
                if nkey in closed:
                    continue

                cell_cost = self._cell_cost(grid_map, nix, niy)
                ng = current.g + move_cost * grid_map.reso * cell_cost
                h = self._h(nix, niy, gix, giy)

                neighbor = _Node(nix, niy, ng, current)
                heapq.heappush(open_heap, (ng + h, neighbor))

        return None  # no path found

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_free(self, grid_map: FixedGaussianGridMap, ix: int, iy: int) -> bool:
        """True if the cell is inside the grid and below the obstacle threshold."""
        if ix < 0 or ix >= grid_map.cells or iy < 0 or iy >= grid_map.cells:
            return False
        return float(grid_map.gmap[ix, iy]) < self.obstacle_threshold

    def _cell_cost(self, grid_map: FixedGaussianGridMap, ix: int, iy: int) -> float:
        """
        Traversal cost multiplier for cell (ix, iy).
        1.0 = free space; higher near obstacles.
        """
        prob = float(grid_map.gmap[ix, iy])
        return 1.0 + self.obstacle_cost_weight * prob

    @staticmethod
    def _h(ix: int, iy: int, gix: int, giy: int) -> float:
        """Euclidean heuristic (admissible for uniform-cost grid)."""
        return math.hypot(gix - ix, giy - iy)

    @staticmethod
    def _extract_path(goal_node: _Node):
        """Walk parent pointers from goal back to start, then reverse."""
        path = []
        node = goal_node
        while node is not None:
            path.append((node.ix, node.iy))
            node = node.parent
        path.reverse()
        return path

    def _nearest_free(self, grid_map: FixedGaussianGridMap, ix: int, iy: int):
        """
        BFS from (ix, iy) to find the nearest free cell.
        Returns (None, None) if the entire grid is blocked.
        """
        from collections import deque
        visited = {(ix, iy)}
        queue = deque([(ix, iy)])
        while queue:
            cx, cy = queue.popleft()
            if self._is_free(grid_map, cx, cy):
                return cx, cy
            for ddx, ddy, _ in self._MOTION:
                nx, ny = cx + ddx, cy + ddy
                if (nx, ny) not in visited:
                    if 0 <= nx < grid_map.cells and 0 <= ny < grid_map.cells:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None, None
