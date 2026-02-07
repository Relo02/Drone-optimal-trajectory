"""
A* Local Planner using Gaussian Grid Map

Plans paths from current position to a global goal using probabilistic
obstacle information from lidar-based Gaussian grid maps.

author: Lorenzo Ortolani
"""

import math
import heapq
import numpy as np
from mujoco_sim.gaussian_grid_map import GaussianGridMap


class Node:
    """A* search node."""

    def __init__(self, ix, iy, cost, parent_index):
        self.ix = ix  # grid index x
        self.iy = iy  # grid index y
        self.cost = cost  # g(n): cost from start
        self.parent_index = parent_index  # parent node index

    def __lt__(self, other):
        return self.cost < other.cost


class AStarLocalPlanner:
    """
    A* local planner that uses Gaussian grid map for obstacle avoidance.

    The planner considers obstacle probability when computing path costs,
    preferring paths that stay away from high-probability obstacle regions.
    """

    # 8-directional motion: [dx, dy, cost_multiplier]
    MOTION = [
        [1, 0, 1.0],    # right
        [0, 1, 1.0],    # up
        [-1, 0, 1.0],   # left
        [0, -1, 1.0],   # down
        [1, 1, 1.414],  # right-up (diagonal)
        [1, -1, 1.414], # right-down
        [-1, 1, 1.414], # left-up
        [-1, -1, 1.414] # left-down
    ]

    def __init__(self, obstacle_threshold=0.5, obstacle_cost_weight=10.0):
        """
        Initialize the A* local planner.

        Args:
            obstacle_threshold: Probability above which a cell is considered blocked
            obstacle_cost_weight: Weight for obstacle probability in cost function
        """
        self.obstacle_threshold = obstacle_threshold
        self.obstacle_cost_weight = obstacle_cost_weight
        self.grid_map = None
        self.path = None
        self.path_world = None

    def set_grid_map(self, grid_map: GaussianGridMap):
        """Set the Gaussian grid map to use for planning."""
        self.grid_map = grid_map

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        if self.grid_map is None:
            return None, None
        ix = int((x - self.grid_map.minx) / self.grid_map.xyreso)
        iy = int((y - self.grid_map.miny) / self.grid_map.xyreso)
        return ix, iy

    def grid_to_world(self, ix, iy):
        """Convert grid indices to world coordinates (cell center)."""
        if self.grid_map is None:
            return None, None
        x = ix * self.grid_map.xyreso + self.grid_map.minx
        y = iy * self.grid_map.xyreso + self.grid_map.miny
        return x, y

    def is_valid_cell(self, ix, iy):
        """Check if grid cell is within bounds and not blocked."""
        if self.grid_map is None or self.grid_map.gmap is None:
            return False

        # Check bounds
        if ix < 0 or ix >= self.grid_map.xw:
            return False
        if iy < 0 or iy >= self.grid_map.yw:
            return False

        # Check obstacle probability
        prob = self.grid_map.gmap[ix, iy]
        if prob >= self.obstacle_threshold:
            return False

        return True

    def get_cell_cost(self, ix, iy):
        """
        Get traversal cost for a cell based on obstacle probability.

        Higher probability = higher cost, encouraging paths away from obstacles.
        """
        if self.grid_map is None or self.grid_map.gmap is None:
            return float('inf')

        if not self.is_valid_cell(ix, iy):
            return float('inf')

        prob = self.grid_map.gmap[ix, iy]
        # Base cost + weighted probability cost
        return 1.0 + self.obstacle_cost_weight * prob

    def heuristic(self, ix, iy, gix, giy):
        """Euclidean distance heuristic."""
        return math.hypot(gix - ix, giy - iy)

    def get_local_goal(self, start_world, goal_world):
        """
        Get local goal within grid bounds.

        If the global goal is outside the grid, returns the boundary point
        closest to the goal along the line from start to goal.

        Args:
            start_world: (x, y) start position in world coordinates
            goal_world: (x, y) goal position in world coordinates

        Returns:
            (gix, giy) local goal in grid indices
        """
        gix, giy = self.world_to_grid(goal_world[0], goal_world[1])

        # Check if goal is within grid
        if 0 <= gix < self.grid_map.xw and 0 <= giy < self.grid_map.yw:
            # Goal is inside grid, but check if it's blocked
            if self.is_valid_cell(gix, giy):
                return gix, giy
            # Goal cell is blocked, find nearest valid cell
            return self._find_nearest_valid_cell(gix, giy)

        # Goal is outside grid - find intersection with grid boundary
        six, siy = self.world_to_grid(start_world[0], start_world[1])

        # Direction from start to goal
        dx = gix - six
        dy = giy - siy

        if dx == 0 and dy == 0:
            return six, siy

        # Find boundary intersection using parametric line
        t_values = []

        if dx != 0:
            t_left = -six / dx if dx != 0 else float('inf')
            t_right = (self.grid_map.xw - 1 - six) / dx if dx != 0 else float('inf')
            if dx > 0:
                t_values.append(t_right)
            else:
                t_values.append(t_left)

        if dy != 0:
            t_bottom = -siy / dy if dy != 0 else float('inf')
            t_top = (self.grid_map.yw - 1 - siy) / dy if dy != 0 else float('inf')
            if dy > 0:
                t_values.append(t_top)
            else:
                t_values.append(t_bottom)

        # Get minimum positive t
        t = min([t for t in t_values if t > 0], default=0)
        t = min(t, 1.0)  # Don't go past the goal

        # Calculate boundary point
        bix = int(six + t * dx * 0.95)  # 0.95 to stay inside
        biy = int(siy + t * dy * 0.95)

        # Clamp to grid bounds
        bix = max(0, min(bix, self.grid_map.xw - 1))
        biy = max(0, min(biy, self.grid_map.yw - 1))

        if self.is_valid_cell(bix, biy):
            return bix, biy

        return self._find_nearest_valid_cell(bix, biy)

    def _find_nearest_valid_cell(self, ix, iy):
        """Find the nearest valid cell to the given position using BFS (Breadth-First Search)."""
        if self.is_valid_cell(ix, iy):
            return ix, iy

        visited = set()
        queue = [(ix, iy)]
        visited.add((ix, iy))

        while queue:
            cx, cy = queue.pop(0)
            for motion in self.MOTION:
                nx = cx + motion[0]
                ny = cy + motion[1]

                if (nx, ny) in visited:
                    continue

                if 0 <= nx < self.grid_map.xw and 0 <= ny < self.grid_map.yw:
                    if self.is_valid_cell(nx, ny):
                        return nx, ny
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        # No valid cell found, return original
        return ix, iy

    def plan(self, start_world, goal_world):
        """
        Plan a path from start to goal using A*.

        Args:
            start_world: (x, y) start position in world coordinates
            goal_world: (x, y) goal position in world coordinates

        Returns:
            List of (x, y) waypoints in world coordinates, or None if no path found
        """
        if self.grid_map is None or self.grid_map.gmap is None:
            print("Error: No grid map set")
            return None

        # Convert to grid coordinates
        six, siy = self.world_to_grid(start_world[0], start_world[1])

        # Validate start position
        if not (0 <= six < self.grid_map.xw and 0 <= siy < self.grid_map.yw):
            print(f"Error: Start position outside grid bounds")
            return None

        # Get local goal (handles goals outside grid)
        gix, giy = self.get_local_goal(start_world, goal_world)

        if gix is None or giy is None:
            print("Error: Could not determine valid goal")
            return None

        # Check if already at goal
        if six == gix and siy == giy:
            self.path = [(six, siy)]
            self.path_world = [self.grid_to_world(six, siy)]
            return self.path_world

        # A* search
        start_node = Node(six, siy, 0.0, -1)
        open_set = []
        heapq.heappush(open_set, (self.heuristic(six, siy, gix, giy), start_node))

        closed_set = {}
        open_dict = {(six, siy): start_node}

        while open_set:
            _, current = heapq.heappop(open_set)

            if (current.ix, current.iy) in closed_set:
                continue

            # Check if goal reached
            if current.ix == gix and current.iy == giy:
                return self._extract_path(current, closed_set)

            # Add to closed set
            current_index = (current.ix, current.iy)
            closed_set[current_index] = current

            # Expand neighbors
            for motion in self.MOTION:
                nix = current.ix + motion[0]
                niy = current.iy + motion[1]

                if not self.is_valid_cell(nix, niy):
                    continue

                if (nix, niy) in closed_set:
                    continue

                # Calculate cost
                move_cost = motion[2] * self.grid_map.xyreso
                cell_cost = self.get_cell_cost(nix, niy)
                new_cost = current.cost + move_cost * cell_cost

                neighbor_key = (nix, niy)

                # Check if we found a better path
                if neighbor_key in open_dict:
                    if open_dict[neighbor_key].cost <= new_cost:
                        continue

                neighbor = Node(nix, niy, new_cost, current_index)
                open_dict[neighbor_key] = neighbor

                f_cost = new_cost + self.heuristic(nix, niy, gix, giy)
                heapq.heappush(open_set, (f_cost, neighbor))

        print("Warning: No path found to goal")
        return None

    def _extract_path(self, goal_node, closed_set):
        """Extract path from goal node back to start."""
        path_grid = [(goal_node.ix, goal_node.iy)]
        current = goal_node

        while current.parent_index != -1:
            current = closed_set[current.parent_index]
            path_grid.append((current.ix, current.iy))

        path_grid.reverse()
        self.path = path_grid

        # Convert to world coordinates
        self.path_world = [self.grid_to_world(ix, iy) for ix, iy in path_grid]

        return self.path_world

    def get_next_waypoint(self, current_pos, lookahead_distance=0.5):
        """
        Get the next waypoint along the path for trajectory tracking.

        Args:
            current_pos: (x, y) current position
            lookahead_distance: Distance ahead to look for waypoint

        Returns:
            (x, y) next waypoint or None if no path
        """
        if self.path_world is None or len(self.path_world) == 0:
            return None

        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0

        for i, (wx, wy) in enumerate(self.path_world):
            dist = math.hypot(wx - current_pos[0], wy - current_pos[1])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Look ahead from closest point
        for i in range(closest_idx, len(self.path_world)):
            wx, wy = self.path_world[i]
            dist = math.hypot(wx - current_pos[0], wy - current_pos[1])
            if dist >= lookahead_distance:
                return (wx, wy)

        # Return last waypoint if we're near the end
        return self.path_world[-1]

    def draw_path(self, ax=None):
        """Draw the planned path on a matplotlib axes."""
        import matplotlib.pyplot as plt

        if self.path_world is None:
            return None

        if ax is None:
            fig, ax = plt.subplots()

        path_x = [p[0] for p in self.path_world]
        path_y = [p[1] for p in self.path_world]

        ax.plot(path_x, path_y, 'b-', linewidth=2, label='A* Path')
        ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'r*', markersize=15, label='Local Goal')

        return ax


# def main():
#     """Test the A* local planner with a simple example."""
#     import matplotlib.pyplot as plt

#     # Create a simple test scenario
#     print("A* Local Planner Test")
#     print("=" * 40)

#     # Generate some obstacle points
#     np.random.seed(42)
#     ox = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 2.0, 2.5, 3.0])
#     oy = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0])
#     points = np.column_stack([ox, oy, np.zeros_like(ox)])

#     # Create Gaussian grid map
#     grid_map = GaussianGridMap(xyreso=0.2, std=0.3, extend_area=2.0)
#     grid_map.update_from_lidar_points(points, drone_pos=[0, 0, 0])

#     print(f"Grid size: {grid_map.xw} x {grid_map.yw}")
#     print(f"Grid bounds: x=[{grid_map.minx:.1f}, {grid_map.maxx:.1f}], "
#           f"y=[{grid_map.miny:.1f}, {grid_map.maxy:.1f}]")

#     # Create planner
#     planner = AStarLocalPlanner(obstacle_threshold=0.3, obstacle_cost_weight=15.0)
#     planner.set_grid_map(grid_map)

#     # Plan path
#     start = (0.0, 0.0)
#     goal = (5.0, 5.0)

#     print(f"\nPlanning from {start} to {goal}...")
#     path = planner.plan(start, goal)

#     if path:
#         print(f"Path found with {len(path)} waypoints")

#         # Visualize
#         fig, ax = plt.subplots(figsize=(10, 8))

#         # Draw grid map
#         grid_map.draw_heatmap(ax=ax, cmap='Reds', alpha=0.6)

#         # Draw obstacles
#         ax.scatter(ox, oy, c='red', s=100, marker='x', linewidths=2, label='Obstacles')

#         # Draw path
#         planner.draw_path(ax=ax)

#         # Draw global goal (if different from local goal)
#         ax.plot(goal[0], goal[1], 'm^', markersize=12, label='Global Goal')

#         ax.set_xlabel('X (m)')
#         ax.set_ylabel('Y (m)')
#         ax.set_title('A* Local Planner with Gaussian Grid Map')
#         ax.legend()
#         ax.grid(True, alpha=0.3)

#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No path found!")


# if __name__ == '__main__':
#     main()
