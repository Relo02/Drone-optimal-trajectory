"""

2D gaussian grid map for lidar-based obstacle detection

author: Lorenzo Ortolani

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class GaussianGridMap:
    """Gaussian grid map that can be updated with lidar point cloud data."""

    def __init__(self, xyreso=0.5, std=1.0, extend_area=2.0):
        """
        Initialize the Gaussian grid map.

        Args:
            xyreso: Grid resolution in meters
            std: Standard deviation for Gaussian distribution
            extend_area: Extension area around obstacles in meters
        """
        self.xyreso = xyreso
        self.std = std
        self.extend_area = extend_area
        self.gmap = None
        self.minx = self.miny = self.maxx = self.maxy = 0
        self.xw = self.yw = 0

    def update_from_lidar_points(self, points, drone_pos=None):
        """
        Update the grid map from 3D lidar points.

        Args:
            points: Nx3 array of lidar hit points in world frame
            drone_pos: Optional drone position [x, y, z] to center the map

        Returns:
            True if map was updated, False if no valid points
        """
        if len(points) == 0:
            return False

        # Extract 2D coordinates (x, y) from 3D points
        ox = points[:, 0]
        oy = points[:, 1]

        # Calculate grid configuration
        self._calc_grid_config(ox, oy, drone_pos)

        # Generate the Gaussian grid map
        self._generate_map(ox, oy)

        return True

    def _calc_grid_config(self, ox, oy, drone_pos=None):
        """Calculate grid map configuration based on obstacle positions."""
        if drone_pos is not None:
            # Center map around drone position with extend_area
            self.minx = drone_pos[0] - self.extend_area * 2
            self.miny = drone_pos[1] - self.extend_area * 2
            self.maxx = drone_pos[0] + self.extend_area * 2
            self.maxy = drone_pos[1] + self.extend_area * 2

            # Expand to include all obstacles
            self.minx = min(self.minx, min(ox) - self.extend_area / 2.0)
            self.miny = min(self.miny, min(oy) - self.extend_area / 2.0)
            self.maxx = max(self.maxx, max(ox) + self.extend_area / 2.0)
            self.maxy = max(self.maxy, max(oy) + self.extend_area / 2.0)
        else:
            self.minx = round(min(ox) - self.extend_area / 2.0)
            self.miny = round(min(oy) - self.extend_area / 2.0)
            self.maxx = round(max(ox) + self.extend_area / 2.0)
            self.maxy = round(max(oy) + self.extend_area / 2.0)

        self.xw = int(round((self.maxx - self.minx) / self.xyreso))
        self.yw = int(round((self.maxy - self.miny) / self.xyreso))

    def _generate_map(self, ox, oy):
        """Generate the Gaussian probability map."""
        self.gmap = np.zeros((self.xw, self.yw))

        for ix in range(self.xw):
            for iy in range(self.yw):
                x = ix * self.xyreso + self.minx
                y = iy * self.xyreso + self.miny

                # Find minimum distance to any obstacle
                distances = np.sqrt((ox - x)**2 + (oy - y)**2)
                mindis = np.min(distances)

                # Convert distance to probability using Gaussian CDF
                pdf = 1.0 - norm.cdf(mindis, 0.0, self.std)
                self.gmap[ix, iy] = pdf

    def get_probability(self, x, y):
        """
        Get obstacle probability at a given world position.

        Args:
            x, y: World coordinates

        Returns:
            Probability value [0, 1] or None if outside map bounds
        """
        if self.gmap is None:
            return None

        ix = int((x - self.minx) / self.xyreso)
        iy = int((y - self.miny) / self.xyreso)

        if 0 <= ix < self.xw and 0 <= iy < self.yw:
            return self.gmap[ix, iy]
        return None

    def draw_heatmap(self, ax=None, cmap='Reds', alpha=0.7):
        """
        Draw the Gaussian grid map as a heatmap.

        Args:
            ax: Matplotlib axes to draw on (creates new if None)
            cmap: Colormap to use
            alpha: Transparency of the heatmap

        Returns:
            The matplotlib axes object
        """
        if self.gmap is None:
            return None

        if ax is None:
            fig, ax = plt.subplots()

        # Create coordinate arrays matching the grid dimensions
        # For pcolormesh with shading='flat', we need xw+1 and yw+1 edge coordinates
        x_edges = np.linspace(self.minx - self.xyreso / 2.0,
                              self.minx + self.xw * self.xyreso - self.xyreso / 2.0,
                              self.xw + 1)
        y_edges = np.linspace(self.miny - self.xyreso / 2.0,
                              self.miny + self.yw * self.xyreso - self.xyreso / 2.0,
                              self.yw + 1)

        pcm = ax.pcolormesh(x_edges, y_edges, self.gmap.T, vmax=1.0, cmap=cmap, alpha=alpha, shading='flat')
        ax.set_aspect('equal')

        return ax, pcm


def generate_gaussian_grid_map(ox, oy, xyreso, std, extend_area=10.0):
    """
    Legacy function for backwards compatibility.
    Generate a Gaussian grid map from obstacle positions.

    Args:
        ox, oy: Arrays of obstacle x and y coordinates
        xyreso: Grid resolution
        std: Standard deviation for Gaussian distribution
        extend_area: Extension area around obstacles

    Returns:
        gmap, minx, maxx, miny, maxy
    """
    grid_map = GaussianGridMap(xyreso=xyreso, std=std, extend_area=extend_area)
    points = np.column_stack([ox, oy, np.zeros_like(ox)])
    grid_map.update_from_lidar_points(points)

    return grid_map.gmap, grid_map.minx, grid_map.maxx, grid_map.miny, grid_map.maxy
