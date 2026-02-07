"""
Real-time point cloud visualization with Gaussian grid map.
Streams lidar data from MuJoCo simulation with continuous updates.
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gaussian_grid_map import GaussianGridMap

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path('/home/lorenzo/Drone-optimal-trajectory/mujoco/model/drone_world.xml')
data = mujoco.MjData(model)

# Lidar configuration
NUM_LIDAR_RAYS = 360
LIDAR_SENSOR_START = 6

# Gaussian grid map parameters
GRID_RESOLUTION = 0.2  # meters
GAUSSIAN_STD = 0.5  # standard deviation for probability distribution
EXTEND_AREA = 2.0  # extension around obstacles

# Animation parameters
UPDATE_INTERVAL = 50  # milliseconds between frames
SIM_STEPS_PER_FRAME = 10  # simulation steps per visualization frame


def get_lidar_points(data):
    """Convert lidar readings to 3D points in world frame."""
    points = []
    drone_pos = data.qpos[:3]
    drone_quat = data.qpos[3:7]

    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, drone_quat)
    rot_mat = rot_mat.reshape(3, 3)

    for i in range(NUM_LIDAR_RAYS):
        distance = data.sensordata[LIDAR_SENSOR_START + i]
        if distance > 0:
            angle = np.radians(i)
            ray_dir_body = np.array([np.cos(angle), np.sin(angle), 0])
            ray_dir_world = rot_mat @ ray_dir_body
            hit_point = drone_pos + distance * ray_dir_world
            points.append(hit_point)

    return np.array(points) if points else np.empty((0, 3))


class RealtimeVisualizer:
    """Real-time visualization of lidar point cloud and Gaussian grid map."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.grid_map = GaussianGridMap(
            xyreso=GRID_RESOLUTION,
            std=GAUSSIAN_STD,
            extend_area=EXTEND_AREA
        )
        self.frame_count = 0
        self.pcm = None  # Store pcolormesh reference for colorbar

        # Setup figure
        self.fig = plt.figure(figsize=(16, 5))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # 3D point cloud subplot
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_zlabel('Z (m)')
        self.ax1.set_title('3D Point Cloud')

        # 2D top-down view subplot
        self.ax2 = self.fig.add_subplot(132)
        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Y (m)')
        self.ax2.set_title('Top-Down View (2D Lidar Scan)')
        self.ax2.set_aspect('equal')
        self.ax2.grid(True)

        # Gaussian grid map subplot
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.set_xlabel('X (m)')
        self.ax3.set_ylabel('Y (m)')
        self.ax3.set_title('Gaussian Grid Map (Obstacle Probability)')

        # Initialize with first frame
        self._step_simulation()
        self._init_plots()

        plt.tight_layout()

    def _on_key(self, event):
        """Handle key press events."""
        if event.key == 'escape' or event.key == 'q':
            plt.close(self.fig)

    def _step_simulation(self):
        """Step the MuJoCo simulation."""
        for _ in range(SIM_STEPS_PER_FRAME):
            mujoco.mj_step(self.model, self.data)

    def _init_plots(self):
        """Initialize plot elements."""
        points = get_lidar_points(self.data)
        drone_pos = self.data.qpos[:3]

        if len(points) > 0:
            # 3D scatter plots
            self.scatter3d_points = self.ax1.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c='green', s=10, alpha=0.7, label='Lidar hits'
            )
            self.scatter3d_drone = self.ax1.scatter(
                *drone_pos, c='blue', s=100, marker='^', label='Drone'
            )
            self.ax1.legend()

            # 2D scatter plots
            self.scatter2d_points = self.ax2.scatter(
                points[:, 0], points[:, 1],
                c='green', s=10, alpha=0.7, label='Lidar hits'
            )
            self.scatter2d_drone = self.ax2.scatter(
                drone_pos[0], drone_pos[1],
                c='blue', s=100, marker='^', label='Drone'
            )
            self.ax2.legend()

            # Gaussian grid map
            self.grid_map.update_from_lidar_points(points, drone_pos=drone_pos)
            _, self.pcm = self.grid_map.draw_heatmap(ax=self.ax3, cmap='Reds', alpha=0.8)
            self.scatter3_points = self.ax3.scatter(
                points[:, 0], points[:, 1],
                c='black', s=5, alpha=0.5, label='Lidar hits'
            )
            self.scatter3_drone = self.ax3.scatter(
                drone_pos[0], drone_pos[1],
                c='blue', s=100, marker='^', label='Drone'
            )
            self.ax3.legend()
            self.cbar = self.fig.colorbar(self.pcm, ax=self.ax3, label='Probability')

    def update(self, frame):
        """Update function called by FuncAnimation."""
        self._step_simulation()

        points = get_lidar_points(self.data)
        drone_pos = self.data.qpos[:3]

        self.frame_count += 1

        if len(points) == 0:
            return

        # Update 3D plot (need to clear and redraw for 3D scatter)
        self.ax1.cla()
        self.ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c='green', s=10, alpha=0.7, label='Lidar hits')
        self.ax1.scatter(*drone_pos, c='blue', s=100, marker='^', label='Drone')
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_zlabel('Z (m)')
        self.ax1.set_title(f'3D Point Cloud (Frame {self.frame_count})')
        self.ax1.legend()

        # Update 2D top-down view
        self.scatter2d_points.set_offsets(points[:, :2])
        self.scatter2d_drone.set_offsets([[drone_pos[0], drone_pos[1]]])
        self.ax2.set_title(f'Top-Down View ({len(points)} points)')

        # Auto-adjust 2D view limits
        margin = 1.0
        self.ax2.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
        self.ax2.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)

        # Update Gaussian grid map
        self.grid_map.update_from_lidar_points(points, drone_pos=drone_pos)

        # Clear and redraw heatmap
        self.ax3.cla()
        _, self.pcm = self.grid_map.draw_heatmap(ax=self.ax3, cmap='Reds', alpha=0.8)
        self.ax3.scatter(points[:, 0], points[:, 1], c='black', s=5, alpha=0.5, label='Lidar hits')
        self.ax3.scatter(drone_pos[0], drone_pos[1], c='blue', s=100, marker='^', label='Drone')
        self.ax3.set_xlabel('X (m)')
        self.ax3.set_ylabel('Y (m)')
        self.ax3.set_title('Gaussian Grid Map (Obstacle Probability)')
        self.ax3.legend()

        return []

    def run(self):
        """Start the real-time visualization."""
        print("Starting real-time visualization...")
        print("Press 'q' or 'Escape' to quit")

        self.anim = FuncAnimation(
            self.fig,
            self.update,
            interval=UPDATE_INTERVAL,
            blit=False,
            cache_frame_data=False
        )
        plt.show()


if __name__ == '__main__':
    visualizer = RealtimeVisualizer(model, data)
    visualizer.run()
