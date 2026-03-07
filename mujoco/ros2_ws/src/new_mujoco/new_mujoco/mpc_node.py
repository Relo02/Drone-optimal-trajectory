"""
MPC tracker ROS2 node for the Skydio X2 drone.

Architecture
------------
  Subscribes:
    /skydio/pose    PoseStamped       — drone pose (position + orientation)
    /skydio/scan3d  PointCloud2       — 3-D lidar hits (world frame), for building
                                        the MPC's own obstacle cost grid
    /a_star/path    nav_msgs/Path     — local A* path produced by a_star_node

  Publishes:
    /mpc/predicted_path  nav_msgs/Path          — N-step MPC predicted trajectory
    /mpc/next_setpoint   geometry_msgs/PoseStamped — lookahead setpoint on MPC trajectory
    /goal_pose           geometry_msgs/PoseStamped — same as next_setpoint, drives
                                                      the cascaded-PID in skydio_sim_node
    /mpc/diagnostics     std_msgs/Float64MultiArray — [success, cost, solve_time_ms]

  Control flow (at mpc_rate_hz):
    1. Estimate drone velocity by differentiating consecutive /skydio/pose messages.
    2. Build FixedGaussianGridMap from the latest lidar scan.
    3. Solve MPCTracker → predicted state trajectory X*(N+1, 7).
    4. Walk the predicted trajectory to find the first state >= mpc_lookahead_dist
       from the drone and publish that as the PID setpoint.  This prevents the
       drone from oscillating when the 1-step-ahead point (only v_ref*dt metres
       away) is closer than the PID's natural settling distance.
    5. Publish setpoint also on /goal_pose to close the control loop with
       the cascaded PID inside skydio_sim_node.

  Logging (throttled):
    Console:  solve status, cost, solve time, next setpoint, path length
    The full predicted path and A* path are visible in Foxglove via their
    respective topic subscriptions (/mpc/predicted_path and /a_star/path).

author: Lorenzo Ortolani
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray

from new_mujoco.gaussian_grid_map import FixedGaussianGridMap
from new_mujoco.mpc_tracker import MPCConfig, MPCTracker


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class MPCNode(Node):

    def __init__(self):
        super().__init__('mpc_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('mpc_N',            30)
        self.declare_parameter('mpc_dt',           0.1)
        self.declare_parameter('mpc_v_max_xy',     2.0)
        self.declare_parameter('mpc_v_max_z',      1.0)
        self.declare_parameter('mpc_a_max_xy',     2.0)
        self.declare_parameter('mpc_a_max_z',      1.5)
        self.declare_parameter('mpc_yaw_rate_max', 1.5)
        self.declare_parameter('mpc_v_ref',        1.0)
        self.declare_parameter('mpc_Q_xy',        15.0)
        self.declare_parameter('mpc_Q_z',         20.0)
        self.declare_parameter('mpc_Q_vel_xy',     1.0)
        self.declare_parameter('mpc_Q_vel_z',      2.0)
        self.declare_parameter('mpc_Q_yaw',        0.2)
        self.declare_parameter('mpc_Q_terminal',  50.0)
        self.declare_parameter('mpc_R_acc_xy',     1.0)
        self.declare_parameter('mpc_R_acc_z',      1.5)
        self.declare_parameter('mpc_R_yaw_rate',   0.1)
        self.declare_parameter('mpc_R_jerk',       0.3)
        self.declare_parameter('mpc_W_obs',              30.0)
        self.declare_parameter('mpc_d_safe_pts',          0.5)
        self.declare_parameter('mpc_W_obs_pts',          50.0)
        self.declare_parameter('mpc_max_obs_constraints', 15)
        self.declare_parameter('mpc_obs_check_radius',    3.0)
        self.declare_parameter('mpc_max_iter',   100)
        self.declare_parameter('mpc_warm_start',  True)
        self.declare_parameter('mpc_rate_hz',       5.0)
        # Lookahead distance [m]: the first predicted state >= this distance from
        # the drone is used as the PID setpoint.  Must be > v_ref*dt to avoid
        # oscillation.  Rule of thumb: ~1–2 × the PID pos_vel_limit / mpc_rate_hz.
        self.declare_parameter('mpc_lookahead_dist', 1.5)
        # Grid (MPC builds its own grid from lidar)
        self.declare_parameter('grid_reso',        0.25)
        self.declare_parameter('grid_half_width',  5.0)
        self.declare_parameter('grid_std',         0.4)
        self.declare_parameter('max_lidar_range',  6.0)
        self.declare_parameter('planning_height',  1.5)

        # ── MPCConfig ─────────────────────────────────────────────────
        cfg = MPCConfig(
            N             = int(self.get_parameter('mpc_N').value),
            dt            = float(self.get_parameter('mpc_dt').value),
            v_max_xy      = float(self.get_parameter('mpc_v_max_xy').value),
            v_max_z       = float(self.get_parameter('mpc_v_max_z').value),
            a_max_xy      = float(self.get_parameter('mpc_a_max_xy').value),
            a_max_z       = float(self.get_parameter('mpc_a_max_z').value),
            yaw_rate_max  = float(self.get_parameter('mpc_yaw_rate_max').value),
            v_ref         = float(self.get_parameter('mpc_v_ref').value),
            Q_xy          = float(self.get_parameter('mpc_Q_xy').value),
            Q_z           = float(self.get_parameter('mpc_Q_z').value),
            Q_vel_xy      = float(self.get_parameter('mpc_Q_vel_xy').value),
            Q_vel_z       = float(self.get_parameter('mpc_Q_vel_z').value),
            Q_yaw         = float(self.get_parameter('mpc_Q_yaw').value),
            Q_terminal    = float(self.get_parameter('mpc_Q_terminal').value),
            R_acc_xy      = float(self.get_parameter('mpc_R_acc_xy').value),
            R_acc_z       = float(self.get_parameter('mpc_R_acc_z').value),
            R_yaw_rate    = float(self.get_parameter('mpc_R_yaw_rate').value),
            R_jerk        = float(self.get_parameter('mpc_R_jerk').value),
            W_obs                = float(self.get_parameter('mpc_W_obs').value),
            d_safe_pts           = float(self.get_parameter('mpc_d_safe_pts').value),
            W_obs_pts            = float(self.get_parameter('mpc_W_obs_pts').value),
            max_obs_constraints  = int(self.get_parameter('mpc_max_obs_constraints').value),
            obs_check_radius     = float(self.get_parameter('mpc_obs_check_radius').value),
            max_iter      = int(self.get_parameter('mpc_max_iter').value),
            warm_start    = bool(self.get_parameter('mpc_warm_start').value),
        )
        self._tracker = MPCTracker(config=cfg)

        self._grid_map = FixedGaussianGridMap(
            reso       = float(self.get_parameter('grid_reso').value),
            half_width = float(self.get_parameter('grid_half_width').value),
            std        = float(self.get_parameter('grid_std').value),
        )

        self._max_lidar_range = float(self.get_parameter('max_lidar_range').value)
        self._planning_height = float(self.get_parameter('planning_height').value)
        self._lookahead_dist  = float(self.get_parameter('mpc_lookahead_dist').value)

        # ── State ─────────────────────────────────────────────────────
        self._pose: PoseStamped | None = None
        self._vel = np.zeros(3)          # estimated world-frame velocity [m/s]
        self._yaw = 0.0
        self._prev_pos: np.ndarray | None = None
        self._prev_stamp_sec: float | None = None
        self._a_star_path: list | None = None
        self._lidar_points: np.ndarray | None = None

        # Logging counters
        self._solve_count     = 0
        self._fail_count      = 0
        self._total_solve_ms  = 0.0

        # ── QoS ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(PoseStamped, '/skydio/pose',  self._pose_cb,  10)
        self.create_subscription(Path,        '/a_star/path',  self._path_cb,  10)
        self.create_subscription(PointCloud2, '/skydio/scan3d', self._lidar_cb, sensor_qos)

        # ── Publishers ────────────────────────────────────────────────
        self._pred_path_pub = self.create_publisher(Path,               '/mpc/predicted_path', 10)
        self._setpoint_pub  = self.create_publisher(PoseStamped,        '/mpc/next_setpoint',  10)
        self._goal_pub      = self.create_publisher(PoseStamped,        '/goal_pose',          10)
        self._diag_pub      = self.create_publisher(Float64MultiArray,  '/mpc/diagnostics',    10)

        # ── MPC timer ─────────────────────────────────────────────────
        rate = float(self.get_parameter('mpc_rate_hz').value)
        self.create_timer(1.0 / rate, self._mpc_cb)

        self.get_logger().info(
            f'MPC node ready | N={cfg.N} dt={cfg.dt}s '
            f'v_ref={cfg.v_ref} m/s rate={rate} Hz'
        )
        self.get_logger().info(
            'Topics:\n'
            '  sub  /skydio/pose\n'
            '  sub  /skydio/scan3d\n'
            '  sub  /a_star/path\n'
            '  pub  /mpc/predicted_path\n'
            '  pub  /mpc/next_setpoint\n'
            '  pub  /goal_pose          (drives cascaded PID)\n'
            '  pub  /mpc/diagnostics'
        )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        cur_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        cur_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Finite-difference velocity estimate
        if self._prev_pos is not None and self._prev_stamp_sec is not None:
            dt = cur_t - self._prev_stamp_sec
            if dt > 1e-3:
                self._vel = (cur_pos - self._prev_pos) / dt

        self._prev_pos        = cur_pos
        self._prev_stamp_sec  = cur_t

        o = msg.pose.orientation
        self._yaw = _quat_to_yaw(o.x, o.y, o.z, o.w)
        self._pose = msg

    def _path_cb(self, msg: Path):
        if not msg.poses:
            self._a_star_path = None
            return
        self._a_star_path = [
            (p.pose.position.x, p.pose.position.y, p.pose.position.z)
            for p in msg.poses
        ]

    def _lidar_cb(self, msg: PointCloud2):
        pts = []
        for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            pts.append([p[0], p[1], p[2]])

        if not pts:
            self._lidar_points = None
            return

        arr = np.array(pts, dtype=float)
        if self._pose is not None:
            px = self._pose.pose.position.x
            py = self._pose.pose.position.y
            dists = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
            arr = arr[dists <= self._max_lidar_range]

        self._lidar_points = arr if len(arr) > 0 else None

    # ── MPC timer ─────────────────────────────────────────────────────

    def _mpc_cb(self):
        if self._pose is None:
            self.get_logger().warn('[MPC] Waiting for /skydio/pose …', throttle_duration_sec=5.0)
            return

        if self._a_star_path is None:
            self.get_logger().warn('[MPC] Waiting for /a_star/path …', throttle_duration_sec=5.0)
            return

        pos = self._pose.pose.position
        drone_pos = np.array([pos.x, pos.y, pos.z])

        # Full 7-D state for MPC
        state = np.array([
            pos.x, pos.y, pos.z,
            self._vel[0], self._vel[1], self._vel[2],
            self._yaw,
        ])

        # Rebuild obstacle grid from latest LiDAR scan
        self._grid_map.update(self._lidar_points, drone_pos)
        self._tracker.update_grid(self._grid_map)

        # Solve MPC
        obs_pts_2d = self._lidar_points[:, :2] if self._lidar_points is not None else None
        try:
            result = self._tracker.solve(
                state,
                self._a_star_path,
                z_ref=self._planning_height,
                obstacle_points_2d=obs_pts_2d,
            )
        except Exception as exc:
            self._fail_count += 1
            self.get_logger().error(f'[MPC] Solve exception: {exc}')
            return

        # ── Counters & logging ────────────────────────────────────────
        self._solve_count    += 1
        self._total_solve_ms += result.solve_time_ms

        # ── Lookahead setpoint selection ──────────────────────────────
        # Walk the predicted trajectory and pick the first state that is
        # >= mpc_lookahead_dist from the current drone position.
        # If the entire horizon stays within that radius (drone is close to
        # the local/global goal), fall back to the last A* waypoint so the
        # PID homes in on the actual goal instead of oscillating around the
        # MPC horizon endpoint.
        drone_xy2 = drone_pos[:2]
        lookahead_idx = len(result.x_pred) - 1
        found_lookahead = False
        for k in range(1, result.x_pred.shape[0]):
            if float(np.linalg.norm(result.x_pred[k, :2] - drone_xy2)) >= self._lookahead_dist:
                lookahead_idx = k
                found_lookahead = True
                break

        if found_lookahead:
            nxt = result.x_pred[lookahead_idx, :3]
        else:
            # Near goal: steer toward the last A* waypoint (= local / global goal)
            last_wp = self._a_star_path[-1]
            nxt = np.array([
                float(last_wp[0]),
                float(last_wp[1]),
                float(last_wp[2]) if len(last_wp) > 2 else self._planning_height,
            ])

        self.get_logger().info(
            f'[MPC] #{self._solve_count:04d} '
            f'ok={result.success} '
            f'cost={result.cost:8.1f} '
            f'solve={result.solve_time_ms:5.1f} ms '
            f'avg={self._total_solve_ms / self._solve_count:5.1f} ms  '
            f'fails={self._fail_count}  '
            f'path_wpts={len(self._a_star_path)}  '
            f'lookahead_k={lookahead_idx}  '
            f'setpt=[{nxt[0]:.2f}, {nxt[1]:.2f}, {nxt[2]:.2f}]',
            throttle_duration_sec=0.5,
        )

        stamp = self.get_clock().now().to_msg()

        # ── Publish predicted path ─────────────────────────────────────
        pred_msg = Path()
        pred_msg.header.stamp    = stamp
        pred_msg.header.frame_id = 'world'
        for k in range(result.x_pred.shape[0]):
            ps = PoseStamped()
            ps.header.stamp    = stamp
            ps.header.frame_id = 'world'
            ps.pose.position.x = float(result.x_pred[k, 0])
            ps.pose.position.y = float(result.x_pred[k, 1])
            ps.pose.position.z = float(result.x_pred[k, 2])
            yaw_k = float(result.x_pred[k, 6])
            ps.pose.orientation.z = math.sin(yaw_k / 2.0)
            ps.pose.orientation.w = math.cos(yaw_k / 2.0)
            pred_msg.poses.append(ps)
        self._pred_path_pub.publish(pred_msg)

        # ── Setpoint (lookahead point on predicted trajectory) ─────────
        sp = PoseStamped()
        sp.header.stamp    = stamp
        sp.header.frame_id = 'world'
        sp.pose.position.x = float(nxt[0])
        sp.pose.position.y = float(nxt[1])
        sp.pose.position.z = float(nxt[2])
        sp.pose.orientation.w = 1.0
        self._setpoint_pub.publish(sp)

        # Publish the same setpoint on /goal_pose to drive the cascaded PID
        self._goal_pub.publish(sp)

        # ── Diagnostics ───────────────────────────────────────────────
        diag = Float64MultiArray()
        diag.data = [
            float(result.success),
            float(result.cost),
            float(result.solve_time_ms),
            float(self._total_solve_ms / self._solve_count),  # running avg solve time
            float(self._fail_count),
        ]
        self._diag_pub.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
