# Run this script inside the ros2-humble docker container to expose Gazebo topics to ROS 2.
cd /workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run ros_gz_bridge parameter_bridge \
  /world/default/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
  --ros-args -r /world/default/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan:=/scan