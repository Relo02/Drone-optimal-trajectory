# Run this script inside the ros2-humble docker container
cd /workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run ros_gz_bridge parameter_bridge /camera@sensor_msgs/msg/Image@ignition.msgs.Image /camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo /depth_camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked /depth_camera@sensor_msgs/msg/Image@ignition.msgs.Image