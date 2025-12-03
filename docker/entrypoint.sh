#!/bin/bash
set -e
# Source ROS 2 if available
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
# Any project-specific setup can go here (PX4 envs, paths, etc).
# Finally exec the passed command
exec "$@"
