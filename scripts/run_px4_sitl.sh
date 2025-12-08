#!/usr/bin/env bash
set -e

cd /workspace/drone_ws/src/PX4-Autopilot

echo "Starting PX4 SITL with gz_x500_depth..."
echo "Command: make px4_sitl gz_x500_depth"

# Initialize and update submodules
echo "Initializing submodules..."
git submodule update --init --recursive

# Clean previous build
echo "Cleaning previous build..."
rm -rf build/

# Source ROS and Gazebo environment
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Run SITL with Gazebo x500
echo "Building PX4 SITL with Gazebo x500..."
make px4_sitl gz_x500_depth -j$(nproc)