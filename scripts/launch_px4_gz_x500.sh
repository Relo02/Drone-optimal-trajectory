#!/bin/bash
set -e

# Source ROS
source /opt/ros/humble/setup.bash

# Set Gazebo simulation paths
export GZ_SIM_RESOURCE_PATH=/opt/px4_source/Tools/simulation/gz/models:/opt/px4_source/Tools/simulation/gz/worlds
export PX4_GZ_MODELS=/opt/px4_source/Tools/simulation/gz/models

# Navigate to PX4 source directory
cd /opt/px4_source

echo "Checking git status..."
git status || echo "Git status check failed"

echo "Building PX4 for the first time (this may take several minutes)..."
if ! make px4_sitl gz_x500_depth; then
    echo "Build failed. Checking git repository state..."
    git fetch --tags || echo "Fetch tags failed"
    git fetch --unshallow || echo "Already unshallow or fetch failed"
    echo "Cleaning and rebuilding..."
    make clean
    make submodulesclean
    git submodule update --init --recursive
    make px4_sitl gz_x500_depth
fi