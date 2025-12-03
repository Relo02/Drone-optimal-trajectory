#!/usr/bin/env bash
set -e

cd ../..
cd PX4-Autopilot

echo "Starting PX4 SITL with gz_x500_depth..."
echo "Command: make px4_sitl gz_x500_depth"

# Navigate to PX4 directory if not already there
if [ -d "PX4-Autopilot" ]; then
    cd PX4-Autopilot
fi

# Run SITL
make px4_sitl gz_x500_depth