#!/bin/bash
# Script to download Crazyflie 2 mesh files from varadVaidya/cf2_mujoco

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF2_DIR="$SCRIPT_DIR/cf2"

echo "Downloading Crazyflie 2 mesh files..."

# Create cf2 directory
mkdir -p "$CF2_DIR"

# Clone the repository to a temp location
TEMP_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/varadVaidya/cf2_mujoco.git "$TEMP_DIR/cf2_mujoco"

# Copy the cf2 directory contents
cp -r "$TEMP_DIR/cf2_mujoco/cf2/"* "$CF2_DIR/"

# Cleanup temp directory
rm -rf "$TEMP_DIR"

echo "Mesh files downloaded successfully to: $CF2_DIR"
echo ""
echo "Files downloaded:"
ls -la "$CF2_DIR"
