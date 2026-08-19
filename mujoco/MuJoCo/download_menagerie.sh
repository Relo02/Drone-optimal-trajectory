#!/bin/bash
# Script to download the MuJoCo Menagerie models used by the Skydio X2 simulation
#
# This replaces the broken `mujoco/MuJoCo` submodule: the commit that used to be
# recorded there (48397aa) does not exist in any public MuJoCo repository, so it
# could never be cloned by anyone except the machine that created it.
#
# NOTE: this only fetches the *stock* menagerie. The scene actually loaded by
# skydio_sim_node is `skydio_x2/skydio_world.xml`, a custom file carrying the 576
# rangefinder sites (lidar3d_0 ... lidar3d_575) produced by `generate_x2_lidar.py`.
# Neither file is in this repository — they must be committed separately.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/mujoco_menagerie-main"

echo "Downloading MuJoCo Menagerie models..."

# Clone the repository to a temp location
TEMP_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git "$TEMP_DIR/menagerie"

# Move it under the name expected by skydio_sim_node / skydio_params.yaml
rm -rf "$DEST_DIR"
mv "$TEMP_DIR/menagerie" "$DEST_DIR"
rm -rf "$DEST_DIR/.git"

# Cleanup temp directory
rm -rf "$TEMP_DIR"

echo "Menagerie downloaded successfully to: $DEST_DIR"
echo ""

if [ ! -f "$DEST_DIR/skydio_x2/skydio_world.xml" ]; then
    echo "WARNING: $DEST_DIR/skydio_x2/skydio_world.xml is missing."
    echo "         The stock menagerie does not ship it — it is the custom scene"
    echo "         with the 3-D lidar sites. Add it before launching the simulation:"
    echo "           ros2 launch new_mujoco skydio_sim.launch.py"
    echo "         will otherwise abort with 'Model not found'."
fi
