#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-mpc_bag}"
if [ $# -gt 0 ]; then
  shift
fi

ros2 bag record -o "$out_dir" \
  /mpc/reference_trajectory \
  /mpc/optimal_trajectory \
  /mpc/cost \
  /fmu/out/vehicle_odometry \
  "$@"
