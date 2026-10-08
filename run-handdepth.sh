#!/usr/bin/env bash
# Public, foreground launcher. Source/venvs stay here; growing data stays on a mounted disk.
set -euo pipefail
HANDDEPTH_APP=$(cd "$(dirname "$0")" && pwd)
HANDDEPTH_DATA_ROOT=${HANDDEPTH_DATA_ROOT:-/mnt/robotics-data/robotics/agibot-g2}
HANDDEPTH_SCENE=${HANDDEPTH_SCENE:-$HANDDEPTH_DATA_ROOT/projects/handdepth/scene.xml}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec "$HANDDEPTH_APP/venv/bin/python" -E -s -m handdepth.cli teleop \
  --host "${HANDDEPTH_HOST:-0.0.0.0}" --port "${HANDDEPTH_PORT:-8765}" \
  --scene "$HANDDEPTH_SCENE" --recording-root "$HANDDEPTH_DATA_ROOT/datasets/handdepth" \
  --pc-pose-model "$HANDDEPTH_DATA_ROOT/models/handdepth/pose_landmarker_full.task" \
  --auto-calibrate --arm-mapping limb --wrist-mapping relative --grip-mapping fist "$@"
