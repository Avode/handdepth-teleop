#!/usr/bin/env bash
set -euo pipefail
HANDDEPTH_APP=$(cd "$(dirname "$0")/../.." && pwd)
HANDDEPTH_DATA=${HANDDEPTH_DATA_ROOT:-/mnt/robotics-data/robotics/agibot-g2}
mountpoint -q /mnt/robotics-data || { echo 'Mount the recording HDD first.' >&2; exit 2; }
exec "$HANDDEPTH_APP/venv/bin/python" -E -s -m handdepth.cli teleop \
 --scene "$HANDDEPTH_DATA/projects/handdepth/scene.xml" \
 --recording-root "$HANDDEPTH_DATA/datasets/handdepth" --auto-calibrate "$@"
