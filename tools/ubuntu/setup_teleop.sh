#!/usr/bin/env bash
set -euo pipefail
HANDDEPTH_APP=$(cd "$(dirname "$0")/../.." && pwd)
HANDDEPTH_DATA=${HANDDEPTH_DATA_ROOT:-/mnt/robotics-data/robotics/agibot-g2}
# Refuse to write growing model/data files onto the root SSD when HDD is absent.
mountpoint -q /mnt/robotics-data || { echo 'Mount /mnt/robotics-data before setup.' >&2; exit 2; }
test -w /mnt/robotics-data || { echo 'Data HDD is not writable.' >&2; exit 2; }
test -f "$HANDDEPTH_DATA/assets/g2-mujoco/g2-original.urdf" || { echo 'Installed G2 assets missing; see TELEOP.md.' >&2; exit 2; }
python3 -m venv "$HANDDEPTH_APP/venv"
mkdir -p "$HANDDEPTH_DATA/cache/pip"
PIP_CACHE_DIR="$HANDDEPTH_DATA/cache/pip" "$HANDDEPTH_APP/venv/bin/pip" install -r "$HANDDEPTH_APP/receiver/requirements-teleop.txt" -e "$HANDDEPTH_APP/receiver"
"$HANDDEPTH_APP/venv/bin/python" "$HANDDEPTH_APP/tools/ubuntu/build_teleop_scene.py" \
 --source "$HANDDEPTH_DATA/assets/g2-mujoco" --output "$HANDDEPTH_DATA/projects/handdepth"
printf 'Ready. Start: %s/tools/ubuntu/run_teleop.sh\n' "$HANDDEPTH_APP"
