#!/usr/bin/env bash
# Independent Ubuntu RGB inference, isolated from MuJoCo's Python/Qt packages.
set -euo pipefail
HANDDEPTH_APP=$(cd "$(dirname "$0")/../.." && pwd)
mountpoint -q /mnt/robotics-data || { echo 'Mount the robotics HDD first.' >&2; exit 2; }
python3 -m venv "$HANDDEPTH_APP/pose-venv"
"$HANDDEPTH_APP/pose-venv/bin/python" -E -s -m pip install --no-cache-dir -r "$HANDDEPTH_APP/receiver/requirements-pc-pose.txt"
"$HANDDEPTH_APP/venv/bin/python" -E -s - <<'PY'
import hashlib,json,urllib.request
from handdepth.recording import check_recording_root
root=check_recording_root('/mnt/robotics-data/robotics/agibot-g2/models/handdepth')
name='pose_landmarker_full.task'
url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/'+name
expected='4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad'
path=root/name
if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
    tmp=path.with_suffix('.part');urllib.request.urlretrieve(url,tmp)
    digest=hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest!=expected:
        tmp.unlink();raise RuntimeError('Upstream model changed: review before installing')
    tmp.replace(path)
path.with_suffix('.json').write_text(json.dumps({'url':url,'sha256':expected,'bytes':path.stat().st_size},indent=2)+'\n')
print(path)
PY
