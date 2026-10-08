# Setup and reproducibility

This public snapshot provides Ubuntu sources. The Swift app remains separate;
the [protocol](../PROTOCOL.md) documents the sender interface. Existing app users
can stream to this receiver. Others can run the tests and synthetic fixtures.

## Install the receiver

On Ubuntu, install Python 3.10 with venv support and the normal OpenGL/OpenCV
desktop libraries (`python3-venv`, `libgl1`, `libglib2.0-0`). In this checkout:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r receiver/requirements-dev.txt -r receiver/requirements-assets.txt -e receiver
venv/bin/handdepth serve --host 0.0.0.0 --port 8765
```

The live server and teleop cannot both bind port 8765. Stop the standalone server
before launching teleop. The main environment uses OpenCV 4; the optional pose
worker uses a separate environment to isolate its dependency versions.

## Prepare external G2 assets

The meshes and upstream model source are not vendored. These commands reproduce
the source revisions used during development. They download code/assets, so review
the upstream repositories and licenses before building.

```bash
export HANDDEPTH_DATA_ROOT=/mnt/robotics-data/robotics/agibot-g2
# Verify the intended disk is really mounted. Do not continue if this fails.
mountpoint -q /mnt/robotics-data
test -w /mnt/robotics-data
mkdir -p "$HANDDEPTH_DATA_ROOT/assets"
git clone https://github.com/AgibotTech/genie_sim_robot_model.git "$HANDDEPTH_DATA_ROOT/assets/genie_sim_robot_model"
git -C "$HANDDEPTH_DATA_ROOT/assets/genie_sim_robot_model" checkout 6d5b2159806ae0e64b432b638e9ee288d8bddc8f
export HANDDEPTH_SRDF="$HANDDEPTH_DATA_ROOT/assets/genie.srdf.xacro"
curl -fL https://raw.githubusercontent.com/AgibotTech/genie_sim/6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46/source/geniesim_ros/src/ros_ws/src/genie_sim_moveit/config/genie.srdf.xacro -o "$HANDDEPTH_SRDF"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python tools/ubuntu/build_g2_assets.py
venv/bin/python tools/ubuntu/build_teleop_scene.py \
  --source "$HANDDEPTH_DATA_ROOT/assets/g2-mujoco" \
  --output "$HANDDEPTH_DATA_ROOT/projects/handdepth"
```

The converter preserves gripper mimic constraints and adjacent-link exclusions,
adds local position servos and gravity compensation, and uses MuJoCo's inertia
balancing for inconsistent upstream inertias. The teleop scene restores the head
and torso joints, then locks the torso. These are documented simulation choices;
the generated model is not a validated dynamics model for physical deployment.
The screenshots show a local torso label; the public converter omits that purely
visual customization.

## Enable Ubuntu RGB pose inference

For the default data mount:

```bash
bash tools/ubuntu/setup_pc_pose.sh
./run-handdepth.sh
```

The worker downloads MediaPipe Pose Landmarker Full from Google's model storage
and checks its expected SHA-256. It runs on CPU in `pose-venv`. For another mounted
data root, install `receiver/requirements-pc-pose.txt` into `pose-venv`, place the
verified model under `$HANDDEPTH_DATA_ROOT/models/handdepth/`, and use that root
with the launcher. The convenience setup scripts currently assume the default
`/mnt/robotics-data` mount; the Python CLI exposes explicit scene/model/output paths.

To run without the independent PC model, use `./run-handdepth.sh --no-pc-pose`.
The separate phone app must still provide compatible calibrated observations.
On the phone, enter Ubuntu's LAN address (`hostname -I`) and port `8765`.
Keep the phone fixed, with shoulders, chest, elbows and hands visible. Show bent
elbows and relaxed wrists briefly; no T-pose is needed.

## Test and record

```bash
export HANDDEPTH_G2_SCENE="$HANDDEPTH_DATA_ROOT/projects/handdepth/scene.xml"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python -m pytest -q tests
./run-handdepth.sh --record
```

Production recording validates a writable mounted ancestor other than `/` and
fails rather than falling back to SSD. `--fixture-storage` is only for synthetic
test data. Do not commit real recordings, calibration dumps, or private logs.

The Python fixtures under `fixtures/` are synthetic protocol agreement data;
their `*-swift.bin` counterparts came from the sender's wire tests and are also
synthetic. Their Swift test source is not included in this release.

The `tools/ubuntu/validate_*.py` programs consume private recordings for physics
comparisons. The original recordings are not public, so the historical human
replay results cannot be independently reproduced from this repository alone.
The full deterministic Python/physics regression suite can be rerun with the
separately built G2 scene. CI runs without those assets and explicitly skips the
asset-dependent tests.
