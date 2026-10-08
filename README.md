# HandDepth Teleop

**An iPhone TrueDepth → Ubuntu RGB-D → AgiBot G2 teleoperation prototype in MuJoCo.**

I started with a goal: teach a simulated robot to pick up and place everyday
objects. This repository is the perception and teleoperation foundation I built
toward that goal: turning human upper-body motion into constrained robot motion,
with explicit handling of uncertainty, occlusion and stale input.

![G2 simulation with human pose overlay and tracking diagnostics](docs/assets/g2-control-diagnostics.png)

**Status:** experimental, simulation only. Robot torso/base stay fixed. Arms and
head move relative to the observed human torso. No learned manipulation policy,
autonomous pick-and-place result, or physical robot deployment is claimed.

**Release scope:** this repository contains the Ubuntu receiver, geometry,
visualization, retargeting/controller code, model-conversion tools and Python
tests. The custom Swift iPhone app is a separate component and is **not included
in this release**. Without that app, start with synthetic fixtures and tests;
live phone capture is not a complete fresh-clone workflow yet.

[Watch the supplied demo](docs/assets/handdepth-demo.mp4) ·
[Read the journey — blog draft](docs/blog/iphone-to-humanoid.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Setup](docs/SETUP.md) ·
[Evidence and limitations](docs/RESULTS.md) · [Wire protocol](PROTOCOL.md)

## What it does

- Receives synchronized depth, camera calibration, phone body/hand observations
  and RGB preview packets over local WebSockets.
- Adds independent Ubuntu RGB pose estimates and registered depth verification
  when exact source frames match.
- Maps shoulder and forearm **directions relative to the torso** into Mink IK;
  tracks palm orientation relative to the forearm; fist closes, open hand releases.
- Keeps coherent measured/inferred arm geometry visible during occlusion, while
  holding robot commands when required measurements are unavailable or stale.
- Separates physics, perception, display and recording. Measured points are cyan,
  held/inferred display points amber, and robot direction targets green.
- Uses bounded processing queues and a current-phone fallback after a 100 ms
  fusion wait, preserving the original measurement age.

![RGB, TrueDepth and human skeleton panels](docs/assets/rgb-depth-arms.png)

Depth measures visible surfaces, not internal anatomical joint centers. A useful
overlay can remain visible while the controller correctly holds.

## Quick start: receiver and tests

Ubuntu 22.04 / Python 3.10 is the development baseline. Keep this checkout and
virtual environments on SSD; use a mounted data disk for recordings and assets.

```bash
git clone https://github.com/Avode/handdepth-teleop.git
cd handdepth-teleop
python3 -m venv venv
venv/bin/python -m pip install -r receiver/requirements-dev.txt -r receiver/requirements-teleop.txt -e receiver
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python -m pytest -q tests
venv/bin/handdepth --help
```

Tests needing the G2 scene skip when its separately installed assets are absent.
The full local validation ran **268 tests with the G2 model installed**; see the
[results](docs/RESULTS.md) for the exact scope. Follow [setup](docs/SETUP.md) to
prepare the model, enable the independent RGB worker and launch the simulation:

```bash
./run-handdepth.sh
# Optional, only with the intended data disk mounted:
./run-handdepth.sh --record
```

`C` arms/resumes tracking and acquires a comfortable head/wrist neutral;
`Space` holds; `R` clears display history; `H` toggles the skeleton;
`G` toggles robot transparency; `Q` exits. Calibration does not require a T-pose.

The transport has no authentication/TLS: use a trusted local network and do not
expose the receiver to the Internet. Stale-input holds are software behavior in
simulation, not a physical robot safety certification.

## Reading the code

| Area | Entry point |
| --- | --- |
| Wire validation, bounded pairing, fallback and source age | `receiver/handdepth/server.py`, `state.py`, `pc_pose.py` |
| Calibration and measured 3D reconstruction | `receiver/handdepth/geometry.py`, `body.py`, `limb_refinement.py` |
| Persistent display and two-segment arm geometry | `receiver/handdepth/display_tracking.py`, `body_tracking.py` |
| Torso-relative mapping and finger-curl gesture | `receiver/handdepth/retarget.py`, `hand_control.py` |
| Direction-task Jacobians and physics controller | `receiver/handdepth/shoulder_task.py`, `teleop.py` |
| Human overlay and RGB/depth dashboard | `receiver/handdepth/pose_overlay.py`, `visualization.py` |
| Regression tests and recorded-input validators | `tests/`, `tools/ubuntu/validate_*.py` |

## Next steps

1. Publish the Swift capture app and verify sustained exact RGB/depth pairing.
2. Measure pose/gesture accuracy and availability against a defined reference.
3. Benchmark movement error, latency and recovery across repeatable occlusions.
4. Add object interaction and demonstration collection before training a policy.

Built by **Umar / [Avode](https://github.com/Avode)** with Codex assistance.
Original code is [MIT licensed](LICENSE). [Third-party credits and release
boundaries](THIRD_PARTY_NOTICES.md) identify the robotics frameworks and assets.
