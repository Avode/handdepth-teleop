# Ubuntu RGB + TrueDepth limb fusion — 2026-10-08

Latest continuity fix: [STREAMING_AND_OVERLAY_FIX.md](STREAMING_AND_OVERLAY_FIX.md).
Unpaired RGB no longer stalls the sensor stream: current phone RGB-D is published
after a bounded PC wait, with original age and explicit fallback provenance.
The historical missing-RGB hold behavior below has been superseded. Exact source
matching is still mandatory for PC fusion and an RGB skeleton overlay.

Control now uses the additional forearm/elbow mapping described in
[ELBOW_MAPPING.md](ELBOW_MAPPING.md). The RGB-D inference described here is
unchanged; `limb` is the new default control mode.

The running teleoperation receiver now adds independent Ubuntu RGB inference.
The phone still captures RGB/TrueDepth and runs Apple Vision body/hand detection.
Ubuntu runs MediaPipe Pose Landmarker Full on each matched RGB preview, compares
its shoulders/elbows/wrists with the phone detections, verifies limb support in
the registered depth image, and reconstructs measured surface XYZ. This shares
processing; it does not disable phone Vision or increase the phone capture rate.
There is no iOS rebuild, network/protocol change or cloud processing.

## Why visible arms appeared lost

The phone's projected-length continuity filter can mark a current elbow inferred
even while its separately transmitted raw Vision detection remains useful.
Ubuntu previously rejected that elbow unconditionally. The new verifier can
reacquire a fresh raw Vision or PC RGB detection when current depth supports the
limb. It never samples the phone's held/inferred pixel or interprets its projected
pixel lengths as metres. It never uses MediaPipe model-predicted XYZ/depth.

The PC person must agree with both current phone shoulders before replacement.
Confident PC detections can replace missing/rejected raw joints; current associated
hand wrists remain a separate fallback. Verification requires plausible metric
segments, depth support along both links, a foreground boundary, no current hand
covering the elbow, and two distinct confirming frames when recovering a rejected
joint. Local depth edges use only the connected surface containing the detected
pixel. Depth holes, unsupported foreground/background and abrupt metric-length
changes are rejected. Upper-arm and forearm validity are reported independently.

## Provenance and freshness

- RGB/depth pairing requires identical session, calibration, frame, capture time
  and color dimensions. Eight frames per channel, one active job and one latest
  pending job bound memory and latency. Old worker results cannot survive reset,
  disconnect, a newer publication or source identity change.
- `body_tracking` remains the original phone continuity channel. `pc_pose` retains
  the independent PC detections. `body.limb_tracking` records comparison distances,
  depth support, rejection reasons and verified joint provenance.
- Current verified observations use `current_pc_rgb_depth_verified` or
  `current_raw_vision_depth_verified`; they can explicitly supply measured robot
  input. Original capture/arrival ages are retained after inference. Held/inferred
  display geometry never passes controller freshness checks.
- RGB: magenta = phone observation, cyan = Ubuntu RGB-D verified replacement,
  amber = held/inferred with age. MuJoCo: cyan = measured, amber = held/inferred,
  green = robot target. Existing metric length retention and bend continuity stay.
- The robot torso/base remain fixed. Shoulder mode still drives only shoulder
  joints 1/2; better forearm estimation does not enable elbow/wrist joint control.
- Worker failure falls back to the phone-based receiver and is shown in the
  MuJoCo status and `pc_pose_fusion.error`. Missing matching RGB causes a hold
  once the last published pose is stale; unmatched RGB is never fused.

## Installation and operation

Software is on SSD in `/path/to/handdepth-teleop`. A separate `pose-venv`
contains pinned MediaPipe dependencies, avoiding changes to MuJoCo's Python/Qt
packages. `tools/ubuntu/setup_pc_pose.sh` installs them and checks the mounted HDD
before downloading the Full model with a pinned SHA-256. The model and its cache
are under `/mnt/robotics-data/robotics/agibot-g2/models/handdepth/`.

Official model documentation:
https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker

Launch as usual:

```bash
./run-handdepth.sh
```

PC inference is enabled by default for live `teleop`. `--no-pc-pose` disables it
when invoking the inner `tools/ubuntu/run_teleop.sh` launcher directly;
`--pc-pose-model PATH` selects another installed model. The endpoint remains
`YOUR_UBUNTU_IP:8765`, using `/sensor` and `/preview`.

All growing files require a mounted data volume. Sessions retain raw packets and
phone-derived poses; a separate `pc-poses.jsonl` contains the published fused
poses. `teleop/latest.json` exposes worker state, inference timing and limb
verification. The raw packet recording is the source of truth for phone outputs.

## Validation

202 Python tests pass, including actual G2 physics, both-arm recovery, inferred
pixel exclusion, background/hand occluder rejection, exact source matching,
bounded queues, session/calibration reset, preserved measurement age, late-result
rejection after disconnect and phone fallback on worker failure.

Two different 450-frame windows from the current recording were replayed through
the real CPU worker, depth reconstruction and G2 shoulder controller. Counts are
accepted measured links, not ground-truth accuracy measurements:

| Window | Link | Before / 450 | Fused / 450 |
| --- | --- | ---: | ---: |
| 35000 | Left upper arm | 75 | 380 |
| 35000 | Left forearm | 75 | 380 |
| 35000 | Right upper arm | 16 | 31 |
| 35000 | Right forearm | 9 | 24 |
| 50500 | Left upper arm | 223 | 228 |
| 50500 | Left forearm | 114 | 130 |
| 50500 | Right upper arm | 49 | 326 |
| 50500 | Right forearm | 40 | 323 |

In the first window a hand covered the right elbow in 318 frames, correctly
preventing promotion. In the second window, poor far-arm forearm depth support
remained rejected. Before/after images show recovered near-arm lines following
the visible silhouette. Median model inference was 19.0 and 20.7 ms, respectively.
Both replays had zero IK failures or physics warnings, zero base displacement,
less than 0.0001 rad torso constraint compliance, and passed disconnect hold.

Validation outputs are on the HDD under
`datasets/handdepth/teleop/limb-tracking-validation/`: `fusion-recording-1/` and
`fusion-recording-2/` contain reports, fused poses and comparison PNGs.
`live-report.json` and `live-status-samples.jsonl` record the live integration.
The 45-second live check retained both arms through observed/held/inferred states
in both views, with one calibration, no worker/IK/physics errors, zero base
displacement and at most 0.000032 rad torso constraint compliance. Source rate
varied up to 7.5 Hz; the control loop ran about 41 Hz. Window captures are saved
as `live-dashboard.png` and `live-mujoco.png` in the same validation directory.
Reproduce a recorded check with `tools/ubuntu/validate_limb_fusion.py SESSION
--start-frame N --frames 450 --output HDD_DIRECTORY`. Ordinary CLI replay still
uses phone-derived reconstruction; this explicit validator runs the PC worker.

## Remaining limits

This measures visible skin surfaces, not internal anatomical joint centers.
Agreement and depth checks reduce obvious errors but do not establish exact
anatomical accuracy. The phone still supplies torso/head/hands and person
association; both phone shoulders must be present to accept PC replacements.
Current recovery conservatively needs depth support for the whole arm. Full
occlusion, overlapping arms, a hand over the elbow, missing wrist depth, or an
arm pressed against the torso can still produce amber retained geometry.
Disabling phone Vision entirely would require an iOS capture-only mode plus
Ubuntu torso/head/hand inference; that is not part of this shared-processing path.
