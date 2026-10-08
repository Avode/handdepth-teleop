# Shoulder and forearm direction tracking — 2026-10-08

Latest extension: [WRIST_GRIP_MAPPING.md](WRIST_GRIP_MAPPING.md) enables measured
forearm-relative wrist rotations and fist gripper control by default. The wrist
hold/pinch statements below describe the earlier elbow-only stage; its limb
direction mapping, static anchor and measurement gates remain intact.

The default launcher/CLI now selects `--arm-mapping limb`. It preserves the
existing measured shoulder direction target and adds measured forearm direction,
elbow bend and bend-plane rotation. `--arm-mapping shoulder` retains the previous
shoulder-only behavior for comparison. Python class defaults remain Cartesian
for compatibility with existing callers.

## Mapping

Use current measured shoulder, elbow and wrist XYZ in the human torso frame.
Normalize shoulder→elbow and elbow→wrist, then rotate both through the existing
anatomical right/down/forward → robot forward/left/up basis. Targets are absolute
directions, independent of human translation, limb length or calibration offsets.
Elbow flexion is the angle between those vectors: zero means straight. The task
uses actual G2 elbow/wrist sites and the model's small link offsets; it does not
copy a human scalar angle directly into the robot joint's encoder value.

G2 joints 1/2 maintain upper-arm direction, joint 3 aligns the bend plane, and
joint 4 bends the elbow. Elbow bend alone cannot reproduce a forearm's full 3D
direction. Joints 5–7 remain held; head and pinch retain their existing behavior.
The upper-arm task has five times the residual weight of the forearm task
(25 times squared error cost), prioritizing shoulders when a combined target is
infeasible. This is a weighted preference, not an exact shoulder constraint.

Forearm targets have the same 80 ms directional filter and 1.4 rad/s target rate
cap as shoulders. Joint velocity limits remain 1 rad/s. Bounded measured-state
servo damping compensation now applies to joints 1–4, with at most 0.12 s extra
lead and 0.12 rad lead displacement. Actual physical velocity can differ slightly
from the commanded IK velocity. No robot XML, actuator gains or physics state
assignment was changed.

When the two human link directions are almost parallel or antiparallel, bend
plane rotation becomes ambiguous. Axial rotation holds below sin(bend)=0.12
(about 6.9° from straight/folded) and re-enables above 0.20 (about 11.5°).
This hysteresis prevents straight-arm noise from spinning the upper arm.

## Measurement and hold behavior

Each elbow requires three distinct current measured frames, independently of the
other arm. Calibration remains comfortable; no T-pose or elbow-zero pose is
required. C arms/resumes directions and acquires head neutral as before.

The elbow/wrist controller accepts the existing explicitly verified Ubuntu
RGB-D observations. It checks body landmark validity, the phone's held/inferred
cues and exact verification provenance. A current associated hand wrist may
supply a missing body wrist only with a matching hand ID/side and measured point,
or the separate current RGB-D verification. Held/inferred display points and
pixel-space lengths never enter control.

Missing wrist: hold elbow and axial rotation while continuing a measured
shoulder. Missing elbow/shoulder: hold that arm. Missing torso orientation or
stale/disconnected stream: hold commands. Reacquisition resets directional
filters to the actual robot link directions and preserves session calibration.
New session/calibration/coordinate geometry requires fresh acquisition.

## Collisions and residual error

Elbow movement exposes collisions that shoulder-only motion avoided. Collision
avoidance now covers distal links 4–7, complete gripper collision geometry and
all torso links, including the lower torso. This prevents hands pressing into
the body and stalling otherwise valid shoulder motion. Physical collisions,
configuration and velocity limits remain enabled.

A forearm target can still be unreachable while preserving the shoulder direction,
especially when a large robot gripper would occupy the torso or the other hand.
The displayed error remains nonzero in that case. Do not remove contact handling
or let torso motion disguise the discrepancy. Wrist orientation imitation is
not enabled by this change and could later improve clearance in some poses.

## Viewer and operation

Run `./run-handdepth.sh`. RGB/depth reception and PC
fusion use the same iPhone endpoint `YOUR_UBUNTU_IP:8765`. The green target layer
now includes elbow→wrist segments at robot link lengths. Viewer rows show each
forearm direction error and elbow-flexion error in degrees, or the hold reason.
Cyan human geometry is measured; amber is retained/inferred display geometry.
Torso/base remain fixed. Space holds, C resumes, R resets display history.

The status JSON now contains `elbows.left/right`, measured/desired forearm
vectors, direction and flexion errors, axial-rotation activity, wrist source,
and explicit `inferred_control: false`. Existing shoulder diagnostics remain.

## Validation

The full suite passes **228 tests**, including 26 new elbow tests. It covers
both-arm angular signs, translation/length independence,
absolute calibration, missing/invalid/held/inferred wrists, associated-hand
provenance, session/geometry reset, bend-plane hysteresis, finite-difference G2
Jacobian verification, actual physics convergence, wrist/torso holds,
disconnect/reacquisition, and body-blocked forearm targets.

For a collision-free reachable step involving both shoulders, upper-arm axial
rotation and elbows, both forearms reach less than 0.01° error after two simulated
seconds. This is a synthetic model-derived target, not an estimate of anatomical
accuracy. A body-blocked regression preserves shoulders within 0.5° while reporting
remaining forearm error and avoiding physical contact.

Three identical-input comparisons replayed 210 seconds of recorded fused poses
through the actual old/new G2 physics controllers. Mean angular errors in degrees:

| Recording / side | Forearm before → after | Bend before → after | Shoulder before → after |
| --- | ---: | ---: | ---: |
| Window 1 / left | 27.94 → 3.14 | 8.65 → 2.54 | 1.09 → 1.02 |
| Window 1 / right | 35.38 → 20.81 | 16.08 → 6.18 | 3.90 → 4.06 |
| Window 2 / left | 52.95 → 34.27 | 26.19 → 16.75 | 10.03 → 9.77 |
| Window 2 / right | 28.98 → 4.79 | 17.77 → 2.37 | 4.29 → 4.41 |
| Recent stream / left | 48.97 → 28.00 | 19.41 → 6.95 | 4.26 → 4.61 |
| Recent stream / right | 36.25 → 11.79 | 19.07 → 4.33 | 3.61 → 3.67 |

These are repeated controller samples against estimated human directions, not
independent observations or anatomical ground truth. Occlusion/reacquisition,
rapid changes and body-blocked poses account for the larger residuals. Only 117
right-arm controller samples were available in window 1. Detailed counts and
95th percentiles are preserved in the reports; do not generalize the best window
to every pose.

All six final controller runs passed disconnect hold, exact fixed torso commands,
zero base displacement, torso compliance below 0.0001 rad, joint-limit checks,
zero IK failures and zero MuJoCo warnings. Peak physical joint speed in the three
new-controller replays was 0.95, 1.08 and 1.07 rad/s respectively.

Reports are on the mounted HDD under
`/mnt/robotics-data/robotics/agibot-g2/datasets/handdepth/teleop/elbow-validation/`:
`recording-1-final.json`, `recording-2-final.json`, `live-recording-final.json`.
Earlier diagnostic reports in the same directory describe pre-fix iterations.
The restarted live receiver/viewer use `limb` mode. A 50-second check sampled 198
distinct source frames with one calibration, zero IK/physics failures, zero base
movement and at most 0.000024 rad torso constraint compliance. On valid tracking
samples, median forearm errors were 1.57° left / 1.16° right and median bend errors
0.61° / 0.47°. Forearm 95th percentiles were 7.10° / 2.69°. The left forearm was
active in 79/200 sampled statuses and the right in 190/200; missing measurements
held correctly, including seven samples with left shoulder tracking while its
elbow held. Control ran at 42.4 Hz, source at approximately 7–7.5 Hz, PC inference
around 20 ms. These measured-pose errors exclude held samples. Live reports and
both window captures are `live-report.json`, `live-status-samples.jsonl`,
`live-mujoco.png` and `live-dashboard.png` in that validation directory.

Reproduce with:

```bash
cd /path/to/handdepth-teleop
venv/bin/python -E -s -m pytest -q tests
venv/bin/python -E -s tools/ubuntu/validate_elbows.py /HDD/session/pc-poses.jsonl \
  --start 120 --seconds 90 --output /HDD/validation.json
```

Software stays on SSD; new recordings, validation files and logs require the
mounted robotics HDD and do not fall back to SSD. The pre-edit backup is
`teleop/elbow-backup-20261008T012458/` under the HandDepth dataset directory.
