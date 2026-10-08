# Torso-relative shoulder direction — 2026-10-08

This documents the preserved shoulder-only stage. The default now extends it
with elbow bend and forearm direction using `--arm-mapping limb`; see
[ELBOW_MAPPING.md](ELBOW_MAPPING.md). Select `--arm-mapping shoulder` for the
behavior and historical results below.

The live launcher now uses `--arm-mapping shoulder`. The former controller used
Mink wrist position cost 1.0 and palm orientation cost 0.18, with elbow position
cost only 0.08. Its neutral offsets and independent segment scaling also differed
from the raw metric human overlay. Thus a hand could reach its Cartesian target
without the upper arm matching the displayed shoulder angle.

## Current behavior

- Use the current measured shoulder and elbow, expressed in the measured human
  torso frame. Normalize `elbow - shoulder`, then apply the established anatomical
  right/down/forward → robot forward/left/up rotation. Translation and human limb
  length do not affect the requested angle. The wrist is not needed.
- A Mink unit-direction task minimizes the difference between the robot and
  human upper-arm unit vectors. Its analytic Jacobian differentiates the actual
  G2 shoulder/elbow sites and the vector normalization. This is a rank-two angular
  task, not a wrist-position task or a copy of human/robot Euler joint values.
- G2 joints 1/2 move. Joint 3 (axial twist), elbow 4, and wrist 5–7 commands hold.
  Shoulder/elbow points cannot measure rotation about the bicep's own axis.
  Model shoulder-to-elbow geometry includes a 10.5 mm lateral offset; the task
  uses that actual geometry rather than assuming an ideal ball joint.
- The existing head orientation and pinch controls remain. Palm orientation does
  not drive the wrist in shoulder mode. The torso/base remain the fixed anchor.
- Three distinct measured frames activate each shoulder. Comfortable C calibration
  arms/resumes tracking and reacquires head neutral; it does not zero the human's
  absolute shoulder direction. Temporary occlusion keeps acquisition and neutral.
- Reject missing/invalid shoulder or elbow depth landmarks and phone held/inferred
  point cues. No image-space arm lengths or retained display geometry enter control.
  Missing torso orientation holds commands. A stale/disconnected stream holds all
  commands. Session, calibration, or display-geometry changes require reacquisition.
- Direction targets use geodesic filtering (80 ms time constant), capped at
  1.4 rad/s. G2 arm IK velocities remain limited to 1 rad/s, with configuration
  and existing collision-avoidance limits. Commands drive the position servos;
  physics still owns actual joint state.
- Shoulder servos add bounded damping compensation. The old measured-state
  command `q + dt*v` produced a very small spring torque against actuator damping,
  causing visible creeping even with correct angle targets. For these four
  shoulder actuators only, the additional lead is
  `clip((actuator_damping + joint_damping)/kp - dt, 0, 0.12) * v`, bounded
  to 0.12 rad and clipped to actuator ranges. It is recomputed from measured
  joint state each tick, never accumulated. Missing measurements cancel it
  through the same hold path. No actuator gains, robot XML, or other joints change.

## Display and operation

Run `./run-handdepth.sh`. The iPhone endpoint remains
`YOUR_UBUNTU_IP:8765`. Source connection/protocol/depth reconstruction are unchanged.

Green upper-arm segments show filtered angle targets at the robot's shoulders and
link lengths. Cyan is current measured human geometry; amber is held/inferred
display geometry. The two skeletons can have different shoulder widths and limb
lengths, so matching angles does not imply coincident endpoints. Viewer rows show
left/right angular error against the current measured direction, including target
filtering and servo lag. Held parts have no current angle target.

Keep the phone fixed with chest, shoulders and elbows visible. Slowly raise/lower
the upper arms; the robot's elbows and wrists remain bent at their held angles.
Space pauses, C resumes, R clears display history. If the shoulder row says
`measured shoulder/elbow required`, expose the elbow until its metric measurement
returns. R cannot manufacture a missing measurement.

For comparison, stop the service intentionally, then start the root launcher with
`HANDDEPTH_ARM_MAPPING=cartesian`. Direct `teleop --arm-mapping cartesian` also
selects the legacy controller. The Python `Controller`/`Retargeter` constructors
retain their Cartesian default for existing callers and replay tools; the CLI and
root launcher explicitly choose the new shoulder mode.

## Validation

- Full suite: **182 passed**, including 29 shoulder tests. Covers anatomical axis
  signs, both arms, absolute calibration, limb-length/wrist independence, missing
  torso, occlusion and contradictory inferred-point data, reacquisition, distinct
  source-frame gating, session/calibration/geometry reset, antipodal filtering,
  finite-difference Jacobian verification, and physical G2 movement/holds/limits.
- A controlled reachable movement on both G2 shoulders converges below 1 degree
  error. Distal and torso servo commands remain constant; stale input holds commands.
- For a larger reachable step (0.7/0.3 rad changes in shoulder joints 1/2), the
  old servo stepping still had about 11.2° direction error after two simulated
  seconds; bounded compensation reduced that to 0.02°. Peak physical joint speed
  was 0.765 rad/s in this test, with no IK failure or limit violation.
- Exact 120-second packet replay from `PRIVATE_RECORDING`, reconstructed
  through the receiver and simulated at 50 Hz, compares both controllers on the
  same fresh measured frames when both mappings are active:

| Shoulder | Legacy mean / median | Shoulder mode mean / median |
| --- | --- | --- |
| Left (590 control samples) | 49.14° / 50.67° | 6.52° / 1.42° |
| Right (1,821 control samples) | 38.40° / 38.26° | 1.80° / 0.97° |

A second 120-second packet replay uses the new live arm-movement recording
`PRIVATE_RECORDING`, with approximately 5,476 common control samples per arm:

| Shoulder | Legacy mean / median | Shoulder mode mean / median |
| --- | --- | --- |
| Left | 52.73° / 57.87° | 2.91° / 1.31° |
| Right | 69.60° / 66.23° | 4.02° / 1.80° |

These are repeated controller samples, not independent human observations. The
comparison measures agreement with the estimated human direction, not anatomical
ground truth. Left/right 95th-percentile shoulder errors are 36.49°/5.17° on the
first recording and 8.54°/15.81° on the live-movement recording: tracking
is not instantaneous, particularly during large changes and reacquisition.
All four controller runs had zero IK failures/physics warnings, zero base
displacement, and passed the disconnect-hold check. Shoulder-mode peak physical
joint speeds were 0.726 and 0.869 rad/s. Torso servo commands are exactly fixed;
MuJoCo's finite-compliance equality constraints deflect by at most 4.06e-5 rad
(0.00233°), within the replay's explicit 1e-4 rad compliance tolerance. The first
replay check used 2e-5 rad and flagged this tiny numerical deflection; reports now
retain every check result and use the documented physical tolerance.
Joint limits/collision constraints remain active and can
leave a residual angle on an unreachable request.

Reproduce with `venv/bin/python -E -s tools/ubuntu/validate_shoulders.py SESSION
--seconds 120 --output HDD_PATH`. Reports, live verification and the pre-change
backup are under the mounted HDD:
`/mnt/robotics-data/robotics/agibot-g2/datasets/handdepth/teleop/`:
`shoulder-comparison-phone032.json`, `shoulder-comparison-live-movements.json`,
`shoulder-live-validation.json`, and
`shoulder-backup-20261008T000406/`. Mount verification occurs before writing;
there is no SSD recording fallback.

Live verification: both shoulders tracked in the first 30-second check, and both
Ubuntu windows displayed the retained human skeleton and green direction targets.
After loading the final servo adjustment, the next 30-second check received 115
distinct source frames, had one calibration, zero IK failures/physics warnings,
and zero base movement. Left shoulder measurements were available in 23/120
samples (median error 0.58°); no fresh measured right elbow was available in that
window, so the right shoulder correctly held. Both final shoulder controllers
were exercised by the physical tests and both recorded-data replays. The native
viewer and receiver remain running under `handdepth-teleop.service`.

## Remaining limits

This is a shoulder-only stage, not full upper-limb imitation. Forearm direction,
elbow flexion, bicep twist and palm orientation need separate subsequent work.
The camera observes skin surfaces and Vision landmarks, not internal joint centers.
Torso-frame or elbow-depth errors still produce incorrect angles; a persistent
amber overlay does not prove that valid control measurements exist. The current
source rate is approximately 7.5 Hz, so smoothing and bounded movement add visible
lag. Gains and dynamics are simulation settings, not physical-robot validation.
