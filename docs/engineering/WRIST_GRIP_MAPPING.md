# Relative wrist rotation and fist gripper control — 2026-10-08

The root launcher now runs `--arm-mapping limb --wrist-mapping relative
--grip-mapping fist`. Shoulder/elbow direction tracking and the fixed torso/base
are preserved. Python Controller/Retargeter defaults remain hold/pinch for
compatibility with existing integrations; the CLI and executable use the new
defaults explicitly.

## Use

Start `./run-handdepth.sh` and stream to
`YOUR_UBUNTU_IP:8765`. Briefly show your hands with comfortably bent elbows and
relaxed wrists. Neutral is acquired independently from three distinct measured
frames. C explicitly reacquires a comfortable neutral; no T-pose is needed.
Rotate your wrists, close a fist to close the corresponding gripper, and open
your hand to release. Space holds; C resumes. R only resets display history.

The viewer reports wrist orientation error, tracking/hold reasons, grip state
and finger-curl score. `teleop/latest.json` adds `wrists`, `wrist_mapping`,
`grips` and `grip_mapping`, retaining the existing shoulder/elbow diagnostics.

## Wrist mapping

Current measured shoulder→elbow and elbow→wrist vectors define a forearm bend
frame in the human torso coordinates: longitudinal z points toward the wrist,
x is the bend-plane normal, and y completes a right-handed frame. The measured
palm rotation is expressed relative to that frame. A comfortable neutral cancels
the human palm/model gripper reference offset; anatomical forearm axes map its
rotation change onto the G2 forearm's axes. Whole-arm rotation/translation does
not change a fixed relative wrist pose.

Mink's installed `RelativeFrameTask` regulates the wrist site orientation
relative to `arm_[lr]_link4`, with zero position cost. Its upstream shoulder,
elbow and torso Jacobian columns cancel. Only joints 5–7 are enabled for this
task. The actual G2 model also has zero forearm-direction Jacobian columns for
those joints. The existing joint/contact limits and shoulder/forearm priorities
remain enabled. Wrist orientation is filtered with a 100 ms time constant and
1.2 rad/s target limit; joint IK speeds stay limited to 1 rad/s with bounded
servo damping compensation. Physical speeds can differ slightly.

Palm orientation now requires three measured anchors: wrist, index MCP and
little MCP. A consistent knuckle-midpoint axis avoids dropping orientation merely
because middle/ring MCP depth is missing. Palm position still averages available
palm anchors. Depth rejection, plausible triangle bounds, nondegeneracy checks
and quaternion flip handling remain in place. No inferred elbow/palm depth is
invented.

Wrist control requires a current associated hand, measured palm and measured
forearm. A nearly straight/folded elbow does not reliably define its bend plane;
the wrist holds there using the existing bend-plane hysteresis. Missing data,
ambiguous hand identity, missing torso orientation and stale/disconnected input
hold commands. Occlusion preserves wrist neutral; new session/geometry or an
explicit C calibration reacquires it. Amber display history remains display-only.

## Fist gesture

The receiver uses current phone Vision finger coordinates in the original color
pixels, separate from measured 3D palm orientation. This allows visible curled
fingers to control the gripper even when fingertip depth is unavailable. It is
explicitly labelled `current_vision_rgb_finger_curl`, not measured 3D finger bend.

For each index/middle/ring/little finger, compare its MCP-to-tip distance with the
sum of its three projected bone lengths. Straight fingers score near 0 and folded
fingers near 1. The calculation is invariant to image translation, rotation,
mirroring and uniform scale; it is still sensitive to projection/foreshortening.
Low confidence, tiny/edge-on palms, degenerate paths and insufficient landmarks
remain unknown. A thumb/index pinch alone does not count as a fist.

- Close: at least three fingers have curl >= **0.65**, with no confidently
  extended finger below **0.30**.
- Release: at least three fingers have curl <= **0.30**.
- Confirm a change on two distinct source frames separated by at least 80 ms.
  The intermediate band keeps the established state without chatter.
- Loss, identity change or a gap over 0.5 s cancels pending confirmation. Current
  evidence must reconfirm before resuming; stale frames cannot advance the latch.

Commands remain -0.85 rad closed / -0.05 rad open, within model limits and filtered
at 0.6 rad/s. Actual MuJoCo gripper-link separation confirms the polarity for both
hands. A missing gesture holds the actual gripper command; it does not complete
a pending closure blindly. Gesture state is independent per anatomical hand.

To compare older behavior, use the launcher environment variables
`HANDDEPTH_WRIST_MAPPING=hold` and/or `HANDDEPTH_GRIP_MAPPING=pinch` before starting
an inactive service. Running the launcher while active does not replace it.

## Validation and limits

**257 Python tests pass**, including 19 new tests. These cover both wrists and
all three axes, whole-arm invariance, measured-data gates, ambiguous/straight-arm
holds, neutral/session behavior, Jacobian finite differences, physical convergence
below 2 degrees for reachable model-derived targets, simultaneous shoulder/elbow
adherence, and actual two-gripper close/release with loss/disconnect holds.
Gesture tests cover scale/mirror/rotation, pinch rejection, threshold hysteresis,
duplicate/stale frames, identity changes and RGB evidence without fingertip depth.

Three recorded-input physics comparisons cover 165 seconds across two recordings.
The first 105 seconds tested the initial five-anchor palm gate; the final 60-second
comparison uses the three-anchor implementation. All preserve joint limits,
fixed torso/base and disconnect holds, with zero IK failures/physics warnings.
Final recorded wrist median tracking errors were 4.5 degrees left and 9.8 degrees
right while active. Right-hand measurements are intermittent; its 95th percentile
error was 91.5 degrees during short reacquisition/moving-target intervals. These
are controller-to-estimated-target errors, not anatomical accuracy measurements.
Shoulder/forearm mean errors changed by less than 0.1 degree in that comparison.

The existing recordings include occlusion, projection ambiguity and source-session
changes. Fist thresholds need a live user check across palm-facing and edge-on
poses; current RGB landmarks cannot fully resolve hidden fingers. Neither these
replays nor synthetic tests establish gesture-classification accuracy.

Only `handdepth-teleop.service` was restarted. The receiver is listening at the
unchanged address and both Ubuntu windows are running with relative/fist modes.
The phone disconnected before deployment and had not resumed by the final check.
The 15-second runtime check therefore verified standby/hold behavior, not live
gestures: zero command updates, IK failures, physics warnings or base motion.
See `runtime-check.json`. Resume the phone stream for the remaining live gesture
check; no receiver/network configuration change is needed.

Source/venvs remain on SSD. The mounted `/dev/sda1` HDD stores all new reports in
`/mnt/robotics-data/robotics/agibot-g2/datasets/handdepth/teleop/wrist-grip-validation/`.
The repeatable validator is `tools/ubuntu/validate_wrists.py`. Pre-change core
files are in `teleop/wrist-grip-backup-20261008T041022/`. The phone wire protocol,
connection endpoint, body tracking history and RGB/metric occlusion fix remain.
