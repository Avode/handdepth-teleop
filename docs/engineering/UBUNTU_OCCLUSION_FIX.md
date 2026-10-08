# Ubuntu occlusion continuity — 2026-10-08

The iPhone wire format and LAN connection are unchanged. The robot base and
torso remain fixed, with the existing neutral-relative retargeting, servo limits
and freshness gates. **No inferred display XYZ enters Mink or retarget.py.**

## Why the lines disappeared

`pose_overlay.py` cleared its visual scene and returned whenever torso
orientation was missing. Validity filtering also discarded individual body
points each frame. `visualization.py` only paired the most recent sensor and
preview packets; arrival on two independent sockets caused intermittent raw RGB
frames without bones. Its camera-space 3D panel had no body history.

## Current behavior

`display_tracking.py` owns display-only state: at most 19 body points, two
coherent arms and five length samples per arm. A point stores its last reliable
source frame/time; time passing does not turn old data into a new measurement.
Observed geometry uses the normal measurement color. Held/inferred geometry is
amber; RGB/camera-panel lines are dashed. Status rows show joint states and ages.
When information is insufficient, an established arm retains all three points
of its last coherent shape. Unknown points remain unknown.

The RGB panel consumes the phone's named `body_tracking` points and constraints
directly, keeping observed-body, observed-hand, held and inferred provenance.
An eight-frame cache per channel pairs matching session, frame, capture time,
calibration and raster dimensions. It displays the newest exact pair while
waiting for the next pair, instead of flashing an unannotated image between
socket arrivals. Stale/disconnected RGB freezes that pair with a **HELD** label,
amber bones and increasing age. It never puts one frame's detections on another
frame's image. Older recordings use bounded per-joint 2D observation history.

The MuJoCo overlay uses measured torso-relative XYZ; the dashboard 3D panel
uses measured camera XYZ. A missing torso orientation freezes the MuJoCo
human shape in the fixed robot torso frame; new camera XYZ is never transformed
through an old torso orientation. The dashboard camera frame can still show
current valid camera measurements independently.

## Metric elbow inference

Lengths come from complete measured 3D arms, including a current unambiguously
associated measured hand wrist when available. Length acquisition requires
three consistent distinct observations (confidence at least 0.4), within a
20% spread. Five samples at most are retained. Broad 5–75 cm segment bounds and
a 0.4–2.5 upper-arm/forearm ratio reject gross surface mismatches; these are
quality gates, not proof of anatomical joint accuracy. Lengths freeze during
loss and can be refreshed by consistent complete measurements on reacquisition.

Given current shoulder and wrist, a missing elbow is placed on the intersection
of two spheres with the retained metric radii. The previous 3D bend is parallel
transported as the wrist direction changes. Reach is clamped to the reachable
annulus, with a 2-degree minimum bend and a tiny folded-arm separation. The
bone lengths are never stretched. The constrained wrist is explicitly marked
**CLAMPED**; a red target/cross and connecting line show the requested measured
wrist separately. A previously straight arm with no known bend plane is held
until a bend is actually observed.

The phone's lengths are **color pixels**, used only by the 2D display. Its
held/inferred elbow state is an explicitly uncertain occlusion cue, never a
metric depth measurement or a uniquely determined 3D bend. Ubuntu uses the
previous measured 3D bend instead. `body.py` skips elbow depth sampling when
that cue rejects the raw Vision elbow, even if raw Vision confidence is high.
This prevents an occluding hand surface from being accepted as elbow XYZ. Raw
recorded packets stay unchanged; derived measured elbow validity becomes false.

## Reset and limits

R clears both local display histories immediately, retaining the source frame
watermark so the same polled frame cannot refill them. New phone sessions and
incompatible projection/coordinate geometry clear history automatically. Rate
or filtering metadata alone does not clear geometrically compatible history.
An authoritative phone point set that shrinks/clears also resets local history,
because the phone otherwise retains established points through occlusion.

Protocol v1 has no explicit phone reset epoch. A phone reset that immediately
recreates the identical full point set cannot be distinguished from ordinary
reacquisition; use R for a guaranteed simultaneous Ubuntu reset in that case.
An additive reset epoch would remove that ambiguity in a future phone update.

Retained shapes can become old or anatomically inaccurate while hidden. Their
ages remain visible, and measured visible surfaces still have depth/detector
error. Show each arm comfortably bent with its endpoints visible briefly to
establish metric geometry; no T-pose is required. Unknown geometry is never
filled from projected pixel lengths. Stale/disconnected input still holds robot
commands even while display bones remain visible.

## Verification

153 tests cover both elbows, torso loss, fixed lengths, bend continuity,
unreachable/coincident wrists, reacquisition, unknown/straight arms, source and
reset changes, exact RGB matching, no inferred depth sampling, and controller
hold despite persistent display geometry. Two packet replays cover 180 seconds
and include MuJoCo physics, both displays and disconnect holds. A 50-second live
check and actual window captures verified the installed running program.

```bash
cd /path/to/handdepth-teleop
venv/bin/python -E -s -m pytest -q tests
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python -E -s tools/ubuntu/validate_occlusion.py \
  /mnt/robotics-data/robotics/agibot-g2/datasets/handdepth/SESSION \
  --seconds 120 \
  --output /mnt/robotics-data/robotics/agibot-g2/datasets/handdepth/teleop/occlusion-check
```

The validator requires a writable mounted data volume before writing. App/code
and venv remain on SSD; logs, packets, pose JSONL, reports and images stay on HDD.
