# Evidence and limitations

The demo shows a working experimental teleoperation integration. Numbers below
describe specific checks, not a general accuracy or reliability guarantee.

## Tests and build

- Historical deployed snapshot: **268 Python tests passed** after stream/overlay fixes.
- Public snapshot validation: see [release-validation.json](evidence/release-validation.json)
  for the fresh test count against a separately rebuilt G2 scene.
- The public asset converter produced 70 meshes, 16 initial actuators and 10
  gripper mimic constraints; the teleop builder restored the head/torso to 24 actuators.
- Synthetic physical targets verify IK/Jacobians, convergence, joint/contact
  limits and holds. They do not measure human anatomical accuracy.
- CI without external G2 assets skips the asset-dependent tests. It must not be
  described as equivalent to the full local physics check.

## Historical recorded-input comparisons

These figures are transcribed from the project's engineering notes. Original
human recordings remain private; they are not part of the public release.

| Check | Result | Interpretation |
| --- | --- | --- |
| 120-second moving-arm shoulder replay | Left mean 52.73° → 2.91°; right 69.60° → 4.02° | Legacy hand-position controller versus shoulder-direction stage, on common active control samples |
| Same replay, shoulder p95 after change | Left 8.54°; right 15.81° | Large changes and reacquisition still introduce lag |
| 210 seconds of elbow comparisons, three windows | Forearm means before 27.94–52.95°; after 3.14–34.27° across six arm/window cases | Improvement varies substantially with visibility, motion and reachability |
| Final 60-second wrist comparison | Active median 4.5° left / 9.8° right; right p95 91.5° | Intermittent measurements and reacquisition remain substantial limitations |
| 50-second elbow live check | Active median forearm 1.57° left / 1.16° right; available 79/200 left and 190/200 right sampled statuses | Held/missing samples are excluded from the angular error statistics |

All angular errors compare the robot to **estimated** human targets, not ground
truth. Controller ticks repeat observations and are not independent camera frames.
Historical shoulder-only results are not claimed as current full-arm benchmarks.

## Streaming and display

The stray shoulder line was reproduced as a 0.7535 m shoulder-to-hip edge joining
a current shoulder to a hip observation 59.86 s older. Coherent pelvis admission
removed that line in replay without changing either arm's coordinates.

The 60-second continuity check after the receiver fallback change measured 7.26 Hz
publication, with zero stale samples in 300 sampled statuses. Measurement age was
282 ms median, 382 ms p95 and 456 ms maximum. Both phone fallback and paired PC
fusion occurred. No person was in view during its final inspection: this validates
transport behavior, not active limb tracking.

The author's supplied screenshots show another session at approximately 3.6–4.3 Hz
with some joints held. Do not present 7.26 Hz as a guaranteed rate or the demo's
entire-session rate. PC model inference time, network arrival time, measurement
age and motion response are different measurements.

## What is not established

- Anatomical joint-center accuracy or calibrated ground-truth pose error.
- Fist/open-hand classification accuracy over users, views and occlusions.
- Complete fresh-clone phone setup: Swift source is not in this release.
- Sustained exact phone RGB/depth pairing under thermal/network load.
- A trained policy, autonomous object manipulation, or real G2 deployment.
- Physical safety certification or validated robot dynamics.

Recorded replays, deterministic regression tests, live transport checks and a
qualitative demo answer different questions. The next benchmark should report
latency distributions, per-joint availability, tracking error while active,
reacquisition time and failures under repeatable motion/occlusion conditions.
