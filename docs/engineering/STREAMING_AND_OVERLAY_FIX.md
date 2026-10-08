# Stray shoulder line and stream freezes — 2026-10-08

## Findings

The long amber line in the supplied MuJoCo image is the right shoulder→right hip
edge. Replaying source frame **178243** reproduces a **0.753508 m** segment to an
isolated hip observation **59.86 seconds** older than the shoulder. This is a
stale lower-body estimate being joined to a current upper body, not an extra arm
or a robot control target.

The apparent stream loss is also reproducible without a disconnected phone.
Both sockets remained established, with around 3 ms TCP RTT, while the last PC
fused pose stopped advancing. In the captured packet log, sensor IDs become even
(`181380,181382,...,181418`) and preview IDs become odd
(`181381,181383,...,181419`). Of the first 2,767 accepted sensor packets, only 490
had exact preview matches. The last match was frame 178243. The two independently
scheduled phone channels can settle on opposite phases and stay there.

Sensor input still arrives around 7.5 Hz: median recorded receive interval
134 ms, p95 212 ms; phone processing median 108 ms, p95 122 ms. The old receiver
published only paired PC results, so an unpaired stream froze indefinitely even
though fresh phone measurements continued arriving. The RGB panel also kept
showing the last matched image indefinitely. This diagnosis does not establish
that all future connection jitter has the same cause.

## Ubuntu changes

- `MetricSkeleton` establishes pelvis geometry only after three coherent current
  shoulder/hip pairs, with confidence and plausible geometry checks. An isolated
  hip cannot establish a shoulder-to-hip line. Valid established lower-body
  geometry is retained relative to the human torso, rather than leaving a stale
  camera-space hip behind when the person moves. Both 3D views share this logic.
  Upper-arm/forearm history, metric lengths, bend continuity and reset semantics
  remain. The replay verifies both arm shapes are numerically unchanged.
- Each accepted sensor frame gets a bounded **100 ms** PC-fusion wait. A missing
  RGB match or slow PC result then publishes the current phone RGB-D reconstruction
  with explicit `phone_rgbd_fallback_unpaired_or_slow_pc` provenance. Original
  capture/receive ages and phone measurement validity are retained. A late PC
  result cannot republish that frame and flicker its validity. Fast matched PC
  results continue to win. Pending sensor history is bounded to eight entries;
  only the latest expired candidate is published when processing falls behind.
- Disconnect, measurement timeout, new session and changed geometry clear pending
  fallback work. Expiry runs at 10 ms intervals in the headless receiver thread.
  FrameGate's 250 ms processing/capture checks and the controller's 500 ms
  freshness limit are unchanged. No synthetic freshness, adjacent-frame fusion
  or inferred display coordinates enter control.
- If the last matched RGB pair is over 500 ms old but a fresh preview is available,
  show the fresh preview with **unpaired RGB / no skeleton overlay**. Do not paint
  a different sensor frame's skeleton onto it. Current 3D measurements continue
  alongside it. Exact matched overlays return automatically when pairing resumes.
- `stream_transport` in `teleop/latest.json` exposes received/accepted/rejected
  counts, rejection reasons, channel source/arrival rates and frame IDs, wire age,
  phone processing/ACK processing times, publication source/rate and pending count.
  This distinguishes source pauses, rejected stale frames, and missing fusion
  pairs. The dashboard labels phone fallback explicitly.

The fixed robot torso/base, measured shoulder/elbow/relative-wrist mapping and
fist-close/open-hand-release control remain enabled. Only the managed receiver/
viewer service was restarted; the phone endpoint and protocol did not change.

## Validation

**268 tests pass**. Eleven new cases exercise missing RGB, fast PC success,
late/duplicate results, source/timeout/disconnect boundaries, bounded pending work,
alternating channel IDs, live unpaired RGB with exact-overlay reacquisition,
isolated false hips, coherent pelvis retention and display reset behavior.

The recorded-frame MuJoCo render removes the stray hip/line while preserving
both arms exactly. New files, including the live runtime report, are on the
verified mounted HDD under:
`/mnt/robotics-data/robotics/agibot-g2/datasets/handdepth/teleop/stream-overlay-validation/`.

- `stray-line-before-after.png` and `stray-line-report.json`
- `live-report.json` (rates, original measurement ages, publication sources,
  channel diagnostics, control/physics health and retained overlay geometry)

The 60-second live check advanced from frame 192742 to 193638, with **zero stale
samples** in 300 status observations. Publication continued at **7.26 Hz** while
sensor and preview IDs were again on opposite phases. Measurement age was
282 ms median, 382 ms p95 and 456 ms maximum, below the unchanged 500 ms control
freshness limit. PC fusion and phone fallback both published during this check.
Later frames resumed PC fusion automatically when exact matches returned.
Both sockets remained established, with empty send/receive queues; the receiver
and both visible windows were verified running after the restart.

No person was in view during the final live inspection, so the robot correctly
held waiting for measured shoulders/chest. This validates transport continuity,
source switching and stale-input protection; the stray-line correction was
verified using recorded human input and the before/after MuJoCo render, rather
than claiming a fresh live arm-motion trial. There were zero IK failures or
physics warnings, zero base displacement and only numerical torso deviation
(approximately 6e-12 rad). Live window screenshots are saved beside the report.

Pre-change files are in `teleop/stream-overlay-backup-20261008T051059/`.

## Remaining phone-side fix

The phone should select synchronized RGB/depth samples once, generate both packets
from that same capture, and coordinate bounded admission/backpressure. Never relabel
adjacent frames as a pair. The Swift source is not included in this Ubuntu release;
paired transport must be validated on the physical phone. Ubuntu's explicit phone
fallback preserves current observations while exact paired fusion is unavailable.
