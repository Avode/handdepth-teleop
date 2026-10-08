# Hidden forearm anchored to a visible hand — 2026-10-08

The user's screenshot is original source frame **105448**, from recording
`PRIVATE_RECORDING`. Its right hand has a valid measured wrist even though
the palm-position/orientation result is invalid. The shoulder is measured and
the elbow is occluded. The retained metric arm solver already connects those
endpoints with measured segment lengths and its previous bend plane. However,
the RGB panel previously drew the phone's separate pixel-circle solution. That
solution disagreed with the metric arm because foreshortening changes projected
segment lengths.

## Changes

- `body.py` retains an independently measured, unambiguously associated hand
  wrist even when the body elbow is missing. The arm's measurement-valid flags
  remain false until the elbow is measured; wrist validity does not imply palm
  validity or a complete observed arm.
- `geometry.py` projects camera XYZ back through the color intrinsics and
  forward lens-distortion mapping. This is the inverse of the existing
  registered-depth ray geometry, not a linear resize of the depth image.
- `visualization.py` updates its camera-space metric skeleton before its RGB
  layer. When a hidden elbow can be solved using reliable metric lengths and
  current endpoints, `display_tracking.py` projects that same coherent arm onto
  the exactly matched RGB frame. Both arms use this path. The measured wrist
  endpoint stays fixed unless the target exceeds the arm's reachable annulus;
  in that case the existing clamp marker identifies the actual hand target.
- RGB inferred geometry is amber, with observation age and a `3D` suffix.
  It carries `ubuntu_metric_arm_projection` provenance, current constraint-frame
  metadata, and `inferred_control: false`. Phone `body_tracking`, raw packets and
  measured pose data are unchanged. Its lengths are still **color pixels**;
  projected metric arm metadata has separate names and an explicit metre unit.
- If metric history is insufficient, the existing phone image-space cue remains
  available. Previously unseen metric geometry is not invented. Reset, session,
  coordinate geometry, source frame and capture-time boundaries are respected.

No depth is sampled at the hidden elbow. A 2D elbow cue is not promoted to
measured XYZ. The metric solver and MuJoCo torso-relative overlay retain their
existing length/bend continuity behavior. The robot torso/base remain fixed,
default mapping stays `limb`, and inferred display geometry does not enter the
controller. An occluded elbow still holds that robot arm pending measured
reacquisition. Hidden elbow anatomy is inherently ambiguous from two endpoints;
the displayed bend is a continuity estimate, not ground truth.

## Validation

The full Python suite passes **238 tests**. Ten new cases cover both sides,
valid wrist with invalid palm, preservation of measured data/controller holds,
fixed segment lengths, unreachable wrist annotation, unseen geometry,
session/geometry/local resets, stale source rejection and calibrated projection
with lens distortion. After the final status/legend changes, the 37 relevant
display tests also pass.

Exact-frame replay uses the original RGB, raw sensor packet, fused pose and
preceding metric history. The right RGB elbow changes from **(209.09, 393.45)**
to **(157.86, 394.37)** color pixels, a **51.24 px** correction onto the visible
bicep. Independent current PC RGB inference places it at **(158.24, 389.40)**;
that image-space agreement is corroboration, not a depth measurement. Retained
metric lengths remain **0.254854 m / 0.385507 m**. Measured wrist endpoint error
and RGB wrist projection error are both zero. Original phone tracking and
measured pose remain unchanged, and the elbow remains inferred/display-only.

The updated live run was monitored for **50 seconds / 250 status samples**.
Projected metric arms appeared in 81 right-arm and 46 left-arm samples; both
camera-space and MuJoCo displays retained inferred/held arms, and both windows
were visually inspected. Source rate was 7.4 Hz, control about 40 Hz and PC
inference about 21 ms. There were zero IK failures/physics warnings, one
calibration, zero base displacement, and at most 0.000034 rad torso compliance.
Three status samples caught measured reacquisition before the slower display
refresh; checking their exact recorded poses confirmed current measured forearms
in all three. Inferred geometry was never used for control.

HDD evidence (mount verified as `/dev/sda1` before writing):
`/mnt/robotics-data/robotics/agibot-g2/datasets/handdepth/teleop/hidden-forearm-validation/`

- `comparison-105448.png` and `corrected-105448.png`
- `exact-frame-report.json`, original packet/pose and metric-history files
- `live-report.json` from the updated receiver and actual MuJoCo run

The exact-frame validator is `tools/ubuntu/validate_projected_arm.py`. Its
`--backup` argument points to the pre-change renderer in HDD directory
`teleop/hidden-forearm-backup-20261008T014641`. It asserts source-frame identity,
unchanged measured data, fixed lengths, wrist anchoring and control separation.

Only `handdepth-teleop.service` was restarted. The root launcher remains
`./run-handdepth.sh`, with automatic iPhone reconnect
to `YOUR_UBUNTU_IP:8765`. Receiver and MuJoCo stay running. Status now includes
`dashboard_display.rgb_arm_methods`, whose projected-arm value is
`projected_metric_lengths_previous_bend`.
