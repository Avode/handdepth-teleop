# HandDepth protocol v1

Two independent unencrypted binary WebSocket endpoints on a trusted LAN:
`ws://HOST:8765/sensor` and `ws://HOST:8765/preview`. One publisher per endpoint.
There is no cloud, microphone, robot interface, ARKit sceneDepth or LiDAR dependency.

## Framing

Each WebSocket message is exactly one packet:

| Bytes | Meaning |
|---|---|
| 0–3 | Unsigned UInt32 JSON header byte length, **big endian** |
| next N | UTF-8 JSON object, no BOM; finite numbers only; invalid measurements use `null` |
| remainder | Binary payload, length exactly `payload_length` |

Header limit: 65,536 bytes; payload limit: 2,097,152 bytes; total limit:
2,162,692 bytes. No base64, compression, concatenated packets, pickle or native
struct padding. Unknown protocol versions/kinds, duplicate JSON keys, non-finite
numbers, unsupported orientation, malformed lengths/dimensions and invalid joint
lists are rejected. JSON key order and numeric text formatting are not significant.

Every header has `protocol_version: 1`, `kind`, `session_id`, `payload_length`.
Data headers have `payload_encoding`. Control payloads are empty.

## Common data fields

| Field | Definition |
|---|---|
| `kind` | `sensor` or `preview`, must match endpoint |
| `session_id` | UUID string for one app process/camera timeline; reconnect retains it |
| `frame_id` | Increasing nonnegative integer assigned at synchronized capture callback, including callbacks subsequently dropped; never a send counter |
| `capture_time_s` | Color sample presentation timestamp converted **from AVCaptureSession.synchronizationClock to CMClockGetHostTimeClock**, seconds |
| `clock` | Exactly `ios_host_monotonic_seconds`; no UTC epoch and no direct comparison with PC clocks |
| `depth_capture_time_s` | Depth synchronized timestamp converted to the same host clock (iPhone data) |
| `sync_delta_s` | Depth timestamp minus color timestamp; pairs more than one capture interval apart are discarded |
| `processing_start_s`, `processing_end_s` | Same iOS host clock, actual processing times for this packet |
| `calibration_id` | SHA-256 of sorted JSON capture settings + calibration; changes with matrices, LUTs, reference/actual dimensions, rate, filtering, zoom, stabilization or geometry settings |
| `color_dimensions` | `[width,height]`, actual color pixel buffer and JPEG dimensions |
| `depth_dimensions` | `[width,height]`, actual depth map (required on sensor; iPhone also supplies it on preview) |
| `orientation` | Exactly `native_up`; native sensor raster with connection rotation angle **0** |
| `mirrored` | Exactly `false`; display mirroring is never transmitted |
| `application_version` | `0.3.0` (older recordings may contain `0.1.0` / `0.2.0`) |
| `clock_estimate` | `null` or `{offset_s,rtt_s,ios_sample_s}` described below |

v1 transmitted rasters support 1–2048 pixels per dimension, subject to payload
limit. Calibration reference dimensions may reach 16384; they need not equal the
transport resolution. iPhone sends
native dimensions and no crop or resize. The UI is locked to landscape right;
rotating the phone never changes transmitted raster geometry. Vision receives
`.up`. The stationary phone defines the camera frame, not a world frame.

## Sensor

`payload_encoding: "depth_u16_mm_le"`. Tightly packed row-major **little-endian
UInt16 millimetres**, exactly `width * height * 2` bytes. Pixel `(x,y)` begins at
`2*(y*width+x)`. **0 means invalid**, never a point at the camera origin.

On iPhone, AVDepthData is converted to DepthFloat32 metres. CVPixelBuffer's actual
bytesPerRow is honored; no row padding is transmitted. Non-finite values,
values below 0.001 m or above 65.535 m are encoded as 0. Other values round to
nearest millimetre (ties away from zero). Quantization error is at most 0.5 mm
apart from floating-point input representation; it does not establish sensor
accuracy. Wire range is 1–65535 mm; hand reconstruction samples only 0–4 m.

Additional header fields:

- `depth_accuracy`: `absolute` or `relative`, from AVDepthData. Relative data is displayed but never claimed as metric 3D.
- `depth_quality`: `high` or `low`, the frame's Apple metadata.
- `depth_filtered`: Boolean from actual `isDepthDataFiltered`, not just requested settings. Default filtering off; configurable. Filtered surfaces can incorporate temporal interpolation.
- `hands`: **zero, one, or two** objects, each `{hand_id,chirality,landmarks}`.
  Each detected hand retains **all 21** entries ordered below, each
  `{name,x,y,confidence}`. Missing/invalid coordinates use
  `x:null,y:null,confidence:0`. Confidence is Vision's [0,1] value.
  `chirality` is Vision's `left`, `right`, or `unknown` classification.
- `hand_id`: session-local `hand-N`, unique within a packet. Bounded image-space
  palm matching associates consecutive observations one-to-one, with chirality
  as a soft hint. Sorting by creation number keeps array order stable when
  Vision reorders detections. IDs can survive a brief disappearance (≤500 ms),
  but **no old landmarks are emitted**. Longer loss creates a new ID. IDs are
  not person identity; crossings/occlusion/large motion can change association.
- `landmarks` and `chirality` at header level are compatibility aliases for the
  first hand, or `[]` / `unknown` when none. They must agree with `hands[0]`.
  New receivers use `hands`. Legacy v1 packets without `hands` remain readable
  as one `legacy-primary` hand. This is an additive protocol-v1 extension.
- `tracking_error`: `null` or Vision error string for this frame.
- `body` (application 0.3.0): `null` or `{landmarks,torso_surface,torso_surface_source}`.
  Exactly **19** body points use the same `{name,x,y,confidence}` representation
  and source frame as both hands. Order:
  `nose, neck, root, leftEye, rightEye, leftEar, rightEar, leftShoulder,
  rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip,
  leftKnee, rightKnee, leftAnkle, rightAnkle`. Left/right are anatomical.
  `VNDetectHumanBodyPoseRequest` runs in the same request pass as hand detection.
  Zero observations yields null; **multiple people yields null + body_error**.
  There is no inferred 3D Vision skeleton or person identity tracking.
- `body_error`: `null` or body-specific error. Legacy packets without body fields
  remain readable with no body output.
- `torso_surface`: null or one `torsoSurface` image sampling location, whose XYZ
  will require a measured depth sample. `torso_surface_source` is one of:
  `image_midpoint_neck_root` (mean neck/root coordinates, both confidence ≥0.4),
  or `image_shoulders_positive_y_normal` when hips/root are unavailable.
  For the latter, let `d = leftShoulder - rightShoulder` in normalized image
  coordinates. Negate d if its X is negative. Require `|d| ≥0.08` and both
  confidences ≥0.4. Site = shoulder midpoint + `0.6 * [-d.y,d.x]`; it must lie
  inside [0,1]². This uses the **normalized-image** positive-Y normal, not a
  metric anatomical torso center. Confidence is the minimum prerequisite
  confidence. Receiver validates the stated transform and does not accept
  arbitrary predicted torso locations.
- `calibration`: complete calibration object on **every sensor frame**, or `null`. Reconnect/dropped packets therefore cannot lose a required calibration announcement.

Joint order:
`wrist, thumbCMC, thumbMP, thumbIP, thumbTip, indexMCP, indexPIP, indexDIP, indexTip,
middleMCP, middlePIP, middleDIP, middleTip, ringMCP, ringPIP, ringDIP, ringTip,
littleMCP, littlePIP, littleDIP, littleTip`.

Landmarks use normalized **unmirrored, distorted color image** edge coordinates:
X right, Y down, origin top-left; pixel centers are `(i+0.5,j+0.5)` in pixel
coordinates. Vision bottom-left output `(xv,yv)` becomes `(xv,1-yv)`. Pixel
coordinates are `(x*color_width,y*color_height)`. No old detection is attached to
a new depth frame. A skipped/failed request yields no landmarks.

## Calibration and registration

Matrices are **nested arrays of rows** on the wire. Swift extracts `m[column][row]`
explicitly rather than dumping SIMD memory.

| Calibration field | Meaning |
|---|---|
| `intrinsics_row_major` | 3×3 K, focal lengths/principal point in reference pixels |
| `reference_dimensions` | `[width,height]` from intrinsicMatrixReferenceDimensions |
| `color_from_reference`, `depth_from_reference` | 3×3 homogeneous pixel-edge transforms from reference pixels to actual rasters; v1 diagonal scaling only, translation zero |
| `distortion_center` | `[x,y]` in reference pixels; can differ from principal point |
| `lens_distortion_lut`, `inverse_lens_distortion_lut` | Arrays of Float32-origin relative radial magnifications; missing table is `null`, not identity |
| `lut_point_mapping` | `sdk_reference_forward_rectified_to_distorted_inverse_distorted_to_rectified` |
| `extrinsics_camera_to_reference_row_major_mm` | 3×4 `[R\|t]`; Apple camera-to-reference-camera pose, rotation unitless, **translation mm** |
| `extrinsic_from`, `extrinsic_to` | `calibrated_camera`, `apple_reference_camera` |
| `extrinsic_translation_units` | `millimetres` |
| `extrinsic_applied_to_pose` | `false` |
| `pixel_size_mm` | Physical pixel size at reference resolution |
| `alignment` | `truedepth_color_fov` |
| `crop`, `resize` | `none`, `native_output_dimensions` |
| `orientation`, `mirrored` | `native_up`, `false` |

Apple's TrueDepth device provides **perspective corrected**, common-FOV depth
aligned to the YUV camera. Converting YUV to BGRA/JPEG preserves this raster.
Different native resolutions are registered by the supplied scale transforms.
The raw extrinsic is preserved as provenance; applying it again as IR-to-RGB
registration would double-transform the already aligned data. Its reference
camera is not a world coordinate system for this application.

Depth remains lens-distorted. The receiver builds a rectified depth grid at
native depth resolution, not a naively resized pinhole map:

1. Convert each rectified depth pixel center to calibration reference coordinates.
2. Pull-map through **lens_distortion_lut** to original distorted coordinates.
3. Scale to depth resolution, choose nearest source pixel, preserve invalid/outside pixels as zero. Do not bilinearly blend depths across surfaces.
4. Transform the distorted landmark through **inverse_lens_distortion_lut** in reference pixels and scale to the same rectified grid.
5. Sample locally and unproject using `K_depth = depth_from_reference * K` only after rectification.

For each LUT, maximum radius is distance from distortion center to farthest
reference image corner. Linearly interpolate entries at `r/rmax*(count-1)`;
apply `center + (point-center)*(1+entry)`. These point directions follow the
installed Apple SDK's reference implementation; image-operation descriptions can
otherwise be misleading. See [source notes](docs/reference/APPLE_CALIBRATION.md).

Absent intrinsics/LUTs, invalid transforms, unequal aspect/FOV or relative depth
accuracy keep RGB/raw depth available while marking all metric output invalid.
v1 explicitly rejects crop, rotation and mirror transforms instead of guessing.

## Preview

`payload_encoding: "jpeg"`; binary JPEG, quality initially 0.65, no EXIF rotation
applied, exactly `payload_length` bytes. Receiver checks JPEG dimensions before
large decoder allocation. Same source frame IDs/timestamps/calibration ID as
sensor packets, but independent rates/channels mean some RGB/depth frames are
unpaired. The dashboard overlays landmarks only when **session_id and frame_id
both match**. Recording/replay retains these identities without interpolating.

## ACK, heartbeat, clock estimation and freshness

Data response on the same socket:
`{kind:"ack",session_id,frame_id,accepted:Boolean,payload_encoding:"none",...}`.
`accepted:false` means a valid but duplicate/stale/mismatched-session frame was
discarded. Malformed input closes with 1007; wrong endpoint/occupied endpoint
closes with 1008.

Each iPhone channel admits **one** data packet and waits for the matching ACK
before admitting another. New frames while busy are dropped; no pending frame
FIFO. The capture callback performs bounded synchronous work on one non-main
serial queue; AVFoundation discards late output. Body/hand processing cadence is
bounded by both requested/thermal rate and `0.7 / EWMA(callback processing seconds)`
(with a 1 Hz floor). A missed deadline leaves a quarter capture interval before new
work, avoiding back-to-back catch-up. JPEG rate is independent. Local iPhone RGB
holds the latest processed source and its matching overlays, expiring after
500 ms; preview-only packets do not change that source image. UI holds one latest snapshot and
polls at 10 Hz. The PC uses max WebSocket queue 1 and serialized CPU work. Recording
slowdowns cause frame drops at the sender through delayed ACKs.

Sensor clock exchange every ~2 s when admission is free:

- iPhone sends empty `clock_ping` with `t0` (iOS host clock).
- PC returns empty `clock_pong` with echoed `t0`, `t1` (PC receive monotonic time), `t2` (PC send monotonic time).
- iPhone records `t3` on receipt and estimates
  `offset_s=((t1-t0)+(t2-t3))/2` (PC minus iOS),
  `rtt_s=(t3-t0)-(t2-t1)`.
- Subsequent data includes `clock_estimate`, with `ios_sample_s=t3`.

Native WebSocket ping/pong also checks the connection (preview every ~2 s; server
ping every 2 s, timeout 3 s). iPhone data/clock ACK timeout 1 s cancels the socket,
discards in-flight state, clears clock estimate, and retries with 0.5–5 s bounded
exponential backoff. Native heartbeat/open timeout is 3 s. Disconnect and
backgrounding cancel both channels; foreground reconnects if connection was
requested. Only newly captured frames are offered after reconnect.

iPhone rejects locally older-than-250 ms results. PC rejects non-increasing
frame IDs/capture times per channel, processing age >250 ms, and aligned capture
age >250 ms plus clock uncertainty. High-water marks survive same-session
reconnect; receiver clock alignment is cleared on each new sensor connection. New sensor sessions reset estimator state. Preview cannot replace an
active sensor session.

Capture-to-receiver latency is `receive_monotonic-(capture_time_s+offset_s)` and
remains `null` until at least **3 distinct** clock samples (≤10 s old, RTT ≤100 ms)
agree within 20 ms. Receiver uses the lowest-RTT recent sample and reports
uncertainty as RTT/2 plus observed offset spread. LAN asymmetry, OS scheduling
and camera presentation timestamps limit interpretation; this is a clock
alignment estimate, not photon-to-motion latency. Without alignment the UI
labels **reception age** only. Depth/hand pose expires after 500 ms without an
accepted sensor packet and becomes invalid immediately on sensor disconnect.

## Public pose output: `handdepth.pose.v1`

One finite JSON object per line in `--pose-jsonl` or recording `poses.jsonl`.
Coordinates are in **metres**, optical camera **X right, Y down, Z forward**.
There is no world/robot frame. Typical fields:

```json
{"schema":"handdepth.pose.v1","protocol_version":1,"session_id":"...","frame_id":42,
 "capture_time_s":100.0,"clock":"ios_host_monotonic_seconds","calibration_id":"...",
 "coordinate_frame":"camera_optical_x_right_y_down_z_forward",
 "tracking_valid":true,"depth_valid":true,"orientation_valid":true,
 "palm_position_m":[0.01,0.02,0.5],"palm_quaternion_xyzw":[0,0,1,0],
 "pinch_distance_m":0.025,"depth_filtered":false,"depth_quality":"high",
 "capture_to_receive_s":null,"clock_uncertainty_s":null,
 "measurement_age_basis":"receiver_arrival_only","received_monotonic_s":1000.01,
 "reason":"measured_visible_surfaces","landmarks":[]}
```

The numbers above are illustrative, not a hardware measurement.

Since application 0.2.0, each frame additionally contains `hands` and
`primary_hand_id`. **Consume `hands` for both hands**. Every entry contains:

```json
{"hand_id":"hand-1","chirality":"left","tracking_valid":true,
 "depth_valid":true,"orientation_valid":true,
 "palm_position_m":[0.01,0.02,0.5],"palm_quaternion_xyzw":[0,0,1,0],
 "pinch_distance_m":0.025,"reason":"measured_visible_surfaces","landmarks":[]}
```

The 21 reconstructed landmarks and palm/pinch/validity fields belong to that
hand only. Source frame, timestamps, calibration, depth metadata and clock age
are common frame-level fields. Rectification runs once per synchronized depth
map; both sets of 2D landmarks sample it independently. Quaternion history is
separate per `hand_id`, never shared between hands. Missing hands are removed
immediately and their orientation history is cleared. No depth from one hand
fills holes in the other.

Frame-level `landmarks`, `chirality`, `tracking_valid`, `depth_valid`,
`orientation_valid`, palm, pinch and reason remain **first-hand aliases** for
older consumers, rather than aggregate validity. `primary_hand_id` identifies
that hand. An empty `hands` clears all aliases; timeout/disconnect likewise
emits `hands:[]` and `primary_hand_id:null`. Old receivers/consumers display only
the compatibility primary hand; update the receiver to show both.

`landmarks` retains all detector points, including when calibration or absolute
depth is unavailable: `{name,confidence,xyz_m,valid,sigma_z_m,sigma_xyz_m,reason,sample_count}`.
Invalid XYZ/uncertainty are `null`; no extrapolation fills invisible fingers.
A 5×5 rectified neighborhood requires valid center, ≥8 valid samples, values ≤4 m,
and rejects 10–90 percentile span >max(20 mm,4% depth) or center/median discrepancy
>max(15 mm,3% depth). Landmark confidence threshold is 0.4. Median depth supplies Z.
MAD×1.4826 with 0.5 mm floor describes local Z spread; lateral uncertainty also
includes a one-pixel footprint. `uncertainty_model` states this **excludes
systematic sensor error**, tissue/joint offset and detector bias; it is not a
certified accuracy bound.

When ≥3 valid wrist/MCP surface samples exist for a hand, a robust median palm
anchor additionally rejects samples more than **0.35 m** away, reporting
`hand_surface_outlier` and null XYZ. This broad plausibility bound rejects a
smooth background mistakenly sampled at a fingertip; it does not infer hidden
joints or certify anatomical correctness. Without enough anchors only local
sampling checks are available. The bound is evaluated independently per hand.

Palm position averages wrist and all four finger MCP surface samples; all five
must be valid. Palm axes expressed in camera coordinates:
X = normalized littleMCP→indexMCP;
Y = wrist→middleMCP with its X component removed;
Z = X×Y. Quaternion **[x,y,z,w]** maps palm axes into the public optical frame.
Wrist→middle and palm width must exceed 15 mm; orthogonal residual must exceed
12 mm. Degenerate geometry returns null orientation. Quaternion sign follows the
previous valid result (`q` and `-q` are equivalent). Chirality changes/loss reset
continuity; >120° rotation jump within 200 ms is invalidated for that sample.
Pinch distance is the Euclidean distance between valid thumbTip/indexTip surface
samples, not an anatomical joint-center distance.

`depth_valid` means at least one valid visible surface point; palm/orientation/
pinch can independently be null. `tracking_valid` means at least one ≥0.4
confidence detector point. On timeout/disconnect an additional `receiver_event:
true` line clears all movement fields, with the last source identity retained.
A future Mink adapter must require current source identity, age, calibration and
per-field validity. This repository never opens a robot connection.

## Body and torso-relative output

This additive `handdepth.pose.v1` extension is present in application **0.3.0**.
Frame identity/timestamps/calibration/age remain shared with the hand measurements.
`body:null` means no current single-operator observation. When present:

| Field | Meaning |
|---|---|
| `body.tracking_valid`, `depth_valid` | At least one confident detector point / at least one measured body surface point; not full-body validity |
| `body.landmarks` | All 19 entries with name/confidence, `xyz_m`, `xyz_torso_m`, `valid`, local uncertainty, sample count and reason |
| `body.torso_surface` | Sample result at the derived image site, with explicit source string ending `_measured_depth`; null when no site is available |
| `body.torso` | `position_m`, `quaternion_xyzw`, independent `position_valid` / `orientation_valid`, frame/origin/axis-source definitions |
| `body.head` | `position_m`, `quaternion_xyzw`, independent validity, `position_torso_m`, `quaternion_torso_xyzw` and `torso_relative_position_valid` / `torso_relative_orientation_valid` |
| `body.arms.left/right` | Dictionaries described below; can be empty on calibration failure |
| `hands[].body_arm_side` | `left`, `right` or null; current anatomical arm association |
| `hands[].palm_position_torso_m`, `palm_quaternion_torso_xyzw` | Measured palm pose relative to observed torso; independent `torso_relative_position_valid` / `torso_relative_orientation_valid` |
| `hands[].landmarks[].xyz_torso_m` | Each valid surface point transformed into the observed torso frame; null if torso orientation is unavailable |

**Camera frame:** all `xyz_m` / `position_m` and camera quaternions remain in the
public optical camera frame X right, Y down, Z forward, in metres. There is no
invented world frame, robot frame or neutral operator frame.

**Torso frame:** origin O = mean measured left/right shoulder positions.
X = normalized leftShoulder→rightShoulder (operator anatomical right).
Y = O→measured torso surface with its X component removed, normalized.
Z = X×Y (operator forward for an upright front-facing operator).
Rotation R has these axes as columns, so its **[x,y,z,w]** quaternion maps torso
vectors into camera coordinates. Require shoulder width 0.15–0.8 m, torso surface
distance from O 0.05–0.65 m and orthogonal Y residual ≥0.04 m. Degeneracy, missing
prerequisites or surface outliers return null orientation, preserving valid
shoulder-midpoint position independently. This is an observed surface frame,
not an anatomical skeleton fit or proof of correct torso orientation.

For camera point P: `P_torso = Rᵀ(P-O)`. For a measured head/palm camera rotation
A: `A_torso = Rᵀ A`. Relative quaternions also use **[x,y,z,w]** and map the
head/palm axes into torso axes. Common translation/rotation cancels; synthetic
geometry tests verify this. If torso orientation is missing, **all relative
fields are null**, even when their camera-space measurements remain valid.

**Head frame:** position = measured nose surface. X = leftEye→rightEye;
Y = eye-midpoint→nose with its X component removed; Z = X×Y. Eye separation must
be 0.025–0.14 m and nose distance from eye midpoint 0.012–0.14 m, with orthogonal
residual ≥0.012 m. This **facial surface landmark frame approximates head motion**;
its axes have a face-geometry-dependent tilt and are not a calibrated anatomical
cranial quaternion. Neutral-pose mapping is needed before robot head retargeting.
Both eyes and nose must be measured; profile/occlusion may invalidate orientation
while nose position remains available. No head orientation is inferred from a
single face point. Nose/eye/ear samples use a 3×3 neighborhood with ≥5 samples;
other points retain the 5×5 / ≥8 policy, same center/edge/uncertainty checks.

**Arms:** each anatomical side includes:

- `positions_camera_m` / `positions_torso_m`: dictionary keyed by
  `leftShoulder,leftElbow,leftWrist` or right equivalents. Each value independently nullable.
- `measurement_valid`: all three measured endpoints valid. `torso_relative_valid`
  additionally requires the torso frame. `upper_arm_length_m` / `forearm_length_m`
  and `elbow_flexion_rad` require all three endpoints; straight elbow = 0 rad.
- `upper_arm_direction_camera` / `forearm_direction_camera` and corresponding
  `_torso`: unit vectors shoulder→elbow and elbow→wrist. No full segment quaternion
  is invented; **`axial_twist_observable:false`** because endpoint locations do
  not measure rotation about the limb's long axis.
- `hand_id`: current matching hand or null. Bounded one-to-one current-frame 2D
  wrist matching uses confidence ≥0.4, separation ≤0.14 normalized units and a
  0.05 chirality mismatch penalty. If best/second-best assignment costs differ
  by <0.025, neither side is assigned. Array order never determines anatomical side.

Body samples farther than 0.65 m (head/neck) or 2 m (other landmarks) from a valid
shoulder midpoint are rejected; torso surface limit 0.7 m. Upper-arm / forearm
lengths outside 0.03–0.75 m reject the distal endpoint. These broad plausibility
checks can reject smooth background samples but do not guarantee occlusion
recognition, anatomical accuracy or suitability for direct joint control.

Torso, head and each relative head/hand quaternion have independent sign
continuity and >120° / <200 ms jump rejection. Missing prerequisites, calibration
change, loss, sensor disconnect or timeout clear histories. No stale body/arm/head
measurements survive invalidation events. All measurements refer to visible
skin/clothing surfaces; depth cannot see internal joints or hidden limbs.

## Recording: HDREC1

Session directory under the checked recording root:

- `metadata.json`: application/protocol versions, format, UTC creation time, receiver clock, fixture-storage status.
- `packets.hdr`: magic ASCII `HDREC1\n`, then records. Each record is a **13-byte** big-endian prefix `>BdI`: channel UInt8 (0 sensor,1 preview), receive-time Float64 seconds (PC monotonic), packet length UInt32; followed by the exact WebSocket bytes. Includes calibration, raw depth, body/hand landmarks and JPEGs with source identities. Truncated/corrupt records fail replay.
- `poses.jsonl`: derived pose and validity for accepted sensor frames and explicit timeout/disconnect invalidation events; receive timestamps.
- `calibration/<sha256>.json`: calibration objects and original session/calibration IDs; filenames hash identity to avoid path injection.

Replay reconstructs poses from original packets with recorded clock/receive
metadata, preserving unmatched RGB/depth IDs. It can run headless and write a new
pose JSONL or dashboard snapshot. Logs contain only accepted frames, so drops are
observable as frame-ID gaps. Abnormal termination can leave a partial final record;
replay reports truncation instead of silently fabricating a final frame.

## Simulation adapter (receiver 0.4.0)

The phone remains compatible with protocol v1 and `handdepth.pose.v1`. The
in-process simulation consumer reads one latest immutable pose snapshot under
the receiver lock; it has no FIFO movement queue. Sensor ACKs still follow
measurement processing, independently of preview delivery.

`body.arms[side].wrist_source` identifies `vision_body_wrist_surface` or
`associated_vision_hand_wrist_surface`. If a body's 2D wrist is low confidence,
its position with confidence ≥0.2 can be used only as an association hint. An
associated, same-frame hand wrist with confidence ≥0.4 and valid measured depth
can supply the arm endpoint. Original body landmark validity is preserved;
missing or ambiguous associations do not supply a wrist. Segment plausibility
checks apply. This uses measured hand surfaces, without synthesizing a joint.

`handdepth.teleop.v1` is the **local simulation status/trajectory JSON schema**,
not a new iPhone wire packet. `source_session_id`, `source_frame_id` identify the
latest measurement; `last_motion_frame_id` identifies the last accepted servo
update. State is `tracking` or `holding`; `active_parts` names valid current
parts. Joint names accompany `ctrl`, actual model `qpos` is in MuJoCo joint
order, angles are radians. Position errors are metres; speed is rad/s;
`receiver_monotonic_s` uses the Ubuntu process clock. Configured rates and
`achieved_control_hz`/`sensor_source_hz` are distinguished. Neutral calibration
files record the source pose and retargeting scales. No simulation coordinates
replace the public optical camera-frame measurements.

The `static-torso-v1` mapping also reports `torso_mode: fixed`, fixed/current
torso joint values, maximum torso displacement/rotation, base displacement,
per-part hold reasons, segment scales, pinch distances, orientation errors,
IK failures and physics warning counts. `pose_overlay` describes the local
visualization (enabled/status, fixed origin in model metres, scale 1.0, geometry
count). It changes neither the wire protocol nor measured pose validity. The
human torso-relative skeleton shares the robot torso origin; robot targets use
neutral offsets and segment scaling separately. Neutral files keep the acquired
arm/head/palm baselines across ordinary tracking loss and hand-ID changes.

Display continuity adds `pose_overlay.history` and `dashboard_display` status
(observed/held counts, ages, metric arm lengths and acquisition validity,
held/inferred arm modes, wrist clamping, and RGB pairing). `inferred_control`
is false. Pose JSON adds `display_geometry_id`, `color_dimensions`, and
`depth_dimensions` to define compatible display-history coordinates. These
local fields are not new iPhone wire requirements. The phone's `body_tracking`
pixel constraints stay distinct from measured XYZ and inferred display XYZ.

See [TELEOP.md](docs/SETUP.md) for the exact mapping, motion bounds, clutch/recovery
policy, head/gripper limitations, CLI, replay and HDD storage conventions.
