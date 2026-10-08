# Apple calibration evidence (inspected 2026-10-07)

Xcode 27.0 (27A266a), iOS SDK 27.0. The implementation targets iOS 17.0+.

Primary sources:

- [TrueDepth device](https://developer.apple.com/documentation/avfoundation/avcapturedevice/devicetype-swift.struct/builtintruedepthcamera): IR depth is perspective corrected to YUV, with matching FOV/aspect even if resolutions differ. Shared image-plane registration uses normalized coordinates. Do not apply a second IR-to-color rigid transform.
- [AVDepthData](https://developer.apple.com/documentation/avfoundation/avdepthdata): depth maps retain accompanying image lens distortion and require calibration rectification for 3D.
- [Intrinsics](https://developer.apple.com/documentation/avfoundation/avcameracalibrationdata/intrinsicmatrix): pixel focal lengths/principal point; origin is image upper left, samples lie at pixel centers. Scale from the supplied reference dimensions.
- [Extrinsics](https://developer.apple.com/documentation/avfoundation/avcameracalibrationdata/extrinsicmatrix): camera-to-reference-camera pose; native SIMD has 3 rows / 4 columns and is column major. Rotation is unitless, translation is millimetres. We serialize by explicit row/column indexing and retain it as provenance, without treating the Apple reference camera as a world frame.
- [Lens LUT](https://developer.apple.com/documentation/avfoundation/avcameracalibrationdata/lensdistortionlookuptable), [inverse LUT](https://developer.apple.com/documentation/avfoundation/avcameracalibrationdata/inverselensdistortionlookuptable), [center](https://developer.apple.com/documentation/avfoundation/avcameracalibrationdata/lensdistortioncenter): radial model centered at the distortion center, which can differ from the principal point.
- [Connection rotation](https://developer.apple.com/documentation/avfoundation/avcaptureconnection/videorotationangle): rotation angles apply to both video and depth outputs. Both connections explicitly use zero degrees; mirroring/stabilization are disabled.
- [Session clock](https://developer.apple.com/documentation/avfoundation/avcapturesession/synchronizationclock): capture timestamps use this clock. We explicitly convert to the CoreMedia host clock before transmission.
- [Vision hand detection](https://developer.apple.com/documentation/vision/vndetecthumanhandposerequest): existing native detector; [`maximumHandCount`](https://developer.apple.com/documentation/vision/vndetecthumanhandposerequest/maximumhandcount) is explicitly set to **2** in version 0.2.0; request orientation `.up` and top-left conversion `(x,1-y)`.

## Body detector source

Version 0.3.0 adds Apple's [VNDetectHumanBodyPoseRequest](https://developer.apple.com/documentation/vision/vndetecthumanbodyposerequest)
and [19 supported joint names](https://developer.apple.com/documentation/vision/vnhumanbodyposeobservation/jointname).
The current official Markdown pages were inspected on 2026-10-07: the API is
available from iOS 14 and returns body observations; `root` denotes the waist.
The app uses the 2D request with `.up` on the same RGB buffer as hand detection.
The PC obtains metric visible-surface samples from synchronized calibrated depth,
not from an inferred 3D body-model scale. Body/head/torso frame construction is
our explicitly documented surface geometry, not an Apple anatomical pose guarantee.

## Resolving LUT direction precisely

The property descriptions discuss image operations, whose pull-resampling direction differs from a point transformation. The installed SDK's reference implementation and its preceding comments resolve the mapping:

`/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/System/Library/Frameworks/AVFoundation.framework/Headers/AVCameraCalibrationData.h`, lines 122–188 (iOS SDK 27.0).

- Rectified output pixel → original distorted input pixel: `lensDistortionLookupTable`.
- Original distorted landmark point → rectified point: `inverseLensDistortionLookupTable`.
- Entries are relative radial magnification: radial vector is multiplied by `1 + interpolated_entry`; zero is identity.
- LUT samples span zero to the farthest image corner, using `max(center.x, width-center.x)` and the corresponding Y quantity.
- Image size, point, and center must share a resolution. We work in reference pixels, then scale to actual dimensions.

`geometry.py` implements these directions separately. A nonzero analytical LUT fixture verifies direction, inverse mapping, nearest-neighbor rectification and out-of-FOV holes. Missing LUTs are unknown distortion, never assumed zero. Resampling preserves invalid holes and avoids interpolation across surfaces. Synthetic geometry validates the implementation; hardware accuracy still needs the physical target tests.
