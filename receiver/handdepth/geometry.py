"""TrueDepth registration and rectification in camera optical X right/Y down/Z forward.

AVDepthData is already perspective aligned to the YUV camera, with equal FOV
and aspect ratio. Never apply the raw calibration extrinsic a second time.
LUT direction follows Apple's AVCameraCalibrationData.h reference implementation.
"""
import math
import hashlib
import json
import numpy as np
from .protocol import ProtocolError, dimensions, sensor_hands

class CalibrationError(ValueError):
    pass


def matrix(value, shape):
    try:
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as e:
        raise CalibrationError("invalid matrix") from e
    if a.shape != shape or not np.isfinite(a).all():
        raise CalibrationError("matrix shape/values")
    return a

class Calibration:
    def __init__(self, raw, color_dimensions, depth_dimensions):
        if not isinstance(raw, dict) or raw.get("alignment") != "truedepth_color_fov":
            raise CalibrationError("missing TrueDepth registration")
        try:
            self.ref = np.array(dimensions(raw["reference_dimensions"], max_dimension=16384), dtype=float)
            self.color = np.array(dimensions(color_dimensions), dtype=float)
            self.depth = np.array(dimensions(depth_dimensions), dtype=float)
            self.k = matrix(raw["intrinsics_row_major"], (3, 3))
            self.center = matrix(raw["distortion_center"], (2,))
            self.forward = np.asarray(raw["lens_distortion_lut"], dtype=float)
            self.inverse = np.asarray(raw["inverse_lens_distortion_lut"], dtype=float)
            self.color_from_ref = matrix(raw["color_from_reference"], (3, 3))
            self.depth_from_ref = matrix(raw["depth_from_reference"], (3, 3))
            matrix(raw["extrinsics_camera_to_reference_row_major_mm"], (3, 4))
        except (KeyError, TypeError, ValueError, ProtocolError) as e:
            raise CalibrationError("missing or invalid calibration") from e
        if self.k[0, 0] <= 0 or self.k[1, 1] <= 0 or not np.allclose(self.k[2], [0, 0, 1]):
            raise CalibrationError("invalid intrinsics")
        for lut in (self.forward, self.inverse):
            if lut.ndim != 1 or not 2 <= len(lut) <= 4096 or not np.isfinite(lut).all() or np.any(lut <= -1) or np.any(np.abs(lut) > 2):
                raise CalibrationError("distortion tables required")
        if not np.all((self.center >= 0) & (self.center <= self.ref)):
            raise CalibrationError("distortion center outside reference")
        # This version deliberately supports uncropped native TrueDepth frames only.
        for size, transform in ((self.color, self.color_from_ref), (self.depth, self.depth_from_ref)):
            if abs(size[0]/size[1] - self.ref[0]/self.ref[1]) > 0.01:
                raise CalibrationError("FOV/aspect mismatch")
            if not np.allclose(transform, np.diag([*(size/self.ref), 1]), atol=1e-6):
                raise CalibrationError("unsupported crop/rotation")
        self.k_depth = self.depth_from_ref @ self.k
        self.inv_k_depth = np.linalg.inv(self.k_depth)
        self.radius_max = np.linalg.norm(np.maximum(self.center, self.ref-self.center))
        # Build a bounded depth-resolution inverse resampling map ONCE per calibration.
        w, h = self.depth.astype(int)
        yy, xx = np.mgrid[:h, :w]
        rect_ref = np.stack([xx+0.5, yy+0.5], axis=-1) * self.ref/self.depth
        distorted_ref = self.map_points(rect_ref, self.forward)
        source = distorted_ref * self.depth/self.ref - 0.5
        self.map_x = np.rint(source[..., 0]).astype(int)
        self.map_y = np.rint(source[..., 1]).astype(int)
        self.inside = (self.map_x >= 0) & (self.map_x < w) & (self.map_y >= 0) & (self.map_y < h)
        self.map_x = self.map_x.clip(0, w-1)
        self.map_y = self.map_y.clip(0, h-1)

    def map_points(self, points, lut):
        points = np.asarray(points, dtype=float)
        delta = points-self.center
        radius = np.linalg.norm(delta, axis=-1)
        t = np.minimum(radius/self.radius_max, 1)*(len(lut)-1)
        i = np.floor(t).astype(int)
        f = t-i
        mag = lut[i]*(1-f) + lut[np.minimum(i+1, len(lut)-1)]*f
        return self.center + delta*(1+mag[..., None])

    def rectify(self, depth):
        if depth.shape != tuple(self.depth.astype(int)[::-1]):
            raise CalibrationError("depth shape mismatch")
        # Nearest-neighbor: interpolation between background/hand depths invents surfaces.
        return np.where(self.inside, depth[self.map_y, self.map_x], 0).astype(np.uint16)

    def landmark_pixel(self, x, y):
        # Input normalized, top-left, image edge coordinates. Output rectified depth edges.
        return self.map_points(np.array([x, y])*self.ref, self.inverse)*self.depth/self.ref

    def unproject(self, pixel, z):
        ray = self.inv_k_depth @ np.array([pixel[0], pixel[1], 1.])
        return ray*(z/ray[2])

    def project_color(self, xyz):
        """Camera metric XYZ -> original, distorted normalized RGB coordinates.

        This reverses the registered depth ray, including the forward lens LUT;
        it is not a linear resize of rectified depth pixels onto the RGB image.
        """
        xyz=np.asarray(xyz,dtype=float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all() or xyz[2]<=0:
            return None
        ray=self.k@xyz
        return self.map_points(ray[:2]/ray[2],self.forward)/self.ref


def sample_surface(calibration, rectified, landmark, radius=2, minimum_samples=8):
    result = {"name": landmark["name"], "confidence": landmark["confidence"], "xyz_m": None,
              "valid": False, "sigma_z_m": None, "sigma_xyz_m": None, "reason": "low_confidence", "sample_count": 0}
    if landmark["confidence"] < 0.4 or landmark["x"] is None or landmark["y"] is None:
        return result
    p = calibration.landmark_pixel(landmark["x"], landmark["y"])
    x, y = np.floor(p).astype(int)  # pixel centers are i+0.5; edges span [0,width]
    h, w = rectified.shape
    if x < radius or y < radius or x >= w-radius or y >= h-radius:
        result["reason"] = "image_boundary"
        return result
    tile = rectified[y-radius:y+radius+1, x-radius:x+radius+1].astype(float)/1000
    values = tile[(tile > 0) & (tile <= 4)]
    result["sample_count"] = len(values)
    if rectified[y, x] == 0 or len(values) < minimum_samples:
        result["reason"] = "insufficient_depth"
        return result
    median = float(np.median(values))
    # Reject an edge, rather than selecting whichever cluster dominates the window.
    spread = float(np.percentile(values, 90)-np.percentile(values, 10))
    if spread > max(0.020, median*0.04) or abs(tile[radius, radius]-median) > max(0.015, median*0.03):
        result["reason"] = "depth_edge"
        return result
    sigma = max(0.0005, float(np.median(np.abs(values-median)))*1.4826)
    xyz = calibration.unproject(p, median)
    footprint = median/min(calibration.k_depth[0, 0], calibration.k_depth[1, 1])
    result.update(xyz_m=xyz.tolist(), valid=True, sigma_z_m=sigma,
                  sigma_xyz_m=[math.hypot(sigma*xyz[0]/median, footprint), math.hypot(sigma*xyz[1]/median, footprint), sigma],
                  reason="surface_sample")
    return result


def quaternion_xyzw(r):
    # Stable conversion near 180 degrees; rotation columns are palm axes in optical frame.
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t+1)*2
        q = np.array([(r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s, s/4])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i+1)%3, (i+2)%3
        s = math.sqrt(max(0, 1+r[i,i]-r[j,j]-r[k,k]))*2
        q = np.zeros(4)
        q[i] = s/4; q[j] = (r[j,i]+r[i,j])/s; q[k] = (r[k,i]+r[i,k])/s
        q[3] = (r[k,j]-r[j,k])/s
    return q/np.linalg.norm(q)

class HandPoseState:
    def __init__(self):
        self.previous_q = None
        self.previous_chirality = None
        self.previous_capture = None


class PoseEstimator:
    def __init__(self):
        from .body import BodyPoseEstimator
        self.calibration_id = None
        self.calibration = None
        self.hand_states = {}
        self.body_estimator = BodyPoseEstimator()
        self.display_geometry_id = None

    def reset(self):
        self.__init__()

    def reset_orientations(self):
        self.hand_states.clear()
        self.body_estimator.reset()

    def estimate(self, header, depth, pc_pose=None):
        if pc_pose is not None and any(pc_pose.get(k)!=header.get(k) for k in
                ('session_id','calibration_id','frame_id','capture_time_s','color_dimensions')):
            raise ValueError('PC pose does not match RGB/depth source frame')
        detections = sensor_hands(header)
        # Forget absent hands immediately: reappearance must not reuse stale motion.
        present = {h["hand_id"] for h in detections}
        self.hand_states = {k:v for k,v in self.hand_states.items() if k in present}
        out = {"schema": "handdepth.pose.v1", "protocol_version": 1, "session_id": header["session_id"],
               "frame_id": header["frame_id"], "capture_time_s": header["capture_time_s"], "clock": header["clock"],
               "calibration_id": header["calibration_id"], "coordinate_frame": "camera_optical_x_right_y_down_z_forward",
               "depth_filtered": header["depth_filtered"], "depth_quality": header["depth_quality"],
               "uncertainty_model": "local_spread_and_pixel_footprint_excludes_systematic_sensor_error"}
        # Display history resets on actual projection/coordinate changes, not
        # merely a sensor-rate/filter setting included in calibration_id.
        if header.get('calibration') is not None:
            geometry = {k:header.get(k) for k in ('calibration','color_dimensions','depth_dimensions')}
            geometry_id = hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest()
            if self.display_geometry_id is not None and geometry_id!=self.display_geometry_id:
                self.calibration=None;self.reset_orientations()
            self.display_geometry_id = geometry_id
        out.update(display_geometry_id=self.display_geometry_id or header['calibration_id'],
                   color_dimensions=header['color_dimensions'],depth_dimensions=header['depth_dimensions'])
        error = None
        try:
            if self.calibration_id != header["calibration_id"] or self.calibration is None:
                self.calibration = Calibration(header.get("calibration"), header["color_dimensions"], header["depth_dimensions"])
                self.calibration_id = header["calibration_id"]
                self.reset_orientations()
            # Registration/rectification is shared by both hands from this source frame.
            rect = self.calibration.rectify(depth)
        except (CalibrationError, np.linalg.LinAlgError) as e:
            error = "calibration_invalid: " + str(e)
            self.reset_orientations()
            rect = depth
        if error is None and header["depth_accuracy"] != "absolute":
            error = "relative_depth_not_metric"
            self.reset_orientations()
        hands = [self._estimate_hand(h, header["capture_time_s"], rect, error) for h in detections]
        from .hand_control import finger_curl
        for raw,hand in zip(detections,hands):
            hand['grip_gesture']=dict(finger_curl(raw,header['color_dimensions']),
                source_frame_id=header['frame_id'],capture_time_s=header['capture_time_s'])
        out["hands"] = hands
        out["primary_hand_id"] = hands[0]["hand_id"] if hands else None
        # Existing pose consumers/recordings keep a primary-hand compatibility view.
        primary = hands[0] if hands else self._empty_hand({"hand_id":None,"chirality":"unknown","landmarks":[]})
        for key in ("landmarks","tracking_valid","depth_valid","orientation_valid","palm_position_m", "palm_quaternion_xyzw", "pinch_distance_m","reason","chirality"):
            out[key] = primary[key]
        if "orientation_rejection" in primary: out["orientation_rejection"] = primary["orientation_rejection"]
        out["body"] = self.body_estimator.estimate(header.get("body"),self.calibration,rect,error,header["capture_time_s"],hands,detections,header.get('body_tracking'),header['frame_id'],pc_pose)
        if out['body'] and 'limb_tracking' in out['body']:
            out['body']['limb_tracking'].update({k:out[k] for k in ('session_id','calibration_id','display_geometry_id')})
        if pc_pose is not None:out['pc_pose']=pc_pose
        out["body_error"] = header.get("body_error")
        # Explicit pixel-space continuity channel. Do not unproject an inferred
        # elbow through the occluding hand's depth or change measured validity.
        from copy import deepcopy
        out["body_tracking"] = deepcopy(header.get("body_tracking"))
        return out, rect

    @staticmethod
    def _empty_hand(hand):
        return {"hand_id":hand["hand_id"], "chirality":hand["chirality"],
                "landmarks":[{"name":l["name"], "confidence":l["confidence"], "xyz_m":None, "valid":False,
                    "sigma_z_m":None,"sigma_xyz_m":None,"reason":"calibration_invalid","sample_count":0} for l in hand["landmarks"]],
                "tracking_valid":any(l["confidence"]>=.4 and l["x"] is not None for l in hand["landmarks"]),
                "depth_valid":False,"orientation_valid":False,"palm_position_m":None,
                "palm_quaternion_xyzw":None,"pinch_distance_m":None,"reason":"no_valid_surface_samples"}

    def _estimate_hand(self, hand, capture, rect, error):
        out = self._empty_hand(hand)
        if error:
            out["reason"] = error
            for point in out["landmarks"]: point["reason"] = error
            return out
        state = self.hand_states.setdefault(hand["hand_id"], HandPoseState())
        if state.previous_capture is not None and not 0 <= capture-state.previous_capture <= .5:
            state.previous_q = None
        out["landmarks"] = [sample_surface(self.calibration, rect, l) for l in hand["landmarks"]]
        # A confidently located fingertip can still fall onto a smooth background
        # patch. Local edge tests cannot detect that: require a plausible hand-sized
        # distance from a robust palm surface anchor when at least three exist.
        palm = ["wrist", "indexMCP", "middleMCP", "ringMCP", "littleMCP"]
        anchors = [l["xyz_m"] for l in out["landmarks"] if l["valid"] and l["name"] in palm]
        if len(anchors) >= 3:
            anchor = np.median(anchors, axis=0)
            for landmark in out["landmarks"]:
                if landmark["valid"] and np.linalg.norm(np.array(landmark["xyz_m"])-anchor) > .35:
                    landmark.update(valid=False, xyz_m=None, sigma_z_m=None, sigma_xyz_m=None, reason="hand_surface_outlier")
        points = {l["name"]: np.array(l["xyz_m"]) for l in out["landmarks"] if l["valid"]}
        out["depth_valid"] = bool(points)
        orientation_anchors=('wrist','indexMCP','littleMCP')
        if all(n in points for n in orientation_anchors):
            out["palm_position_m"] = np.mean([points[n] for n in palm if n in points], axis=0).tolist()
            x = points["indexMCP"] - points["littleMCP"]
            # Use a consistent measured triangle. Ring/middle depth availability
            # must not gate the wrist or switch its reference axis between frames.
            y = (points["indexMCP"]+points["littleMCP"])/2 - points["wrist"]
            nx, ny = np.linalg.norm(x), np.linalg.norm(y)
            if .015 < nx < .18 and .015 < ny < .23:
                x /= nx
                y -= np.dot(x, y)*x
                if np.linalg.norm(y) > .012:
                    y /= np.linalg.norm(y)
                    z = np.cross(x, y)
                    q = quaternion_xyzw(np.column_stack((x, y, z)))
                    chirality = hand["chirality"]
                    if chirality != state.previous_chirality: state.previous_q = None
                    if state.previous_q is not None and np.dot(q, state.previous_q) < 0:
                        q = -q
                    jump = state.previous_q is not None and state.previous_capture is not None and 0 < capture-state.previous_capture < .2 and abs(np.dot(q, state.previous_q)) < .5
                    if not jump:
                        out["palm_quaternion_xyzw"] = q.tolist()
                        out["orientation_valid"] = True
                        out['orientation_source']='measured_wrist_index_little_mcp'
                        state.previous_q = q
                        state.previous_chirality = chirality
                        state.previous_capture = capture
                    else:
                        out["orientation_rejection"] = "abrupt_orientation_flip"
        if not out["orientation_valid"]: state.previous_q = None
        if "thumbTip" in points and "indexTip" in points:
            out["pinch_distance_m"] = float(np.linalg.norm(points["thumbTip"]-points["indexTip"]))
        out["reason"] = "measured_visible_surfaces" if out["depth_valid"] else "no_valid_surface_samples"
        return out
