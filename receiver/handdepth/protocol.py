"""Strict, bounded v1 binary framing. No implicit pickle or NaN JSON."""
import json
import math
import struct
from dataclasses import dataclass
import numpy as np

VERSION = 1
MAX_HEADER = 65_536
MAX_PAYLOAD = 2_097_152
MAX_MESSAGE = 4 + MAX_HEADER + MAX_PAYLOAD
JOINTS = ["wrist", "thumbCMC", "thumbMP", "thumbIP", "thumbTip",
          "indexMCP", "indexPIP", "indexDIP", "indexTip",
          "middleMCP", "middlePIP", "middleDIP", "middleTip",
          "ringMCP", "ringPIP", "ringDIP", "ringTip",
          "littleMCP", "littlePIP", "littleDIP", "littleTip"]
BODY_JOINTS = ["nose", "neck", "root", "leftEye", "rightEye", "leftEar", "rightEar",
               "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
               "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]

class ProtocolError(ValueError):
    pass

@dataclass(frozen=True)
class Packet:
    header: dict
    payload: bytes


def _bad_constant(value):
    raise ProtocolError("non-finite JSON: " + value)

def _finite_float(value):
    f = float(value)
    if not math.isfinite(f):
        raise ProtocolError("non-finite JSON number")
    return f

def _unique_object(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ProtocolError("duplicate JSON key")
        d[k] = v
    return d

def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError("invalid " + name)
    return value

def integer(value, name, lo=0, hi=2**53-1):
    if type(value) is not int or not lo <= value <= hi:
        raise ProtocolError("invalid " + name)
    return value

def dimensions(value, max_dimension=2048):
    if not isinstance(value, list) or len(value) != 2:
        raise ProtocolError("dimensions must be [width,height]")
    return tuple(integer(v, "dimension", 1, max_dimension) for v in value)

def validate_landmarks(landmarks, allow_empty=True, names=JOINTS):
    if not isinstance(landmarks, list) or len(landmarks) not in ((0,len(names)) if allow_empty else (len(names),)):
        raise ProtocolError("unexpected landmark count")
    for j, point in zip(names, landmarks):
        if not isinstance(point, dict) or point.get("name") != j:
            raise ProtocolError("joint order")
        if not 0 <= finite(point.get("confidence"), "confidence") <= 1:
            raise ProtocolError("confidence range")
        for axis in ("x", "y"):
            if point.get(axis) is not None and not 0 <= finite(point[axis], axis) <= 1:
                raise ProtocolError("landmark range")
        if (point.get("x") is None) != (point.get("y") is None):
            raise ProtocolError("partial landmark")

def sensor_hands(header):
    """Additive v1 hands array; legacy recordings retain their single-hand data."""
    if "hands" in header:
        return header["hands"]
    return [{"hand_id":"legacy-primary", "chirality":header.get("chirality","unknown"),
             "landmarks":header["landmarks"]}] if header["landmarks"] else []

def encode(header, payload=b""):
    h = dict(header, protocol_version=VERSION, payload_length=len(payload))
    try:
        raw = json.dumps(h, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    except (TypeError, ValueError) as e:
        raise ProtocolError(str(e)) from e
    if not 0 < len(raw) <= MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ProtocolError("message too large")
    return struct.pack(">I", len(raw)) + raw + payload

def decode(data):
    if not isinstance(data, bytes) or not 4 <= len(data) <= MAX_MESSAGE:
        raise ProtocolError("binary message size")
    n, = struct.unpack(">I", data[:4])
    if not 0 < n <= MAX_HEADER or 4+n > len(data):
        raise ProtocolError("header length")
    try:
        h = json.loads(data[4:4+n].decode("utf-8"), parse_constant=_bad_constant, parse_float=_finite_float, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as e:
        raise ProtocolError("invalid JSON") from e
    if not isinstance(h, dict) or h.get("protocol_version") != VERSION or type(h.get("protocol_version")) is not int:
        raise ProtocolError("protocol version")
    payload = data[4+n:]
    if integer(h.get("payload_length"), "payload_length", hi=MAX_PAYLOAD) != len(payload):
        raise ProtocolError("payload length mismatch")
    kind = h.get("kind")
    if kind not in {"sensor", "preview", "clock_ping", "clock_pong", "clock_sync", "heartbeat", "ack"}:
        raise ProtocolError("unknown packet kind")
    if not isinstance(h.get("session_id"), str) or not 1 <= len(h["session_id"]) <= 64:
        raise ProtocolError("session_id")
    if kind in {"sensor", "preview"}:
        integer(h.get("frame_id"), "frame_id")
        finite(h.get("capture_time_s"), "capture_time_s")
        if h.get("clock") != "ios_host_monotonic_seconds":
            raise ProtocolError("capture clock")
        finite(h.get("processing_start_s"), "processing_start_s")
        finite(h.get("processing_end_s"), "processing_end_s")
        if not h["capture_time_s"] <= h["processing_start_s"] <= h["processing_end_s"]:
            raise ProtocolError("processing timestamps")
        if h.get("orientation") != "native_up" or h.get("mirrored") is not False:
            raise ProtocolError("unsupported orientation/mirroring")
        if not isinstance(h.get("calibration_id"), str) or len(h["calibration_id"]) > 64:
            raise ProtocolError("calibration_id")
        cw, ch = dimensions(h.get("color_dimensions"))
        if kind == "sensor":
            w, ht = dimensions(h.get("depth_dimensions"))
            if h.get("payload_encoding") != "depth_u16_mm_le" or len(payload) != w*ht*2:
                raise ProtocolError("depth encoding/dimensions")
            if h.get("depth_accuracy") not in {"absolute", "relative"} or h.get("depth_quality") not in {"high", "low"}:
                raise ProtocolError("depth metadata")
            if type(h.get("depth_filtered")) is not bool:
                raise ProtocolError("depth filtering")
            landmarks = h.get("landmarks")
            validate_landmarks(landmarks)
            if h.get("chirality", "unknown") not in ("left", "right", "unknown"):
                raise ProtocolError("chirality")
            if "hands" in h:
                hands = h["hands"]
                if not isinstance(hands,list) or len(hands)>2:
                    raise ProtocolError("expected at most two hands")
                identities = set()
                for hand in hands:
                    if not isinstance(hand,dict): raise ProtocolError("hand object")
                    identity = hand.get("hand_id")
                    if not isinstance(identity,str) or not 1 <= len(identity) <= 64 or identity in identities:
                        raise ProtocolError("unique hand_id required")
                    identities.add(identity)
                    if hand.get("chirality") not in ("left","right","unknown"):
                        raise ProtocolError("hand chirality")
                    validate_landmarks(hand.get("landmarks"),allow_empty=False)
                if landmarks != (hands[0]["landmarks"] if hands else []) or h.get("chirality") != (hands[0]["chirality"] if hands else "unknown"):
                    raise ProtocolError("primary hand aliases disagree")
            if h.get("body") is not None:
                body = h["body"]
                if not isinstance(body,dict): raise ProtocolError("body object")
                validate_landmarks(body.get("landmarks"),allow_empty=False,names=BODY_JOINTS)
                surface = body.get("torso_surface")
                if surface is not None:
                    validate_landmarks([surface],allow_empty=False,names=["torsoSurface"])
                    source=body.get('torso_surface_source')
                    if source=='image_midpoint_neck_root':
                        a,b=body['landmarks'][1:3]
                        if any(p['x'] is None or p['confidence']<.4 for p in (a,b)):raise ProtocolError('torso midpoint prerequisites')
                        expected=((a['x']+b['x'])/2,(a['y']+b['y'])/2)
                    elif source=='image_shoulders_positive_y_normal':
                        a,b=body['landmarks'][7:9]
                        if any(p['x'] is None or p['confidence']<.4 for p in (a,b)):raise ProtocolError('torso shoulder prerequisites')
                        dx,dy=a['x']-b['x'],a['y']-b['y']
                        if math.hypot(dx,dy)<.08:raise ProtocolError('torso shoulder width')
                        if dx<0:dx,dy=-dx,-dy
                        expected=((a['x']+b['x'])/2-.6*dy,(a['y']+b['y'])/2+.6*dx)
                    else:raise ProtocolError('torso surface source')
                    if any(surface[k] is None or abs(surface[k]-v)>1e-6 for k,v in zip(('x','y'),expected)):
                        raise ProtocolError('torso image transform')
                    if abs(surface['confidence']-min(a['confidence'],b['confidence']))>1e-6:raise ProtocolError('torso confidence')
            if "body_tracking" in h:
                from .body_tracking import validate_body_tracking
                validate_body_tracking(h["body_tracking"], h)
            if h.get("calibration") is not None and not isinstance(h["calibration"], dict):
                raise ProtocolError("calibration")
        elif h.get("payload_encoding") != "jpeg" or not payload.startswith(b"\xff\xd8"):
            raise ProtocolError("JPEG encoding")
    elif payload:
        raise ProtocolError("control payload must be empty")
    return Packet(h, payload)

def depth_array(packet):
    w, h = dimensions(packet.header["depth_dimensions"])
    return np.frombuffer(packet.payload, dtype="<u2").reshape(h, w)
