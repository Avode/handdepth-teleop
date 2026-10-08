"""Deterministic synthetic fixture producer; no camera measurements are fabricated as real."""
from pathlib import Path
import copy
import json
import numpy as np
import cv2
from handdepth.protocol import encode, JOINTS, BODY_JOINTS

ROOT = Path(__file__).resolve().parents[1]

def sensor_header(frame=42, capture=100., size=(64,48), depth_size=(32,24)):
    w,h = size; dw,dh = depth_size
    positions = [(0.5,.72),(.32,.62),(.25,.52),(.20,.42),(.18,.32),
                 (.38,.45),(.36,.34),(.34,.23),(.32,.15),
                 (.5,.42),(.5,.3),(.5,.19),(.5,.1),
                 (.58,.43),(.6,.32),(.62,.22),(.64,.14),
                 (.65,.45),(.7,.37),(.74,.3),(.78,.24)]
    return {"kind":"sensor","session_id":"fixture-session","frame_id":frame,"capture_time_s":capture,
        "clock":"ios_host_monotonic_seconds","processing_start_s":capture+.001,"processing_end_s":capture+.015,
        "calibration_id":"synthetic-zero-distortion", "application_version":"0.1.0",
        "orientation":"native_up","mirrored":False,"color_dimensions":[w,h],"depth_dimensions":[dw,dh],
        "payload_encoding":"depth_u16_mm_le", "depth_accuracy":"absolute","depth_quality":"high","depth_filtered":False,
        "landmarks":[{"name":n,"x":x,"y":y,"confidence":.95} for n,(x,y) in zip(JOINTS,positions)],
        "chirality":"right","tracking_error":None,
        "calibration":{"alignment":"truedepth_color_fov","intrinsics_row_major":[[w*1.25,0,w/2],[0,w*1.25,h/2],[0,0,1]],
            "reference_dimensions":[w,h],"distortion_center":[w/2,h/2],"lens_distortion_lut":[0.,0.,0.],
            "inverse_lens_distortion_lut":[0.,0.,0.],"color_from_reference":[[1,0,0],[0,1,0],[0,0,1]],
            "depth_from_reference":[[dw/w,0,0],[0,dh/h,0],[0,0,1]],
            "extrinsics_camera_to_reference_row_major_mm":[[1,0,0,0],[0,1,0,0],[0,0,1,0]],
            "extrinsic_from":"calibrated_camera","extrinsic_to":"apple_reference_camera","extrinsic_translation_units":"millimetres",
            "extrinsic_applied_to_pose":False,"pixel_size_mm":.001,"crop":"none","resize":"native_output_dimensions",
            "orientation":"native_up","mirrored":False,
            "lut_point_mapping":"sdk_reference_forward_rectified_to_distorted_inverse_distorted_to_rectified"}}

def two_hand_header(header=None):
    h = copy.deepcopy(header if header is not None else sensor_header())
    left = copy.deepcopy(h["landmarks"])
    for p in left: p["x"] = p["x"]*.65+.03
    right = copy.deepcopy(left)
    for p in right: p["x"] = 1-p["x"]
    h.update(application_version="0.2.0", hands=[
        {"hand_id":"hand-1","chirality":"left","landmarks":left},
        {"hand_id":"hand-2","chirality":"right","landmarks":right}],landmarks=left,chirality="left")
    return h

def body_header(header=None):
    h=two_hand_header(header)
    h['hands'][0]['chirality']='right';h['hands'][1]['chirality']='left';h['chirality']='right'
    positions=[(.5,.2),(.5,.35),(.5,.85),(.54,.14),(.46,.14),(.58,.18),(.42,.18),
               (.72,.37),(.28,.37),(.78,.54),(.22,.54),(.645,.72),(.355,.72),
               (.62,.85),(.38,.85),(.62,.92),(.38,.92),(.62,.98),(.38,.98)]
    h['body']={'landmarks':[{'name':name,'x':x,'y':y,'confidence':.95} for name,(x,y) in zip(BODY_JOINTS,positions)],
               'torso_surface':{'name':'torsoSurface','x':.5,'y':.6,'confidence':.95},
               'torso_surface_source':'image_midpoint_neck_root'}
    h.update(body_error=None,application_version='0.3.0')
    return h

def preview_packet(header):
    w,h = header["color_dimensions"]
    image = np.zeros((h,w,3),np.uint8)
    image[:] = [70,35,20]
    points = [p for hand in header.get("hands",[{"landmarks":header["landmarks"]}]) for p in hand["landmarks"]]
    if header.get('body'):points += header['body']['landmarks']
    for p in points:
        cv2.circle(image,(int(p['x']*w),int(p['y']*h)),1,(50,220,100),-1)
    ok,jpeg = cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,80])
    assert ok
    h = {k:v for k,v in header.items() if k not in ('landmarks','hands','body','body_error','calibration','chirality','tracking_error')}
    h.update(kind="preview",payload_encoding="jpeg")
    return encode(h,jpeg.tobytes())

if __name__ == '__main__':
    h = sensor_header()
    depth = np.full((24,32),500,dtype='<u2')
    depth[0,:4] = [0,1,256,65535]
    (ROOT/'fixtures/sensor-python.bin').write_bytes(encode(h,depth.tobytes()))
    (ROOT/'fixtures/sensor-header.json').write_text(json.dumps(h,indent=2,allow_nan=False))
    (ROOT/'fixtures/depth-u16le.bin').write_bytes(depth.tobytes())
    (ROOT/'fixtures/preview.bin').write_bytes(preview_packet(h))
    (ROOT/'fixtures/two-hand-python.bin').write_bytes(encode(two_hand_header(h),depth.tobytes()))
    (ROOT/'fixtures/body-python.bin').write_bytes(encode(body_header(h),depth.tobytes()))
