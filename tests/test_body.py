import copy
import io
import json
from pathlib import Path
import numpy as np
import pytest
from handdepth.protocol import encode,decode,ProtocolError,BODY_JOINTS
from handdepth.geometry import PoseEstimator
from handdepth.body import associate_arms,rotation_matrix
from handdepth.server import Receiver
from handdepth.recording import Recorder,replay_packets
from generate_fixtures import body_header,preview_packet

def depth():return np.full((24,32),500,dtype='<u2')

def test_swift_python_body_fixture():
    root=Path(__file__).resolve().parents[1]/'fixtures'
    a=decode((root/'body-python.bin').read_bytes());b=decode((root/'body-swift.bin').read_bytes())
    assert a==b and len(a.header['hands'])==2
    assert [p['name'] for p in a.header['body']['landmarks']]==BODY_JOINTS

@pytest.mark.parametrize('fault',['count','order','confidence','partial','midpoint','source','not_object'])
def test_malformed_body(header,fault):
    h=body_header(header);b=h['body']
    if fault=='count':b['landmarks'].pop()
    elif fault=='order':b['landmarks'][0]['name']='head'
    elif fault=='confidence':b['landmarks'][5]['confidence']=2
    elif fault=='partial':b['landmarks'][0]['x']=None
    elif fault=='midpoint':b['torso_surface']['y']=.4
    elif fault=='source':b['torso_surface_source']='world_prediction'
    elif fault=='not_object':h['body']=[]
    with pytest.raises(ProtocolError):decode(encode(h,depth().tobytes()))

def test_measured_upper_body_and_torso_relative_poses(header):
    h=body_header(header);pose,_=PoseEstimator().estimate(h,depth());b=pose['body']
    assert len(b['landmarks'])==19 and b['tracking_valid'] and b['depth_valid']
    assert b['torso']['orientation_valid'] and b['head']['orientation_valid']
    assert np.allclose(rotation_matrix(b['torso']['quaternion_xyzw']),np.diag([-1,1,-1]))
    assert np.allclose(b['torso']['position_m'],[0,-.039,.5])
    assert np.allclose(b['head']['position_m'],[0,-.09,.5])
    assert np.allclose(b['head']['position_torso_m'],[0,-.051,0])
    assert np.allclose(b['head']['quaternion_torso_xyzw'],[0,0,0,1])
    assert b['arms']['left']['hand_id']=='hand-2' and b['arms']['right']['hand_id']=='hand-1'
    for side,arm in b['arms'].items():
        assert arm['measurement_valid'] and 0<arm['elbow_flexion_rad']<np.pi
        assert arm['upper_arm_length_m']>.03 and arm['forearm_length_m']>.03
        assert arm['torso_relative_valid'] and not arm['axial_twist_observable']
        assert np.isclose(np.linalg.norm(arm['upper_arm_direction_torso']),1)
    assert all(x['torso_relative_position_valid'] and x['torso_relative_orientation_valid'] for x in pose['hands'])
    assert pose['hands'][0]['palm_position_torso_m'][0]>0  # operator's anatomical right

def test_relative_geometry_is_invariant_to_camera_translation_and_roll(header):
    h=body_header(header);est=PoseEstimator();a,_=est.estimate(h,depth())
    angle=.06;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    points=[p for hand in h['hands'] for p in hand['landmarks']]+h['body']['landmarks']+[h['body']['torso_surface']]
    for p in points:
        pixel=r@(np.array([p['x']*64,p['y']*48])-np.array([32,24]))+np.array([34,24])
        p.update(x=pixel[0]/64,y=pixel[1]/48)
    for key in ('capture_time_s','processing_start_s','processing_end_s'):h[key]+=.05
    h['frame_id']+=1
    decode(encode(h,depth().tobytes()))
    b,_=est.estimate(h,depth())
    assert not np.allclose(a['body']['torso']['position_m'],b['body']['torso']['position_m'])
    assert np.allclose(a['body']['head']['position_torso_m'],b['body']['head']['position_torso_m'])
    assert np.allclose(a['body']['head']['quaternion_torso_xyzw'],b['body']['head']['quaternion_torso_xyzw'])
    for x,y in zip(a['hands'],b['hands']):
        assert np.allclose(x['palm_position_torso_m'],y['palm_position_torso_m'])
        assert np.allclose(x['palm_quaternion_torso_xyzw'],y['palm_quaternion_torso_xyzw'])

def test_body_loss_or_missing_calibration_invalidates_relative_motion(header):
    h=body_header(header);est=PoseEstimator();a,_=est.estimate(h,depth())
    assert a['body']['head']['orientation_valid']
    h['body']=None
    b,_=est.estimate(h,depth())
    assert b['body'] is None and all(x['palm_position_torso_m'] is None for x in b['hands'])
    assert all(x['depth_valid'] for x in b['hands'])
    h=body_header(header);h['calibration']=None
    bad,_=PoseEstimator().estimate(h,depth())
    assert len(bad['body']['landmarks'])==19 and not bad['body']['depth_valid']
    assert not bad['body']['torso']['orientation_valid'] and all(x['palm_position_torso_m'] is None for x in bad['hands'])

def test_degenerate_torso_and_missing_head_are_not_invented(header):
    h=body_header(header)
    h['body']['torso_surface']=None
    for p in h['body']['landmarks']:
        if p['name'] in ('leftEye','rightEye'):p.update(x=None,y=None,confidence=0)
    pose,_=PoseEstimator().estimate(h,depth())
    assert pose['body']['torso']['position_valid'] and not pose['body']['torso']['orientation_valid']
    assert pose['body']['head']['position_valid'] and not pose['body']['head']['orientation_valid']
    assert pose['body']['head']['quaternion_xyzw'] is None
    assert all(x['palm_position_torso_m'] is None for x in pose['hands'])

def test_seated_torso_surface_is_measured_without_hips(header):
    h=body_header(header);b=h['body']
    for p in b['landmarks']:
        if p['name'] in ('root','leftHip','rightHip'):p.update(x=None,y=None,confidence=0)
    a,c=b['landmarks'][7:9];dx,dy=a['x']-c['x'],a['y']-c['y']
    b['torso_surface'].update(x=(a['x']+c['x'])/2-.6*dy,y=(a['y']+c['y'])/2+.6*dx)
    b['torso_surface_source']='image_shoulders_positive_y_normal'
    decode(encode(h,depth().tobytes()))
    pose,_=PoseEstimator().estimate(h,depth())
    assert pose['body']['torso']['orientation_valid']
    assert not pose['body']['landmarks'][2]['valid']
    assert pose['body']['torso_surface']['source']=='image_shoulders_positive_y_normal_measured_depth'
    b['torso_surface']['y']+=.1
    with pytest.raises(ProtocolError):decode(encode(h,depth().tobytes()))

def test_arm_association_is_one_to_one_and_ambiguous_crossing_invalid(header):
    h=body_header(header);mapping=associate_arms(h['body'],h['hands'])
    assert mapping=={'left':1,'right':0}
    for p in h['body']['landmarks']:
        if p['name'] in ('leftWrist','rightWrist'):p.update(x=.5,y=.72)
    for hand in h['hands']:
        hand['chirality']='unknown';hand['landmarks'][0].update(x=.5,y=.72)
    assert associate_arms(h['body'],h['hands'])=={'left':None,'right':None}

def test_abrupt_head_axis_flip_is_invalid_without_invalidating_torso_or_hands(header):
    h=body_header(header);est=PoseEstimator();first,_=est.estimate(h,depth())
    assert first['body']['head']['torso_relative_orientation_valid']
    a,b=h['body']['landmarks'][3:5]
    a['x'],b['x']=b['x'],a['x'];h['capture_time_s']+=.05
    bad,_=est.estimate(h,depth())
    assert not bad['body']['head']['orientation_valid']
    assert bad['body']['head']['quaternion_torso_xyzw'] is None
    assert bad['body']['torso']['orientation_valid']
    assert all(p['torso_relative_orientation_valid'] for p in bad['hands'])

def test_body_record_replay_and_disconnect_clear_all_fields(tmp_path,header):
    h=body_header(header);rec=Recorder(tmp_path,fixture=True);r=Receiver(rec,io.StringIO())
    assert r.process('sensor',encode(h,depth().tobytes()),1000)
    assert r.process('preview',preview_packet(h),1000.01)
    assert r.latest_pose['body']['torso']['orientation_valid']
    assert r.dashboard.render(now=1000.02).shape==(615,1440,3)
    r.invalidate('sensor_disconnected')
    assert r.latest_pose['body'] is None and not r.estimator.body_estimator.history
    rec.close();replay=Receiver()
    for channel,t,raw in replay_packets(rec.path):replay.process(channel,raw,t,replay=True)
    assert replay.latest_pose['body']['head']['orientation_valid']

def test_replay_source_fps_uses_capture_timestamps_not_cpu_replay_speed(header):
    h=body_header(header);r=Receiver()
    for i in range(16):
        t=100+i/15
        h.update(frame_id=i,capture_time_s=t,processing_start_s=t,processing_end_s=t)
        r.process('sensor',encode(h,depth().tobytes()),1000+i/1000,replay=True)
    assert np.isclose(r.dashboard.source_fps,15)


def test_same_frame_hand_wrist_can_fill_low_confidence_body_wrist(header):
    h=body_header(header)
    for p in h['body']['landmarks']:
        if p['name']=='leftWrist':p['confidence']=.25
    pose,_=PoseEstimator().estimate(h,depth())
    b=pose['body'];original=next(p for p in b['landmarks'] if p['name']=='leftWrist')
    assert not original['valid'] and original['reason']=='low_confidence'
    assert b['arms']['left']['measurement_valid'] and b['arms']['left']['torso_relative_valid']
    assert b['arms']['left']['wrist_source']=='associated_vision_hand_wrist_surface'
    wrist=next(p for hand in pose['hands'] if hand['body_arm_side']=='left' for p in hand['landmarks'] if p['name']=='wrist')
    assert np.allclose(b['arms']['left']['positions_camera_m']['leftWrist'],wrist['xyz_m'])
    h['hands']=[]
    missing,_=PoseEstimator().estimate(h,depth())
    assert not missing['body']['arms']['left']['measurement_valid']
