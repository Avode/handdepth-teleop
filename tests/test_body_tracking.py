"""Cross-language tests for explicit occlusion estimates (never measured XYZ)."""
import copy
import io
import json
import math
from pathlib import Path
import numpy as np
import pytest
from handdepth.protocol import decode, encode, ProtocolError
from handdepth.geometry import PoseEstimator
from handdepth.server import Receiver
from handdepth.recording import Recorder, replay_packets
from handdepth.visualization import tracking_overlay

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def packet():
    return decode((ROOT/'fixtures/body-tracking-swift.bin').read_bytes())

def test_swift_geometry_and_python_wire_agreement(packet):
    other = decode((ROOT/'fixtures/body-tracking-python.bin').read_bytes())
    assert packet == other
    h = packet.header
    tracking = h['body_tracking']
    assert tracking['frame_id'] == h['frame_id'] == 43
    points = {p['name']:p for p in tracking['points']}
    for side,arm in tracking['arms'].items():
        assert arm['elbow_inferred'] and arm['length_source_frame_id'] == 42
        elbow = points[side+'Elbow']
        assert elbow['state']=='inferred' and not elbow['measurement_valid']
        assert elbow['source_frame_id']==42 and elbow['confidence']==0
        xyz = [np.array([points[side+n]['x']*64,points[side+n]['y']*48]) for n in ('Shoulder','Elbow','Wrist')]
        assert np.linalg.norm(xyz[1]-xyz[0]) == pytest.approx(arm['upper_arm_length_px'])
        assert np.linalg.norm(xyz[2]-xyz[1]) == pytest.approx(arm['forearm_length_px'])
        a,b=xyz[2]-xyz[0],xyz[1]-xyz[0]
        assert abs(a[0]*b[1]-a[1]*b[0]) > .01
        # Raw missing elbows remain missing despite the inferred visible skeleton.
        raw = next(p for p in h['body']['landmarks'] if p['name']==side+'Elbow')
        assert raw['x'] is None and raw['confidence']==0
    assert len(encode(h,packet.payload)) < 65536

@pytest.mark.parametrize('fault',['future_frame','future_time','wrong_frame','count','order','unknown_joint',
    'nan','bad_state','claimed_measurement','bad_confidence','length','length_source','arm_type',
    'point_bounds','hand_joint','wrist_flag','unlabelled_inference','wrong_units','wrong_orientation','point_type'])
def test_malformed_tracking_is_rejected(packet,fault):
    h = copy.deepcopy(packet.header);t=h['body_tracking'];p=t['points'][9];a=t['arms']['left']
    assert p['name']=='leftElbow'
    if fault=='future_frame':p['source_frame_id']=100
    elif fault=='future_time':p['source_time_s']=200
    elif fault=='wrong_frame':t['frame_id']=42
    elif fault=='count':t['points']*=2
    elif fault=='order':t['points'].reverse()
    elif fault=='unknown_joint':p['name']='worldElbow'
    elif fault=='nan':p['x']=float('nan')
    elif fault=='bad_state':p['state']='valid'
    elif fault=='claimed_measurement':p['measurement_valid']=True
    elif fault=='bad_confidence':p['confidence']=.99
    elif fault=='length':a['upper_arm_length_px']*=1.1
    elif fault=='length_source':a['length_source_time_s']=200
    elif fault=='arm_type':t['arms']['left']=[]
    elif fault=='point_bounds':p['y']=1e5
    elif fault=='hand_joint':p.update(state='observed_hand',measurement_valid=True,confidence=.9,source_frame_id=43,source_time_s=100.1)
    elif fault=='wrist_flag':a['wrist_clamped']=True
    elif fault=='unlabelled_inference':del t['arms']['left']
    elif fault=='wrong_units':t['length_unit']='metres'
    elif fault=='wrong_orientation':t['coordinate_frame']='mirrored'
    elif fault=='point_type':t['points'][0]=None
    with pytest.raises(ProtocolError):decode(encode(h,packet.payload))

def test_pose_passes_estimates_without_inventing_depth_or_modifying_measurements(packet):
    h=packet.header;depth=np.frombuffer(packet.payload,dtype='<u2').reshape(24,32)
    pose,_=PoseEstimator().estimate(h,depth)
    assert pose['body_tracking']==h['body_tracking']
    for side in ('left','right'):
        assert not pose['body']['arms'][side]['measurement_valid']
        elbow=next(p for p in pose['body']['landmarks'] if p['name']==side+'Elbow')
        assert not elbow['valid'] and elbow['xyz_m'] is None
    pose['body_tracking']['points'][0]['x']=0
    assert h['body_tracking']['points'][0]['x'] != 0
    h=copy.deepcopy(h);h['calibration']=None
    pose,_=PoseEstimator().estimate(h,depth)
    assert pose['body_tracking'] and not pose['body']['depth_valid']

def test_record_replay_timeout_disconnect_and_reconnect(packet,tmp_path):
    rec=Recorder(tmp_path,fixture=True);output=io.StringIO();receiver=Receiver(rec,output)
    raw=encode(packet.header,packet.payload)
    assert receiver.process('sensor',raw,1000)
    assert not receiver.process('sensor',raw,1000.1)  # duplicate must not refresh it
    receiver.expire(1001)
    assert receiver.latest_pose['body_tracking'] is None
    assert receiver.latest_pose['receiver_event']
    h=copy.deepcopy(packet.header)
    h['session_id']='new-phone-session'
    assert receiver.process('sensor',encode(h,packet.payload),1002)
    assert receiver.latest_pose['body_tracking']['frame_id']==43
    receiver.invalidate('sensor_disconnected')
    assert receiver.latest_pose['body_tracking'] is None
    rec.close()
    replay=Receiver()
    for channel,now,raw in replay_packets(rec.path):assert replay.process(channel,raw,now,replay=True)
    assert replay.latest_pose['body_tracking']==h['body_tracking']
    poses=[json.loads(line) for line in output.getvalue().splitlines()]
    assert poses[0]['body_tracking'] and poses[-1]['body_tracking'] is None

def test_rgb_overlay_draws_two_bones_and_does_not_mutate_input(packet):
    image=np.zeros((480,640,3),dtype=np.uint8)
    result=tracking_overlay(image,packet.header['body_tracking'])
    assert not image.any() and result.any()
    assert ((result==[0,180,255]).all(axis=2)).sum()>100  # amber inference


def test_older_phone_and_recordings_remain_compatible(packet):
    h=copy.deepcopy(packet.header);del h['body_tracking']
    decode(encode(h,packet.payload))
    pose,_=PoseEstimator().estimate(h,np.frombuffer(packet.payload,dtype='<u2').reshape(24,32))
    assert pose['body_tracking'] is None
