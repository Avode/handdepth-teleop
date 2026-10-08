import copy
import io
import json
from pathlib import Path
import numpy as np
import pytest
from handdepth.protocol import encode, decode, depth_array, ProtocolError
from handdepth.geometry import PoseEstimator, Calibration
from handdepth.server import Receiver
from handdepth.recording import Recorder, replay_packets
from generate_fixtures import two_hand_header, preview_packet

def depth():
    return np.full((24,32),500,dtype='<u2')

def set_hands(h, hands):
    h['hands'] = hands
    h['landmarks'] = hands[0]['landmarks'] if hands else []
    h['chirality'] = hands[0]['chirality'] if hands else 'unknown'

def test_two_hand_swift_python_fixture_agreement():
    root = Path(__file__).resolve().parents[1]/'fixtures'
    python = decode((root/'two-hand-python.bin').read_bytes())
    swift = decode((root/'two-hand-swift.bin').read_bytes())
    assert python == swift
    assert [h['hand_id'] for h in python.header['hands']] == ['hand-1','hand-2']
    assert all(len(h['landmarks']) == 21 for h in python.header['hands'])
    assert depth_array(python)[0,:4].tolist() == [0,1,256,65535]

@pytest.mark.parametrize('fault',['too_many','duplicate_id','missing_id','bad_chirality','missing_joint','bad_confidence','aliases','not_object'])
def test_reject_malformed_second_hand(header, fault):
    h = two_hand_header(header)
    if fault == 'too_many': h['hands'].append(copy.deepcopy(h['hands'][1]))
    elif fault == 'duplicate_id': h['hands'][1]['hand_id'] = h['hands'][0]['hand_id']
    elif fault == 'missing_id': del h['hands'][1]['hand_id']
    elif fault == 'bad_chirality': h['hands'][1]['chirality'] = 'both'
    elif fault == 'missing_joint': h['hands'][1]['landmarks'].pop()
    elif fault == 'bad_confidence': h['hands'][1]['landmarks'][5]['confidence'] = 2
    elif fault == 'aliases': h['landmarks'] = h['hands'][1]['landmarks']
    elif fault == 'not_object': h['hands'][1] = None
    with pytest.raises(ProtocolError): decode(encode(h,depth().tobytes()))

def test_two_metric_poses_share_rectification_and_independent_history(header,monkeypatch):
    h = two_hand_header(header); estimator = PoseEstimator()
    calls = []
    rectify = Calibration.rectify
    def counted(self,d):
        calls.append(1)
        return rectify(self,d)
    monkeypatch.setattr(Calibration,'rectify',counted)
    first,_ = estimator.estimate(h,depth())
    assert len(calls)==1
    a,b = first['hands']
    assert a['orientation_valid'] and b['orientation_valid']
    assert a['pinch_distance_m'] > 0 and b['pinch_distance_m'] > 0
    assert a['palm_position_m'][0] < b['palm_position_m'][0]
    assert a['palm_position_m'][2] == pytest.approx(.5)
    assert abs(np.dot(a['palm_quaternion_xyzw'],b['palm_quaternion_xyzw'])) < .01
    h['capture_time_s'] += .05
    for p in h['hands'][0]['landmarks']: p['x'] = 1-p['x']
    flipped,_ = estimator.estimate(h,depth())
    assert not flipped['hands'][0]['orientation_valid']
    assert flipped['hands'][0]['orientation_rejection']=='abrupt_orientation_flip'
    assert flipped['hands'][1]['orientation_valid']
    assert np.allclose(flipped['hands'][1]['palm_position_m'],b['palm_position_m'])

def test_detection_reordering_and_loss_do_not_mix_pose_states(header):
    h = two_hand_header(header); estimator = PoseEstimator()
    first,_ = estimator.estimate(h,depth())
    h['capture_time_s'] += .05
    set_hands(h,list(reversed(h['hands'])))
    reordered,_ = estimator.estimate(h,depth())
    original = {x['hand_id']:x for x in first['hands']}
    for hand in reordered['hands']:
        assert hand['orientation_valid']
        assert np.allclose(hand['palm_position_m'],original[hand['hand_id']]['palm_position_m'])
    set_hands(h,h['hands'][:1])
    one,_ = estimator.estimate(h,depth())
    assert len(one['hands'])==1 and set(estimator.hand_states)=={one['hands'][0]['hand_id']}
    set_hands(h,[])
    none,_ = estimator.estimate(h,depth())
    assert none['hands']==[] and none['palm_position_m'] is None and not estimator.hand_states

def test_invalid_hand_does_not_invalidate_other_hand(header):
    h = two_hand_header(header)
    for point in h['hands'][0]['landmarks']: point.update(confidence=0,x=None,y=None)
    pose,_ = PoseEstimator().estimate(h,depth())
    assert not pose['hands'][0]['tracking_valid'] and not pose['hands'][0]['depth_valid']
    assert pose['hands'][1]['orientation_valid'] and pose['hands'][1]['pinch_distance_m'] is not None
    h['calibration']=None
    invalid,_ = PoseEstimator().estimate(h,depth())
    assert len(invalid['hands'])==2 and all(len(x['landmarks'])==21 for x in invalid['hands'])
    assert all(not x['depth_valid'] and x['palm_position_m'] is None for x in invalid['hands'])

def test_smooth_background_sample_at_fingertip_is_rejected(header):
    h=two_hand_header(header);d=depth();cal=Calibration(h['calibration'],h['color_dimensions'],h['depth_dimensions'])
    thumb=h['hands'][0]['landmarks'][4]
    x,y=np.floor(cal.landmark_pixel(thumb['x'],thumb['y'])).astype(int)
    d[y-2:y+3,x-2:x+3]=3355  # locally smooth background, no local edge at this tip
    pose,_=PoseEstimator().estimate(h,d)
    sampled=pose['hands'][0]['landmarks'][4]
    assert sampled['sample_count']==25 and sampled['reason']=='hand_surface_outlier'
    assert not sampled['valid'] and sampled['xyz_m'] is None
    assert pose['hands'][0]['pinch_distance_m'] is None
    assert pose['hands'][1]['orientation_valid']

def test_both_hands_record_replay_render_and_disconnect(tmp_path,header):
    h = two_hand_header(header);recorder=Recorder(tmp_path,fixture=True);stream=io.StringIO()
    receiver=Receiver(recorder,stream)
    assert receiver.process('sensor',encode(h,depth().tobytes()),1000)
    assert receiver.process('preview',preview_packet(h),1000.01)
    assert receiver.dashboard.render(now=1000.02).shape==(615,1440,3)
    receiver.invalidate('sensor_disconnected')
    assert receiver.latest_pose['hands']==[] and not receiver.estimator.hand_states
    recorder.close()
    replay=Receiver()
    for channel,t,raw in replay_packets(recorder.path): replay.process(channel,raw,t,replay=True)
    assert len(replay.latest_pose['hands'])==2
    emitted=list(map(json.loads,stream.getvalue().splitlines()))
    assert len(emitted[0]['hands'])==2 and emitted[1]['hands']==[]
