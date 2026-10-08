import copy
import time
from types import SimpleNamespace
import numpy as np
import pytest
from handdepth.protocol import encode,decode
from handdepth.server import Receiver
from handdepth.display_tracking import MetricSkeleton
from handdepth.retarget import Retargeter
from generate_fixtures import preview_packet,body_header
from test_limb_refinement import scene,pc_observation
from test_display_tracking import measured_pose,next_frame,point
from test_retarget import home


class FusionStub:
    def __init__(self):self.stats={'error':None};self.frames=[]
    def add(self,*args):self.frames.append(args)
    def reset(self):self.frames.clear()


def receiver():
    r=Receiver();r.pc_fusion=FusionStub();h,d=scene()
    h.update(landmarks=[],chirality='unknown');h.pop('body_tracking')
    return r,h,d


def test_missing_preview_publishes_current_phone_pose_after_bounded_wait():
    r,h,d=receiver();now=time.monotonic()
    assert r.process('sensor',encode(h,d.tobytes()),now)
    assert r.latest_pose is None
    r.expire(now+.09);assert r.latest_pose is None
    r.expire(now+.11);p=r.latest_pose
    assert p['frame_id']==h['frame_id'] and p['processing_location']=='phone_rgbd_fallback_unpaired_or_slow_pc'
    assert p['received_monotonic_s']==now and p['body'] is not None
    assert Retargeter(home()).fresh(p,now+.3)
    assert not Retargeter(home()).fresh(p,now+.7)
    # Late inference cannot flicker validity or renew freshness for the same frame.
    r._pc_result(h,d,now,p,pc_observation(h));assert r.latest_pose is p
    assert not r.pending_sensor


def test_fast_pc_result_wins_and_is_not_followed_by_duplicate_fallback():
    r,h,d=receiver();now=time.monotonic();r.process('sensor',encode(h,d.tobytes()),now)
    original=r.pending_sensor[h['frame_id']][3]
    r._pc_result(h,d,now,original,pc_observation(h));p=r.latest_pose
    assert p['processing_location']=='ubuntu_rgb_pose_plus_registered_depth_fusion'
    r.expire(now+.2);assert r.latest_pose is p and not r.pending_sensor


@pytest.mark.parametrize('event',['disconnect','timeout','new_session'])
def test_pending_fallback_cannot_cross_freshness_or_session_boundary(event):
    r,h,d=receiver();now=time.monotonic();r.process('sensor',encode(h,d.tobytes()),now)
    if event=='disconnect':r.invalidate('sensor_disconnected')
    elif event=='timeout':r.expire(now+.6)
    else:
        h=dict(h,session_id='new',frame_id=1)
        r.process('sensor',encode(h,d.tobytes()),now+.2)
    r.expire(now+.25)
    assert r.latest_pose is None
    if event=='new_session':
        r.expire(now+.32);assert r.latest_pose['session_id']=='new'


def test_alternating_channels_keep_pose_and_unpaired_video_moving(header):
    r=Receiver();r.pc_fusion=FusionStub();now=time.monotonic();h=body_header(header)
    for i in range(6):
        t=now+i*.14
        sensor=dict(h,frame_id=42+2*i,capture_time_s=100+i*.14,processing_start_s=100+i*.14,processing_end_s=100+i*.14+.01)
        preview=dict(sensor,frame_id=sensor['frame_id']+1,capture_time_s=sensor['capture_time_s']+.066,
                     processing_start_s=sensor['capture_time_s']+.067,processing_end_s=sensor['capture_time_s']+.068)
        r.process('sensor',encode(sensor,np.full((24,32),500,dtype='<u2').tobytes()),t)
        r.process('preview',preview_packet(preview),t+.01)
        r.expire(t+.11)
        assert r.latest_pose['frame_id']==sensor['frame_id']
        r.dashboard.render(t+.12)
        assert 'unpaired RGB frame '+str(preview['frame_id']) in r.dashboard.rgb_status
        assert not r.dashboard.rgb_overlay_matched
    assert r.stream_status(now+.9)['publications']['phone_rgbd_fallback_unpaired_or_slow_pc']==6
    # Matching resumes normally without resetting the phone session or display.
    p=decode(preview_packet(sensor));r.dashboard.preview(p.header,p.payload,t+.13)
    r.dashboard.render(t+.14)
    assert r.dashboard.rgb_overlay_matched and 'matched RGB frame' in r.dashboard.rgb_status


def test_pending_sensor_queue_is_bounded_and_gate_rejections_are_diagnosable():
    r,h,d=receiver();now=time.monotonic()
    for i in range(20):
        q=dict(h,frame_id=h['frame_id']+i,capture_time_s=h['capture_time_s']+.01*i,
               processing_start_s=h['capture_time_s']+.01*i,processing_end_s=h['capture_time_s']+.01*i+.01)
        r.process('sensor',encode(q,d.tobytes()),now+.01*i)
    assert len(r.pending_sensor)==8
    assert not r.process('sensor',encode(q,d.tobytes()),now+.2)
    assert r.stream_status(now+.2)['channels']['sensor']['rejected']['frame_order']==1
    r.expire(now+.4)
    assert r.latest_pose['frame_id']==q['frame_id'] and not r.pending_sensor


def pelvis(p,bad=False):
    p['body']['torso'].update(position_m=[0,0,1],quaternion_xyzw=[0,0,0,1])
    for n,v in [('leftHip',[-.16,.45,0]),('rightHip',[.16,.45,0])]:
        if bad and n=='leftHip':continue
        if bad:v=[-.5,.7,.2]
        point(p,n).update(valid=True,confidence=.9,xyz_torso_m=v,xyz_m=(np.array(v)+[0,0,1]).tolist())
    return p


@pytest.mark.parametrize('space',['camera','torso'])
def test_isolated_outlier_hip_cannot_grow_a_stray_shoulder_line(space):
    s=MetricSkeleton(space);p=pelvis(measured_pose(),bad=True)
    for _ in range(20):s.update(p);p=next_frame(p)
    assert 'rightHip' not in s.points
    assert all(side+n in s.points for side in ('left','right') for n in ('Shoulder','Elbow','Wrist'))


@pytest.mark.parametrize('space',['camera','torso'])
def test_known_pelvis_is_retained_coherently_but_not_learned_from_single_frame(space):
    s=MetricSkeleton(space);p=pelvis(measured_pose());s.update(p)
    assert 'rightHip' not in s.points
    for _ in range(2):p=next_frame(p);s.update(p)
    assert 'rightHip' in s.points
    p=next_frame(p)
    for name in ('rightHip','leftHip'):point(p,name).update(valid=False,xyz_m=None,xyz_torso_m=None)
    p['body']['torso']['position_m']=[.5,.2,1.1]
    s.update(p)
    assert s.points['rightHip'].state=='inferred'
    expected=np.array([.16,.45,0])+(p['body']['torso']['position_m'] if space=='camera' else np.zeros(3))
    assert np.allclose(s.points['rightHip'].xyz,expected)
    s.reset();s.update(next_frame(p));assert 'rightHip' not in s.points
