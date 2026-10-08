import copy
from pathlib import Path
import threading
import time
from handdepth.pc_pose import PCFusion,same_frame
from handdepth.server import Receiver
from handdepth.protocol import encode
from test_limb_refinement import scene,pc_observation


class FakeWorker:
    def __init__(self):
        self.started=threading.Event();self.release=threading.Event();self.closed=False
    def detect(self,h,jpeg):
        self.started.set();assert self.release.wait(2)
        return dict(h,inference_ms=1)
    def close(self):self.closed=True


def test_async_queue_replaces_pending_frames_and_discards_reset_results():
    worker=FakeWorker();results=[];done=threading.Event()
    def result(h,*args):results.append(h['frame_id']);done.set()
    f=PCFusion(result,Path('/unused/model.task'),worker)
    h,_=scene()
    try:
        for channel in ['preview','sensor']:f.add(channel,h,b'data',time.monotonic(),{})
        assert worker.started.wait(1)
        for i in range(1,8):
            q=dict(h,frame_id=h['frame_id']+i,capture_time_s=h['capture_time_s']+i*.01)
            for channel in ['sensor','preview']:f.add(channel,q,b'data',time.monotonic(),{})
        assert f.stats['replaced']==6
        f.reset();worker.release.set()
        q=dict(h,frame_id=h['frame_id']+9)
        for channel in ['sensor','preview']:f.add(channel,q,b'data',time.monotonic(),{})
        assert done.wait(1) and results==[q['frame_id']]
        assert f.stats['discarded']>=1
    finally:worker.release.set();f.close()
    assert worker.closed


def test_pair_caches_are_bounded_and_never_mix_frames_or_calibration():
    worker=FakeWorker();worker.release.set();f=PCFusion(lambda *a:None,Path('/unused/model.task'),worker)
    h,_=scene()
    try:
        for i in range(30):f.add('sensor',dict(h,frame_id=i),b'data',time.monotonic(),{})
        assert len(f.sensor)==8 and f.stats['submitted']==0
        f.add('preview',dict(h,frame_id=29,capture_time_s=999),b'jpeg',time.monotonic())
        assert f.stats['submitted']==0 and f.stats['discarded']==1
        f.add('preview',dict(h,frame_id=28,calibration_id='different'),b'jpeg',time.monotonic())
        assert not f.sensor and f.stats['submitted']==0
    finally:f.close()


def test_disconnect_before_first_result_cannot_publish_robot_input():
    h,d=scene();r=Receiver();now=time.monotonic()
    h.update(landmarks=[],chirality='unknown');h.pop('body_tracking')
    r.process('sensor',encode(h,d.tobytes()),now)
    original=copy.deepcopy(r.latest_pose);r.latest_pose=None
    r.invalidate('sensor_disconnected')
    r._pc_result(h,d,now,original,pc_observation(h))
    assert r.latest_pose is None and not r.fusion_input_live


def test_fused_publication_keeps_original_age_and_rejects_late_results():
    h,d=scene();r=Receiver();now=time.monotonic()
    h.update(landmarks=[],chirality='unknown');h.pop('body_tracking')
    r.process('sensor',encode(h,d.tobytes()),now)
    original=copy.deepcopy(r.latest_pose);r.latest_pose=None
    original['capture_to_receive_s']=.07
    r._pc_result(h,d,now,original,pc_observation(h))
    fused=r.latest_pose
    assert fused['processing_location']=='ubuntu_rgb_pose_plus_registered_depth_fusion'
    assert fused['received_monotonic_s']==now and fused['capture_to_receive_s']==.07
    assert fused['pc_pose']['frame_id']==h['frame_id']
    r._pc_result(h,d,now,original,pc_observation(h))
    assert r.latest_pose is fused # a duplicate cannot refresh freshness
    stale=dict(h,frame_id=h['frame_id']+1)
    r._pc_result(stale,d,now-1,original,pc_observation(stale))
    assert r.latest_pose is fused
    new_session=dict(stale,session_id='unmatched-session')
    r._pc_result(new_session,d,now,original,pc_observation(new_session))
    assert r.latest_pose is fused
    r.invalidate('sensor_disconnected')
    assert r.latest_pose['receiver_event'] and r.latest_pose['body'] is None
    r._pc_result(stale,d,now,original,pc_observation(stale))
    assert r.latest_pose['receiver_event']


def test_failed_pc_worker_preserves_phone_receiver_fallback():
    from types import SimpleNamespace
    h,d=scene();r=Receiver()
    h.update(landmarks=[],chirality='unknown');h.pop('body_tracking')
    r.pc_fusion=SimpleNamespace(stats={'error':'worker unavailable'})
    assert r.process('sensor',encode(h,d.tobytes()),time.monotonic())
    assert r.latest_pose['frame_id']==h['frame_id']
    assert r.latest_pose['body'] is not None
