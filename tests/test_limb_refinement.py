import copy
import json
import time
import cv2
import numpy as np
import pytest
from generate_fixtures import sensor_header,body_header
from handdepth.geometry import PoseEstimator,Calibration
from handdepth.limb_refinement import verified_limb_point,PC_SOURCE
from handdepth.display_tracking import ImageSkeleton,MetricSkeleton
from handdepth.retarget import measured_upper_arm


def scene():
    h=body_header(sensor_header(size=(640,480),depth_size=(320,240)))
    h['hands']=[]
    d=np.full((240,320),2500,dtype='<u2')
    cv2.rectangle(d,(90,70),(230,200),1300,-1)
    points={p['name']:p for p in h['body']['landmarks']}
    for side in ('left','right'):
        xy=[tuple(np.floor([points[side+j]['x']*320,points[side+j]['y']*240]).astype(int)) for j in ('Shoulder','Elbow','Wrist')]
        for a,b in zip(xy,xy[1:]):cv2.line(d,a,b,1300,12)
    h['body_tracking']={'points':[dict(name=s+'Elbow',x=.05,y=.05,confidence=0.,
        state='inferred',measurement_valid=False,source_frame_id=h['frame_id']-1,
        source_time_s=h['capture_time_s']-.1) for s in ('left','right')],'arms':{}}
    return h,d


def pc_observation(h):
    pc={k:copy.deepcopy(h[k]) for k in ('session_id','calibration_id','frame_id','capture_time_s','color_dimensions')}
    pc.update(inference_ms=20.,landmarks={p['name']:dict(p,confidence=.99) for p in h['body']['landmarks']
        if p['name'].endswith(('Shoulder','Elbow','Wrist'))})
    return pc


def advance(h):
    h['frame_id']+=1;h['capture_time_s']+=.13


@pytest.mark.parametrize('use_pc',[False,True])
def test_both_links_recover_from_current_rgbd_never_from_inferred_xy(use_pc,monkeypatch):
    import handdepth.limb_refinement as lr
    h,d=scene();pc=pc_observation(h);est=PoseEstimator();sample=lr.sample_surface;calls=[]
    if use_pc:
        for p in h['body']['landmarks']:
            if p['name'].endswith('Elbow'):p.update(confidence=0.,x=None,y=None)
    def spy(c,depth,p,**kw):
        if p['name'].endswith('Elbow'):calls.append((p['x'],p['y']))
        return sample(c,depth,p,**kw)
    monkeypatch.setattr(lr,'sample_surface',spy)
    pose,_=est.estimate(h,d,pc if use_pc else None)
    assert not any(verified_limb_point(pose,s+'Elbow') for s in ('left','right'))
    for _ in range(3): # one source frame cannot confirm itself
        pose,_=est.estimate(h,d,pc if use_pc else None)
        assert not verified_limb_point(pose,'leftElbow')
    advance(h);pc.update(frame_id=h['frame_id'],capture_time_s=h['capture_time_s'])
    pose,_=est.estimate(h,d,pc if use_pc else None)
    assert (.05,.05) not in calls
    json.dumps(pose,allow_nan=False)
    assert pose['body_tracking']==h['body_tracking']
    for s in ('left','right'):
        assert verified_limb_point(pose,s+'Elbow')
        arm=pose['body']['arms'][s]
        assert arm['upper_arm_measurement_valid'] and arm['forearm_measurement_valid']
        assert arm['torso_relative_valid'] and measured_upper_arm(pose,s) is not None
        e=next(p for p in pose['body']['landmarks'] if p['name']==s+'Elbow')
        assert e['xyz_m'][2]==pytest.approx(1.3)
        if use_pc:assert e['source']==PC_SOURCE
    image=ImageSkeleton().update(pose,h['body'],time.monotonic())
    assert all(p['state']=='observed_rgbd' for p in image['points'] if p['name'].endswith('Elbow'))
    metric=MetricSkeleton();metric.update(pose)
    assert metric.points['leftElbow'].state=='observed' and metric.arms['left']['mode']=='observed'


@pytest.mark.parametrize('failure',['background','depth_hole','missing_both_detectors','low_pc_visibility','different_person','hand_occluder'])
def test_bad_or_occluded_candidates_do_not_become_measured(failure):
    h,d=scene();pc=pc_observation(h)
    if failure=='background':d[:]=1300
    elif failure=='depth_hole':
        for s in ('left','right'):
            p=pc['landmarks'][s+'Elbow'];d[int(p['y']*240),int(p['x']*320)]=0
    elif failure in ('missing_both_detectors','low_pc_visibility','different_person'):
        for p in h['body']['landmarks']:
            if p['name'].endswith('Elbow'):p.update(x=None,y=None,confidence=0.)
        if failure=='missing_both_detectors':pc['landmarks']={}
        elif failure=='low_pc_visibility':
            for n,p in pc['landmarks'].items():
                if n.endswith('Elbow'):p['confidence']=.3
        else:
            for p in pc['landmarks'].values():p['x']=1-p['x']
    else:
        h['hands']=[]
        for s in ('left','right'):
            elbow=pc['landmarks'][s+'Elbow'];points=copy.deepcopy(h['landmarks'])
            for i,p in enumerate(points):
                theta=i*2*np.pi/len(points);p.update(x=elbow['x']+.035*np.cos(theta),y=elbow['y']+.04*np.sin(theta))
            h['hands'].append({'hand_id':s,'chirality':s,'landmarks':points})
    est=PoseEstimator()
    for _ in range(4):
        pc.update(frame_id=h['frame_id'],capture_time_s=h['capture_time_s'])
        pose,_=est.estimate(h,d,pc)
        assert not verified_limb_point(pose,'leftElbow') and measured_upper_arm(pose,'left') is None
        advance(h)


@pytest.mark.parametrize('field',['frame_id','session_id','calibration_id','capture_time_s','color_dimensions'])
def test_rgb_pose_must_match_depth_source_exactly(field):
    h,d=scene();pc=pc_observation(h);pc[field]='wrong'
    with pytest.raises(ValueError,match='source frame'):PoseEstimator().estimate(h,d,pc)


def test_recovered_point_cannot_bypass_stale_provenance_or_reset():
    h,d=scene();est=PoseEstimator()
    est.estimate(h,d);advance(h);p,_=est.estimate(h,d)
    assert verified_limb_point(p,'leftElbow')
    q=copy.deepcopy(p);q['frame_id']+=1
    assert not verified_limb_point(q,'leftElbow') and measured_upper_arm(q,'left') is None
    h['calibration_id']='changed';advance(h);p,_=est.estimate(h,d)
    assert not verified_limb_point(p,'leftElbow')
    advance(h);p,_=est.estimate(h,d);assert verified_limb_point(p,'leftElbow')
    est.reset();advance(h);p,_=est.estimate(h,d)
    assert not verified_limb_point(p,'leftElbow')


def test_foreground_edge_layer_is_selected_without_crossing_to_background():
    from handdepth.limb_refinement import surface_candidate
    h,d=scene();cal=Calibration(h['calibration'],h['color_dimensions'],h['depth_dimensions'])
    p={'name':'leftWrist','x':.5,'y':.5,'confidence':.9}
    d[:]=2500;d[118:123,158:161]=1300
    out=surface_candidate(cal,d,p)
    assert out['valid'] and out['reason']=='limb_connected_depth_layer' and out['xyz_m'][2]==pytest.approx(1.3)
    d[120,160]=0;assert not surface_candidate(cal,d,p)['valid']
