import copy
import math
import numpy as np
import pytest
from handdepth.display_tracking import MetricSkeleton, ImageSkeleton
from handdepth.protocol import BODY_JOINTS
from handdepth.geometry import PoseEstimator
from handdepth.visualization import Dashboard
from generate_fixtures import body_header, preview_packet
from handdepth.protocol import decode


def measured_pose():
    xyz={}
    for side,sign in [('left',-1),('right',1)]:
        xyz.update({side+'Shoulder':[sign*.2,0,0],side+'Elbow':[sign*.35,.23,.05],
                    side+'Wrist':[sign*.24,.25,.32]})
    return {'session_id':'phone','frame_id':10,'capture_time_s':10.,'received_monotonic_s':1000.,
        'calibration_id':'geometry','coordinate_frame':'camera_optical_x_right_y_down_z_forward',
        'color_dimensions':[640,480],'body':{'torso':{'orientation_valid':True},'arms':{},
            'landmarks':[{'name':n,'valid':n in xyz,'confidence':.9,'xyz_torso_m':xyz.get(n),
                'xyz_m':(np.array(xyz[n])+[0,0,1]).tolist() if n in xyz else None} for n in BODY_JOINTS]},'hands':[]}


def next_frame(p):
    p=copy.deepcopy(p);p['frame_id']+=1;p['capture_time_s']+=.1;p['received_monotonic_s']+=.1
    return p


def point(p,name):return next(x for x in p['body']['landmarks'] if x['name']==name)


def establish(s,p):
    for i in (2,1,0):
        f=copy.deepcopy(p);f['frame_id']-=i;f['capture_time_s']-=i*.1;f['received_monotonic_s']-=i*.1
        s.update(f)


def hide(p,*names):
    for n in names:point(p,n).update(valid=False,xyz_m=None,xyz_torso_m=None)
    return p


def assert_lengths(s,side):
    arm=s.arms[side];xyz=[s.points[side+n].xyz for n in ('Shoulder','Elbow','Wrist')]
    assert np.allclose(np.linalg.norm(np.diff(xyz,axis=0),axis=1),arm['lengths'],atol=1e-9)
    assert np.isfinite(xyz).all()


@pytest.mark.parametrize('side',['left','right'])
def test_occluded_elbow_preserves_metric_lengths_and_bend_continuity(side):
    p=measured_pose();s=MetricSkeleton();establish(s,p);original=copy.deepcopy(p)
    lengths=s.arms[side]['lengths'].copy();previous=s.arms[side]['bend'].copy()
    for i in range(80):
        p=hide(next_frame(p),side+'Elbow')
        shoulder=np.array(point(p,side+'Shoulder')['xyz_torso_m'])
        point(p,side+'Wrist')['xyz_torso_m']=(shoulder+[.07*math.sin(i*.025),.25,.25+.04*math.sin(i*.04)]).tolist()
        before=copy.deepcopy(p);s.update(p)
        assert p==before  # inferred XYZ must never leak into measured pose/control
        assert s.points[side+'Elbow'].state=='inferred'
        assert s.arms[side]['mode']=='inferred'
        assert not s.arms[side]['clamped']
        assert_lengths(s,side)
        assert np.array_equal(s.arms[side]['lengths'],lengths)
        assert s.arms[side]['bend']@previous>.995
        previous=s.arms[side]['bend'].copy()
    assert len(s.points)==6 and all(len(a['samples'])<=5 for a in s.arms.values())
    assert original['body']['landmarks'][BODY_JOINTS.index(side+'Elbow')]['valid']


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('offset',[[3,0,0],[0,0,0],[1e-12,0,0],[-3,0,0]])
def test_unreachable_and_degenerate_wrist_clamps_endpoint_never_stretches(side,offset):
    p=measured_pose();s=MetricSkeleton();establish(s,p)
    p=hide(next_frame(p),side+'Elbow')
    shoulder=np.array(point(p,side+'Shoulder')['xyz_torso_m'])
    target=shoulder+offset;point(p,side+'Wrist')['xyz_torso_m']=target.tolist()
    s.update(p);assert_lengths(s,side)
    assert s.arms[side]['clamped'] and s.points[side+'Wrist'].state=='inferred'
    assert np.array_equal(s.arms[side]['target'],target)
    assert not np.allclose(s.points[side+'Wrist'].xyz,target)


def test_partial_torso_loss_and_stale_source_hold_coherent_shape_with_age():
    p=measured_pose();s=MetricSkeleton();s.update(p)
    previous={n:v.xyz.copy() for n,v in s.points.items()}
    p=next_frame(p);p['body']['torso']['orientation_valid']=False
    for x in p['body']['landmarks']:x['xyz_torso_m']=None
    s.update(p)
    assert s.arms.keys()=={'left','right'}
    assert all(v.state=='held' and np.array_equal(v.xyz,previous[n]) for n,v in s.points.items())
    assert s.summary(1002)['oldest_age_s']==pytest.approx(2.)
    event=dict(p,body=None,body_tracking=None,receiver_event=True,received_monotonic_s=1002.)
    s.update(event,fresh=False,now=1003.)
    assert len(s.points)==6 and s.summary(1003)['oldest_age_s']==pytest.approx(3.)
    # Reacquisition is explicitly measured again; no reset/recalibration needed.
    p=next_frame(measured_pose());p['frame_id']=30;s.update(p)
    assert all(v.state=='observed' for v in s.points.values())


@pytest.mark.parametrize('side',['left','right'])
def test_missing_endpoint_holds_whole_arm_and_unseen_elbow_stays_unknown(side):
    p=measured_pose();s=MetricSkeleton();s.update(p)
    old={n:s.points[side+n].xyz.copy() for n in ('Shoulder','Elbow','Wrist')}
    p=hide(next_frame(p),side+'Elbow',side+'Wrist')
    point(p,side+'Shoulder')['xyz_torso_m'][0]+=.2;s.update(p)
    assert all(np.array_equal(s.points[side+n].xyz,old[n]) for n in old)
    assert all(s.points[side+n].state=='held' for n in old)
    blank=MetricSkeleton();blank.update(p)
    assert side not in blank.arms and side+'Elbow' not in blank.points


@pytest.mark.parametrize('change',['session','geometry','coordinate_frame','dimensions','explicit_reset','phone_reset'])
def test_history_resets_on_session_geometry_and_explicit_reset(change):
    p=measured_pose()
    p['body_tracking']={'points':[dict(name=n,x=.5,y=.5,measurement_valid=True,state='observed_body',
        source_frame_id=1,source_time_s=10.,confidence=.9) for n in BODY_JOINTS if point(p,n)['valid']], 'arms':{}}
    s=MetricSkeleton();s.update(p);assert s.arms
    if change=='explicit_reset':
        s.reset();s.update(p);assert not s.points # same frame cannot repopulate
    p=next_frame(p);p['body']=None
    if change=='session':p['session_id']='new'
    elif change=='geometry':p['calibration_id']='different projection'
    elif change=='coordinate_frame':p['coordinate_frame']='different'
    elif change=='dimensions':p['color_dimensions']=[480,640]
    elif change=='phone_reset':p['body_tracking']['points']=[]
    s.update(p);assert not s.points and not s.arms


def test_2d_lengths_never_become_metric_and_phone_occlusion_cue_rejects_raw_elbow():
    p=measured_pose();s=MetricSkeleton();establish(s,p);lengths=s.arms['left']['lengths'].copy()
    p=next_frame(p)
    p['body_tracking']={'points':[{'name':'leftElbow','measurement_valid':False}],
        'arms':{'left':{'elbow_inferred':True,'upper_arm_length_px':8192,'forearm_length_px':4096}}}
    point(p,'leftElbow')['xyz_torso_m']=[.2,.1,.3]  # confident occluding surface
    s.update(p);assert s.points['leftElbow'].state=='inferred'
    assert np.array_equal(s.arms['left']['lengths'],lengths);assert_lengths(s,'left')


def test_inferred_elbow_is_never_depth_sampled_even_if_raw_vision_is_confident(header,monkeypatch):
    import handdepth.body as b
    h=body_header(header);sample=b.sample_surface;called=[]
    def spy(cal,depth,p,**kw):
        called.append(p['name']);return sample(cal,depth,p,**kw)
    monkeypatch.setattr(b,'sample_surface',spy)
    h['body_tracking']={'points':[{'name':side+'Elbow','measurement_valid':False} for side in ('left','right')]}
    pose,_=PoseEstimator().estimate(h,np.full((24,32),500,dtype='<u2'))
    for side in ('left','right'):
        assert side+'Elbow' not in called
        elbow=point(pose,side+'Elbow')
        assert not elbow['valid'] and elbow['xyz_m'] is None
        assert elbow['reason']=='elbow_occluded_2d_tracking_cue'
        assert not pose['body']['arms'][side]['measurement_valid']
    assert 'leftShoulder' in called


def test_rgb_uses_exact_cached_pair_during_socket_interleaving_and_stale_hold(header,monkeypatch):
    import handdepth.visualization as v
    dash=Dashboard();h=body_header(header);depth=np.full((24,32),500,dtype='<u2')
    estimator=PoseEstimator();seen=[];original=v.tracking_overlay
    def capture(image,tracking,*a,**kw):
        seen.append((tracking['frame_id'],kw.get('held_all',False)))
        return original(image,tracking,*a,**kw)
    monkeypatch.setattr(v,'tracking_overlay',capture)
    for i in range(12):
        h=copy.deepcopy(h);h['frame_id']=i;h['capture_time_s']=100+i*.1
        h['processing_start_s']=h['processing_end_s']=h['capture_time_s']
        p,_=estimator.estimate(h,depth);p['received_monotonic_s']=1000+i*.1
        dash.sensor(h,depth,p,1000+i*.1)
        if i:
            dash.render(1000+i*.1)
            assert seen[-1][0]==i-1 # newest sensor must not draw on older RGB
        preview=decode(preview_packet(h));dash.preview(preview.header,preview.payload,1000+i*.1+.01)
        dash.render(1000+i*.1+.02);assert seen[-1]==(i,False)
    assert len(dash.sensor_frames)<=8 and len(dash.preview_frames)<=8
    dash.status='sensor_disconnected';result=dash.render(1005.)
    assert seen[-1]==(11,True) and 'HELD matched' in dash.rgb_status
    assert result.shape==(615,1440,3) and dash.metric_skeleton.points
    dash.reset_display();assert dash.matched is None and not dash.metric_skeleton.points


def test_rgb_phone_points_states_constraints_and_ages_are_preserved():
    p=measured_pose();p['body_tracking']={'points':[
        {'name':'leftElbow','x':.4,'y':.5,'state':'inferred','measurement_valid':False,
         'source_frame_id':0,'source_time_s':8.,'confidence':0.}],
        'arms':{'left':{'upper_arm_length_px':99,'forearm_length_px':42,'wrist_clamped':True}}}
    image=ImageSkeleton();tracking=image.update(p,None,1000.)
    assert tracking['points']==p['body_tracking']['points'] and tracking['arms']==p['body_tracking']['arms']
    assert image.source.clock(1002)-tracking['points'][0]['source_time_s']==4.
    image.reset();p=next_frame(p)
    assert image.update(p,None,1000.1)['points']==[] # reset never resurrects old held points


def test_geometry_identity_does_not_clear_on_rate_only_calibration_id_change(header):
    est=PoseEstimator();h=body_header(header);d=np.full((24,32),500,dtype='<u2')
    first,_=est.estimate(h,d);h['calibration_id']='rate-only-change';second,_=est.estimate(h,d)
    assert first['display_geometry_id']==second['display_geometry_id']
    h['calibration']['intrinsics_row_major'][0][0]*=1.1
    h['calibration_id']='intrinsics-change';third,_=est.estimate(h,d)
    assert second['display_geometry_id']!=third['display_geometry_id']


def test_lengths_need_three_stable_samples_and_reject_gross_occluder_shape():
    s=MetricSkeleton();p=measured_pose();s.update(p)
    assert not s.arms['left']['lengths_reliable']
    q=hide(next_frame(p),'leftElbow');s.update(q)
    assert s.arms['left']['mode']=='held'  # no single-frame metric extrapolation
    for i in range(3):
        p=next_frame(p);p['frame_id']+=10;s.update(p)
    assert s.arms['left']['lengths_reliable']
    bad=MetricSkeleton();p=measured_pose()
    point(p,'leftElbow')['xyz_torso_m']=[-.2,.5,0]
    point(p,'leftWrist')['xyz_torso_m']=[-.2,.56,0]
    establish(bad,p)
    assert not bad.arms['left']['lengths_reliable']


def test_previously_straight_arm_holds_without_inventing_unseen_bend_plane():
    p=measured_pose();s=MetricSkeleton()
    point(p,'leftElbow')['xyz_torso_m']=[-.2,.25,0]
    point(p,'leftWrist')['xyz_torso_m']=[-.2,.5,0]
    establish(s,p);old=[s.points['left'+n].xyz.copy() for n in ('Shoulder','Elbow','Wrist')]
    p=hide(next_frame(p),'leftElbow');point(p,'leftWrist')['xyz_torso_m']=[-.1,.4,.1]
    s.update(p)
    assert s.arms['left']['mode']=='held'
    assert np.array_equal(old,[s.points['left'+n].xyz for n in ('Shoulder','Elbow','Wrist')])


def test_matching_frame_number_with_different_preview_geometry_is_not_paired(header):
    d=Dashboard();h=body_header(header);depth=np.full((24,32),500,dtype='<u2')
    p,_=PoseEstimator().estimate(h,depth);d.sensor(h,depth,p,1000.)
    preview=decode(preview_packet(h));preview.header['calibration_id']='different'
    d.preview(preview.header,preview.payload,1000.01)
    assert d.matched is None
