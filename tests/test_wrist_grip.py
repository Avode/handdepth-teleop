import copy
import numpy as np
import pytest
import mink
from handdepth.hand_control import FistLatch,finger_curl,forearm_frame
from handdepth.geometry import quaternion_xyzw,PoseEstimator
from generate_fixtures import body_header
from handdepth.retarget import TORSO_TO_ROBOT
from handdepth.teleop import Controller,DEFAULT_SCENE
from test_retarget import measured,advance
from test_elbow_mapping import set_limb


def fingers(closed=False):
    points=[dict(name='wrist',x=.5,y=.8,confidence=.99)]
    for finger,x in zip(('index','middle','ring','little'),(.35,.45,.55,.65)):
        shape=[(x,.55),(x,.42),(x+.03,.47),(x+.02,.57)] if closed else [(x,.55),(x,.42),(x,.32),(x,.22)]
        points += [dict(name=finger+n,x=p[0],y=p[1],confidence=.99) for n,p in zip(('MCP','PIP','DIP','Tip'),shape)]
    return {'hand_id':'hand','landmarks':points}


def evidence(closed,frame=1,t=1.):
    return dict(finger_curl(fingers(closed),(640,480)),source_frame_id=frame,capture_time_s=t)


@pytest.mark.parametrize('closed',[False,True])
def test_fist_shape_scale_rotation_and_mirror_invariant(closed):
    h=fingers(closed);baseline=finger_curl(h,(640,480));assert baseline['valid']
    assert baseline['score']>.65 if closed else baseline['score']<.3
    for scale in (.5,1.2):
        for mirror in (-1,1):
            transformed=copy.deepcopy(h);a=.65;r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            for p in transformed['landmarks']:
                q=(np.array([p['x'],p['y']])*[640,480]-[320,240])*[mirror,1]
                q=(scale*r@q+[320,240])/[640,480];p.update(x=q[0],y=q[1])
            assert finger_curl(transformed,(640,480))['score']==pytest.approx(baseline['score'])


def test_pinching_one_finger_is_not_fist_and_hidden_fingers_are_unknown():
    h=fingers();closed=fingers(True)
    for i,p in enumerate(h['landmarks']):
        if p['name'].startswith('index'):h['landmarks'][i]=closed['landmarks'][i]
    g=finger_curl(h,(640,480));l=FistLatch()
    for frame in (1,2,3):
        g.update(source_frame_id=frame,capture_time_s=frame*.13)
        assert l.update(g,frame,frame*.13,'hand') in (None,-.05)
    for p in h['landmarks']:
        if p['name'].startswith(('ring','little')) and not p['name'].endswith('MCP'):p['confidence']=.1
    assert not finger_curl(h,(640,480))['valid']


def test_fist_hysteresis_distinct_frames_loss_and_identity():
    l=FistLatch();g=evidence(True)
    for _ in range(10):assert l.update(g,1,1.,'a') is None
    assert l.update(evidence(True,2,1.13),2,1.13,'a')==-.85
    mid=evidence(True,3,1.26);mid['fingers']={n:.45 for n in mid['fingers']}
    assert l.update(mid,3,1.26,'a')==-.85
    assert l.update(evidence(False,4,1.39),4,1.39,'a')==-.85
    assert l.update(evidence(False,5,1.52),5,1.52,'a')==-.05
    assert l.update(None,6,1.65,'a') is None
    assert l.update(evidence(True,7,1.78),7,1.78,'b') is None
    assert l.update(evidence(True,8,1.91),8,1.91,'b')==-.85
    assert l.update(evidence(True,9,2.6),9,2.6,'b') is None
    assert l.update(evidence(True,10,2.7),11,2.7,'b') is None


def limb_controller(header):
    if not DEFAULT_SCENE.exists():pytest.skip('installed G2 scene required')
    c=Controller(DEFAULT_SCENE,arm_mapping='limb',wrist_mapping='relative',grip_mapping='fist')
    p=measured(header)
    for side in ('left','right'):
        u=c.anchor.rotation.T@c.shoulder_tasks[side].direction(c.cfg)
        f=c.anchor.rotation.T@c.forearm_tasks[side].direction(c.cfg)
        set_limb(p,side,u,f);set_palm(p,side,np.eye(3))
    assert c.calibrate(p,10.);advance(c,p,70)
    assert set(c.retarget.wrist_neutral)=={'left','right'}
    return c,p


def set_palm(p,side,relative):
    points=p['body']['arms'][side]['positions_torso_m']
    s,e,w=[np.array(points[side+j]) for j in ('Shoulder','Elbow','Wrist')]
    u=TORSO_TO_ROBOT@(e-s)/np.linalg.norm(e-s);f=TORSO_TO_ROBOT@(w-e)/np.linalg.norm(w-e)
    frame=forearm_frame(u,f)
    h=next(h for h in p['hands'] if h['body_arm_side']==side)
    h.update(orientation_valid=True,torso_relative_orientation_valid=True,
             palm_quaternion_torso_xyzw=quaternion_xyzw(TORSO_TO_ROBOT.T@frame@relative).tolist())


@pytest.mark.parametrize('side',['left','right'])
def test_wrist_relative_target_ignores_whole_arm_rotation_and_translation(header,side):
    c,p=limb_controller(header);r=c.retarget
    baseline=r.targets(p,p['received_monotonic_s'])['arms'][side]['wrist_rotation_relative']
    rotation=mink.SO3.exp(np.array([.2,-.3,.15])).as_matrix()
    points=p['body']['arms'][side]['positions_torso_m']
    p['body']['arms'][side]['positions_torso_m']={n:(TORSO_TO_ROBOT.T@rotation@TORSO_TO_ROBOT@np.array(v)+[.1,-.2,.3]).tolist() for n,v in points.items()}
    set_palm(p,side,np.eye(3));p['frame_id']+=1
    target=r.targets(p,p['received_monotonic_s'])['arms'][side]['wrist_rotation_relative']
    assert np.allclose(target,baseline)
    assert r.calibrations==1


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('axis',[0,1,2])
def test_relative_wrist_rotation_axes_and_missing_measurements(header,side,axis):
    c,p=limb_controller(header);r=c.retarget;n=r.wrist_neutral[side]
    v=np.zeros(3);v[axis]=.25;delta=mink.SO3.exp(v).as_matrix()
    set_palm(p,side,delta@n['human'])
    result=r.targets(p,p['received_monotonic_s'])
    assert np.allclose(result['arms'][side]['wrist_rotation_relative'],n['axes']@delta@n['axes'].T@n['robot'])
    hand=next(h for h in p['hands'] if h['body_arm_side']==side)
    hand['orientation_valid']=False
    assert 'wrist_rotation_relative' not in r.targets(p,p['received_monotonic_s'])['arms'][side]
    hand['orientation_valid']=True
    p['body_tracking']={'points':[{'name':side+'Elbow','measurement_valid':False,'state':'inferred'}]}
    assert r.targets(p,p['received_monotonic_s'])['arms'][side] is None
    assert side in r.wrist_neutral


def test_wrist_jacobian_does_not_recruit_upstream_joints():
    c=Controller(DEFAULT_SCENE,arm_mapping='limb',wrist_mapping='relative');q=c.cfg.q.copy()
    for side,task in c.wrist_tasks.items():
        task.set_target_from_configuration(c.cfg);jac=task.compute_jacobian(c.cfg)[3:]
        assert np.max(np.abs(jac[:,c.groups[side][:4]+c.groups['torso']]))<1e-10
        for dof in c.groups[side][4:]:
            dq=np.zeros(c.model.nv);dq[dof]=1e-6
            c.cfg.update(q);c.cfg.integrate_inplace(dq,1);a=task.compute_error(c.cfg)[3:]
            c.cfg.update(q);c.cfg.integrate_inplace(-dq,1);b=task.compute_error(c.cfg)[3:]
            assert np.allclose((a-b)/2e-6,jac[:,dof],atol=1e-6)
        c.cfg.update(q)


def test_both_wrists_converge_in_g2_physics_with_limb_and_torso_preserved(header):
    c,p=limb_controller(header);q=c.cfg.q.copy()
    for side in ('left','right'):
        target=q.copy()
        indices=[int(c.model.jnt_qposadr[c.model.dof_jntid[i]]) for i in c.groups[side][4:]]
        target[indices]+=[.25,-.2,.18]
        c.cfg.update(target);desired=c.wrist_rotation(side);c.cfg.update(q)
        n=c.retarget.wrist_neutral[side]
        relative=n['axes'].T@desired@n['robot'].T@n['axes']@n['human']
        set_palm(p,side,relative)
    advance(c,p,220)
    for side in ('left','right'):
        assert c.wrist_status()[side]['active']
        assert c.wrist_status()[side]['angular_error_deg']<2
        assert c.shoulder_status()[side]['angular_error_deg']<1
        assert c.elbow_status()[side]['angular_error_deg']<2
    assert c.ik_failures==0 and sum(w.number for w in c.data.warning)==0
    assert c.status()['base_displacement_m']==0 and c.max_torso_deviation<1e-4
    assert c.max_limit_violation<.02 and c.max_joint_speed<1.2
    # Measurement loss cancels pending wrist servo motion while shoulders continue.
    p['hands'][0]['torso_relative_orientation_valid']=False;advance(c,p,1)
    side=p['hands'][0]['body_arm_side'];aids=[i for i,d in enumerate(c.dofs) if d in c.groups[side][4:]]
    held=c.data.ctrl[aids].copy();advance(c,p,20);assert np.array_equal(c.data.ctrl[aids],held)
    p['hands'][0]['torso_relative_orientation_valid']=True;advance(c,p,80)
    assert c.wrist_status()[side]['active'] and c.retarget.calibrations==1
    c.step(None,100);held=c.data.ctrl.copy();updates=c.command_updates
    for i in range(20):c.step(None,100+i*.02)
    assert c.command_updates==updates and np.array_equal(c.data.ctrl,held)


def test_fist_gripper_commands_and_session_reset(header):
    c,p=limb_controller(header);r=c.retarget
    for closed in (True,False):
        for _ in range(3):
            p['frame_id']+=1;p['capture_time_s']+=.13
            for hand in p['hands']:hand['grip_gesture']=evidence(closed,p['frame_id'],p['capture_time_s'])
            result=r.targets(p,p['received_monotonic_s'])
        assert all(result['grippers'][s]==(-.85 if closed else -.05) for s in ('left','right'))
    p['session_id']='new'
    assert r.targets(p,p['received_monotonic_s']) is None and not r.armed
    assert r.calibrate(p,p['received_monotonic_s'],partial=True)
    assert not r.wrist_neutral and all(f.closed is None for f in r.fists.values())


def test_gesture_uses_current_rgb_even_without_fingertip_depth(header):
    h=body_header(header);hand=h['hands'][0];raw=fingers(True)
    known={p['name']:p for p in raw['landmarks']}
    for p in hand['landmarks']:
        if p['name'] in known:p.update(known[p['name']])
    p,_=PoseEstimator().estimate(h,np.zeros((24,32),dtype='<u2'))
    hand=p['hands'][0];g=hand['grip_gesture']
    assert not hand['depth_valid'] and not hand['orientation_valid']
    assert g['valid'] and g['score']>.65 and g['space']=='color_pixels'
    assert g['source_frame_id']==h['frame_id'] and g['capture_time_s']==h['capture_time_s']


def test_palm_orientation_needs_three_measured_anchors_not_five(header):
    estimator=PoseEstimator();depth=np.full((24,32),500,dtype='<u2')
    full,_=estimator.estimate(header,depth)
    for joint in header['landmarks']:
        if joint['name'] in ('ringMCP','middleMCP'):joint['confidence']=.1
    header['frame_id']+=1;header['capture_time_s']+=.13
    partial,_=estimator.estimate(header,depth)
    assert partial['orientation_valid']
    assert np.allclose(partial['palm_quaternion_xyzw'],full['palm_quaternion_xyzw'])
    next(p for p in header['landmarks'] if p['name']=='littleMCP')['confidence']=.1
    missing,_=estimator.estimate(header,depth)
    assert not missing['orientation_valid']


def test_straight_arm_and_ambiguous_association_hold_wrist(header):
    c,p=limb_controller(header);saved=copy.deepcopy(p)
    u=c.anchor.rotation.T@c.shoulder_tasks['left'].direction(c.cfg)
    set_limb(p,'left',u,u)
    r=c.retarget;result=r.targets(p,p['received_monotonic_s'])
    assert 'wrist_rotation_relative' not in result['arms']['left']
    assert 'wrist_rotation_relative' in result['arms']['right']
    p=saved;p['frame_id']+=1
    h=next(h for h in p['hands'] if h['body_arm_side']=='left');h['chirality']='right'
    assert 'wrist_rotation_relative' not in r.targets(p,p['received_monotonic_s'])['arms']['left']
    h['chirality']='left';p['frame_id']+=1
    assert 'wrist_rotation_relative' in r.targets(p,p['received_monotonic_s'])['arms']['left']
    assert r.calibrations==1


def test_fist_physically_closes_and_open_hand_releases_both_grippers(header):
    c,p=limb_controller(header);gaps=[]
    for closed in (True,False):
        for i in range(180):
            now=10+c.steps*.02;p['received_monotonic_s']=now
            if i%7==0:p['frame_id']+=1;p['capture_time_s']+=.14
            for h in p['hands']:h['grip_gesture']=evidence(closed,p['frame_id'],p['capture_time_s'])
            c.step(p,now)
        current=[]
        for side in ('l','r'):
            a=c.data.xpos[c.model.body('gripper_'+side+'_left_support_link').id]
            b=c.data.xpos[c.model.body('gripper_'+side+'_right_support_link').id]
            current.append(np.linalg.norm(a-b))
        gaps.append(current)
    assert np.all(np.array(gaps[1])-gaps[0]>.05)
    assert c.ik_failures==0 and sum(w.number for w in c.data.warning)==0
    # No current gesture -> hold, even if the previous state was a confirmed fist.
    for h in p['hands']:h['grip_gesture']=None
    advance(c,p,1);aids=[c.model.actuator(n).id for n in ('idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1')]
    hold=c.data.ctrl[aids].copy();advance(c,p,30);assert np.array_equal(hold,c.data.ctrl[aids])
