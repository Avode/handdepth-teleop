import copy
from pathlib import Path
import os
import time
import numpy as np
import pytest
from handdepth.retarget import TORSO_TO_ROBOT, FrameTarget, Retargeter
from handdepth.geometry import PoseEstimator, quaternion_xyzw
from handdepth.body import rotation_matrix
from generate_fixtures import body_header


def measured(header):
    p,_=PoseEstimator().estimate(body_header(header),np.full((24,32),500,dtype='<u2'))
    p['received_monotonic_s']=10.;p['capture_to_receive_s']=.1
    return p


def home():
    return {'torso':FrameTarget(np.array([0.,0.,1.]),np.eye(3)),
        'head':FrameTarget(np.array([0.,0.,.3]),np.eye(3)),
        'arms':{s:{'length':.6,'elbow':np.array([.2,sign*.25,0.]),'wrist':np.array([.4,sign*.25,0.]),'rotation':np.eye(3)} for s,sign in [('left',1),('right',-1)]}}


def acquire(r,p):
    for _ in range(6):
        p['frame_id']+=1;p['capture_time_s']+=.13
        r.targets(p,10.)


def calibrated(header):
    p=measured(header);r=Retargeter(home());assert r.calibrate(p,10.)
    acquire(r,p)
    return p,r


def real_controller():
    pytest.importorskip('mink')
    from handdepth.teleop import Controller
    scene=Path(os.environ.get('HANDDEPTH_G2_SCENE','/mnt/robotics-data/robotics/agibot-g2/projects/handdepth/scene.xml'))
    if not scene.exists():pytest.skip('installed G2 scene requires Ubuntu asset deployment')
    return Controller(scene)


def advance(c,p,count=120):
    for i in range(count):
        now=10.+c.steps*.02
        p['received_monotonic_s']=now
        if i%7==0:p['frame_id']+=1;p['capture_time_s']+=.14
        c.step(p,now)


def test_axis_mapping_is_right_handed_and_anatomical():
    c=TORSO_TO_ROBOT
    assert np.allclose(c.T@c,np.eye(3)) and np.isclose(np.linalg.det(c),1)
    assert np.allclose(c@[1,0,0],[0,-1,0])
    assert np.allclose(c@[0,1,0],[0,0,-1])
    assert np.allclose(c@[0,0,1],[1,0,0])


def test_neutral_does_not_move_robot(header):
    p,r=calibrated(header);t=r.targets(p,10.)
    assert np.allclose(t['torso'].position,home()['torso'].position)
    for s in ('left','right'):
        assert np.allclose(t['arms'][s]['wrist'].position,home()['arms'][s]['wrist'])
        assert np.allclose(t['arms'][s]['wrist'].rotation,home()['arms'][s]['rotation'])
        assert np.allclose(t['arms'][s]['elbow'],home()['arms'][s]['elbow'])
    assert np.allclose(t['head'].rotation,np.eye(3))


def test_failed_manual_calibration_preserves_home_and_filter_state(header):
    c=real_controller();p=measured(header)
    assert c.calibrate(p,10.)
    advance(c,p,30)
    old_home=c.retarget.home
    old_filtered=dict(c.filtered);old_tasks=c.active_tasks.copy()
    assert old_tasks
    assert not c.calibrate(None,20.)
    assert c.retarget.home is old_home
    assert c.filtered==old_filtered and c.active_tasks==old_tasks
    assert c.retarget.calibrations==1 and c.retarget.armed


def test_persistent_display_cannot_resume_stale_robot_commands(header):
    from handdepth.display_tracking import MetricSkeleton
    c=real_controller();p=measured(header);assert c.calibrate(p,10.)
    advance(c,p,35);display=MetricSkeleton();display.update(p)
    assert display.points
    before=c.command_updates
    now=p['received_monotonic_s']+1.
    for i in range(20):
        display.update(p,fresh=False,now=now+i*.02)
        c.step(p,now+i*.02)
    assert display.points and all(v.state=='held' for v in display.points.values())
    assert c.command_updates==before and c.holding
    assert c.status()['base_displacement_m']==0 and c.retarget.calibrations==1


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('axis,expected',[(0,[0,-1,0]),(1,[0,0,-1]),(2,[1,0,0])])
def test_independent_anatomical_reach_directions(header,side,axis,expected):
    p,r=calibrated(header)
    p['body']['arms'][side]['positions_torso_m'][side+'Wrist'][axis]+=.05
    t=r.targets(p,10.)
    assert np.allclose(t['arms'][side]['wrist'].position-home()['arms'][side]['wrist'],np.array(expected)*r.segment_scales[side][1]*.05)
    other='right' if side=='left' else 'left'
    assert np.allclose(t['arms'][other]['wrist'].position,home()['arms'][other]['wrist'])


def test_pinch_open_close_and_invalid_hold(header):
    p,r=calibrated(header)
    h=next(h for h in p['hands'] if h['body_arm_side']=='left')
    h['pinch_distance_m']=.02
    assert np.isclose(r.targets(p,10.)['grippers']['left'],-.85)
    h['pinch_distance_m']=.08
    assert np.isclose(r.targets(p,10.)['grippers']['left'],-.05)
    for value in [None,float('nan'),.4,-.1]:
        h['pinch_distance_m']=value
        assert 'left' not in r.targets(p,10.)['grippers']


def test_invalid_parts_and_valid_wrist_with_missing_elbow(header):
    p,r=calibrated(header)
    p['body']['arms']['left']['torso_relative_valid']=False
    p['body']['head']['torso_relative_orientation_valid']=False
    t=r.targets(p,10.)
    assert t['arms']['left'] is None and t['arms']['right'] and t['head'] is None
    p['body']['arms']['left']['positions_torso_m']['leftElbow']=None
    t=r.targets(p,10.)
    assert t['arms']['left']['elbow'] is None and t['arms']['left']['wrist'] is not None


@pytest.mark.parametrize('failure',['stale','event','session','calibration','old_frame','wrong_frame','nonfinite_time'])
def test_freshness_identity_and_order_hold_without_replaying_stale_motion(header,failure):
    p,r=calibrated(header);saved=copy.deepcopy(p)
    if failure=='stale':p['received_monotonic_s']=9
    elif failure=='event':p['receiver_event']=True
    elif failure=='session':p['session_id']='different'
    elif failure=='calibration':p['calibration_id']='changed'
    elif failure=='old_frame':p['frame_id']-=1
    elif failure=='wrong_frame':p['coordinate_frame']='world'
    elif failure=='nonfinite_time':p['received_monotonic_s']=float('nan')
    assert r.targets(p,10.) is None
    assert r.targets(saved,10.) is None
    if failure in ('session','calibration','old_frame'):assert not r.armed
    else:
        assert r.armed and r.calibrations==1
        saved['frame_id']+=1
        assert r.targets(saved,10.) is not None


def test_common_camera_rigid_motion_leaves_all_robot_targets_fixed(header):
    p,r=calibrated(header);a=r.targets(p,10.)
    import mink
    rot=mink.SO3.exp(np.array([.24,-.35,.12])).as_matrix();shift=np.array([.5,-.3,.6])
    b=p['body'];torso=b['torso']
    r0=rotation_matrix(torso['quaternion_xyzw']);o=np.array(torso['position_m'])
    rt=rot@r0;ot=rot@o+shift
    torso.update(position_m=ot.tolist(),quaternion_xyzw=quaternion_xyzw(rt).tolist())
    for arm in b['arms'].values():
        for name,pos in arm['positions_camera_m'].items():
            if pos is not None:
                new=rot@np.array(pos)+shift
                arm['positions_camera_m'][name]=new.tolist()
                arm['positions_torso_m'][name]=(rt.T@(new-ot)).tolist()
    for item,pkey,qkey in [(b['head'],'position_m','quaternion_xyzw'),*[(h,'palm_position_m','palm_quaternion_xyzw') for h in p['hands']]]:
        if item[pkey] is not None:item[pkey]=(rot@np.array(item[pkey])+shift).tolist()
        if item[qkey] is not None:item[qkey]=quaternion_xyzw(rot@rotation_matrix(item[qkey])).tolist()
        relative_p='position_torso_m' if pkey=='position_m' else 'palm_position_torso_m'
        relative_q='quaternion_torso_xyzw' if qkey=='quaternion_xyzw' else 'palm_quaternion_torso_xyzw'
        item[relative_p]=(rt.T@(np.array(item[pkey])-ot)).tolist()
        item[relative_q]=quaternion_xyzw(rt.T@rotation_matrix(item[qkey])).tolist()
    t=r.targets(p,10.)
    assert np.allclose(a['torso'].position,t['torso'].position)
    assert np.allclose(a['torso'].rotation,t['torso'].rotation)
    assert np.allclose(a['head'].rotation,t['head'].rotation)
    for s in ('left','right'):
        assert np.allclose(a['arms'][s]['wrist'].position,t['arms'][s]['wrist'].position)
        assert np.allclose(a['arms'][s]['wrist'].rotation,t['arms'][s]['wrist'].rotation)


@pytest.mark.parametrize('axis',[0,1,2])
def test_head_and_palm_rotation_signs_and_neutral_tilt(header,axis):
    import mink
    p=measured(header)
    tilted=mink.SO3.exp(np.array([.3,-.2,.1])).as_matrix()
    p['body']['head']['quaternion_torso_xyzw']=quaternion_xyzw(tilted).tolist()
    h=next(x for x in p['hands'] if x['body_arm_side']=='left')
    h['palm_quaternion_torso_xyzw']=quaternion_xyzw(tilted).tolist()
    r=Retargeter(home());assert r.calibrate(p,10.);acquire(r,p)
    delta=np.zeros(3);delta[axis]=.15;turn=mink.SO3.exp(delta).as_matrix()
    p['body']['head']['quaternion_torso_xyzw']=quaternion_xyzw(turn@tilted).tolist()
    h['palm_quaternion_torso_xyzw']=quaternion_xyzw(turn@tilted).tolist()
    t=r.targets(p,10.)
    assert np.allclose(mink.SO3.from_matrix(t['head'].rotation).log(),TORSO_TO_ROBOT@delta)
    assert np.allclose(mink.SO3.from_matrix(t['arms']['left']['wrist'].rotation).log(),TORSO_TO_ROBOT@delta)


def test_occlusion_and_temporary_hand_id_change_preserve_neutral(header):
    p,r=calibrated(header);original=copy.deepcopy(r.arm_neutral);head0=r.head_neutral.copy()
    baseline=r.targets(p,10.)
    hidden=copy.deepcopy(p);hidden['body']['torso']['orientation_valid']=False
    for i in range(10):
        hidden['received_monotonic_s']=11+i
        assert r.targets(hidden,11+i) is None and r.armed
    p['frame_id']+=1;p['received_monotonic_s']=21
    p['hands'].reverse()
    for h in p['hands']:
        side=h['body_arm_side'];h['hand_id']='reacquired-'+side;p['body']['arms'][side]['hand_id']=h['hand_id']
    after=r.targets(p,21)
    assert r.calibrations==1 and np.allclose(r.head_neutral,head0)
    for side in ('left','right'):
        assert np.allclose(r.arm_neutral[side],original[side])
        assert np.allclose(after['arms'][side]['wrist'].rotation,baseline['arms'][side]['wrist'].rotation)
        assert np.allclose(after['arms'][side]['wrist'].position,baseline['arms'][side]['wrist'].position)
    # Known contradictory chirality never drives the other gripper/palm.
    h=p['hands'][0];side=h['body_arm_side'];h['chirality']='right' if side=='left' else 'left'
    after=r.targets(p,21)
    assert side not in after['grippers'] and after['arms'][side]['wrist'].rotation is None


def test_neutral_uses_distinct_frames_and_independent_parts(header):
    p=measured(header);r=Retargeter(home());arm=copy.deepcopy(p['body']['arms']['left'])
    p['body']['arms']['left']['torso_relative_valid']=False
    assert r.calibrate(p,10.,partial=True)
    for _ in range(50):r.targets(p,10.)
    assert not r.arm_neutral  # one camera frame is still just one sample
    acquire(r,p)
    assert 'right' in r.arm_neutral and 'left' not in r.arm_neutral
    p['body']['arms']['left']=arm;acquire(r,p)
    assert 'left' in r.arm_neutral and r.calibrations==1


def test_real_g2_mink_physics_static_anchor_convergence_and_loss(header):
    c=real_controller();p=measured(header);assert c.calibrate(p,10.)
    advance(c,p,80);start=c.data.qpos.copy();wrist0=c.data.site('l_wrist').xpos.copy()
    p['body']['arms']['left']['positions_torso_m']['leftWrist'][2]+=.04
    p['body']['arms']['left']['positions_torso_m']['leftElbow'][2]+=.02
    p['body']['torso']['position_m']=[1.,2.,3.]
    p['body']['torso']['quaternion_xyzw']=quaternion_xyzw(np.array([[0,-1,0],[1,0,0],[0,0,1]])).tolist()
    advance(c,p,200)
    assert c.error is None and c.command_updates>=200
    assert c.data.site('l_wrist').xpos[0]>wrist0[0]+.025
    assert c.max_limit_violation<.01 and np.isfinite(c.data.qvel).all()
    assert c.errors['l_wrist']<.025
    assert c.max_torso_deviation<1e-5 and c.max_torso_translation<1e-5
    assert np.allclose(c.data.ctrl[c.torso_aids],c.torso_anchor_q,atol=1e-12)
    assert sum(w.number for w in c.data.warning)==0
    n=c.command_updates;loss_time=30.;c.step(p,loss_time)
    assert c.retarget.armed and c.holding and c.command_updates==n
    frozen=c.data.ctrl.copy();p['received_monotonic_s']=loss_time
    for i in range(10):c.step(p,loss_time+i*.02)
    assert np.allclose(c.data.ctrl,frozen) and c.command_updates==n
    p['frame_id']+=1;p['received_monotonic_s']=30.2;c.step(p,30.2)
    assert c.command_updates>n and c.retarget.calibrations==1
    # Recovery target starts from actual held pose and remains rate bounded.
    assert np.max(np.abs(c.data.ctrl-frozen))<.021


@pytest.mark.parametrize('axis',[0,1,2])
def test_real_g2_head_axes_and_wrist_seventh_joint(header,axis):
    import mink
    c=real_controller();p=measured(header);assert c.calibrate(p,10.);advance(c,p,80)
    start=c.data.site('head_frame').xmat.reshape(3,3).copy()
    h=p['body']['head'];delta=np.zeros(3);delta[axis]=.08
    h['quaternion_torso_xyzw']=quaternion_xyzw(mink.SO3.exp(delta).as_matrix()@rotation_matrix(h['quaternion_torso_xyzw'])).tolist()
    advance(c,p,140)
    actual=mink.SO3.from_matrix(c.data.site('head_frame').xmat.reshape(3,3)@start.T).log()
    assert np.dot(actual,TORSO_TO_ROBOT@delta)>0.003
    assert c.orientation_errors['head_frame']<.035
    jacp=np.zeros((3,c.model.nv));jacr=np.zeros_like(jacp)
    c.mj.mj_jacSite(c.model,c.data,jacp,jacr,c.model.site('l_wrist').id)
    seventh=int(c.model.joint('idx27_arm_l_joint7').dofadr[0])
    assert np.linalg.norm(jacr[:,seventh])>.99
    assert c.max_torso_deviation<1e-5


@pytest.mark.asyncio
async def test_websocket_pose_to_real_g2_and_disconnect(header):
    import asyncio
    import websockets
    from handdepth.server import Receiver
    from handdepth.protocol import encode,decode
    receiver=Receiver();c=real_controller()
    async with websockets.serve(receiver.handler,'127.0.0.1',0,max_queue=1) as server:
        port=server.sockets[0].getsockname()[1]
        h=body_header(header);depth=np.full((24,32),500,dtype='<u2')
        async with websockets.connect(f'ws://127.0.0.1:{port}/sensor') as ws:
            await ws.send(encode(h,depth.tobytes()));assert decode(await ws.recv()).header['accepted']
            assert c.calibrate(receiver.latest_pose,time.monotonic())
            start=c.data.qpos.copy()
            for i in range(10):
                h['frame_id']+=1
                for key in ('capture_time_s','processing_start_s','processing_end_s'):h[key]+=.04
                if i>4:
                    for point in h['body']['landmarks']:
                        if point['name']=='leftWrist':point['y']+=.005
                await ws.send(encode(h,depth.tobytes()));assert decode(await ws.recv()).header['accepted']
                for _ in range(10):c.step(receiver.latest_pose,time.monotonic())
            assert c.error is None and c.command_updates>50
            assert np.linalg.norm(c.data.qpos-start)>.02
            await ws.send(encode(h,depth.tobytes()));assert not decode(await ws.recv()).header['accepted']
        await asyncio.sleep(.02)
        assert receiver.latest_pose['receiver_event']
        n=c.command_updates;c.step(receiver.latest_pose,time.monotonic())
        assert c.command_updates==n and c.holding
        async with websockets.connect(f'ws://127.0.0.1:{port}/sensor') as ws:
            # A duplicate capture remains rejected even on a new connection.
            await ws.send(encode(h,depth.tobytes()));assert not decode(await ws.recv()).header['accepted']
            c.step(receiver.latest_pose,time.monotonic());assert c.command_updates==n
            h['frame_id']+=1
            for key in ('capture_time_s','processing_start_s','processing_end_s'):h[key]+=.04
            await ws.send(encode(h,depth.tobytes()));assert decode(await ws.recv()).header['accepted']
            c.step(receiver.latest_pose,time.monotonic())
            assert c.command_updates>n and c.retarget.calibrations==1
