import copy
import numpy as np
import pytest
from handdepth.retarget import Retargeter, TORSO_TO_ROBOT, measured_upper_arm
from handdepth.shoulder_task import angle, approach_direction, unit
from handdepth.teleop import Controller, DEFAULT_SCENE
from handdepth.geometry import quaternion_xyzw
from test_retarget import measured, home, acquire, advance


def set_direction(pose, side, direction, length=.3):
    points=pose['body']['arms'][side]['positions_torso_m']
    points[side+'Elbow']=(np.asarray(points[side+'Shoulder'])+
                         length*TORSO_TO_ROBOT.T@unit(direction)).tolist()


def armed(header):
    p=measured(header)
    r=Retargeter(home(),arm_mapping='shoulder')
    assert r.calibrate(p,10.,partial=True)
    acquire(r,p)
    return p,r


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('direction',[[1,0,0],[0,1,0],[0,-1,0],[0,0,-1],[.4,.3,-.8]])
def test_absolute_direction_not_neutral_offset_or_wrist_target(header,side,direction):
    p,r=armed(header)
    for length in [.12,.3,.6]:
        set_direction(p,side,direction,length)
        arm=p['body']['arms'][side]
        arm['positions_torso_m'][side+'Wrist']=None
        arm['measurement_valid']=arm['torso_relative_valid']=False
        a=r.targets(p,10.)['arms'][side]
        assert np.allclose(a['upper_arm_direction'],unit(direction))
        assert 'wrist' not in a and 'elbow' not in a
    # C does not zero away the measured angle.
    assert r.calibrate(p,10.,partial=True)
    assert not r.shoulder_ready
    acquire(r,p)
    assert np.allclose(r.targets(p,10.)['arms'][side]['upper_arm_direction'],unit(direction))


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('invalid',['missing','invalid_landmark','inferred','held','zero_length'])
def test_occluded_elbow_cannot_become_control_from_persistent_display(header,side,invalid):
    p,r=armed(header);saved=copy.deepcopy(p)
    if invalid=='missing':p['body']['arms'][side]['positions_torso_m'][side+'Elbow']=None
    elif invalid=='invalid_landmark':
        next(x for x in p['body']['landmarks'] if x['name']==side+'Elbow')['valid']=False
    elif invalid in ('inferred','held'):
        # Even contradictory retained XYZ must not bypass the phone veto.
        p['body_tracking']={'points':[{'name':side+'Elbow','state':invalid,'measurement_valid':False}]}
    else:set_direction(p,side,[0,0,-1],length=0)
    assert r.targets(p,10.)['arms'][side] is None
    assert r.targets(p,10.)['arms']['right' if side=='left' else 'left'] is not None
    saved['frame_id']+=1
    assert r.targets(saved,10.)['arms'][side] is not None
    assert r.calibrations==1


def test_shoulder_camera_motion_and_partial_torso_loss(header):
    import mink
    p,r=armed(header);before=r.targets(p,10.)
    # Camera/torso pose moves; its local limb directions do not.
    p['body']['torso']['position_m']=[.3,.2,1.2]
    p['body']['torso']['quaternion_xyzw']=quaternion_xyzw(mink.SO3.exp(np.array([.3,.2,.5])).as_matrix()).tolist()
    after=r.targets(p,10.)
    for s in ['left','right']:
        assert np.allclose(before['arms'][s]['upper_arm_direction'],after['arms'][s]['upper_arm_direction'])
    p['body']['torso']['orientation_valid']=False
    assert r.targets(p,10.) is None and r.calibrations==1
    p['body']['torso']['orientation_valid']=True;p['frame_id']+=1
    assert r.targets(p,10.)['arms']['left'] is not None


@pytest.mark.parametrize('field',['session_id','calibration_id','display_geometry_id'])
def test_new_source_and_reset_require_new_shoulder_acquisition(header,field):
    p,r=armed(header);p[field]='changed'
    assert r.targets(p,10.) is None and not r.armed
    assert r.calibrate(p,10.,partial=True) and not r.shoulder_ready
    # Repeated controller ticks cannot count as distinct source measurements.
    for _ in range(10):assert r.targets(p,10.)['arms']['left'] is None
    acquire(r,p)
    assert r.targets(p,10.)['arms']['left'] is not None


def test_direction_filter_is_bounded_for_antipodal_and_degenerate_inputs():
    a=np.array([0.,0.,-1.]);desired=-a
    for _ in range(200):
        b=approach_direction(a,desired,.02)
        assert np.isclose(np.linalg.norm(b),1.) and angle(a,b)<=1.4*.02+1e-12
        a=b
    assert angle(a,desired)<1e-6
    with pytest.raises(ValueError):unit([0,0,0])


def controller():
    if not DEFAULT_SCENE.exists():pytest.skip('installed G2 scene required')
    return Controller(DEFAULT_SCENE,arm_mapping='shoulder')


def test_direction_task_jacobian_matches_finite_difference():
    c=controller();q=c.cfg.q.copy();eps=1e-6
    for task in c.shoulder_tasks.values():
        task.set_target(task.direction(c.cfg))
        j=task.compute_jacobian(c.cfg)
        for i in range(c.model.nv):
            dq=np.zeros(c.model.nv);dq[i]=eps
            c.cfg.update(q);c.cfg.integrate_inplace(dq,1.);plus=task.compute_error(c.cfg)
            c.cfg.update(q);c.cfg.integrate_inplace(-dq,1.);minus=task.compute_error(c.cfg)
            assert np.allclose(j[:,i],(plus-minus)/(2*eps),atol=1e-6)
        c.cfg.update(q)


def test_real_g2_both_shoulders_converge_distal_and_torso_commands_fixed(header):
    c=controller();p=measured(header)
    initial=c.cfg.q.copy()
    desired={}
    for side in ['left','right']:
        q=initial.copy()
        indices=[int(c.model.jnt_qposadr[c.model.dof_jntid[i]]) for i in c.shoulder_dofs[side]]
        q[indices]+=np.array([.3 if side=='left' else -.3,-.35])
        c.cfg.update(q)
        desired[side]=c.anchor.rotation.T@c.shoulder_tasks[side].direction(c.cfg)
        set_direction(p,side,desired[side])
    c.cfg.update(initial)
    assert c.calibrate(p,10.)
    advance(c,p,35)
    frozen=[i for i,x in enumerate(c.dofs) if x in c.groups['torso']+c.groups['left'][2:]+c.groups['right'][2:]]
    ctrl=c.data.ctrl[frozen].copy()
    advance(c,p,350)
    assert np.allclose(c.data.ctrl[frozen],ctrl,atol=1e-12)
    for side,s in c.shoulder_status().items():
        assert s['active'] and s['angular_error_deg']<1
        assert not s['axial_twist_observed'] and not s['inferred_control']
    assert c.active_tasks=={'l_upper_arm','r_upper_arm','head_frame'}
    assert c.status()['base_displacement_m']==0 and c.max_torso_deviation<2e-5
    assert c.ik_failures==0 and sum(w.number for w in c.data.warning)==0
    # Fresh source, missing left elbow: cancel its shoulder trajectory and hold.
    left=[i for i,x in enumerate(c.dofs) if x in c.groups['left']]
    original=copy.deepcopy(p)
    p['body']['arms']['left']['positions_torso_m']['leftElbow']=None
    advance(c,p,1);held=c.data.ctrl[left].copy();advance(c,p,20)
    assert np.allclose(c.data.ctrl[left],held) and not c.shoulder_status()['left']['active']
    original['frame_id']=p['frame_id']+1
    advance(c,original,50)
    assert c.shoulder_status()['left']['active'] and c.retarget.calibrations==1
    # Stale/disconnected input keeps commands constant, even with XYZ retained.
    before=c.command_updates;now=original['received_monotonic_s']+2
    c.step(original,now);held=c.data.ctrl.copy()
    for i in range(20):c.step(original,now+(i+1)*.02)
    assert c.command_updates==before and np.allclose(c.data.ctrl,held)


def test_large_shoulder_request_keeps_limits_and_reports_residual(header):
    c=controller();p=measured(header)
    for s in ['left','right']:set_direction(p,s,[0,1 if s=='right' else -1,0])
    assert c.calibrate(p,10.);advance(c,p,200)
    assert np.isfinite(c.data.qpos).all() and c.max_limit_violation<.02
    assert c.ik_failures==0 and c.status()['base_displacement_m']==0
    assert all(x['angular_error_deg'] is not None for x in c.shoulder_status().values())


def test_shoulder_servos_follow_a_large_reachable_step_without_creeping(header):
    c=controller();p=measured(header);q0=c.cfg.q.copy()
    shoulder_dofs=set(c.shoulder_dofs['left']+c.shoulder_dofs['right'])
    assert all(v==0 for i,v in enumerate(c.shoulder_servo_lead) if c.dofs[i] not in shoulder_dofs)
    assert np.max(c.shoulder_servo_lead)<=.12
    for side in ['left','right']:
        q=q0.copy()
        indices=[int(c.model.jnt_qposadr[c.model.dof_jntid[i]]) for i in c.shoulder_dofs[side]]
        q[indices]+=np.array([-.7 if side=='left' else .7,.3])
        c.cfg.update(q)
        set_direction(p,side,c.anchor.rotation.T@c.shoulder_tasks[side].direction(c.cfg))
    c.cfg.update(q0);assert c.calibrate(p,10.)
    advance(c,p,100)
    assert all(s['angular_error_deg']<1 for s in c.shoulder_status().values())
    assert c.max_joint_speed<1.1 and c.ik_failures==0
