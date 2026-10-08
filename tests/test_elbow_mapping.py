import copy
import numpy as np
import pytest
from handdepth.retarget import Retargeter,TORSO_TO_ROBOT,measured_forearm
from handdepth.shoulder_task import unit,angle
from handdepth.teleop import Controller,DEFAULT_SCENE
from test_retarget import measured,home,acquire,advance


def set_limb(p,side,upper,forearm,lengths=(.30,.25)):
    arm=p['body']['arms'][side];points=arm['positions_torso_m']
    shoulder=np.asarray(points[side+'Shoulder'])
    elbow=shoulder+lengths[0]*TORSO_TO_ROBOT.T@unit(upper)
    wrist=elbow+lengths[1]*TORSO_TO_ROBOT.T@unit(forearm)
    points[side+'Elbow']=elbow.tolist();points[side+'Wrist']=wrist.tolist()
    arm.update(measurement_valid=True,torso_relative_valid=True,wrist_source='vision_body_wrist_surface')
    for joint in p['body']['landmarks']:
        if joint['name'] in (side+'Shoulder',side+'Elbow',side+'Wrist'):joint['valid']=True


def armed(header):
    p=measured(header)
    for side in ('left','right'):set_limb(p,side,[0,0,-1],[1,0,0])
    r=Retargeter(home(),arm_mapping='limb');assert r.calibrate(p,10.,partial=True)
    acquire(r,p)
    return p,r


def controller():
    if not DEFAULT_SCENE.exists():pytest.skip('installed G2 scene required')
    return Controller(DEFAULT_SCENE,arm_mapping='limb')


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('forearm',[[1,0,0],[0,1,0],[0,-1,0],[.5,.4,-.5]])
def test_forearm_absolute_angles_independent_of_scale_translation_and_palm(header,side,forearm):
    p,r=armed(header)
    for lengths in ((.2,.18),(.4,.5)):
        set_limb(p,side,[0,0,-1],forearm,lengths)
        p['body']['arms'][side]['positions_torso_m']={n:(np.array(v)+[.2,-.1,.7]).tolist() for n,v in p['body']['arms'][side]['positions_torso_m'].items()}
        a=r.targets(p,10.)['arms'][side]
        assert np.allclose(a['forearm_direction'],unit(forearm))
        assert a['elbow_flexion_rad']==pytest.approx(angle(np.array([0,0,-1]),unit(forearm)))
        assert 'wrist' not in a
    assert r.calibrate(p,10.,partial=True) and not r.elbow_ready
    acquire(r,p)
    assert np.allclose(r.targets(p,10.)['arms'][side]['forearm_direction'],unit(forearm))


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('invalid',['missing','invalid_landmark','held','inferred','zero','nan'])
def test_bad_wrist_holds_elbow_but_keeps_shoulder_and_other_arm(header,side,invalid):
    p,r=armed(header);saved=copy.deepcopy(p)
    points=p['body']['arms'][side]['positions_torso_m']
    if invalid=='missing':points[side+'Wrist']=None
    elif invalid=='invalid_landmark':next(x for x in p['body']['landmarks'] if x['name']==side+'Wrist')['valid']=False
    elif invalid in ('held','inferred'):p['body_tracking']={'points':[dict(name=side+'Wrist',measurement_valid=False,state=invalid)]}
    elif invalid=='zero':points[side+'Wrist']=points[side+'Elbow']
    else:points[side+'Wrist']=[float('nan'),0,0]
    a=r.targets(p,10.)['arms']
    assert 'upper_arm_direction' in a[side] and 'forearm_direction' not in a[side]
    assert 'forearm_direction' in a['left' if side=='right' else 'right']
    saved['frame_id']+=1
    assert 'forearm_direction' in r.targets(saved,10.)['arms'][side] and r.calibrations==1


def test_current_associated_hand_wrist_requires_matching_measured_point(header):
    p,r=armed(header);side='left';arm=p['body']['arms'][side]
    next(x for x in p['body']['landmarks'] if x['name']=='leftWrist')['valid']=False
    arm['wrist_source']='associated_vision_hand_wrist_surface';arm['hand_id']='current-hand'
    point=dict(name='wrist',valid=True,xyz_torso_m=arm['positions_torso_m']['leftWrist'])
    p['hands']=[dict(hand_id='current-hand',body_arm_side='left',landmarks=[point])]
    assert measured_forearm(p,side)['wrist_source']=='associated_current_hand'
    point['valid']=False;assert measured_forearm(p,side) is None
    point['valid']=True;p['hands'][0]['hand_id']='old-hand';assert measured_forearm(p,side) is None


def test_bend_plane_hysteresis_and_source_reset(header):
    p,r=armed(header)
    for degrees,expected in [(5,False),(9,False),(14,True),(9,True),(5,False),(175,False)]:
        theta=np.radians(degrees);set_limb(p,'left',[0,0,-1],[np.sin(theta),0,-np.cos(theta)])
        assert r.targets(p,10.)['arms']['left']['bend_plane_observable']==expected
    p['display_geometry_id']='new'
    assert r.targets(p,10.) is None and not r.armed
    assert r.calibrate(p,10.,partial=True) and not r.elbow_ready
    for _ in range(10):r.targets(p,10.)
    assert not r.elbow_ready
    acquire(r,p);assert r.elbow_ready=={'left','right'}


def test_forearm_jacobian_matches_actual_g2_finite_difference():
    c=controller();q=c.cfg.q.copy();eps=1e-6
    for task in c.forearm_tasks.values():
        task.set_target(task.direction(c.cfg));j=task.compute_jacobian(c.cfg)
        for i in range(c.model.nv):
            dq=np.zeros(c.model.nv);dq[i]=eps
            c.cfg.update(q);c.cfg.integrate_inplace(dq,1.);plus=task.compute_error(c.cfg)
            c.cfg.update(q);c.cfg.integrate_inplace(-dq,1.);minus=task.compute_error(c.cfg)
            assert np.allclose(j[:,i],(plus-minus)/(2*eps),atol=1e-6)
        c.cfg.update(q)


def reachable_pose(c,header,delta):
    p=measured(header);q0=c.cfg.q.copy()
    for side,sign in [('left',1),('right',-1)]:
        q=q0.copy();indices=[int(c.model.jnt_qposadr[c.model.dof_jntid[i]]) for i in c.groups[side][:4]]
        q[indices]+=np.asarray(delta)*[sign,1,sign,1]
        c.cfg.update(q)
        set_limb(p,side,c.anchor.rotation.T@c.shoulder_tasks[side].direction(c.cfg),c.anchor.rotation.T@c.forearm_tasks[side].direction(c.cfg))
    c.cfg.update(q0)
    return p


def test_both_g2_forearms_converge_with_shoulders_wrist_and_torso_preserved(header):
    # Outward bend planes keep the two grippers separated throughout the move.
    c=controller();p=reachable_pose(c,header,[.1,-.15,-.5,.5]);assert c.calibrate(p,10.)
    advance(c,p,40)
    wrist=[i for i,x in enumerate(c.dofs) if x in c.groups['left'][4:]+c.groups['right'][4:]]
    wrist_ctrl=c.data.ctrl[wrist].copy();advance(c,p,300)
    assert np.allclose(c.data.ctrl[wrist],wrist_ctrl,atol=1e-12)
    for side in ('left','right'):
        assert c.shoulder_status()[side]['angular_error_deg']<1
        elbow=c.elbow_status()[side]
        assert elbow['active'] and elbow['angular_error_deg']<2 and elbow['flexion_error_deg']<2
        assert elbow['axial_rotation_active'] and not elbow['inferred_control']
    assert c.ik_failures==0 and sum(w.number for w in c.data.warning)==0
    assert c.status()['base_displacement_m']==0 and c.max_torso_deviation<1e-4
    assert c.max_limit_violation<.02 and c.max_joint_speed<1.2
    saved=copy.deepcopy(p);p['body']['arms']['left']['positions_torso_m']['leftWrist']=None
    advance(c,p,1)
    elbow_aids=[i for i,x in enumerate(c.dofs) if x in c.groups['left'][2:4]]
    held=c.data.ctrl[elbow_aids].copy();advance(c,p,30)
    assert np.array_equal(c.data.ctrl[elbow_aids],held)
    assert c.shoulder_status()['left']['active'] and not c.elbow_status()['left']['active']
    saved['frame_id']=p['frame_id']+1;advance(c,saved,60)
    assert c.elbow_status()['left']['active'] and c.retarget.calibrations==1
    before=c.command_updates;c.step(None,20);held=c.data.ctrl.copy()
    for i in range(20):c.step(None,20+i*.02)
    assert c.command_updates==before and np.array_equal(c.data.ctrl,held)


def test_near_straight_keeps_axial_rotation_fixed_and_unreachable_is_bounded(header):
    c=controller();p=reachable_pose(c,header,[0,0,0,0])
    for side in ('left','right'):
        upper=c.anchor.rotation.T@c.shoulder_tasks[side].direction(c.cfg)
        set_limb(p,side,upper,upper)
    assert c.calibrate(p,10.);advance(c,p,30)
    twist=[i for i,x in enumerate(c.dofs) if x in [c.groups[s][2] for s in ('left','right')]]
    held=c.data.ctrl[twist].copy();advance(c,p,100)
    assert np.array_equal(c.data.ctrl[twist],held)
    assert all(not s['axial_rotation_active'] and s['active'] for s in c.elbow_status().values())
    for side in ('left','right'):set_limb(p,side,[0,0,-1],[0,0,1])
    advance(c,p,160)
    assert c.max_limit_violation<.02 and c.ik_failures==0 and np.isfinite(c.data.qpos).all()
    assert c.status()['base_displacement_m']==0


def test_body_blocked_forearm_preserves_shoulder_instead_of_contact_stall(header):
    # Regression from a live pose: forearms point into the lower robot torso.
    # Human and robot shapes differ, so exact simultaneous imitation is infeasible.
    c=controller();p=measured(header)
    for side,upper,fore in [('left',[-.58,.257,-.773],[.013,-.708,-.706]),
                            ('right',[-.629,-.386,-.675],[.289,.654,-.699])]:
        set_limb(p,side,upper,fore)
    assert c.calibrate(p,10.);advance(c,p,400)
    assert all(s['angular_error_deg']<1 for s in c.shoulder_status().values())
    assert all(s['angular_error_deg']>10 for s in c.elbow_status().values())
    assert c.data.ncon==0 and c.ik_failures==0 and c.max_joint_speed<1.2
    assert c.status()['base_displacement_m']==0
