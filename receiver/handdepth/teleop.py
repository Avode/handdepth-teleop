"""G2 simulation controller: measured poses -> Mink -> bounded position servos."""
import asyncio
from collections import deque
import json
import logging
import os
from pathlib import Path
from queue import SimpleQueue
import threading
import time
import numpy as np
from .retarget import Retargeter, FrameTarget, SIDES, bounded
from .recording import check_recording_root, Recorder, DEFAULT_ROOT
from .server import Receiver

log=logging.getLogger('handdepth.teleop')
DEFAULT_SCENE=Path(os.environ.get('HANDDEPTH_G2_SCENE','/mnt/robotics-data/robotics/agibot-g2/projects/handdepth/scene.xml'))


class Controller:
    def __init__(self, scene, dt=.02, arm_mapping='cartesian', wrist_mapping='hold', grip_mapping='pinch'):
        import mink
        import mujoco
        self.mk,self.mj=mink,mujoco
        self.model=mujoco.MjModel.from_xml_path(str(scene))
        self.data=mujoco.MjData(self.model)
        m,d=self.model,self.data
        if abs(dt/m.opt.timestep-round(dt/m.opt.timestep))>1e-6:raise ValueError('control dt must be a multiple of physics dt')
        self.dt=dt;self.substeps=round(dt/m.opt.timestep)
        mujoco.mj_resetDataKeyframe(m,d,m.key('home').id);mujoco.mj_forward(m,d)
        self.cfg=mink.Configuration(m);self.cfg.update(d.qpos)
        self.joint_names=[m.joint(int(m.actuator_trnid[i,0])).name for i in range(m.nu)]
        self.qadr=np.array([m.joint(n).qposadr[0] for n in self.joint_names])
        self.dofs=np.array([m.joint(n).dofadr[0] for n in self.joint_names])
        self.groups={}
        for group,token in [('torso','_body_joint'),('head','_head_joint'),('left','_arm_l_'),('right','_arm_r_')]:
            self.groups[group]=[int(m.joint(n).dofadr[0]) for n in self.joint_names if token in n]
        if [len(self.groups[x]) for x in ('torso','head','left','right')]!=[5,3,7,7]:
            raise ValueError('scene must have restored G2 torso/head and both seven-joint arms')
        self.torso_aids=np.array([i for i,x in enumerate(self.dofs) if x in self.groups['torso']])
        self.torso_qadr=self.qadr[self.torso_aids]
        self.torso_anchor_q=d.qpos[self.torso_qadr].copy()
        if any(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,f'fixed_torso_{i}')<0 for i in range(1,6)):
            raise ValueError('scene needs fixed_torso_1..5 physical joint locks; rebuild the HandDepth scene')
        self.anchor=self.transform('torso_frame')
        self.base_body_id=max(1,mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base_link'))
        self.base_anchor=d.xpos[self.base_body_id].copy()
        speeds={n:(.35 if '_body_joint' in n else .65 if '_head_joint' in n else 1.) for n in self.joint_names}
        self.limits=[mink.ConfigurationLimit(m,gain=.85),mink.VelocityLimit(m,speeds)]
        # Signed distances between selected distal arm meshes and torso/opposite arm.
        # Adjacent links are excluded. Physics also retains model contact handling.
        collision={s:[i for i in range(m.ngeom) if m.geom(i).name and m.geom(i).name.endswith('collision_0') and any(k in m.geom(i).name for k in (f'arm_{s}_link5',f'arm_{s}_link6',f'arm_{s}_link7',f'gripper_{s}_base'))] for s in ('l','r')}
        body=[i for i in range(m.ngeom) if m.geom(i).name and m.geom(i).name.startswith('body_link5_collision')]
        if arm_mapping=='limb':
            # Moving elbows can bring wrists into lower torso links and gripper
            # fingers into the opposite hand. The shoulder-only subset did not
            # cover these contacts and physical collisions could stall a shoulder.
            collision={s:[i for i in range(m.ngeom) if m.geom(i).name and '_collision_' in m.geom(i).name
                          and (any(f'arm_{s}_link{k}_' in m.geom(i).name for k in (4,5,6,7))
                               or m.geom(i).name.startswith(f'gripper_{s}_'))] for s in ('l','r')}
            body=[i for i in range(m.ngeom) if m.geom(i).name and m.geom(i).name.startswith('body_link') and '_collision_' in m.geom(i).name]
        self.limits.append(mink.CollisionAvoidanceLimit(m,[(collision['l'],collision['r']),(collision['l'],body),(collision['r'],body)],minimum_distance_from_collisions=.015,collision_detection_distance=.06))
        self.tasks={}
        for name,pc,rc in [('head_frame',0.,.7),('l_elbow',.08,0),('r_elbow',.08,0),('l_wrist',1.,.18),('r_wrist',1.,.18)]:
            self.tasks[name]=mink.FrameTask(name,'site',pc,rc,gain=.7,lm_damping=1e-3)
        from .shoulder_task import UpperArmDirectionTask, ForearmDirectionTask
        self.arm_mapping=arm_mapping
        self.wrist_mapping=wrist_mapping;self.grip_mapping=grip_mapping
        self.wrist_tasks={s:mink.RelativeFrameTask(p+'_wrist','site','arm_'+p+'_link4','body',
                            position_cost=0.,orientation_cost=.7,gain=.7,lm_damping=1e-3)
                          for s,p in [('left','l'),('right','r')]}
        self.wrist_desired={};self.wrist_filtered={}
        self.shoulder_tasks={s:UpperArmDirectionTask(m,p) for s,p in [('left','l'),('right','r')]}
        self.shoulder_desired={}
        self.forearm_tasks={s:ForearmDirectionTask(m,p) for s,p in [('left','l'),('right','r')]}
        self.forearm_desired={}
        self.elbow_measurements={}
        self.twist_active=set()
        if arm_mapping=='limb':
            # Preserve upper-arm adherence when joint limits make simultaneous
            # directions infeasible. This is a weighted preference, not a hard lock.
            for task in self.shoulder_tasks.values():task.cost[:]=5.
        # Joints 1/2 orient the upper arm. Limb mode additionally observes the
        # bend plane (3) and elbow bend (4); shoulder-only mode holds both.
        self.shoulder_dofs={s:self.groups[s][:2] for s in SIDES}
        # A position servo commanded only q_measured + dt*v sees a tiny spring
        # error on every tick. Its damping then slows motion far below v. Add
        # bounded damping feedforward for the active arm-direction joints. This is
        # recomputed from current measured state (never an accumulating target).
        self.shoulder_servo_lead=np.zeros(m.nu)
        if arm_mapping in ('shoulder','limb'):
            for i,dof in enumerate(self.dofs):
                if any(dof in self.groups[s][:(7 if arm_mapping=='limb' and wrist_mapping=='relative' else 4 if arm_mapping=='limb' else 2)] for s in SIDES):
                    kp=float(m.actuator_gainprm[i,0])
                    damping=max(0.,-float(m.actuator_biasprm[i,2]))+float(m.dof_damping[dof])
                    # The normal IK position step already contributes dt*v.
                    self.shoulder_servo_lead[i]=np.clip(damping/max(kp,1e-9)-self.dt,0.,.12)
        self.posture=mink.PostureTask(m,cost=.002)
        self.posture.set_target(d.qpos.copy())
        home=self.home_from_configuration()
        self.retarget=Retargeter(home,arm_mapping=arm_mapping,wrist_mapping=wrist_mapping,grip_mapping=grip_mapping)
        self.ik_ms=0.;self.steps=0;self.command_updates=0;self.error=None;self.last_motion_frame=None
        self.max_joint_speed=0.;self.max_limit_violation=0.;self.solve_times=deque(maxlen=500)
        self.errors={};self.holding=False
        self.orientation_errors={};self.filtered={};self.active_tasks=set();self.held_groups=set()
        self.max_torso_deviation=0.;self.max_torso_translation=0.;self.max_torso_rotation=0.
        self.ik_failures=0;self.grip_filtered={};self.last_grip_command={}

    def home_from_configuration(self):
        torso=self.anchor
        home={'torso':torso,'head':self.relative('head_frame',torso),'arms':{}}
        for side,s in [('left','l'),('right','r')]:
            shoulder=self.relative(s+'_shoulder',torso).position
            elbow=self.relative(s+'_elbow',torso).position
            wrist=self.relative(s+'_wrist',torso)
            segments=[float(np.linalg.norm(elbow-shoulder)),float(np.linalg.norm(wrist.position-elbow))]
            home['arms'][side]={'shoulder':shoulder,'elbow':elbow,'wrist':wrist.position,'rotation':wrist.rotation,
                'forearm_rotation':torso.rotation.T@self.cfg.get_transform_frame_to_world('arm_'+s+'_link4','body').rotation().as_matrix(),
                'segment_lengths':segments,'length':sum(segments)}
        return home

    def calibrate(self,pose,now):
        # A failed C press must not rebase an existing neutral or discard filters.
        if not self.retarget.calibrate(pose,now,partial=True):
            return False
        self.cfg.update(self.data.qpos)
        self.retarget.home=self.home_from_configuration()
        self.posture.set_target(self.data.qpos.copy())
        self.filtered.clear();self.active_tasks.clear()
        return True

    def transform(self,name):
        t=self.cfg.get_transform_frame_to_world(name,'site')
        return FrameTarget(t.translation().copy(),t.rotation().as_matrix().copy())

    def relative(self,name,origin):
        t=self.transform(name)
        return FrameTarget(origin.rotation.T@(t.position-origin.position),origin.rotation.T@t.rotation)

    def set_task(self,name,target):
        current=self.transform(name)
        previous=self.filtered.get(name) if name in self.active_tasks else current
        if previous is None:previous=current
        alpha=1-np.exp(-self.dt/.08)
        position=previous.position+bounded(alpha*(target.position-previous.position),.30*self.dt)
        desired=previous.rotation if target.rotation is None else target.rotation
        dr=self.mk.SO3.from_matrix(previous.rotation.T@desired).log()
        rr=previous.rotation@self.mk.SO3.exp(bounded(alpha*dr,1.4*self.dt)).as_matrix()
        t=FrameTarget(position,rr);self.filtered[name]=t
        self.tasks[name].set_target(self.mk.SE3.from_rotation_and_translation(
            self.mk.SO3.from_matrix(t.rotation),t.position))

    def set_shoulder_task(self,side,direction):
        from .shoulder_task import approach_direction
        task=self.shoulder_tasks[side]
        name=side[0]+'_upper_arm'
        previous=task.target if name in self.active_tasks else task.direction(self.cfg)
        desired=self.anchor.rotation@direction
        task.set_target(approach_direction(previous,desired,self.dt))
        self.shoulder_desired[side]=direction.copy()

    def wrist_rotation(self,side):
        return self.cfg.get_transform(side[0]+'_wrist','site','arm_'+side[0]+'_link4','body').rotation().as_matrix()

    def set_wrist_task(self,side,desired):
        name=side[0]+'_wrist_relative'
        previous=self.wrist_filtered[side] if name in self.active_tasks else self.wrist_rotation(side)
        delta=self.mk.SO3.from_matrix(previous.T@desired).log()
        filtered=previous@self.mk.SO3.exp(bounded((1-np.exp(-self.dt/.1))*delta,1.2*self.dt)).as_matrix()
        self.wrist_desired[side]=desired.copy();self.wrist_filtered[side]=filtered
        self.wrist_tasks[side].set_target(self.mk.SE3.from_rotation_and_translation(self.mk.SO3.from_matrix(filtered),np.zeros(3)))

    def wrist_status(self):
        result={}
        for side in SIDES:
            active=side[0]+'_wrist_relative' in self.active_tasks
            actual=self.wrist_rotation(side);desired=self.wrist_desired.get(side) if active else None
            result[side]={'active':active,'source':'current_measured_palm_forearm' if active else 'held',
                'angular_error_deg':None if desired is None else float(np.degrees(np.linalg.norm(self.mk.SO3.from_matrix(actual.T@desired).log()))),
                'robot_rotation_relative':actual.tolist(),'desired_rotation_relative':None if desired is None else desired.tolist(),
                'neutral_acquired':side in self.retarget.wrist_neutral,'inferred_control':False}
        return result

    def shoulder_status(self):
        from .shoulder_task import angle
        result={}
        for side,task in self.shoulder_tasks.items():
            actual=self.anchor.rotation.T@task.direction(self.cfg)
            active=side[0]+'_upper_arm' in self.active_tasks
            desired=self.shoulder_desired.get(side) if active else None
            filtered=self.anchor.rotation.T@task.target if active else None
            result[side]={'active':active,'source':'current_measured_shoulder_elbow' if active else 'held',
                'robot_direction_torso':actual.tolist(),
                'desired_direction_torso':None if desired is None else desired.tolist(),
                'filtered_direction_torso':None if filtered is None else filtered.tolist(),
                'angular_error_deg':None if desired is None else float(np.degrees(angle(actual,desired))),
                'filtered_angular_error_deg':None if filtered is None else float(np.degrees(angle(actual,filtered))),
                'axial_twist_observed':side in self.twist_active,'inferred_control':False}
        return result

    def set_forearm_task(self,side,measurement):
        from .shoulder_task import approach_direction
        task=self.forearm_tasks[side]
        previous=task.target if side[0]+'_forearm' in self.active_tasks else task.direction(self.cfg)
        direction=measurement['forearm_direction']
        task.set_target(approach_direction(previous,self.anchor.rotation@direction,self.dt))
        self.forearm_desired[side]=direction.copy()
        self.elbow_measurements[side]={k:measurement[k] for k in ('elbow_flexion_rad','bend_plane_observable','wrist_source')}

    def elbow_status(self):
        from .shoulder_task import angle
        result={}
        for side,task in self.forearm_tasks.items():
            upper=self.anchor.rotation.T@self.shoulder_tasks[side].direction(self.cfg)
            actual=self.anchor.rotation.T@task.direction(self.cfg)
            active=side[0]+'_forearm' in self.active_tasks
            desired=self.forearm_desired.get(side) if active else None
            observation=self.elbow_measurements.get(side,{}) if active else {}
            flexion=angle(upper,actual)
            target_flexion=observation.get('elbow_flexion_rad')
            result[side]={'active':active,'source':'current_measured_shoulder_elbow_wrist' if active else 'held',
                'robot_forearm_direction_torso':actual.tolist(),
                'desired_forearm_direction_torso':None if desired is None else desired.tolist(),
                'angular_error_deg':None if desired is None else float(np.degrees(angle(actual,desired))),
                'robot_flexion_deg':float(np.degrees(flexion)),
                'desired_flexion_deg':None if target_flexion is None else float(np.degrees(target_flexion)),
                'flexion_error_deg':None if target_flexion is None else float(np.degrees(abs(flexion-target_flexion))),
                'bend_plane_observable':observation.get('bend_plane_observable',False),
                'axial_rotation_active':side in self.twist_active,'wrist_source':observation.get('wrist_source'),
                'inferred_control':False}
        return result

    def forearm_overlay(self):
        result={}
        for side,task in self.forearm_tasks.items():
            if side[0]+'_forearm' in self.active_tasks:
                elbow=self.cfg.data.site_xpos[self.model.site(side[0]+'_elbow').id].copy()
                wrist=self.cfg.data.site_xpos[self.model.site(side[0]+'_wrist').id]
                result[side]=(elbow,elbow+np.linalg.norm(wrist-elbow)*task.target)
        return result

    def shoulder_overlay(self):
        result={}
        for side,task in self.shoulder_tasks.items():
            if side[0]+'_upper_arm' in self.active_tasks:
                shoulder=self.cfg.data.site_xpos[task.shoulder].copy()
                length=np.linalg.norm(self.cfg.data.site_xpos[task.elbow]-shoulder)
                result[side]=(shoulder,shoulder+length*task.target)
        return result

    def hold_group(self,name,aids):
        if name not in self.held_groups:
            self.data.ctrl[aids]=np.clip(self.data.qpos[self.qadr[aids]],self.model.actuator_ctrlrange[aids,0],self.model.actuator_ctrlrange[aids,1])
        self.held_groups.add(name)

    def hold(self):
        # Cancel a pending servo movement when its measurement disappears.
        if not self.holding:self.data.ctrl[:]=np.clip(self.data.qpos[self.qadr],self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])
        self.holding=True
        self.data.ctrl[self.torso_aids]=self.torso_anchor_q
        self.active_tasks.clear()
        self.twist_active.clear()
        self.held_groups.update(['left','right','head','left_grip','right_grip','left_distal','right_distal'])
        self.held_groups.update(s+'_'+part for s in SIDES for part in ('twist','elbow','wrist'))
        self.grip_filtered.clear()

    def step(self,pose,now=None):
        now=time.monotonic() if now is None else now
        m,d=self.model,self.data
        q=d.qpos.copy()
        for i in range(m.njnt):
            if m.jnt_limited[i]:q[m.jnt_qposadr[i]]=np.clip(q[m.jnt_qposadr[i]],*m.jnt_range[i])
        self.cfg.update(q)
        target=self.retarget.targets(pose,now,self.home_from_configuration())
        if target is None:
            self.hold();self.errors={}
        else:
            self.holding=False
            # Torso is never recruited by hand/head IK, including posture regularization.
            active=set();tasks=[self.posture];next_tasks=set();self.twist_active.clear()
            torso=self.anchor
            def world(t):return FrameTarget(torso.position+torso.rotation@t.position,None if t.rotation is None else torso.rotation@t.rotation)
            if target['head']:
                self.set_task('head_frame',world(target['head']));tasks.append(self.tasks['head_frame']);active.update(self.groups['head']);next_tasks.add('head_frame');self.held_groups.discard('head')
            for side,s in [('left','l'),('right','r')]:
                a=target['arms'][side]
                if a:
                    self.held_groups.discard(side)
                    if self.arm_mapping in ('shoulder','limb'):
                        active.update(self.shoulder_dofs[side])
                        self.set_shoulder_task(side,a['upper_arm_direction'])
                        tasks.append(self.shoulder_tasks[side]);next_tasks.add(s+'_upper_arm')
                        if self.arm_mapping=='limb':
                            following='forearm_direction' in a
                            if following:
                                self.set_forearm_task(side,a)
                                tasks.append(self.forearm_tasks[side]);next_tasks.add(s+'_forearm')
                            wrist_following=following and 'wrist_rotation_relative' in a
                            if wrist_following:
                                self.set_wrist_task(side,a['wrist_rotation_relative'])
                                tasks.append(self.wrist_tasks[side]);next_tasks.add(s+'_wrist_relative')
                            for part,indices,enabled in [
                                    ('twist',self.groups[side][2:3],following and a.get('bend_plane_observable',False)),
                                    ('elbow',self.groups[side][3:4],following),
                                    ('wrist',self.groups[side][4:],wrist_following)]:
                                if enabled:
                                    active.update(indices);self.held_groups.discard(side+'_'+part)
                                    if part=='twist':self.twist_active.add(side)
                                else:
                                    aids=[i for i,x in enumerate(self.dofs) if x in indices]
                                    self.hold_group(side+'_'+part,aids)
                        else:
                            aids=[i for i,x in enumerate(self.dofs) if x in self.groups[side][2:]]
                            self.hold_group(side+'_distal',aids)
                    else:
                        active.update(self.groups[side])
                        if a['elbow'] is not None:
                            self.set_task(s+'_elbow',world(FrameTarget(a['elbow'],None)))
                            tasks.append(self.tasks[s+'_elbow']);next_tasks.add(s+'_elbow')
                        self.set_task(s+'_wrist',world(a['wrist']))
                        tasks.append(self.tasks[s+'_wrist']);next_tasks.add(s+'_wrist')
                        # Missing palm orientation preserves current wrist orientation.
                else:
                    aids=[i for i,x in enumerate(self.dofs) if x in self.groups[side]]
                    self.hold_group(side,aids)
            if not target['head']:
                aids=[i for i,x in enumerate(self.dofs) if x in self.groups['head']]
                self.hold_group('head',aids)
            freeze=self.mk.DofFreezingTask(m,[i for i in range(m.nv) if i not in active])
            started=time.perf_counter()
            try:
                velocity=self.mk.solve_ik(self.cfg,tasks,self.dt,'daqp',damping=1e-4,limits=self.limits,constraints=[freeze])
                if not np.isfinite(velocity).all():raise ValueError('nonfinite IK solution')
                self.cfg.integrate_inplace(velocity,self.dt)
                for i,dof in enumerate(self.dofs):
                    if dof in active:
                        lead=np.clip(self.shoulder_servo_lead[i]*velocity[dof],-.12,.12)
                        d.ctrl[i]=np.clip(self.cfg.q[self.qadr[i]]+lead,*m.actuator_ctrlrange[i])
                for side,value in target['grippers'].items():
                    idx=m.actuator(f'idx{31 if side=="left" else 71}_gripper_{"l" if side=="left" else "r"}_inner_joint1').id
                    previous=self.grip_filtered.get(side,float(d.ctrl[idx]))
                    # Fist hysteresis is upstream; retain the pinch deadband only
                    # for that optional legacy mode, then rate-limit both modes.
                    deadband=.8*.002/.06 if self.grip_mapping=='pinch' else .001
                    if abs(value-previous)>deadband:previous+=(1-np.exp(-self.dt/.1))*(value-previous)
                    self.grip_filtered[side]=previous
                    d.ctrl[idx]=np.clip(d.ctrl[idx]+np.clip(previous-d.ctrl[idx],-.6*self.dt,.6*self.dt),*m.actuator_ctrlrange[idx])
                    self.held_groups.discard(side+'_grip')
                for side in SIDES:
                    if side not in target['grippers']:
                        idx=m.actuator(f'idx{31 if side=="left" else 71}_gripper_{"l" if side=="left" else "r"}_inner_joint1').id
                        self.hold_group(side+'_grip',[idx]);self.grip_filtered.pop(side,None)
                if active or target['grippers']:
                    self.command_updates+=1;self.last_motion_frame=pose['frame_id']
                else:self.holding=True
                self.error=None
                self.active_tasks=next_tasks
            except (ValueError,RuntimeError,self.mk.exceptions.NoSolutionFound) as e:
                self.error=str(e);self.ik_failures+=1;self.retarget.reason='Holding: IK failed; neutral retained';self.hold()
            self.ik_ms=(time.perf_counter()-started)*1000;self.solve_times.append(self.ik_ms)
        d.ctrl[self.torso_aids]=self.torso_anchor_q
        for _ in range(self.substeps):self.mj.mj_step(m,d)
        self.mj.mj_forward(m,d)
        self.cfg.update(d.qpos)
        self.errors={name:float(np.linalg.norm(self.tasks[name].compute_error(self.cfg)[:3])) for name in self.active_tasks if name in self.tasks}
        self.orientation_errors={name:float(np.linalg.norm(self.tasks[name].compute_error(self.cfg)[3:])) for name in self.active_tasks if name in self.tasks and 'elbow' not in name}
        torso_now=self.transform('torso_frame')
        self.max_torso_deviation=max(self.max_torso_deviation,float(np.max(np.abs(d.qpos[self.torso_qadr]-self.torso_anchor_q))))
        self.max_torso_translation=max(self.max_torso_translation,float(np.linalg.norm(torso_now.position-self.anchor.position)))
        self.max_torso_rotation=max(self.max_torso_rotation,float(np.linalg.norm(self.mk.SO3.from_matrix(self.anchor.rotation.T@torso_now.rotation).log())))
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('MuJoCo produced nonfinite state')
        self.max_joint_speed=max(self.max_joint_speed,float(np.max(np.abs(d.qvel))))
        violation=max([max(m.jnt_range[i,0]-d.qpos[m.jnt_qposadr[i]],d.qpos[m.jnt_qposadr[i]]-m.jnt_range[i,1],0.) for i in range(m.njnt) if m.jnt_limited[i]],default=0.)
        self.max_limit_violation=max(self.max_limit_violation,float(violation))
        if violation>.08:raise RuntimeError('physical joint limit exceeded by >.08 rad')
        self.steps+=1

    def status(self,pose=None):
        return {'schema':'handdepth.teleop.v1','simulation_only':True,'state':'tracking' if self.retarget.armed and not self.holding else 'holding',
            'reason':self.retarget.reason,'source_session_id':pose.get('session_id') if pose else None,
            'source_frame_id':pose.get('frame_id') if pose else None,'last_motion_frame_id':self.last_motion_frame,
            'measurement_age_s':self.retarget.age,'active_parts':self.retarget.active_parts,'calibrations':self.retarget.calibrations,
            'mapping_version':{'limb':'static-torso-limb-wrist-v2','shoulder':'static-torso-shoulder-v2','cartesian':'static-torso-v1'}[self.arm_mapping],
            'arm_mapping':self.arm_mapping,'shoulders':self.shoulder_status(),'elbows':self.elbow_status(),
            'wrist_mapping':self.wrist_mapping,'wrists':self.wrist_status(),
            'grip_mapping':self.grip_mapping,'grips':self.retarget.grip_status,
            'shoulder_servo_lead_s':{n:float(v) for n,v in zip(self.joint_names,self.shoulder_servo_lead) if v>0},
            'torso_mode':'fixed','part_status':self.retarget.part_status,
            'arm_scales':self.retarget.scales,'segment_scales':{s:v.tolist() for s,v in self.retarget.segment_scales.items()},
            'pinch_distances_m':self.retarget.pinch_distances,'target_orientation_errors_rad':self.orientation_errors,
            'torso_anchor_qpos':self.torso_anchor_q.tolist(),'torso_qpos':self.data.qpos[self.torso_qadr].tolist(),
            'max_torso_joint_deviation_rad':self.max_torso_deviation,'max_torso_translation_m':self.max_torso_translation,
            'max_torso_rotation_rad':self.max_torso_rotation,'physics_warnings':sum(w.number for w in self.data.warning),
            'ik_failures':self.ik_failures,
            'base_position_m':self.data.xpos[self.base_body_id].tolist(),
            'base_displacement_m':float(np.linalg.norm(self.data.xpos[self.base_body_id]-self.base_anchor)),
            'sim_time_s':float(self.data.time),'control_hz':1/self.dt,'physics_hz':1/self.model.opt.timestep,
            'ik_ms':self.ik_ms,'target_position_errors_m':self.errors,'ik_error':self.error,
            'steps':self.steps,'command_updates':self.command_updates,'joint_names':self.joint_names,
            'qpos':self.data.qpos.tolist(),'ctrl':self.data.ctrl.tolist(),
            'max_physical_joint_speed_rad_s':self.max_joint_speed,'max_physical_limit_violation_rad':self.max_limit_violation}


def run(args):
    from datetime import datetime,timezone
    root=check_recording_root(args.recording_root,args.fixture_storage)
    run_dir=root/'teleop'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%f')
    run_dir.mkdir(parents=True)
    # MuJoCo warnings/output and all growing logs stay on the data mount.
    import os
    os.chdir(run_dir)
    recorder=Recorder(root,args.fixture_storage) if args.record else None
    receiver=Receiver(recorder)
    if not args.no_pc_pose and not args.replay:receiver.start_pc_pose(args.pc_pose_model)
    control=Controller(args.scene,arm_mapping=args.arm_mapping,wrist_mapping=args.wrist_mapping,grip_mapping=args.grip_mapping)
    from .pose_overlay import PoseOverlay
    overlay=PoseOverlay(control.anchor)
    stopped=threading.Event();failures=[];keys=SimpleQueue()
    import signal
    previous_handlers={s:signal.signal(s,lambda *_:stopped.set()) for s in (signal.SIGTERM,signal.SIGINT)}
    def receiving():
        try:
            if args.replay:
                from .cli import replay
                from types import SimpleNamespace
                asyncio.run(replay(SimpleNamespace(session=args.replay,speed=args.speed,headless=True,snapshot=None),receiver))
                stopped.set()
            else:asyncio.run(receiver.run(args.host,args.port,True,stop_event=stopped))
        except Exception as e:failures.append(e);stopped.set()
    thread=threading.Thread(target=receiving,name='HandDepth receiver',daemon=True);thread.start()
    viewer=None;trajectory=(run_dir/'trajectory.jsonl').open('w');pose=None
    start=time.monotonic();last_saved=0.;last_render=0.;auto_pending=args.auto_calibrate;automatic=args.auto_calibrate
    neutral_signature=None
    try:
        if not args.headless:
            import mujoco.viewer
            import cv2
            cv2.namedWindow('HandDepth -> G2 simulation',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('HandDepth -> G2 simulation',900,420)
            cv2.moveWindow('HandDepth -> G2 simulation',970,630)
            viewer=mujoco.viewer.launch_passive(control.model,control.data,key_callback=keys.put,show_left_ui=False,show_right_ui=False)
            viewer.cam.lookat[:]=[.2,0,1.0];viewer.cam.distance=2.8;viewer.cam.azimuth=135;viewer.cam.elevation=-18
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]=True
        while not stopped.is_set() and (args.duration is None or time.monotonic()-start<args.duration):
            tick=time.monotonic()
            if viewer and not viewer.is_running():break
            with receiver.lock:
                pose=receiver.latest_pose
                if args.replay and pose:pose=dict(pose,received_monotonic_s=receiver.latest_received)
            calibrate=auto_pending
            if not args.headless and tick-last_render>=1/15:
                import cv2
                with receiver.lock:image=receiver.dashboard.render()
                cv2.putText(image,f'G2 {control.arm_mapping.upper()} | C arm/neutral | Space hold | H skeleton | G transparency | R reset | Q quit',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                cv2.putText(image,control.retarget.reason[:110],(15,image.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                cv2.imshow('HandDepth -> G2 simulation',image)
                if viewer:
                    parts=control.retarget.part_status
                    with viewer.lock():
                        overlay.draw(viewer.user_scn,pose,fresh=control.retarget.fresh(pose,tick),
                                     targets={n:control.filtered[n] for n in control.active_tasks if n in control.filtered},
                                     shoulder_targets=control.shoulder_overlay(),forearm_targets=control.forearm_overlay(),now=tick)
                    shoulder_status=control.shoulder_status()
                    labels=['Arm mapping','Torso/base','Left arm','Right arm','Head','Left palm','Right palm','Left grip','Right grip','L shoulder error','R shoulder error','Sensor','Control','Skeleton','L display age','R display age','Colors','Keys']
                    values=[control.arm_mapping,'FIXED']+[parts.get(k,'waiting') for k in ['left','right','head','left_palm','right_palm','left_grip','right_grip']]
                    values += [f"{shoulder_status[s]['angular_error_deg']:.1f} deg" if shoulder_status[s]['active'] else 'held / no angle target' for s in SIDES]
                    values += [f'{receiver.dashboard.source_fps:.1f} Hz',f'{control.steps/max(tick-start,.001):.1f} Hz',overlay.status,
                               overlay.skeleton.arm_text('left',tick),overlay.skeleton.arm_text('right',tick),
                               'cyan: measured / amber: held or inferred / green: targets','C: arm/neutral / Space: hold / H: skeleton / G: transparency / R: reset display / Q: quit']
                    if control.arm_mapping=='limb':
                        for side,elbow in control.elbow_status().items():
                            index=labels.index('Sensor')
                            labels.insert(index,side[0].upper()+' forearm / bend')
                            values.insert(index,(f"{elbow['angular_error_deg']:.1f} deg / {elbow['flexion_error_deg']:.1f} deg" if elbow['active'] else parts.get(side+'_elbow','held')))
                        for side,wrist in control.wrist_status().items():
                            index=labels.index('Sensor');labels.insert(index,side[0].upper()+' wrist rotation')
                            values.insert(index,f"{wrist['angular_error_deg']:.1f} deg" if wrist['active'] else parts.get(side+'_palm','held'))
                    if receiver.pc_fusion:
                        stats=receiver.pc_fusion.stats
                        labels.insert(0,'Ubuntu RGB pose')
                        values.insert(0,('phone fallback: '+stats['error'][:70]) if stats['error'] else
                                      (f"CPU {stats['inference_ms']:.0f} ms + TrueDepth" if stats['inference_ms'] is not None else 'starting / awaiting matched RGB'))
                    viewer.set_texts((control.mj.mjtFontScale.mjFONTSCALE_100,control.mj.mjtGridPos.mjGRID_TOPLEFT,'\n'.join(labels),'\n'.join(values)))
                key=cv2.waitKey(1)&255
                if key!=255:keys.put(key)
                last_render=tick
            while not keys.empty():
                key=keys.get()
                if key in (ord('q'),ord('Q'),27):stopped.set()
                elif key==32:control.retarget.disarm();auto_pending=False;automatic=False
                elif key in (ord('c'),ord('C')):calibrate=True
                elif key in (ord('h'),ord('H')):overlay.enabled=not overlay.enabled
                elif key in (ord('r'),ord('R')):
                    overlay.reset()
                    with receiver.lock:receiver.dashboard.reset_display()
                elif key in (ord('g'),ord('G')) and viewer:
                    with viewer.lock():
                        flag=control.mj.mjtVisFlag.mjVIS_TRANSPARENT
                        viewer.opt.flags[flag]=not viewer.opt.flags[flag]
            if calibrate and control.calibrate(pose,tick):
                auto_pending=False
                log.info('neutral acquisition started at frame %s (relaxed pose)',pose['frame_id'])
            control.step(pose,tick)
            r=control.retarget
            signature=(r.calibrations,tuple(r.arm_neutral),tuple(sorted(r.shoulder_ready)),tuple(sorted(r.elbow_ready)),tuple(r.hand_neutral),tuple(r.wrist_neutral),r.head_neutral is not None)
            if r.calibrations and signature!=neutral_signature:
                (run_dir/f'neutral-{r.calibrations}.json').write_text(json.dumps(r.calibration_summary(),allow_nan=False,indent=2))
                neutral_signature=signature
            if automatic and not control.retarget.armed:auto_pending=True
            if viewer:viewer.sync()
            if tick-last_saved>.2:
                status=control.status(pose);status['receiver_monotonic_s']=tick
                status['achieved_control_hz']=control.steps/max(tick-start,.001)
                status['sensor_source_hz']=receiver.dashboard.source_fps
                status['pc_pose_fusion']=dict(receiver.pc_fusion.stats) if receiver.pc_fusion else {'enabled':False}
                status['stream_transport']=receiver.stream_status(tick)
                status['limb_tracking']=(pose.get('body') or {}).get('limb_tracking') if pose else None
                status['pose_overlay']={'enabled':overlay.enabled,'status':overlay.status,
                    'origin_m':overlay.origin.tolist(),'scale':1.0,
                    'geometries':int(viewer.user_scn.ngeom) if viewer else 0,
                    'history':overlay.skeleton.summary(tick),'inferred_control':False}
                with receiver.lock:
                    dash=receiver.dashboard
                    status['dashboard_display']={'rgb_status':dash.rgb_status,
                        'rgb_overlay_matched':dash.rgb_overlay_matched,
                        'matched_frame':dash.matched[0]['frame_id'] if dash.matched else None,
                        'rgb_joints':len(dash.matched[1]['points']) if dash.matched else 0,
                        'rgb_arm_methods':{side:arm.get('method','phone_pixel_constraint')
                            for side,arm in dash.matched[1]['arms'].items()} if dash.matched else {},
                        'metric_history':dash.metric_skeleton.summary(tick)}
                trajectory.write(json.dumps(status,allow_nan=False)+'\n');trajectory.flush()
                temporary=run_dir/'status.tmp';temporary.write_text(json.dumps(status,allow_nan=False,indent=2));temporary.replace(run_dir/'status.json')
                if args.status_file:
                    tmp=args.status_file.with_suffix('.tmp');tmp.write_text(json.dumps(status,allow_nan=False));tmp.replace(args.status_file)
                last_saved=tick
            delay=control.dt-(time.monotonic()-tick)
            if delay>0:stopped.wait(delay)
        if failures:raise RuntimeError('receiver failed: '+str(failures[0]))
    finally:
        stopped.set();thread.join(timeout=3)
        receiver.invalidate('teleop_stopped');control.retarget.disarm('Teleoperation stopped')
        receiver.close()
        if viewer:viewer.close()
        if not args.headless:
            import cv2
            cv2.destroyAllWindows()
        trajectory.close()
        if recorder:recorder.close()
        report=control.status(pose)
        report.update(run=str(run_dir),wall_seconds=time.monotonic()-start,recording=str(recorder.path) if recorder else None,
                      mink='1.1.0',mujoco=control.mj.__version__,solver='daqp',
                      median_ik_ms=float(np.median(control.solve_times)) if control.solve_times else None)
        (run_dir/'report.json').write_text(json.dumps(report,allow_nan=False,indent=2))
        log.info('teleop report: %s',run_dir/'report.json')
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
