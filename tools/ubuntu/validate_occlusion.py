"""Exact-packet replay of both displays plus physical controller, without sockets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from handdepth.recording import replay_packets, check_recording_root
from handdepth.server import Receiver
from handdepth.teleop import Controller, DEFAULT_SCENE
from handdepth.pose_overlay import PoseOverlay


def validate(session, output, start=0., seconds=120.):
    output=check_recording_root(output)
    receiver=Receiver();controller=Controller(DEFAULT_SCENE)
    overlay=PoseOverlay(controller.anchor)
    scene=mujoco.MjvScene(controller.model,maxgeom=1000)
    counts=Counter();errors=[];snapshots={};first=None;tick=None;pose=None
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
    cam.lookat[:]=[.15,0,1.13];cam.distance=2.6;cam.azimuth=140;cam.elevation=-10
    opt=mujoco.MjvOption();opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]=True
    renderer=mujoco.Renderer(controller.model,height=720,width=960)

    def snapshot(tag,now):
        if tag in snapshots:return
        dash=receiver.dashboard
        if dash.matched is None:return
        cv2.imwrite(str(output/(tag+'-dashboard.png')),dash.render(now))
        renderer.update_scene(controller.data,camera=cam,scene_option=opt)
        overlay.draw(renderer.scene,receiver.latest_pose,now=now,clear=False,fresh=not receiver.last_expired)
        rgb=renderer.render()
        cv2.putText(rgb,tag+' | amber: held/inferred display only',(15,25),0,.55,(255,205,60),1)
        for i,side in enumerate(('left','right')):
            cv2.putText(rgb,side+' '+overlay.skeleton.arm_text(side,now),(15,47+20*i),0,.45,(255,205,60),1)
        cv2.imwrite(str(output/(tag+'-mujoco.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        snapshots[tag]={'source_frame':pose['frame_id'],'matched_rgb_frame':dash.matched[0]['frame_id'],
                        'metric':overlay.skeleton.summary(now)}

    try:
        for channel,received,raw in replay_packets(session):
            if first is None:first=received
            if received<first+start:continue
            if received>first+start+seconds:break
            if tick is None:tick=received
            while tick<received:
                if not controller.retarget.armed:controller.calibrate(pose,tick)
                controller.step(pose,tick);tick+=controller.dt
            receiver.process(channel,raw,received) # virtual recorded arrival clock
            pose=receiver.latest_pose
            if pose is None:continue
            if channel=='sensor':
                counts['sensor_frames']+=1
                overlay.draw(scene,pose,now=received)
                skeleton=overlay.skeleton
                for side,arm in skeleton.arms.items():
                    counts[side+'_'+arm['mode']]+=1
                    assert all(side+n in skeleton.points for n in ('Shoulder','Elbow','Wrist'))
                    if arm['mode']=='inferred':
                        xyz=[skeleton.points[side+n].xyz for n in ('Shoulder','Elbow','Wrist')]
                        err=float(np.max(np.abs(np.linalg.norm(np.diff(xyz,axis=0),axis=1)-arm['lengths'])))
                        errors.append(err);assert err<1e-8
                    if arm['clamped']:counts[side+'_clamped']+=1
                if not skeleton.torso_current and len(skeleton.arms)==2:
                    counts['torso_loss_both_arms_retained']+=1
                    assert scene.ngeom>10
            else:
                modes=[a['mode'] for a in overlay.skeleton.arms.values()]
                if 'inferred' in modes:snapshot('elbow-inference',received)
                if len(modes)==2 and not overlay.skeleton.torso_current:snapshot('torso-held',received)
                if any(a['clamped'] for a in overlay.skeleton.arms.values()):snapshot('wrist-clamped',received)
                if receiver.dashboard.matched:
                    counts['matched_rgb_frames']+=1
                    t=receiver.dashboard.matched[1];names={p['name'] for p in t['points']}
                    for side in ('left','right'):
                        if all(side+n in names for n in ('Shoulder','Elbow','Wrist')):counts[side+'_rgb_arm_visible']+=1
        receiver.invalidate('validation_disconnect')
        previous=controller.command_updates
        for i in range(30):
            controller.step(receiver.latest_pose,tick+i*controller.dt)
            overlay.draw(scene,receiver.latest_pose,fresh=False,now=tick+i*controller.dt)
        assert controller.command_updates==previous
        assert controller.status()['base_displacement_m']==0
        assert controller.ik_failures==0 and controller.status()['physics_warnings']==0
        assert overlay.skeleton.points and scene.ngeom>4
        snapshot('disconnected-held',tick+1.)
        report={'session':str(session),'offset_s':start,'duration_s':seconds,'counts':dict(counts),
            'max_inferred_length_error_m':max(errors,default=None),'snapshots':snapshots,
            'disconnect_commands_held':controller.command_updates==previous,'inferred_control':False,
            'controller':controller.status(pose)}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('snapshots','controller')},indent=2))
        return report
    finally:renderer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--seconds',type=float,default=120.)
    a=p.parse_args();validate(a.session,a.output,a.start,a.seconds)
