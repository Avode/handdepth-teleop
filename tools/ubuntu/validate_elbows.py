"""Compare shoulder-only and full limb directions on identical measured poses."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from handdepth.recording import check_recording_root
from handdepth.teleop import Controller,DEFAULT_SCENE
from handdepth.retarget import measured_upper_arm,measured_forearm
from handdepth.shoulder_task import angle


def stats(values):
    return {'samples':len(values),'mean':float(np.mean(values)) if values else None,
            'median':float(np.median(values)) if values else None,
            'p95':float(np.percentile(values,95)) if values else None}


def validate(path,seconds=60,start=0):
    controls={mode:Controller(DEFAULT_SCENE,arm_mapping=mode) for mode in ('shoulder','limb')}
    errors={mode:{s:{k:[] for k in ('upper_arm','forearm','flexion')} for s in ('left','right')} for mode in controls}
    availability=Counter();first=None;tick=None;pose=None;count=0;started=time.monotonic()
    with path.open() as f:
        for line in f:
            try:next_pose=json.loads(line)
            except json.JSONDecodeError:break # an active recorder may have a partial final line
            received=next_pose['received_monotonic_s']
            if first is None:first=received
            if received<first+start:continue
            if received>first+start+seconds:break
            if tick is None:tick=received
            while tick<received:
                for c in controls.values():
                    if not c.retarget.armed:c.calibrate(pose,tick)
                    c.step(pose,tick)
                new,old=controls['limb'],controls['shoulder']
                for side in ('left','right'):
                    active=side[0]+'_forearm' in new.active_tasks
                    availability[side+('_forearm_tracking' if active else '_forearm_held')]+=1
                    if pose is None:continue
                    upper=measured_upper_arm(pose,side)
                    fore=measured_forearm(pose,side)
                    both_shoulders=all(side[0]+'_upper_arm' in c.active_tasks for c in controls.values())
                    if not both_shoulders or upper is None:continue
                    for mode,c in controls.items():
                        u=c.anchor.rotation.T@c.shoulder_tasks[side].direction(c.cfg)
                        v=c.anchor.rotation.T@c.forearm_tasks[side].direction(c.cfg)
                        errors[mode][side]['upper_arm'].append(float(np.degrees(angle(u,upper))))
                        if active and fore is not None:
                            errors[mode][side]['forearm'].append(float(np.degrees(angle(v,fore['forearm_direction']))))
                            errors[mode][side]['flexion'].append(float(np.degrees(abs(angle(u,v)-fore['elbow_flexion_rad']))))
                tick+=new.dt
            pose=next_pose;count+=1
    reports={}
    for mode,c in controls.items():
        before=c.command_updates;c.step(None,tick);held=c.data.ctrl.copy()
        for i in range(25):c.step(None,tick+(i+1)*c.dt)
        report=c.status(pose)
        report['angle_errors_deg']={s:{k:stats(v) for k,v in segments.items()} for s,segments in errors[mode].items()}
        report['validation_checks']={
            'disconnect_hold':c.command_updates==before and bool(np.array_equal(held,c.data.ctrl)),
            'base_fixed':report['base_displacement_m']==0,
            'torso_commands_fixed':bool(np.array_equal(c.data.ctrl[c.torso_aids],c.torso_anchor_q)),
            'torso_compliance_below_1e_4_rad':report['max_torso_joint_deviation_rad']<1e-4,
            'no_ik_failures':c.ik_failures==0,'no_physics_warnings':report['physics_warnings']==0,
            'physical_joint_limits':report['max_physical_limit_violation_rad']<.02}
        reports[mode]=report
    return {'poses_file':str(path),'start_s':start,'duration_s':seconds,'source_poses':count,
            'availability_controller_ticks':dict(availability),'controllers':reports,
            'wall_seconds':time.monotonic()-started}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('poses',type=Path);p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--start',type=float,default=0);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();check_recording_root(a.output.parent)
    result=validate(a.poses,a.seconds,a.start);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:{k:r[k] for k in ('angle_errors_deg','validation_checks','max_physical_joint_speed_rad_s')} for m,r in result['controllers'].items()},indent=2))
    assert all(all(r['validation_checks'].values()) for r in result['controllers'].values()),'see saved checks'
