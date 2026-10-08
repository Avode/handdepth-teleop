"""Reconstruct current hand evidence and replay recorded poses in real G2 physics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from handdepth.geometry import PoseEstimator
from handdepth.protocol import decode,depth_array
from handdepth.recording import check_recording_root,replay_packets
from handdepth.teleop import Controller,DEFAULT_SCENE


def summary(values):
    return {'samples':len(values),'mean':float(np.mean(values)) if values else None,
            'median':float(np.median(values)) if values else None,
            'p95':float(np.percentile(values,95)) if values else None}


def validate(session,output,seconds=60,start=0):
    check_recording_root(output.parent);started=time.monotonic();cached={};first=None
    for line in (session/'pc-poses.jsonl').open():
        try:p=json.loads(line)
        except json.JSONDecodeError:break
        received=p['received_monotonic_s']
        if first is None:first=received
        if received>first+start+seconds:break
        if received>=first+start:cached[p['frame_id']]=p
    assert cached,'no recorded fused poses in interval'
    estimator=PoseEstimator();controllers={mode:Controller(DEFAULT_SCENE,arm_mapping='limb',wrist_mapping=mode,grip_mapping='fist') for mode in ('hold','relative')}
    errors={mode:{s:{k:[] for k in ('shoulder','forearm','wrist')} for s in ('left','right')} for mode in controllers}
    availability=Counter();scores={s:[] for s in ('left','right')};grips=Counter();count=0;pose=None;tick=None
    last=max(p['received_monotonic_s'] for p in cached.values())
    for channel,received,raw in replay_packets(session):
        if received>last+1:break
        if channel!='sensor':continue
        packet=decode(raw);h=packet.header;old=cached.get(h['frame_id'])
        p,_=estimator.estimate(h,depth_array(packet),old.get('pc_pose') if old else None)
        if old is None:continue
        p.update({k:old.get(k) for k in ('received_monotonic_s','capture_to_receive_s','clock_uncertainty_s')})
        received=p['received_monotonic_s']
        if tick is None:tick=received
        while tick<received:
            for mode,c in controllers.items():
                if not c.retarget.armed:c.calibrate(pose,tick)
                c.step(pose,tick)
                for s in ('left','right'):
                    for name,states in (('shoulder',c.shoulder_status()),('forearm',c.elbow_status()),('wrist',c.wrist_status())):
                        if states[s]['active']:errors[mode][s][name].append(states[s]['angular_error_deg'])
                    if mode=='relative':
                        availability[s+('_wrist_tracking' if c.wrist_status()[s]['active'] else '_wrist_held')]+=1
                        grips[s+'_'+c.retarget.grip_status.get(s,{}).get('state','held')]+=1
            tick+=controllers['relative'].dt
        pose=p;count+=1
        for h in p['hands']:
            s=h.get('body_arm_side');g=h['grip_gesture']
            if s in scores and g['valid']:scores[s].append(g['score'])
    reports={}
    for mode,c in controllers.items():
        before=c.command_updates;c.step(None,tick+1);hold=c.data.ctrl.copy()
        for i in range(15):c.step(None,tick+1+i*c.dt)
        s=c.status();reports[mode]={'errors_deg':{side:{k:summary(v) for k,v in segments.items()} for side,segments in errors[mode].items()},
            'calibrations':c.retarget.calibrations,'max_joint_speed_rad_s':c.max_joint_speed,
            'checks':{'disconnect_hold':c.command_updates==before and bool(np.array_equal(c.data.ctrl,hold)),
                'no_ik_failures':c.ik_failures==0,'no_physics_warnings':s['physics_warnings']==0,
                'base_fixed':s['base_displacement_m']==0,'torso_fixed':c.max_torso_deviation<1e-4,
                'joint_limits':c.max_limit_violation<.02}}
    report={'session':str(session),'start_s':start,'duration_s':seconds,'source_poses':count,
            'availability':dict(availability),'grip_states':dict(grips),
            'rgb_curl_scores':{s:summary(v) for s,v in scores.items()},'controllers':reports,
            'wall_seconds':time.monotonic()-started}
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    assert all(all(r['checks'].values()) for r in reports.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=60);p.add_argument('--start',type=float,default=0)
    a=p.parse_args();validate(a.session,a.output,a.seconds,a.start)
