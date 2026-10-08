"""Compare legacy wrist IK and shoulder-direction control on identical packets."""
import argparse
from collections import Counter
import json
import time
import numpy as np
from handdepth.recording import replay_packets, check_recording_root
from handdepth.server import Receiver
from handdepth.teleop import Controller, DEFAULT_SCENE
from handdepth.retarget import measured_upper_arm
from handdepth.shoulder_task import angle


def validate(session,start,seconds):
    receiver=Receiver()
    controls={mode:Controller(DEFAULT_SCENE,arm_mapping=mode) for mode in ('cartesian','shoulder')}
    errors={mode:{s:[] for s in ('left','right')} for mode in controls}
    availability=Counter();first=None;tick=None;pose=None;packets=0;started=time.monotonic()
    for channel,received,raw in replay_packets(session):
        if first is None:first=received
        if received<first+start:continue
        if received>first+start+seconds:break
        if channel!='sensor':continue
        if tick is None:tick=received
        while tick<received:
            for c in controls.values():
                if not c.retarget.armed:c.calibrate(pose,tick)
                c.step(pose,tick)
            new,old=controls['shoulder'],controls['cartesian']
            for side in ('left','right'):
                s=side[0]
                if s+'_upper_arm' not in new.active_tasks:
                    availability[side+'_held']+=1;continue
                availability[side+'_tracking']+=1
                if s+'_wrist' not in old.active_tasks:continue
                target=measured_upper_arm(pose,side)
                for mode,c in controls.items():
                    actual=c.anchor.rotation.T@c.shoulder_tasks[side].direction(c.cfg)
                    errors[mode][side].append(float(np.degrees(angle(actual,target))))
            tick+=new.dt
        receiver.process(channel,raw,received,replay=True)
        pose=receiver.latest_pose;packets+=1
    receiver.invalidate('shoulder_validation_disconnect')
    reports={}
    for mode,c in controls.items():
        before=c.command_updates;c.step(receiver.latest_pose,tick);held=c.data.ctrl.copy()
        for i in range(25):c.step(receiver.latest_pose,tick+(i+1)*c.dt)
        report=c.status(pose)
        report['disconnect_hold']=c.command_updates==before and bool(np.array_equal(c.data.ctrl,held))
        report['upper_arm_error_deg_on_common_tracking_frames']={s:{
            'samples':len(v),'mean':float(np.mean(v)) if v else None,
            'median':float(np.median(v)) if v else None,
            'p95':float(np.percentile(v,95)) if v else None} for s,v in errors[mode].items()}
        # MuJoCo equality constraints have finite compliance. 1e-4 rad is
        # 0.0057 degrees; commanded torso joints must still be exactly fixed.
        report['validation_checks']={
            'disconnect_hold':report['disconnect_hold'],
            'base_fixed':report['base_displacement_m']==0,
            'torso_compliance_below_1e_4_rad':report['max_torso_joint_deviation_rad']<1e-4,
            'torso_commands_fixed':bool(np.array_equal(c.data.ctrl[c.torso_aids],c.torso_anchor_q)),
            'no_physics_warnings':report['physics_warnings']==0,
            'no_ik_failures':report['ik_failures']==0,
            'physical_joint_limits':report['max_physical_limit_violation_rad']<.02}
        reports[mode]=report
    return {'session':str(session),'start_s':start,'duration_s':seconds,'source_packets':packets,
            'availability_controller_ticks':dict(availability),'controllers':reports,
            'wall_seconds':time.monotonic()-started}


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument('session',type=Path);p.add_argument('--start',type=float,default=0)
    p.add_argument('--seconds',type=float,default=120);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();check_recording_root(a.output.parent)
    report=validate(a.session,a.start,a.seconds)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({mode:r['upper_arm_error_deg_on_common_tracking_frames'] for mode,r in report['controllers'].items()},indent=2))
    print(json.dumps({mode:{k:r[k] for k in ['max_torso_joint_deviation_rad','max_physical_joint_speed_rad_s','validation_checks']} for mode,r in report['controllers'].items()},indent=2))
    assert all(all(r['validation_checks'].values()) for r in report['controllers'].values()), 'see saved validation_checks'
