"""Replay exact recorded packets through reconstruction and G2 physics, offline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from handdepth.recording import replay_packets,check_recording_root
from handdepth.server import Receiver
from handdepth.teleop import Controller,DEFAULT_SCENE


def validate(session, start, seconds):
    receiver=Receiver();control=Controller(DEFAULT_SCENE)
    first=None;tick=None;end=None;pose=None;states=Counter();n=0;ctrl_min=ctrl_max=None
    started=time.monotonic()
    for channel,received,raw in replay_packets(session):
        if first is None:first=received
        if received<first+start:continue
        if received>first+start+seconds:break
        if channel!='sensor':continue
        if tick is None:tick=received
        # Advance physical time using the previous measurement until this arrives.
        while tick<received:
            if not control.retarget.armed:control.calibrate(pose,tick)
            control.step(pose,tick);states['tracking' if not control.holding else 'holding']+=1
            ctrl=control.data.ctrl
            ctrl_min=ctrl.copy() if ctrl_min is None else np.minimum(ctrl_min,ctrl)
            ctrl_max=ctrl.copy() if ctrl_max is None else np.maximum(ctrl_max,ctrl)
            tick+=control.dt
        receiver.process(channel,raw,received,replay=True)
        pose=receiver.latest_pose;n+=1
    # Explicitly end source and ensure no old movement target survives.
    receiver.invalidate('validation_disconnect')
    stopped_updates=control.command_updates
    control.step(receiver.latest_pose,tick)
    stopped_ctrl=control.data.ctrl.copy()
    for i in range(25):control.step(receiver.latest_pose,tick+(i+1)*control.dt)
    report=control.status(pose)
    report.update(session=str(session),offset_s=start,duration_s=seconds,source_packets=n,
                  wall_seconds=time.monotonic()-started,states=dict(states),
                  disconnect_commands_unchanged=control.command_updates==stopped_updates,
                  disconnect_ctrl_unchanged=bool(np.allclose(stopped_ctrl,control.data.ctrl)),
                  servo_peak_to_peak_rad=(ctrl_max-ctrl_min).tolist() if ctrl_min is not None else [])
    assert report['max_torso_joint_deviation_rad']<1e-5,report['max_torso_joint_deviation_rad']
    assert report['base_displacement_m']==0
    assert report['physics_warnings']==0
    assert report['max_physical_limit_violation_rad']<.02
    assert report['disconnect_commands_unchanged'] and report['disconnect_ctrl_unchanged']
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('--start',type=float,default=0);p.add_argument('--seconds',type=float,default=45);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    check_recording_root(a.output.parent)
    r=validate(a.session,a.start,a.seconds);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ['source_packets','states','calibrations','ik_failures','max_torso_joint_deviation_rad','max_physical_limit_violation_rad','wall_seconds']},indent=2),flush=True)
