"""Replay matched recorded RGB/depth through independent PC inference and G2."""
import argparse
from collections import Counter,OrderedDict
import json
from pathlib import Path
import time
import cv2
import numpy as np
from handdepth.recording import replay_packets,check_recording_root
from handdepth.protocol import decode,depth_array
from handdepth.geometry import PoseEstimator
from handdepth.pc_pose import PoseProcess,same_frame
from handdepth.visualization import Dashboard
from handdepth.teleop import Controller,DEFAULT_SCENE


class NoRefinement:
    def reset(self):pass
    def refine(self,*args,**kwargs):return {'points':[],'arms':{}}


def validate(session,start_frame,frames,output):
    output=check_recording_root(output)
    baseline=PoseEstimator();baseline.body_estimator.limb_refiner=NoRefinement()
    fused=PoseEstimator();process=PoseProcess();cache=OrderedDict();n=0;counts=Counter();timings=[]
    dash_old=Dashboard();dash_new=Dashboard();controller=Controller(DEFAULT_SCENE,arm_mapping='shoulder')
    tick=None;previous=None;shots=0;started=time.monotonic();reasons=Counter()
    with (output/'poses.jsonl').open('w') as log:
        try:
            for channel,arrival,raw in replay_packets(session):
                packet=decode(raw);h=packet.header
                if h['frame_id']<start_frame:continue
                key=(h['session_id'],h['frame_id']);pair=cache.setdefault(key,{})
                pair[channel]=(packet,arrival)
                while len(cache)>8:cache.popitem(last=False)
                if len(pair)!=2:continue
                cache.pop(key)
                (sensor,received),(preview,_)=pair['sensor'],pair['preview']
                if not same_frame(sensor.header,preview.header):continue
                h=sensor.header;d=depth_array(sensor);pc=process.detect(h,preview.payload)
                a,_=baseline.estimate(h,d);b,_=fused.estimate(h,d,pc)
                for pose in (a,b):pose.update(received_monotonic_s=received,capture_to_receive_s=0.,clock_uncertainty_s=0.)
                n+=1;timings.append(pc['inference_ms'])
                for label,pose in [('before',a),('fused',b)]:
                    for side in ('left','right'):
                        arm=(pose.get('body') or {}).get('arms',{}).get(side,{})
                        for segment in ('upper_arm','forearm'):
                            counts[label,side,segment]+=int(arm.get(segment+'_measurement_valid',False))
                for side,info in (b.get('body') or {}).get('limb_tracking',{}).get('arms',{}).items():reasons[side,info['reason']]+=1
                log.write(json.dumps(b,allow_nan=False)+'\n')
                if tick is None:tick=received
                while tick<received:
                    if not controller.retarget.armed:controller.calibrate(previous,tick)
                    controller.step(previous,tick);tick+=controller.dt
                previous=b
                dash_old.sensor(h,d,a,received);dash_new.sensor(h,d,b,received)
                dash_old.preview(preview.header,preview.payload,received);dash_new.preview(preview.header,preview.payload,received)
                improved=sum(x['valid'] for x in (b.get('body') or {}).get('landmarks',[]))>sum(x['valid'] for x in (a.get('body') or {}).get('landmarks',[]))
                if improved and shots<3 and n in range(10+shots*20,20+shots*20):
                    before=dash_old.render(received);after=dash_new.render(received)
                    cv2.putText(before,'BEFORE: phone continuity veto',(15,25),0,.7,(0,255,255),2)
                    cv2.putText(after,'AFTER: Ubuntu RGB pose + registered depth',(15,25),0,.7,(0,255,255),2)
                    cv2.imwrite(str(output/f'comparison-{h["frame_id"]}.png'),np.vstack([before,after]));shots+=1
                if n>=frames:break
        finally:process.close()
    updates=controller.command_updates
    controller.step(None,tick);held=controller.data.ctrl.copy()
    for i in range(25):controller.step(None,tick+(i+1)*controller.dt)
    status=controller.status(previous)
    report={'session':str(session),'start_frame':start_frame,'matched_frames':n,'wall_seconds':time.monotonic()-started,
        'valid_link_counts':{'/'.join(k):v for k,v in counts.items()},
        'reasons':{'/'.join(k):v for k,v in reasons.items()},
        'pc_inference_ms':{'median':float(np.median(timings)),'p95':float(np.percentile(timings,95))},
        'controller':status,'disconnect_hold':controller.command_updates==updates and bool(np.array_equal(held,controller.data.ctrl))}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='controller'},indent=2),flush=True)
    assert report['disconnect_hold'] and status['ik_failures']==0 and status['physics_warnings']==0
    assert status['base_displacement_m']==0 and status['max_torso_joint_deviation_rad']<1e-4
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('--start-frame',type=int,default=35000)
    p.add_argument('--frames',type=int,default=450);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();validate(a.session,a.start_frame,a.frames,a.output)
