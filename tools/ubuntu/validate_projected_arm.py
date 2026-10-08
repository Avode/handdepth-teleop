"""Reproduce a recorded occluded-arm RGB/metric mismatch with exact source RGB."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from handdepth.recording import check_recording_root
from handdepth.protocol import decode,depth_array
from handdepth.geometry import PoseEstimator
from handdepth.visualization import Dashboard
from handdepth.retarget import measured_forearm,measured_upper_arm


def validate(session,output,backup,frame=105448):
    root=check_recording_root(output)
    spec=importlib.util.spec_from_file_location('handdepth.pre_projection_visualization',backup/'visualization.py')
    old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
    before=old_module.Dashboard();after=Dashboard();target=None
    for line in (session/'pc-poses.jsonl').open():
        pose=json.loads(line)
        if pose['frame_id']>frame:break
        if pose['frame_id']==frame:target=pose;break
        before.metric_skeleton.update(pose);after.metric_skeleton.update(pose)
    assert target is not None
    sensor=decode((root/f'frame-{frame}-sensor.bin').read_bytes())
    preview=decode((root/f'frame-{frame}-preview.bin').read_bytes())
    h=sensor.header;depth=depth_array(sensor);now=target['received_monotonic_s']
    reconstructed,_=PoseEstimator().estimate(h,depth,target['pc_pose'])
    for field in ('received_monotonic_s','capture_to_receive_s','clock_uncertainty_s'):
        reconstructed[field]=target[field]
    # Compare the same received pose in both displays. Separately verify the
    # current receiver also preserves the measured wrist with the elbow absent.
    unchanged=copy.deepcopy(target)
    for dashboard in (before,after):
        dashboard.sensor(h,depth,target,now)
        dashboard.preview(preview.header,preview.payload,now)
    a=before.render(now);b=after.render(now)
    cv2.putText(a,'BEFORE: pixel-only arm constraint',(15,28),0,.7,(0,255,255),2)
    cv2.putText(b,'AFTER: projected metric arm, anchored to measured hand',(15,28),0,.7,(0,255,255),2)
    cv2.imwrite(str(root/f'comparison-{frame}.png'),np.vstack([a,b]))
    cv2.imwrite(str(root/f'corrected-{frame}.png'),b)
    def point(d,name):return next(p for p in d.matched[1]['points'] if p['name']==name)
    pixel=lambda v:np.array([v['x'],v['y']])*h['color_dimensions']
    old=point(before,'rightElbow');new=point(after,'rightElbow')
    hand=next(x for x in target['hands'] if x['body_arm_side']=='right')
    wrist=next(x for x in hand['landmarks'] if x['name']=='wrist')
    metric=after.metric_skeleton
    geometry=np.array([metric.points['right'+n].xyz for n in ('Shoulder','Elbow','Wrist')])
    report={'frame':frame,'session':str(session),'before_elbow_rgb_px':pixel(old).tolist(),
        'after_elbow_rgb_px':pixel(new).tolist(),
        'shift_px':float(np.linalg.norm(pixel(old)-pixel(new))),
        'pc_elbow_rgb_px':pixel(target['pc_pose']['landmarks']['rightElbow']).tolist(),
        'metric_lengths_m':metric.arms['right']['lengths'].tolist(),
        'actual_lengths_m':np.linalg.norm(np.diff(geometry,axis=0),axis=1).tolist(),
        'wrist_endpoint_error_m':float(np.linalg.norm(geometry[2]-wrist['xyz_m'])),
        'projected_wrist_error_px':float(np.linalg.norm(pixel(point(after,'rightWrist'))-after.projection.project_color(wrist['xyz_m'])*h['color_dimensions'])),
        'elbow_state':new['state'],'elbow_measurement_valid':new['measurement_valid'],
        'source_frame_match':new['constraint_frame_id']==sensor.header['frame_id']==preview.header['frame_id'],
        'measured_pose_unchanged':target==unchanged,
        'robot_control_held':measured_upper_arm(reconstructed,'right') is None and measured_forearm(reconstructed,'right') is None,
        'independent_body_wrist_preserved':reconstructed['body']['arms']['right']['positions_camera_m']['rightWrist'] is not None,
        'phone_tracking_preserved':reconstructed['body_tracking']==h['body_tracking']}
    (root/'exact-frame-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert report['wrist_endpoint_error_m']<1e-10 and report['projected_wrist_error_px']<1e-6
    assert np.allclose(report['metric_lengths_m'],report['actual_lengths_m'])
    assert report['elbow_state']=='inferred' and not report['elbow_measurement_valid']
    assert all(report[n] for n in ('source_frame_match','measured_pose_unchanged','robot_control_held','independent_body_wrist_preserved','phone_tracking_preserved'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backup',type=Path,required=True);p.add_argument('--frame',type=int,default=105448)
    a=p.parse_args();validate(a.session,a.output,a.backup,a.frame)
