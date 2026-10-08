"""Summarize actual HDREC1 body recordings; never relabel fixtures as hardware."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from handdepth.protocol import decode
from handdepth.recording import replay_packets
from handdepth.server import Receiver

def summarize(paths, consoles=(), snapshot=None, physical=False):
    counts=Counter();valid=Counter();sources=Counter();versions=Counter();calibrations=set();sizes=set()
    last={};captures=[];ages=[];uncertainties=[];processing=[];monotonic=True;best_score=-1
    session_spans={}
    for path in paths:
        metadata=json.loads((path/'metadata.json').read_text())
        r=Receiver()
        for channel,received,raw in replay_packets(path):
            h=decode(raw).header;versions[h.get('application_version','unknown')]+=1
            key=(h['session_id'],channel)
            if key in last:monotonic &= h['frame_id']>last[key][0] and h['capture_time_s']>last[key][1]
            last[key]=(h['frame_id'],h['capture_time_s']);counts[channel]+=1
            accepted=r.process(channel,raw,received,replay=True)
            if not accepted:valid['replay_rejected']+=1;continue
            if channel=='sensor':
                pose=r.latest_pose;b=pose.get('body');captures.append(h['capture_time_s']);calibrations.add(h['calibration_id'])
                sizes.add((tuple(h['color_dimensions']),tuple(h['depth_dimensions'])))
                span=session_spans.setdefault(h['session_id'],[h['capture_time_s'],h['capture_time_s']]);span[1]=h['capture_time_s']
                processing.append((h['processing_end_s']-h['capture_time_s'])*1000)
                if pose['capture_to_receive_s'] is not None:
                    ages.append(pose['capture_to_receive_s']*1000);uncertainties.append(pose['clock_uncertainty_s']*1000)
                if len(pose['hands'])==2:valid['two_hand_frames']+=1
                if b:
                    valid['body_observation_frames']+=1
                    valid['body_depth_valid_frames']+=b['depth_valid']
                    valid['torso_position_frames']+=b['torso']['position_valid'];valid['torso_orientation_frames']+=b['torso']['orientation_valid']
                    valid['head_position_frames']+=b['head']['position_valid'];valid['head_orientation_frames']+=b['head']['orientation_valid']
                    valid['head_relative_position_frames']+=b['head']['torso_relative_position_valid'];valid['head_relative_orientation_frames']+=b['head']['torso_relative_orientation_valid']
                    if b['torso_surface']:sources[b['torso_surface']['source']]+=1
                    for side in ('left','right'):valid[side+'_arm_measurement_frames']+=b['arms'].get(side,{}).get('measurement_valid',False)
                    valid['both_arms_measurement_frames']+=len(b['arms'])==2 and all(a['measurement_valid'] for a in b['arms'].values())
                    valid['both_arms_relative_frames']+=len(b['arms'])==2 and all(a['torso_relative_valid'] for a in b['arms'].values())
                valid['any_hand_relative_position_frames']+=any(p['torso_relative_position_valid'] for p in pose['hands'])
                valid['both_hand_relative_position_frames']+=len(pose['hands'])==2 and all(p['torso_relative_position_valid'] for p in pose['hands'])
                valid['both_hand_relative_orientation_frames']+=len(pose['hands'])==2 and all(p['torso_relative_orientation_valid'] for p in pose['hands'])
            elif snapshot and r.dashboard.sensor_header and r.dashboard.preview_header['frame_id']==r.dashboard.sensor_header['frame_id']:
                p=r.latest_pose;b=p.get('body')
                score=(0 if not b else 3*b['torso']['orientation_valid']+3*b['head']['orientation_valid']+sum(a['measurement_valid'] for a in b['arms'].values()))+sum(x['torso_relative_orientation_valid'] for x in p['hands'])
                if score>best_score:
                    import cv2
                    cv2.imwrite(str(snapshot),r.dashboard.render());best_score=score
    diagnostics=[]
    for path in consoles:
        for line in path.read_text().splitlines():
            if not line.startswith('HandDepth: running=true'):continue
            values={}
            for key in ('capture_fps','processed_fps','effective_sensor_hz','rtt_ms','sensor_age_ms'):
                m=re.search(r'\b'+key+r'=([\d.eE+-]+)',line)
                if m:values[key]=float(m[1])
            diagnostics.append(values)
    def stats(values):return None if not values else {'samples':len(values),'median':float(np.median(values)),'p95':float(np.percentile(values,95))}
    return {'physical_measurement':physical,'device':'iPhone XR','os':'18.7.3','receiver_platform':'macOS','application_version':'0.3.0','protocol_version':1,
            'recordings':[str(p) for p in paths],'packet_counts':dict(counts),'packet_application_versions':dict(versions),'validity_frame_counts':dict(valid),
            'torso_surface_sources':dict(sources),'strictly_increasing_per_session_channel':bool(monotonic),
            'capture_span_s':max(captures)-min(captures),'session_capture_spans_s':{k:b-a for k,(a,b) in session_spans.items()},
            'dimensions':[{'color':c,'depth':d} for c,d in sorted(sizes)],'calibration_ids':len(calibrations),
            'clock_supported_capture_to_receive_ms':stats(ages),'clock_uncertainty_ms':stats(uncertainties),'source_to_sensor_processing_end_ms':stats(processing),
            'native_diagnostics':{k:stats([d[k] for d in diagnostics if k in d]) for k in ('capture_fps','processed_fps','effective_sensor_hz','rtt_ms','sensor_age_ms')},
            'pose_reconstruction':'Final receiver estimator replayed from actual synchronized sensor packets; JPEG/depth are recorded hardware data.',
            'remaining_unverified':['anatomical/metric ground truth','Ubuntu target performance','head anatomical orientation','arm axial twist (unobservable from endpoints)','all physical occlusion and multi-person cases']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sessions',nargs='+',type=Path);p.add_argument('--console',action='append',type=Path,default=[]);p.add_argument('--output',required=True,type=Path);p.add_argument('--snapshot',type=Path);p.add_argument('--physical',action='store_true',help='explicitly declare independently confirmed camera input; never use for synthetic fixtures');a=p.parse_args()
    result=summarize(a.sessions,a.console,a.snapshot,a.physical);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
