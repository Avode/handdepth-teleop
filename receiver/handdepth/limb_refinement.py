"""Verify fresh RGB joint detections against the current registered depth image.

Never sample a held/inferred phone point. A phone continuity veto may be
superseded only by a current raw Vision or PC RGB detection with depth evidence.
"""
from collections import deque
import math
import cv2
import numpy as np
from .geometry import sample_surface


SOURCE = 'current_raw_vision_depth_verified'
PC_SOURCE = 'current_pc_rgb_depth_verified'


def verified_limb_point(pose, name):
    """Provenance gate shared by display and control; old recovery is not fresh."""
    body = (pose or {}).get('body') or {}
    evidence = body.get('limb_tracking') or {}
    if (evidence.get('frame_id') != pose.get('frame_id') or
            evidence.get('capture_time_s') != pose.get('capture_time_s') or
            any(evidence.get(k)!=pose.get(k) for k in ('session_id','calibration_id','display_geometry_id'))):
        return False
    return any(p.get('name') == name and p.get('source') in (SOURCE,PC_SOURCE)
               and p.get('measurement_valid') is True and p.get('confidence',0) >= .4
               for p in evidence.get('points', []))


def hand_covers(point, hands, dimensions):
    """Current hand silhouette, rather than a large rectangular hand box."""
    size=np.asarray(dimensions,dtype=float)
    xy=np.array([point['x'],point['y']])*size
    for hand in hands:
        pts=np.array([[p['x'],p['y']] for p in hand['landmarks']
                      if p.get('x') is not None and p.get('confidence',0)>=.4],dtype=np.float32)
        if len(pts)<4:continue
        hull=cv2.convexHull((pts*size).astype(np.float32))
        if cv2.contourArea(hull)>1 and cv2.pointPolygonTest(hull,tuple(xy),True)>=-.007*size[1]:
            return True
    return False


def surface_candidate(calibration, depth, point):
    sample=sample_surface(calibration,depth,point)
    if sample['valid'] or sample['reason']!='depth_edge':return sample
    # At a limb outline, foreground and background can occupy one 5x5 patch.
    # Use ONLY the connected depth layer containing the actual detected pixel.
    # A hole at that pixel is never filled with an arbitrary neighbouring depth.
    pixel=calibration.landmark_pixel(point['x'],point['y'])
    x,y=np.floor(pixel).astype(int)
    z=float(depth[y,x])/1000
    if not .15<=z<=4:return sample
    tile=depth[y-2:y+3,x-2:x+3].astype(float)/1000
    mask=((tile>0)&(np.abs(tile-z)<=max(.012,.02*z))).astype(np.uint8)
    _,labels=cv2.connectedComponents(mask,connectivity=4)
    label=labels[2,2]
    if label==0:return sample
    values=tile[labels==label]
    if len(values)<8:return sample
    median=float(np.median(values));sigma=max(.001,float(np.median(np.abs(values-median)))*1.4826)
    xyz=calibration.unproject(pixel,median)
    footprint=2*median/min(calibration.k_depth[0,0],calibration.k_depth[1,1])
    sample.update(valid=True,xyz_m=xyz.tolist(),sigma_z_m=sigma,
        sigma_xyz_m=[math.hypot(sigma*xyz[0]/median,footprint),math.hypot(sigma*xyz[1]/median,footprint),sigma],
        reason='limb_connected_depth_layer',sample_count=len(values))
    return sample


def segment_evidence(calibration,depth,a,b,sa,sb):
    """Depth continuity along a detected bone and a visible foreground boundary."""
    pa=calibration.landmark_pixel(a['x'],a['y']);pb=calibration.landmark_pixel(b['x'],b['y'])
    delta=pb-pa;span=np.linalg.norm(delta)
    if span<4:return {'support':0.,'boundary':0.}
    normal=np.array([-delta[1],delta[0]])/span
    za,zb=sa['xyz_m'][2],sb['xyz_m'][2]
    supported=0;boundaries=0;h,w=depth.shape
    for t in np.linspace(.15,.85,9):
        pixel=pa+t*delta;x,y=np.floor(pixel).astype(int)
        if not 1<=x<w-1 or not 1<=y<h-1:continue
        z=1/((1-t)/za+t/zb)
        tile=depth[y-1:y+2,x-1:x+2].astype(float)/1000
        values=tile[(tile>0)&(tile<=4)]
        tolerance=.045+.035*z
        if len(values)<3 or np.min(np.abs(values-z))>tolerance:continue
        supported+=1
        # One exposed edge is enough for an arm beside the torso. A uniform
        # background plane supplies no silhouette evidence and cannot recover it.
        radius=np.clip(calibration.k_depth[0,0]*.12/z,5,35)
        for sign in (-1,1):
            q=np.floor(pixel+sign*radius*normal).astype(int)
            if 0<=q[0]<w and 0<=q[1]<h:
                outside=depth[q[1],q[0]]/1000
                if outside==0 or outside-z>.12:
                    boundaries+=1;break
    return {'support':supported/9,'boundary':boundaries/9}


class LimbRefiner:
    """Verify two arms with bounded metric history and optional PC RGB joints."""
    def __init__(self):
        self.history={}
        self.pending={}

    def reset(self):
        self.history.clear();self.pending.clear()

    def refine(self,raw,calibration,depth,samples,hands,hand_detections,association,tracking,capture,frame,pc_pose=None):
        points={p['name']:p for p in raw['landmarks']}
        cues={p['name']:p for p in (tracking or {}).get('points',[])}
        result={'frame_id':frame,'capture_time_s':capture,'points':[],'arms':{},
                'method':'fresh_rgb_joints_and_registered_depth','samples_inferred_pixels':False}
        pc=(pc_pose or {}).get('landmarks',{})
        # Associate the PC's person with this phone skeleton before using joints.
        # This prevents an unrelated person or mirrored labels taking over control.
        anchors=[n for n in ('leftShoulder','rightShoulder') if n in pc and
                 points[n]['x'] is not None and points[n]['confidence']>=.4 and pc[n]['confidence']>=.65]
        agreement=[np.linalg.norm(np.array([points[n]['x']-pc[n]['x'],points[n]['y']-pc[n]['y']])) for n in anchors]
        matched=bool(len(anchors)==2 and max(agreement)<.09)
        result['pc_person_matched']=matched
        for side in ('left','right'):
            names=[side+j for j in ('Shoulder','Elbow','Wrist')]
            raw_points=[points[n] for n in names]
            sources=[SOURCE]*3
            disagreements={}
            if matched:
                for i,n in enumerate(names):
                    observed=pc.get(n)
                    if not observed or observed['confidence']<.65 or not 0<=observed['x']<=1 or not 0<=observed['y']<=1:continue
                    cue=cues.get(n)
                    if points[n]['x'] is not None:
                        disagreements[n]=float(np.linalg.norm((np.array([points[n]['x'],points[n]['y']])-np.array([observed['x'],observed['y']]))*calibration.color))
                    if not samples[n]['valid'] or cue is not None and not cue['measurement_valid']:
                        raw_points[i]=dict(observed);sources[i]=PC_SOURCE
            info={'state':'unavailable','reason':'current RGB/depth endpoints required'}
            info['pc_phone_disagreement_px']=disagreements
            result['arms'][side]=info
            if raw_points[1]['x'] is not None and hand_covers(raw_points[1],hand_detections,calibration.color):
                info['reason']='current_hand_over_elbow';self.pending.pop(side,None);continue
            candidate=[surface_candidate(calibration,depth,p) for p in raw_points]
            # Current one-to-one hand wrist can establish the forearm when the
            # body wrist is weak. It never provides an elbow or an old hand ID.
            match=association.get(side)
            wrist_from_hand=False
            if not candidate[2]['valid'] and match is not None:
                wrist=next((p for p in hand_detections[match]['landmarks'] if p['name']=='wrist'),None)
                if wrist is not None:
                    hand_sample=surface_candidate(calibration,depth,wrist)
                    if hand_sample['valid']:
                        raw_points[2]=dict(wrist,name=names[2]);candidate[2]=dict(hand_sample,name=names[2])
                        wrist_from_hand=True
                        sources[2]=SOURCE
            if not all(p['valid'] for p in candidate):
                self.pending.pop(side,None);continue
            xyz=np.array([p['xyz_m'] for p in candidate]);lengths=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
            if not np.all((lengths>=.08)&(lengths<=.55)) or not .35<=lengths[0]/lengths[1]<=2.8:
                info['reason']='implausible_metric_segments';self.pending.pop(side,None);continue
            upper=segment_evidence(calibration,depth,*raw_points[:2],*candidate[:2])
            fore=segment_evidence(calibration,depth,*raw_points[1:],*candidate[1:])
            info.update(upper_arm=upper,forearm=fore,lengths_m=lengths.tolist())
            supported=min(upper['support'],fore['support'])>=7/9 and max(upper['boundary'],fore['boundary'])>=3/9
            if not supported:
                info['reason']='insufficient_limb_depth_support';self.pending.pop(side,None);continue
            old=self.history.get(side)
            if old and capture-old['capture']<=2.:
                median=np.median(old['lengths'],axis=0)
                if np.any(np.abs(lengths-median)>np.maximum(.09,.4*median)):
                    info['reason']='abrupt_metric_length_change';self.pending.pop(side,None);continue
            rejected=not samples[names[1]]['valid'] or any(not samples[n]['valid'] for n in names)
            if rejected:
                previous=self.pending.get(side)
                continuing=(previous is not None and 0<capture-previous['capture']<=.5
                            and frame>previous['frame'] and np.linalg.norm(xyz[1]-previous['elbow'])<=.2+2*(capture-previous['capture']))
                count=min(2,previous['count']+1) if continuing else 1
                # Repeated polling of one frame cannot confirm an observation.
                if previous and frame==previous['frame']:count=previous['count']
                self.pending[side]={'frame':frame,'capture':capture,'elbow':xyz[1],'count':count}
                if count<2:
                    info['state']='acquiring';info['reason']='confirming_current_rgbd_observation';continue
            info.update(state='measured',reason='current_rgbd_limb_supported')
            history=self.history.setdefault(side,{'lengths':deque(maxlen=5),'capture':capture})
            if capture-history['capture']>2.:history['lengths'].clear()
            if not old or capture>history['capture']:history['lengths'].append(lengths)
            if not history['lengths']:history['lengths'].append(lengths)
            history['capture']=capture
            for name,point,c,source in zip(names,raw_points,candidate,sources):
                cue=cues.get(name)
                needs_refinement=(not samples[name]['valid'] or cue is not None and not cue['measurement_valid'])
                if not needs_refinement:continue
                c.update(source=source,reason='rgbd_verified_pc_joint' if source==PC_SOURCE else 'rgbd_verified_raw_joint',pixel_xy=[point['x'],point['y']])
                # Body wrist provenance remains available separately when a hand
                # provides it; original low-confidence body landmark stays invalid.
                if name.endswith('Wrist') and wrist_from_hand:
                    info['associated_wrist']=c
                else:samples[name].update(c)
                result['points'].append(dict(name=name,x=point['x'],y=point['y'],confidence=point['confidence'],
                    state='observed_rgbd',measurement_valid=True,source=source,
                    source_frame_id=frame,source_time_s=capture))
        return result
