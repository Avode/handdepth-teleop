"""Bounded, explicitly uncertain display history. Never supplies control targets.

Metric lengths come only from reliable measured XYZ. Phone color-pixel lengths
are never converted into metres, nor are hidden image points depth-sampled.
"""
from collections import deque
from dataclasses import dataclass, replace
import copy
import math
import time
import numpy as np
from .protocol import BODY_JOINTS
from .retarget import vector
from .limb_refinement import verified_limb_point
from .body import rotation_matrix

LOWER_BODY={'root','leftHip','rightHip','leftKnee','rightKnee','leftAnkle','rightAnkle'}


class DisplaySource:
    """Identity, ordering, and reset boundary shared by the two displays."""
    def __init__(self):
        self.identity = None
        self.frame = -1
        self.names = set()
        self.capture = 0.
        self.arrival = 0.

    def inspect(self, pose, now):
        if not pose or pose.get('receiver_event'):
            return False, False
        identity = (pose.get('session_id'), pose.get('display_geometry_id', pose.get('calibration_id')),
                    pose.get('coordinate_frame'), tuple(pose.get('color_dimensions', [])))
        changed = self.identity is not None and identity != self.identity
        frame = pose.get('frame_id', -1)
        if not changed and frame <= self.frame:
            return False, False
        tracking = pose.get('body_tracking')
        names = {p['name'] for p in tracking['points']} if tracking is not None else None
        # v1 has no reset epoch. Its history is append-only until reset; a
        # smaller authoritative point set signals the phone cleared history.
        reset = changed or (names is not None and bool(self.names - names))
        if changed:self.names = set()
        if names is not None:self.names = names
        self.identity, self.frame = identity, frame
        self.capture = pose.get('capture_time_s', 0.)
        self.arrival = pose.get('received_monotonic_s', now)
        return True, reset

    def clock(self, now):
        return self.capture + max(0., now-self.arrival)


@dataclass
class DisplayPoint:
    xyz: np.ndarray
    state: str
    source_frame: int
    source_time: float
    reason: str = ''

    def held(self):
        return replace(self, state='held', reason='last coherent geometry')


def solve_arm(shoulder, target, lengths, previous_axis, previous_bend):
    """Sphere intersection with parallel-transported bend and bounded reach.

    The endpoint is clamped to the reachable annulus, never the bone lengths.
    A known prior bend is required; this function does not invent an unseen arm.
    """
    if previous_axis is None or previous_bend is None:return None
    a, b = lengths
    delta = target-shoulder
    distance = float(np.linalg.norm(delta))
    axis = delta/distance if distance > 1e-8 else previous_axis.copy()
    dot = float(np.clip(previous_axis@axis, -1., 1.))
    cross = np.cross(previous_axis, axis)
    if dot > -1+1e-7:
        bend = previous_bend + np.cross(cross, previous_bend)
        bend += np.cross(cross, np.cross(cross, previous_bend))/(1+dot)
    else:
        # At the antipode rotate about the previous bend, preserving its side.
        bend = previous_bend.copy()
    bend -= axis*(bend@axis)
    norm = np.linalg.norm(bend)
    if norm < 1e-8:return None
    bend /= norm
    minimum = max(abs(a-b)+1e-6, 1e-5)
    maximum = math.sqrt(a*a+b*b+2*a*b*math.cos(math.radians(2)))
    reach = float(np.clip(distance, minimum, maximum))
    wrist = shoulder+reach*axis
    along = (a*a-b*b+reach*reach)/(2*reach)
    height = math.sqrt(max(0., a*a-along*along))
    elbow = shoulder+along*axis+height*bend
    return elbow, wrist, axis, bend, abs(reach-distance)>1e-7


class MetricSkeleton:
    """At most 19 joints, two coherent arms, and five length samples per arm."""
    def __init__(self, space='torso'):
        self.space = space
        self.source = DisplaySource()
        self.clear()

    def clear(self):
        self.points = {}
        self.arms = {}
        self.torso_time = None
        self.torso_current = False
        self.lower_samples=0;self.lower_sample_time=None;self.lower_relative={}

    def lower_body(self,measured,confidence,body):
        """Do not attach an isolated desk-depth hip to a current shoulder.

        Establish a coherent pelvis from three current shoulder/hip pairs. Known
        lower-body geometry is then retained in the human torso frame, keeping
        camera motion from stretching a current shoulder to an old camera point.
        This affects display only; original measurements are not modified.
        """
        lower={n:measured.pop(n) for n in list(measured) if n in LOWER_BODY}
        torso=body.get('torso',{});origin=vector(torso.get('position_m'))
        quat=vector(torso.get('quaternion_xyzw'),4)
        if not self.torso_current or origin is None or quat is None:return
        r=rotation_matrix(quat)
        def relative(p):return p.xyz if self.space=='torso' else r.T@(p.xyz-origin)
        names=('leftShoulder','rightShoulder','leftHip','rightHip')
        all_points={**measured,**lower}
        valid=all(n in all_points and confidence.get(n,0)>=.6 for n in names)
        if valid:
            ls,rs,lh,rh=[relative(all_points[n]) for n in names]
            down=(lh+rh-ls-rs)/2
            valid=(.12<=np.linalg.norm(lh-rh)<=.60 and .15<=down[1]<=.65
                   and abs(down[0])<.20 and abs(down[2])<.35
                   and all(.20<=np.linalg.norm(h-s)<=.65 for h,s in ((lh,ls),(rh,rs))))
        if valid:
            if self.lower_sample_time is None or not 0<self.source.capture-self.lower_sample_time<=.5:self.lower_samples=0
            self.lower_samples=min(3,self.lower_samples+1);self.lower_sample_time=self.source.capture
            if self.lower_samples>=3:
                allowed={'leftHip','rightHip'}
                if 'root' in lower and np.linalg.norm(relative(lower['root'])-(lh+rh)/2)<.2:allowed.add('root')
                for side in ('left','right'):
                    for start,end in ((side+'Hip',side+'Knee'),(side+'Knee',side+'Ankle')):
                        if start in allowed and end in lower and .1<=np.linalg.norm(lower[end].xyz-lower[start].xyz)<=.7:allowed.add(end)
                for n in allowed:self.lower_relative[n]=replace(lower[n],xyz=relative(lower[n]).copy())
        else:self.lower_samples=0;self.lower_sample_time=None
        for n,p in self.lower_relative.items():
            current=p.source_frame==self.source.frame
            measured[n]=replace(p,xyz=p.xyz.copy() if self.space=='torso' else origin+r@p.xyz,
                state='observed' if current else 'inferred',reason='' if current else 'retained coherent torso-relative lower body')

    def reset(self):
        # Keep the source watermark: polling the same frame cannot undo reset.
        self.clear()

    def hold(self):
        self.points = {n:p.held() for n,p in self.points.items()}
        for arm in self.arms.values():
            arm.update(mode='held',shape=[p.held() for p in arm['shape']])

    def update(self, pose, fresh=True, now=None):
        now = (pose or {}).get('received_monotonic_s', time.monotonic()) if now is None else now
        new, reset = self.source.inspect(pose, now)
        if reset:self.clear()
        if not fresh or (pose or {}).get('receiver_event'):
            self.hold()
            self.torso_current = False
            return self.points
        if not new:return self.points
        self.points = {n:p.held() for n,p in self.points.items()}
        body = (pose or {}).get('body') or {}
        self.torso_current = bool(body.get('torso', {}).get('orientation_valid'))
        if self.torso_current:self.torso_time = self.source.capture
        # Do not transform new camera XYZ through an old torso orientation.
        # Hold the entire coherent torso-relative shape until its frame returns.
        if self.space == 'torso' and not self.torso_current:
            self.hold();return self.points
        key = 'xyz_torso_m' if self.space == 'torso' else 'xyz_m'
        tracking = (pose or {}).get('body_tracking') or {}
        cues = {p['name']:p for p in tracking.get('points', [])}
        measured = {}
        confidence = {}
        for p in body.get('landmarks', []):
            name = p['name'];xyz = vector(p.get(key))
            cue = cues.get(name)
            occluded = name.endswith('Elbow') and cue is not None and not cue['measurement_valid'] and not verified_limb_point(pose,name)
            if name in BODY_JOINTS and p.get('valid') and xyz is not None and not occluded:
                measured[name] = DisplayPoint(xyz.copy(), 'observed', self.source.frame, self.source.capture)
                confidence[name] = p.get('confidence', 0.)
        for side in ('left','right'):
            arm = body.get('arms', {}).get(side, {})
            wrist = side+'Wrist'
            # A current, unambiguous associated hand wrist is a measured surface,
            # even when the body elbow is missing. This is DISPLAY-only fallback.
            matches = [h for h in pose.get('hands', []) if arm.get('hand_id') is not None
                       and h.get('hand_id') == arm['hand_id'] and h.get('body_arm_side') == side
                       and h.get('chirality', 'unknown') in (side, 'unknown')]
            if wrist not in measured and len(matches) == 1:
                for p in matches[0].get('landmarks', []):
                    xyz = vector(p.get(key))
                    if p['name'] == 'wrist' and p.get('valid') and xyz is not None:
                        measured[wrist] = DisplayPoint(xyz.copy(), 'observed', self.source.frame, self.source.capture,
                                                      'associated measured hand wrist')
                        confidence[wrist] = p.get('confidence', 0.)
        self.lower_body(measured,confidence,body)
        self.points.update(measured)
        for side in ('left','right'):
            names = [side+j for j in ('Shoulder','Elbow','Wrist')]
            old = self.arms.get(side)
            complete = all(n in measured for n in names)
            reliable = complete and min(confidence[n] for n in names) >= .4
            cue = tracking.get('arms', {}).get(side, {})
            if cue.get('elbow_inferred') and not verified_limb_point(pose,names[1]):reliable = False
            if reliable:
                s,e,w = [measured[n].xyz for n in names]
                lengths = np.linalg.norm(np.diff([s,e,w], axis=0), axis=1)
                axis = w-s;norm = np.linalg.norm(axis)
                if np.all((lengths >= .05) & (lengths <= .75)):
                    axis = axis/norm if norm>1e-6 else None
                    bend = e-s-axis*((e-s)@axis) if axis is not None else np.zeros(3)
                    bn = np.linalg.norm(bend)
                    samples = old['samples'] if old else deque(maxlen=5)
                    # Reject grossly unequal apparent segments (often a smooth
                    # occluder surface). Require three consistent metric samples
                    # before solving a hidden elbow; a single frame is fragile.
                    quality = .4 <= lengths[0]/lengths[1] <= 2.5
                    if quality:
                        if old and self.source.capture-old['sample_time']>2.:samples.clear()
                        samples.append(lengths)
                    ready = old.get('lengths_reliable',False) if old else False
                    retained = old['lengths'] if old else lengths
                    length_time = old['length_time'] if old else self.source.capture
                    if quality and len(samples)>=3:
                        window=np.array(samples)[-3:];median=np.median(window,axis=0)
                        if np.all(np.ptp(window,axis=0)<=.2*median):
                            retained=median;ready=True;length_time=self.source.capture
                    bend = bend/bn if bn>1e-5 else (old['bend'] if old else None)
                    self.arms[side] = {'samples':samples, 'lengths':retained,
                        'lengths_reliable':ready,'sample_time':self.source.capture if quality else (old['sample_time'] if old else 0.),
                        'axis':axis, 'bend':bend, 'shape':[measured[n] for n in names],
                        'length_time':length_time,'clamped':False,'target':None,
                        'mode':'observed','image_cue':'2D only; not metric'}
                    continue
            if old is None:
                # Previously unseen geometry stays unknown; isolated observations
                # may be shown but do not establish imaginary segment lengths.
                continue
            if old['lengths_reliable'] and names[0] in measured and names[2] in measured:
                shoulder, wrist = measured[names[0]], measured[names[2]]
                result = solve_arm(shoulder.xyz,wrist.xyz,old['lengths'],old['axis'],old['bend'])
                if result is not None:
                    elbow,end,axis,bend,clamped = result
                    prior = old['shape'][1]
                    inferred = DisplayPoint(elbow,'inferred',prior.source_frame,prior.source_time,
                                             'metric lengths + previous 3D bend; not measured')
                    endpoint = replace(wrist,xyz=end,state='inferred',reason='clamped to metric reach') if clamped else wrist
                    shape = [shoulder,inferred,endpoint]
                    old.update(axis=axis,bend=bend,shape=shape,clamped=clamped,target=wrist.xyz.copy(),mode='inferred')
                    self.points.update(zip(names,shape))
                    continue
            # Insufficient current endpoints: hold all three together, never
            # connect a new shoulder to an unrelated held elbow/wrist.
            shape = [p.held() for p in old['shape']]
            old.update(shape=shape,mode='held')
            self.points.update(zip(names,shape))
        return self.points

    def summary(self, now=None):
        now = time.monotonic() if now is None else now
        clock = self.source.clock(now)
        held = [p for p in self.points.values() if p.state != 'observed']
        return {'observed':len(self.points)-len(held),'held_or_inferred':len(held),
                'oldest_age_s':max([max(0.,clock-p.source_time) for p in held],default=0.),
                'torso_current':self.torso_current,
                'arms':{side:{'mode':a['mode'],'lengths_m':a['lengths'].tolist(),
                    'lengths_reliable':a['lengths_reliable'],
                    'wrist_clamped':a['clamped'],'length_age_s':max(0.,clock-a['length_time'])}
                    for side,a in self.arms.items()}}

    def arm_text(self, side, now):
        now=time.monotonic() if now is None else now
        clock=self.source.clock(now);parts=[]
        for joint in ('Shoulder','Elbow','Wrist'):
            p=self.points.get(side+joint)
            parts.append(joint[0]+(': unknown' if p is None else f": {p.state} {max(0.,clock-p.source_time):.1f}s"))
        arm=self.arms.get(side)
        suffix=''
        if arm and not arm['lengths_reliable']:suffix=' | learning lengths'
        if arm and arm['clamped']:suffix+=' | WRIST CLAMPED'
        return ' / '.join(parts)+suffix


class ImageSkeleton:
    """Phone history with an explicit, same-frame projected metric arm layer."""
    def __init__(self):
        self.source = DisplaySource()
        self.points = {}
        self.arms = {}
        self.reset_frame = -1
        self.generation = 0

    def reset(self):
        self.points.clear();self.arms.clear();self.reset_frame = self.source.frame
        self.generation += 1

    def update(self, pose, raw_body, now, metric=None, projection=None):
        new, reset = self.source.inspect(pose, now)
        if reset:self.reset();self.reset_frame = -1
        if not new:return self.snapshot()
        tracking = pose.get('body_tracking')
        if tracking is not None:
            self.points = {p['name']:copy.deepcopy(p) for p in tracking['points']
                           if p['source_frame_id']>self.reset_frame or p['name'] in self.points}
            self.arms = copy.deepcopy(tracking['arms'])
        else:
            for p in self.points.values():p.update(state='held',measurement_valid=False)
            for p in (raw_body or {}).get('landmarks', []):
                if p.get('x') is not None and p.get('confidence',0) >= .4:
                    self.points[p['name']] = dict(p,state='observed_body',measurement_valid=True,
                        source_frame_id=pose['frame_id'],source_time_s=pose['capture_time_s'])
        # Preserve the phone channel verbatim; this separate receiver layer uses
        # the SAME source frame and a fresh raw RGB joint with depth verification.
        evidence=((pose or {}).get('body') or {}).get('limb_tracking') or {}
        for p in evidence.get('points',[]):
            if verified_limb_point(pose,p['name']) and pose['frame_id']>self.reset_frame:
                self.points[p['name']]=copy.deepcopy(p)
                self.arms.pop('left' if p['name'].startswith('left') else 'right',None)
        # Pixel-space lengths change under foreshortening. When measured 3D
        # shoulder/wrist endpoints and learned metric lengths permit a solution,
        # render that SAME solution instead of the phone's different 2D circles.
        # These projected hidden joints remain explicitly inferred, never measured.
        if (metric is not None and projection is not None and metric.space=='camera'
                and metric.source.identity==self.source.identity
                and metric.source.frame==self.source.frame
                and metric.source.capture==self.source.capture):
            for side,arm in metric.arms.items():
                if arm['mode']!='inferred' or not arm['lengths_reliable']:
                    continue
                names=[side+n for n in ('Shoulder','Elbow','Wrist')]
                shape=[metric.points.get(n) for n in names]
                if any(p is None for p in shape):continue
                if shape[1].source_frame<=self.reset_frame:continue
                projected=[projection.project_color(p.xyz) for p in shape]
                if any(v is None or not np.isfinite(v).all() or np.max(np.abs(v))>16 for v in projected):continue
                for name,p,xy in zip(names,shape,projected):
                    # Explicit R reset cannot resurrect an old elbow observation.
                    # The metric tracker independently needs fresh length history.
                    current=p.state=='observed'
                    previous=self.points.get(name,{})
                    self.points[name]={'name':name,'x':float(xy[0]),'y':float(xy[1]),
                        'state':('observed_hand' if 'hand wrist' in p.reason else previous.get('state','observed_body')) if current else p.state,
                        'measurement_valid':current,'confidence':previous.get('confidence',.4) if current else 0.,
                        'source_frame_id':p.source_frame,'source_time_s':p.source_time,
                        'source':'ubuntu_metric_arm_projection','constraint_frame_id':pose['frame_id'],
                        'constraint_time_s':pose['capture_time_s'],'inferred_control':False}
                    if current and self.points[name]['state'] not in ('observed_body','observed_hand','observed_rgbd'):
                        self.points[name]['state']='observed_rgbd'
                target=projection.project_color(arm['target']) if arm['target'] is not None else None
                self.arms[side]={'method':'projected_metric_lengths_previous_bend','length_unit':'metres',
                    'upper_arm_length_m':float(arm['lengths'][0]),'forearm_length_m':float(arm['lengths'][1]),
                    'elbow_inferred':True,'wrist_clamped':arm['clamped'],
                    'target_wrist_xy':None if target is None else target.tolist(),
                    'source_frame_id':pose['frame_id'],'inferred_control':False}
        return self.snapshot()

    def snapshot(self):
        return {'points':copy.deepcopy(list(self.points.values())),'arms':copy.deepcopy(self.arms),
                'frame_id':self.source.frame,'capture_time_s':self.source.capture}
