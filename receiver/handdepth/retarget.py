"""Torso-relative human motion into a fixed G2 reference, with persistent neutral."""
from dataclasses import dataclass
import copy
import math
import numpy as np
from .body import rotation_matrix
from .hand_control import forearm_frame,FistLatch

# Human anatomical right/down/forward -> robot forward/left/up. det(C) = +1.
TORSO_TO_ROBOT = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
SIDES = ('left', 'right')


def vector(value, count=3):
    try:
        a = np.asarray(value, dtype=float)
        return a if a.shape == (count,) and np.isfinite(a).all() else None
    except (TypeError, ValueError):
        return None


def rotation(value):
    q = vector(value, 4)
    if q is None or abs(np.linalg.norm(q)-1) > .02:
        return None
    return rotation_matrix(q/np.linalg.norm(q))


def bounded(v, limit):
    return v * min(1., limit/max(np.linalg.norm(v), 1e-12))


def arm_points(body, side):
    arm = body.get('arms', {}).get(side, {})
    if not arm.get('torso_relative_valid'):
        return None
    p = arm.get('positions_torso_m', {})
    values = [vector(p.get(side+x)) for x in ('Shoulder', 'Elbow', 'Wrist')]
    return np.stack(values) if all(x is not None for x in values) else None


def measured_upper_arm(pose, side):
    """Current metric shoulder/elbow only; never use persistent display geometry.

    Full-arm validity also requires a wrist, which this two-DOF task does not.
    Require each original depth landmark and respect the phone's occlusion veto.
    """
    body = pose.get('body') or {}
    if body.get('torso', {}).get('frame') != 'torso_x_anatomical_right_y_down_z_forward':
        return None
    names = [side+'Shoulder', side+'Elbow']
    landmarks = {p['name']: p for p in body.get('landmarks', [])}
    cues = {p['name']: p for p in (pose.get('body_tracking') or {}).get('points', [])}
    points = body.get('arms', {}).get(side, {}).get('positions_torso_m', {})
    for name in names:
        if not landmarks.get(name, {}).get('valid'):
            return None
        if name in cues and not cues[name].get('measurement_valid'):
            from .limb_refinement import verified_limb_point
            if not verified_limb_point(pose,name):return None
    values = [vector(points.get(n)) for n in names]
    if any(v is None for v in values):
        return None
    upper = values[1]-values[0]
    length = np.linalg.norm(upper)
    return TORSO_TO_ROBOT @ (upper/length) if .05 <= length <= .75 else None


def measured_forearm(pose, side):
    """A current measured elbow/wrist, including the explicit hand-wrist path.

    Never trust retained display XYZ or a standalone elbow-flexion scalar. The
    shoulder gate is shared so bend and upper-arm directions use one torso frame.
    """
    upper = measured_upper_arm(pose, side)
    body = pose.get('body') or {}
    arm = body.get('arms', {}).get(side, {})
    if upper is None or not arm.get('measurement_valid') or not arm.get('torso_relative_valid'):
        return None
    points = arm.get('positions_torso_m', {})
    elbow, wrist = (vector(points.get(side+n)) for n in ('Elbow', 'Wrist'))
    if elbow is None or wrist is None:
        return None
    name = side+'Wrist'
    from .limb_refinement import verified_limb_point
    verified = verified_limb_point(pose, name)
    landmark = next((p for p in body.get('landmarks', []) if p['name']==name), {})
    associated = False
    if arm.get('wrist_source') == 'associated_vision_hand_wrist_surface':
        for hand in pose.get('hands', []):
            if hand.get('hand_id') != arm.get('hand_id') or hand.get('body_arm_side') != side:
                continue
            for p in hand.get('landmarks', []):
                xyz = vector(p.get('xyz_torso_m'))
                if p['name']=='wrist' and p.get('valid') and xyz is not None and np.allclose(xyz,wrist,atol=1e-6):
                    associated = True
        # Connected-surface verification can recover a current associated wrist
        # even when the ordinary hand depth sampler rejected its edge pixel.
        associated = associated or verified
    if not landmark.get('valid') and not associated:
        return None
    cue = next((p for p in (pose.get('body_tracking') or {}).get('points', []) if p['name']==name), None)
    if cue is not None and not cue.get('measurement_valid') and not (verified or associated):
        return None
    fore = wrist-elbow
    length = np.linalg.norm(fore)
    if not .05 <= length <= .75:
        return None
    fore = TORSO_TO_ROBOT @ (fore/length)
    sine = float(np.linalg.norm(np.cross(upper,fore)))
    return {'forearm_direction':fore,'elbow_flexion_rad':float(np.arctan2(sine,np.clip(upper@fore,-1.,1.))),
            'bend_sine':sine,'wrist_source':'associated_current_hand' if associated else 'current_body_wrist'}


def mean_rotation(values):
    u, _, vt = np.linalg.svd(np.sum(values, axis=0))
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


@dataclass
class FrameTarget:
    position: np.ndarray
    rotation: np.ndarray | None


class Retargeter:
    def __init__(self, robot_home, max_age=.5, neutral_samples=3, arm_mapping='cartesian', wrist_mapping='hold', grip_mapping='pinch'):
        if arm_mapping not in ('cartesian', 'shoulder', 'limb'):
            raise ValueError('arm_mapping must be cartesian, shoulder or limb')
        self.arm_mapping = arm_mapping
        if wrist_mapping not in ('hold','relative') or grip_mapping not in ('pinch','fist'):
            raise ValueError('invalid wrist/grip mapping')
        self.wrist_mapping=wrist_mapping;self.grip_mapping=grip_mapping
        self.wrist_neutral={};self.fists={s:FistLatch() for s in SIDES};self.grip_status={}
        self.shoulder_ready = set()
        self.elbow_ready = set()
        self.bend_observable = {}
        self.home = copy.deepcopy(robot_home)
        self.max_age = max_age
        self.neutral_samples = neutral_samples
        self.armed = False
        self.reason = 'Relax arms; press C to set neutral'
        self.neutral = None
        self.identity = None
        self.frame_id = -1
        self.blocked_frame = -1
        self.age = None
        self.scales = {}
        self.segment_scales = {}
        self.hand_neutral = {}
        self.head_neutral = None
        self.calibrations = 0
        self.active_parts = []
        self.part_status = {}
        self.samples = {}
        self.arm_neutral = {}
        self.sample_frames = {}
        self.pinch_distances = {}

    def disarm(self, reason='Paused; press C to set neutral and resume'):
        self.armed = False
        self.reason = reason
        self.active_parts = []
        for latch in self.fists.values():latch.invalidate()

    def fresh(self, pose, now):
        self.age = None
        if not pose:
            return False
        received = pose.get('received_monotonic_s')
        if not isinstance(received, (int, float)) or not math.isfinite(received):
            return False
        age = now - received
        if age < -.05:
            return False
        aligned = pose.get('capture_to_receive_s')
        if isinstance(aligned, (int, float)) and math.isfinite(aligned):
            age += max(0., aligned)
        self.age = max(0., age)
        return (not pose.get('receiver_event') and self.age <= self.max_age
                and pose.get('schema') == 'handdepth.pose.v1'
                and pose.get('coordinate_frame') == 'camera_optical_x_right_y_down_z_forward')

    def calibrate(self, pose, now, partial=False):
        if not self.fresh(pose, now):
            self.reason = 'Waiting for fresh measurements; relax arms'
            return False
        b = pose.get('body') or {}
        t = b.get('torso', {})
        if (not t.get('orientation_valid') or rotation(t.get('quaternion_xyzw')) is None
                or vector(t.get('position_m')) is None):
            self.reason = 'Waiting for measured shoulders/chest; no T-pose needed'
            return False
        if not partial and any(arm_points(b, side) is None for side in SIDES):
            return False
        self.neutral = copy.deepcopy(pose)
        self.identity = (pose['session_id'], pose['calibration_id'])
        self.geometry_id = pose.get('display_geometry_id')
        self.frame_id = pose['frame_id']
        self.blocked_frame = -1
        self.scales.clear()
        self.segment_scales.clear()
        self.arm_neutral.clear()
        self.shoulder_ready.clear()
        self.elbow_ready.clear()
        self.bend_observable.clear()
        self.hand_neutral.clear()
        self.wrist_neutral.clear()
        self.fists={s:FistLatch() for s in SIDES}
        self.head_neutral = None
        self.samples.clear()
        self.sample_frames.clear()
        self.armed = True
        self.calibrations += 1
        self.reason = 'Acquiring comfortable neutral independently for each part'
        return True

    def collect(self, key, value, frame, capture):
        """Three distinct nearby valid source frames, not three controller ticks."""
        if self.sample_frames.get(key) == frame:
            return None
        self.sample_frames[key] = frame
        samples = self.samples.setdefault(key, [])
        if samples and capture - samples[-1][0] > 1.0:
            samples.clear()
        samples.append((capture, value))
        if len(samples) < self.neutral_samples:
            return None
        result = [v for _, v in samples[-self.neutral_samples:]]
        samples.clear()
        return result

    def calibration_summary(self):
        return {'source_pose': self.neutral, 'calibrations': self.calibrations,
                'arm_mapping': self.arm_mapping, 'shoulder_ready': sorted(self.shoulder_ready),
                'elbow_ready': sorted(self.elbow_ready),
                'wrist_mapping':self.wrist_mapping,'grip_mapping':self.grip_mapping,
                'wrist_neutral':{s:{k:v.tolist() for k,v in n.items()} for s,n in self.wrist_neutral.items()},
                'torso_mode': 'fixed', 'arm_scales': self.scales,
                'segment_scales': {s: x.tolist() for s, x in self.segment_scales.items()},
                'arm_neutral_positions_torso_m': {s: x.tolist() for s, x in self.arm_neutral.items()},
                'head_neutral_rotation_torso': None if self.head_neutral is None else self.head_neutral.tolist(),
                'palm_neutral_rotations': {s: {'human_torso': h.tolist(), 'robot_anchor': r.tolist()}
                                          for s, (h, r) in self.hand_neutral.items()}}

    def targets(self, pose, now, current_home=None):
        self.active_parts = []
        self.part_status = {'torso': 'fixed robot anchor', 'left': 'held: unavailable',
                            'right': 'held: unavailable', 'head': 'held: orientation unavailable',
                            'left_palm': 'held: orientation unavailable', 'right_palm': 'held: orientation unavailable',
                            'left_grip': 'held: pinch unavailable', 'right_grip': 'held: pinch unavailable'}
        self.pinch_distances = {}
        self.grip_status={s:{'mode':self.grip_mapping,'valid':False,'state':'held','score':None} for s in SIDES}
        if not self.armed:
            return None
        if not self.fresh(pose, now):
            for latch in self.fists.values():latch.invalidate()
            # A temporary gap stops commands but does NOT erase the neutral.
            self.blocked_frame = max(self.blocked_frame, self.frame_id)
            self.reason = 'Holding: stream stale/disconnected; neutral retained'
            return None
        if (pose.get('session_id'), pose.get('calibration_id')) != self.identity:
            self.disarm('New source session/calibration; acquire comfortable neutral')
            return None
        if pose.get('display_geometry_id') != self.geometry_id:
            self.disarm('Changed source geometry; acquire comfortable neutral')
            return None
        frame = pose.get('frame_id')
        if not isinstance(frame, int) or frame < self.frame_id:
            self.disarm('Frame order violation; press C to resume')
            return None
        if frame <= self.blocked_frame:
            self.reason = 'Holding: waiting for a newer source frame'
            return None
        self.frame_id = frame
        b = pose.get('body') or {}
        t = b.get('torso', {})
        if not t.get('orientation_valid') or rotation(t.get('quaternion_xyzw')) is None or vector(t.get('position_m')) is None:
            for latch in self.fists.values():latch.invalidate()
            self.reason = 'Holding: torso depth/orientation unavailable; neutral retained'
            return None
        c = TORSO_TO_ROBOT
        current = current_home or self.home
        capture = pose.get('capture_time_s', now)
        result = {'torso': copy.deepcopy(self.home['torso']), 'head': None,
                  'arms': {s: None for s in SIDES}, 'grippers': {}}
        for side in SIDES:
            arm = b.get('arms', {}).get(side, {})
            a = arm_points(b, side)
            h = self.home['arms'][side]
            if self.arm_mapping in ('shoulder','limb'):
                direction = measured_upper_arm(pose, side)
                self.part_status[side] = 'held: measured shoulder/elbow required'
                self.part_status[side+'_palm'] = 'held: wrist orientation not enabled'
                self.part_status[side+'_elbow'] = 'held: current shoulder/elbow/wrist required'
                if direction is not None:
                    if side not in self.shoulder_ready:
                        self.part_status[side] = 'acquiring measured upper arm'
                        if self.collect(side+'_shoulder', direction, frame, capture) is not None:
                            self.shoulder_ready.add(side)
                    if side in self.shoulder_ready:
                        # Absolute torso-relative direction: C arms the stream but
                        # never subtracts the human's real shoulder angle away.
                        result['arms'][side] = {'upper_arm_direction': direction}
                        self.active_parts.append(side+' shoulder')
                        self.part_status[side] = 'tracking shoulder direction; distal joints held'
                        if self.arm_mapping == 'limb':
                            forearm = measured_forearm(pose,side)
                            if forearm is not None:
                                if side not in self.elbow_ready:
                                    self.part_status[side+'_elbow'] = 'acquiring measured forearm'
                                    if self.collect(side+'_elbow',forearm['forearm_direction'],frame,capture) is not None:
                                        self.elbow_ready.add(side)
                                if side in self.elbow_ready:
                                    # Hysteresis avoids twist chatter near straight/folded
                                    # arms, where the bend plane is poorly observable.
                                    threshold = .12 if self.bend_observable.get(side,False) else .20
                                    observable = forearm['bend_sine'] > threshold
                                    self.bend_observable[side] = observable
                                    result['arms'][side].update(forearm,bend_plane_observable=observable)
                                    self.active_parts.append(side+' elbow')
                                    self.part_status[side] = 'tracking shoulder + forearm; wrist held'
                                    self.part_status[side+'_elbow'] = 'tracking bend + bend plane' if observable else 'tracking bend; axial rotation held near straight/folded'
            if self.arm_mapping == 'cartesian' and side not in self.arm_neutral and a is not None:
                lengths = np.linalg.norm(np.diff(a, axis=0), axis=1)
                if np.all((lengths > .05) & (lengths < .75)):
                    samples = self.collect(side, a, frame, capture)
                    self.part_status[side] = 'acquiring relaxed neutral'
                    if samples is not None:
                        a0 = np.median(samples, axis=0)
                        human = np.median([np.linalg.norm(np.diff(x, axis=0), axis=1) for x in samples], axis=0)
                        h = copy.deepcopy(current['arms'][side])
                        self.home['arms'][side] = h
                        robot_lengths = np.asarray(h.get('segment_lengths', [h['length']/2]*2))
                        self.segment_scales[side] = np.clip(robot_lengths/human, .5, 2.)
                        self.scales[side] = float(np.clip(h['length']/sum(human), .5, 2.))
                        self.arm_neutral[side] = a0
            if side in self.arm_neutral:
                a0 = self.arm_neutral[side]
                points = arm.get('positions_torso_m', {})
                shoulder = vector(points.get(side+'Shoulder'))
                wrist_point = vector(points.get(side+'Wrist'))
                elbow_point = vector(points.get(side+'Elbow'))
                partial_wrist = elbow_point is None and shoulder is not None and wrist_point is not None
                if (arm.get('torso_relative_valid') or partial_wrist) and shoulder is not None and wrist_point is not None:
                    upper0, lower0 = np.diff(a0, axis=0)
                    elbow = None
                    if a is not None:
                        upper, lower = np.diff(a, axis=0)
                        su, sf = self.segment_scales[side]
                        elbow = h['elbow'] + bounded(c @ (su*(upper-upper0)), .4)
                        displacement = c @ (su*(upper-upper0) + sf*(lower-lower0))
                    else:
                        # Current measured wrist remains useful when only elbow depth is lost.
                        displacement = self.scales[side]*c @ ((wrist_point-shoulder)-(a0[2]-a0[0]))
                    wrist = h['wrist'] + bounded(displacement, .55)
                    # Respect measured robot reach; elbow remains a soft redundancy hint.
                    shoulder_robot = h.get('shoulder')
                    if shoulder_robot is not None:
                        wrist = shoulder_robot + bounded(wrist-shoulder_robot, h['length']*.97)
                    result['arms'][side] = {'elbow': elbow, 'wrist': FrameTarget(wrist, None)}
                    self.active_parts.append(side+' arm')
                    self.part_status[side] = 'tracking' if elbow is not None else 'tracking wrist; elbow held'
            # Association is anatomical and current-frame. IDs are temporary tracker labels.
            matches = [x for x in pose.get('hands', []) if arm.get('hand_id') is not None
                       and x.get('hand_id') == arm.get('hand_id') and x.get('body_arm_side') == side
                       and x.get('chirality', 'unknown') in (side, 'unknown')]
            hand = matches[0] if len(matches) == 1 else None
            if self.wrist_mapping=='relative' and self.arm_mapping=='limb':
                self.part_status[side+'_palm']='held: measured palm/forearm required'
            if hand:
                hr = rotation(hand.get('palm_quaternion_torso_xyzw')) if hand.get('torso_relative_orientation_valid') else None
                target=result['arms'][side]
                if (self.wrist_mapping=='relative' and self.arm_mapping=='limb' and target is not None
                        and 'forearm_direction' in target and hr is not None and hand.get('orientation_valid')):
                    frame_rotation=forearm_frame(target['upper_arm_direction'],target['forearm_direction'])
                    if not target['bend_plane_observable'] or frame_rotation is None:
                        self.part_status[side+'_palm']='held: forearm bend plane ambiguous'
                    else:
                        relative=frame_rotation.T@c@hr
                        if side not in self.wrist_neutral:
                            samples=self.collect(side+'_relative_wrist',relative,frame,capture)
                            self.part_status[side+'_palm']='acquiring comfortable wrist neutral'
                            if samples is not None:
                                robot=current['arms'][side]
                                up=robot['elbow']-robot['shoulder'];fore=robot['wrist']-robot['elbow']
                                robot_frame=forearm_frame(up/np.linalg.norm(up),fore/np.linalg.norm(fore))
                                parent=robot.get('forearm_rotation')
                                if robot_frame is not None and parent is not None:
                                    self.wrist_neutral[side]={'human':mean_rotation(samples),
                                        'robot':parent.T@robot['rotation'],'axes':parent.T@robot_frame}
                        if side in self.wrist_neutral:
                            n=self.wrist_neutral[side]
                            target['wrist_rotation_relative']=n['axes']@(relative@n['human'].T)@n['axes'].T@n['robot']
                            self.active_parts.append(side+' wrist')
                            self.part_status[side+'_palm']='tracking palm relative to forearm'
                            self.part_status[side]='tracking shoulder + forearm + wrist'
                if self.arm_mapping == 'cartesian' and hr is not None and result['arms'][side] is not None:
                    if side not in self.hand_neutral:
                        samples = self.collect(side+'_palm', hr, frame, capture)
                        self.part_status[side+'_palm'] = 'acquiring relaxed palm neutral'
                        if samples is not None:
                            self.hand_neutral[side] = (mean_rotation(samples), current['arms'][side]['rotation'].copy())
                    if side in self.hand_neutral:
                        human0, robot0 = self.hand_neutral[side]
                        # Spatial rotation in the torso frame; neutral surface tilt cancels.
                        result['arms'][side]['wrist'].rotation = c @ (hr @ human0.T) @ c.T @ robot0
                        self.part_status[side+'_palm'] = 'tracking'
                pinch = hand.get('pinch_distance_m')
                if self.grip_mapping=='pinch' and hand.get('depth_valid') and isinstance(pinch, (int, float)) and math.isfinite(pinch) and 0 <= pinch <= .2:
                    self.pinch_distances[side] = pinch
                    result['grippers'][side] = float(-.85 + .8*np.clip((pinch-.02)/.06, 0, 1))
                    self.part_status[side+'_grip'] = 'tracking'
                if self.grip_mapping=='fist':
                    gesture=hand.get('grip_gesture')
                    value=self.fists[side].update(gesture,frame,capture,hand['hand_id'])
                    self.grip_status[side].update(score=(gesture or {}).get('score'),
                        fingers=(gesture or {}).get('fingers',{}),source=(gesture or {}).get('source'),
                        close_threshold=FistLatch.close_threshold,open_threshold=FistLatch.open_threshold,
                        reason=(gesture or {}).get('reason','current finger landmarks unavailable'))
                    if value is not None:
                        result['grippers'][side]=value
                        state='closed' if self.fists[side].closed else 'open'
                        self.grip_status[side].update(valid=True,state=state)
                        self.part_status[side+'_grip']=f"{state} | finger curl {gesture['score']:.2f}"
                    else:self.part_status[side+'_grip']='held: confirming fist/open hand'
            else:
                self.fists[side].invalidate()
                if self.grip_mapping=='fist':self.part_status[side+'_grip']='held: current hand unavailable'
        head = b.get('head', {})
        hr = rotation(head.get('quaternion_torso_xyzw')) if head.get('torso_relative_orientation_valid') else None
        if hr is not None:
            if self.head_neutral is None:
                samples = self.collect('head', hr, frame, capture)
                self.part_status['head'] = 'acquiring relaxed head neutral'
                if samples is not None:
                    self.home['head'] = copy.deepcopy(current['head'])
                    self.head_neutral = mean_rotation(samples)
            if self.head_neutral is not None:
                # Neck has rotational DOFs: do not chase nose translation with torso motion.
                h = self.home['head']
                result['head'] = FrameTarget(h.position.copy(), c @ (hr @ self.head_neutral.T) @ c.T @ h.rotation)
                self.active_parts.append('head')
                self.part_status['head'] = 'tracking orientation'
        self.reason = ('Tracking: '+', '.join(self.active_parts) if self.active_parts else 'Holding/acquiring measured parts') + '; torso fixed'
        return result
