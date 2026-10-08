"""Validate the phone's explicit 2D occlusion estimates; never sample their depth.

These are projected color-pixel constraints, not metric anatomical bone lengths.
Raw body measurements and their 3D validity remain independent of this channel.
"""
import math
from .protocol import BODY_JOINTS, ProtocolError, finite, integer


def validate_body_tracking(value, header):
    def require(condition, reason):
        if not condition:
            raise ProtocolError('body_tracking: '+reason)

    def provenance(obj, frame_key, time_key):
        frame = integer(obj.get(frame_key), frame_key)
        time = finite(obj.get(time_key), time_key)
        require(frame <= header['frame_id'] and 0 < time <= header['capture_time_s'], 'future/invalid source')
        return frame, time

    require(isinstance(value, dict), 'object required')
    require(value.get('schema') == 'handdepth.body_tracking.v1', 'schema')
    require(value.get('coordinate_frame') == 'color_normalized_top_left_unmirrored', 'coordinate frame')
    require(value.get('length_unit') == 'color_pixels', 'length unit')
    f,t = provenance(value, 'frame_id', 'capture_time_s')
    require(f == header['frame_id'] and t == header['capture_time_s'], 'source frame mismatch')
    points = value.get('points')
    require(isinstance(points, list) and len(points) <= 19, 'point count')
    w,h = header['color_dimensions']
    by_name = {}
    last_index = -1
    for point in points:
        require(isinstance(point, dict), 'point object')
        name = point.get('name')
        require(name in BODY_JOINTS, 'point name')
        index = BODY_JOINTS.index(name)
        require(index > last_index, 'unique ordered points')
        last_index = index
        state = point.get('state')
        require(state in ('observed_body','observed_hand','held','inferred'), 'point state')
        valid = point.get('measurement_valid')
        require(type(valid) is bool and valid == (state in ('observed_body','observed_hand')), 'measurement validity')
        pf,pt = provenance(point, 'source_frame_id', 'source_time_s')
        confidence = finite(point.get('confidence'), 'confidence')
        require(0 <= confidence <= 1, 'confidence range')
        if valid:
            require(pf == f and pt == t and confidence >= .4, 'fresh observation provenance')
        if state == 'observed_hand':
            require(name in ('leftWrist','rightWrist'), 'hand association joint')
        if state == 'inferred':
            require(confidence == 0, 'inference has no detector confidence')
        for key,dimension in [('x',w),('y',h)]:
            coordinate = finite(point.get(key), key)
            # Inference may project outside the raster. It is clipped only for
            # display; clamping XY independently would break the bone lengths.
            require(abs(coordinate*dimension) <= 8192 if state == 'inferred' else 0 <= coordinate <= 1,
                    'point bounds')
        by_name[name] = point
    arms = value.get('arms')
    require(isinstance(arms, dict) and set(arms) <= {'left','right'}, 'arms')
    for side, arm in arms.items():
        require(isinstance(arm, dict), 'arm object')
        require(arm.get('method') == 'two_circle_previous_bend', 'constraint method')
        lf,lt = provenance(arm, 'length_source_frame_id','length_source_time_s')
        lengths = [finite(arm.get(k),k) for k in ('upper_arm_length_px','forearm_length_px')]
        require(all(2 < length <= math.hypot(w,h)+1e-6 for length in lengths), 'bone lengths')
        for key in ('elbow_inferred','wrist_clamped'):
            require(type(arm.get(key)) is bool, 'arm state')
        target = arm.get('target_wrist_xy')
        require(isinstance(target,list) and len(target)==2 and all(0 <= finite(v,'wrist target') <= 1 for v in target), 'wrist target')
        names = [side+j for j in ('Shoulder','Elbow','Wrist')]
        require(all(n in by_name for n in names), 'arm points')
        require(arm['elbow_inferred'] == (by_name[names[1]]['state']=='inferred'), 'elbow state mismatch')
        require(arm['wrist_clamped'] == (by_name[names[2]]['state']=='inferred'), 'wrist state mismatch')
        if arm['elbow_inferred']:
            xyz = [(by_name[n]['x']*w,by_name[n]['y']*h) for n in names]
            for a,b,length in zip(xyz,xyz[1:],lengths):
                require(abs(math.dist(a,b)-length) <= 1e-3, 'inferred bone length mismatch')
        else:
            require(not arm['wrist_clamped'], 'clamped without inferred elbow')
    for name,p in by_name.items():
        if p['state']=='inferred':
            side = 'left' if name.startswith('left') else 'right'
            require(name in (side+'Elbow',side+'Wrist') and side in arms, 'unsupported inferred joint')
