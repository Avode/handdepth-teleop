"""Measured body surfaces and an observed torso reference; no world/robot frame."""
import itertools
import math
import numpy as np
from .geometry import sample_surface, quaternion_xyzw

BODY_EDGES = [(0,3),(0,4),(3,5),(4,6),(0,1),(1,7),(1,8),(7,8),
              (7,9),(9,11),(8,10),(10,12),(7,13),(8,14),(13,14),(13,2),(14,2),
              (1,2),(13,15),(15,17),(14,16),(16,18)]

def rotation_matrix(q):
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def invalidate_point(point, reason):
    point.update(valid=False,xyz_m=None,sigma_z_m=None,sigma_xyz_m=None,reason=reason)

def associate_arms(raw, hands):
    """One-to-one image wrist association; ambiguous crossings stay unassigned."""
    body_points={p['name']:p for p in raw['landmarks']}
    hand_points=[next((p for p in h['landmarks'] if p['name']=='wrist' and p['confidence']>=.4 and p['x'] is not None),None) for h in hands]
    costs={}
    for side in ('left','right'):
        b=body_points[side+'Wrist']
        if b['confidence']<.2 or b['x'] is None:continue
        for i,p in enumerate(hand_points):
            if p is None:continue
            distance=math.hypot(p['x']-b['x'],p['y']-b['y'])
            if distance<=.14:
                costs[side,i]=distance+(0.05 if hands[i]['chirality'] not in (side,'unknown') else 0)
    possibilities=[]
    for a,b in itertools.product([None,*range(len(hands))],repeat=2):
        if a is not None and a==b:continue
        assignment={'left':a,'right':b}
        if any(i is not None and (side,i) not in costs for side,i in assignment.items()):continue
        cost=sum(.2 if i is None else costs[side,i] for side,i in assignment.items())
        possibilities.append((cost,assignment))
    possibilities.sort(key=lambda p:p[0])
    if len(possibilities)>1 and possibilities[1][0]-possibilities[0][0]<.025:
        return {'left':None,'right':None}
    return possibilities[0][1]

class BodyPoseEstimator:
    def __init__(self):
        from .limb_refinement import LimbRefiner
        self.history={}
        self.limb_refiner=LimbRefiner()

    def reset(self):
        self.history.clear()
        self.limb_refiner.reset()

    def continuous_quaternion(self,name,q,capture):
        previous=self.history.get(name)
        if previous and 0<capture-previous[1]<.5:
            if np.dot(q,previous[0])<0:q=-q
            if capture-previous[1]<.2 and abs(np.dot(q,previous[0]))<.5:
                self.history.pop(name,None);return None
        self.history[name]=(q,capture)
        return q.tolist()

    def orientation(self, name, x, down, capture, min_width=.025, min_down=.012):
        width=np.linalg.norm(x)
        if width<min_width or width>1:
            self.history.pop(name,None);return None,None
        x=x/width
        y=down-np.dot(down,x)*x
        if np.linalg.norm(y)<min_down:
            self.history.pop(name,None);return None,None
        y=y/np.linalg.norm(y)
        r=np.column_stack((x,y,np.cross(x,y)))
        q=quaternion_xyzw(r)
        result=self.continuous_quaternion(name,q,capture)
        return (r,result) if result is not None else (None,None)

    def estimate(self, raw, calibration, depth, error, capture, hands, hand_detections, tracking=None, frame=None, pc_pose=None):
        for hand in hands:
            hand.update(body_arm_side=None,palm_position_torso_m=None,palm_quaternion_torso_xyzw=None,
                        torso_relative_position_valid=False,torso_relative_orientation_valid=False)
            for point in hand['landmarks']:point['xyz_torso_m']=None
        if raw is None:
            self.reset();return None
        out={'tracking_valid':any(p['confidence']>=.4 and p['x'] is not None for p in raw['landmarks']),
             'depth_valid':False,'landmarks':[], 'torso_surface':None,
             'coordinate_frame':'camera_optical_x_right_y_down_z_forward',
             'torso':{'position_m':None,'quaternion_xyzw':None,'position_valid':False,'orientation_valid':False,
                      'frame':'torso_x_anatomical_right_y_down_z_forward',
                      'origin':'midpoint_measured_shoulders','axes_source':'shoulders_and_measured_torso_surface'},
             'head':{'position_m':None,'quaternion_xyzw':None,'position_valid':False,'orientation_valid':False,
                     'position_torso_m':None,'quaternion_torso_xyzw':None,
                     'torso_relative_position_valid':False,'torso_relative_orientation_valid':False,
                     'axes_source':'measured_eyes_and_nose_surface_frame'},'arms':{},'reason':error}
        if error:
            out['landmarks']=[{'name':p['name'],'confidence':p['confidence'],'valid':False,'xyz_m':None,
                'xyz_torso_m':None,'sigma_z_m':None,'sigma_xyz_m':None,'sample_count':0,'reason':error} for p in raw['landmarks']]
            self.reset();return out
        # The phone can reject a high-confidence Vision elbow covered by a hand.
        # Reject that raw sample initially. The independent RGB-D verifier below
        # can reacquire a CURRENT RGB detection with explicit depth evidence;
        # it never unprojects the phone's held/inferred pixel coordinates.
        hidden = {p['name'] for p in (tracking or {}).get('points', [])
                  if p['name'].endswith('Elbow') and not p['measurement_valid']}
        out['landmarks']=[]
        for p in raw['landmarks']:
            if p['name'] in hidden:
                sample = {'name':p['name'],'confidence':p['confidence'],'valid':False,
                          'xyz_m':None,'sigma_z_m':None,'sigma_xyz_m':None,'sample_count':0,
                          'reason':'elbow_occluded_2d_tracking_cue'}
            elif p['name'] in ('nose','leftEye','rightEye','leftEar','rightEar'):
                sample = sample_surface(calibration,depth,p,radius=1,minimum_samples=5)
            else:sample = sample_surface(calibration,depth,p)
            out['landmarks'].append(sample)
        samples={p['name']:p for p in out['landmarks']}
        association=associate_arms(raw,hand_detections)
        out['limb_tracking']=self.limb_refiner.refine(raw,calibration,depth,samples,hands,hand_detections,
                                                   association,tracking,capture,frame,pc_pose)
        if raw.get('torso_surface') is not None:
            out['torso_surface']=sample_surface(calibration,depth,raw['torso_surface'])
            out['torso_surface']['source']=raw['torso_surface_source']+'_measured_depth'
        def points():return {p['name']:np.array(p['xyz_m']) for p in out['landmarks'] if p['valid']}
        p=points()
        # Broad surface/body-length checks reject smooth background samples. They
        # cannot identify every occlusion or prove an internal anatomical joint.
        if all(k in p for k in ('leftShoulder','rightShoulder')):
            anchor=(p['leftShoulder']+p['rightShoulder'])/2
            for point in out['landmarks']:
                limit=.65 if point['name'] in ('nose','leftEye','rightEye','leftEar','rightEar','neck') else 2.0
                if point['valid'] and np.linalg.norm(np.array(point['xyz_m'])-anchor)>limit:invalidate_point(point,'body_surface_outlier')
            surface=out['torso_surface']
            if surface and surface['valid'] and np.linalg.norm(np.array(surface['xyz_m'])-anchor)>.7:invalidate_point(surface,'torso_surface_outlier')
        p=points()
        for side in ('left','right'):
            for a,b in ((side+'Shoulder',side+'Elbow'),(side+'Elbow',side+'Wrist')):
                if a in p and b in p and not .03<=np.linalg.norm(p[a]-p[b])<=.75:
                    invalidate_point(samples[b],'limb_length_outlier');p.pop(b)
        out['depth_valid']=bool(p)
        torso_r=None;origin=None
        surface=out['torso_surface']
        if all(k in p for k in ('leftShoulder','rightShoulder')):
            origin=(p['leftShoulder']+p['rightShoulder'])/2
            out['torso'].update(position_m=origin.tolist(),position_valid=True)
            if surface and surface['valid']:
                width=np.linalg.norm(p['rightShoulder']-p['leftShoulder'])
                down=np.array(surface['xyz_m'])-origin
                if .15<=width<=.8 and .05<=np.linalg.norm(down)<=.65:
                    torso_r,q=self.orientation('torso',p['rightShoulder']-p['leftShoulder'],down,capture,min_width=.15,min_down=.04)
                    out['torso'].update(quaternion_xyzw=q,orientation_valid=q is not None)
        if torso_r is None:self.history.pop('torso',None)
        def relative(v):return (torso_r.T@(np.array(v)-origin)).tolist() if torso_r is not None and v is not None else None
        for point in out['landmarks']:point['xyz_torso_m']=relative(point['xyz_m'])
        if surface:surface['xyz_torso_m']=relative(surface['xyz_m'])
        head=out['head'];head_r=None
        if 'nose' in p:head.update(position_m=p['nose'].tolist(),position_valid=True,position_torso_m=relative(p['nose']))
        if all(k in p for k in ('leftEye','rightEye','nose')):
            eyes=(p['leftEye']+p['rightEye'])/2
            eye_width=np.linalg.norm(p['rightEye']-p['leftEye'])
            nose_distance=np.linalg.norm(p['nose']-eyes)
            if .025<=eye_width<=.14 and .012<=nose_distance<=.14:
                head_r,q=self.orientation('head',p['rightEye']-p['leftEye'],p['nose']-eyes,capture)
                head.update(quaternion_xyzw=q,orientation_valid=q is not None)
                if head_r is not None and torso_r is not None:head['quaternion_torso_xyzw']=self.continuous_quaternion('head_relative',quaternion_xyzw(torso_r.T@head_r),capture)
        if head_r is None:self.history.pop('head',None)
        if head['quaternion_torso_xyzw'] is None:self.history.pop('head_relative',None)
        head['torso_relative_position_valid']=head['position_torso_m'] is not None
        head['torso_relative_orientation_valid']=head['quaternion_torso_xyzw'] is not None
        for hand in hands:
            hand['palm_position_torso_m']=relative(hand['palm_position_m'])
            hand['torso_relative_position_valid']=hand['palm_position_torso_m'] is not None
            if torso_r is not None and hand['orientation_valid']:
                hand['palm_quaternion_torso_xyzw']=self.continuous_quaternion('hand_relative_'+hand['hand_id'],quaternion_xyzw(torso_r.T@rotation_matrix(hand['palm_quaternion_xyzw'])),capture)
                hand['torso_relative_orientation_valid']=hand['palm_quaternion_torso_xyzw'] is not None
            for point in hand['landmarks']:point['xyz_torso_m']=relative(point['xyz_m'])
        active={'hand_relative_'+h['hand_id'] for h in hands if h['torso_relative_orientation_valid']}
        for key in list(self.history):
            if key.startswith('hand_relative_') and key not in active:self.history.pop(key)
        # Associate against the current raw 2D wrists, not against result order.
        for side in ('left','right'):
            match=association[side]
            wrist_source='vision_body_wrist_surface'
            # Vision's hand request often locates an exposed wrist when the body
            # request has low confidence. Use only its associated, same-frame,
            # measured wrist; keep the original body landmark validity intact.
            wrist_name=side+'Wrist'
            if wrist_name not in p and match is not None:
                wrist=next((x for x in hands[match]['landmarks'] if x['name']=='wrist' and x['valid']),None)
                verified=out['limb_tracking']['arms'].get(side,{}).get('associated_wrist')
                if wrist is None and verified is not None:wrist=verified
                # A hidden elbow does not invalidate an independently measured
                # hand wrist. Retain that endpoint for constrained display IK;
                # full-arm/control validity below still requires a measured elbow.
                if wrist is not None and (side+'Elbow' not in p or .03<=np.linalg.norm(np.array(wrist['xyz_m'])-p[side+'Elbow'])<=.75):
                    p[wrist_name]=np.array(wrist['xyz_m']);wrist_source='associated_vision_hand_wrist_surface'
            names=[side+joint for joint in ('Shoulder','Elbow','Wrist')]
            camera={name:p[name].tolist() if name in p else None for name in names}
            arm={'positions_camera_m':camera,'positions_torso_m':{k:relative(v) for k,v in camera.items()},
                 'measurement_valid':all(n in p for n in names),'hand_id':None,'elbow_flexion_rad':None,
                 'upper_arm_length_m':None,'forearm_length_m':None,
                 'upper_arm_direction_camera':None,'forearm_direction_camera':None,
                 'upper_arm_direction_torso':None,'forearm_direction_torso':None,
                 'torso_relative_valid':all(n in p for n in names) and torso_r is not None,
                 'axial_twist_observable':False,'wrist_source':wrist_source,'association_method':'current_2d_wrists_with_chirality_hint'}
            # Track each link independently: a missing wrist must not erase a
            # measured bicep direction, nor vice versa for a visible forearm.
            for segment,a,b in [('upper_arm',names[0],names[1]),('forearm',names[1],names[2])]:
                valid=a in p and b in p
                arm[segment+'_measurement_valid']=valid
                arm[segment+'_torso_relative_valid']=valid and torso_r is not None
                if valid:
                    v=p[b]-p[a];length=np.linalg.norm(v)
                    if length>1e-8:
                        arm[segment+'_length_m']=float(length)
                        arm[segment+'_direction_camera']=(v/length).tolist()
                        arm[segment+'_direction_torso']=(torso_r.T@(v/length)).tolist() if torso_r is not None else None
            if arm['measurement_valid']:
                upper=p[names[0]]-p[names[1]];lower=p[names[2]]-p[names[1]]
                arm.update(upper_arm_length_m=float(np.linalg.norm(upper)),forearm_length_m=float(np.linalg.norm(lower)),
                           elbow_flexion_rad=float(math.pi-math.acos(np.clip(np.dot(upper,lower)/(np.linalg.norm(upper)*np.linalg.norm(lower)),-1,1))))
                upper_direction=-upper/np.linalg.norm(upper);lower_direction=lower/np.linalg.norm(lower)
                arm.update(upper_arm_direction_camera=upper_direction.tolist(),forearm_direction_camera=lower_direction.tolist(),
                           upper_arm_direction_torso=(torso_r.T@upper_direction).tolist() if torso_r is not None else None,
                           forearm_direction_torso=(torso_r.T@lower_direction).tolist() if torso_r is not None else None)
            if match is not None:
                arm['hand_id']=hands[match]['hand_id'];hands[match]['body_arm_side']=side
            out['arms'][side]=arm
        out['reason']='measured_visible_body_surfaces' if out['depth_valid'] else 'no_valid_body_surface_samples'
        return out
