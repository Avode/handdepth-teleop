"""Measured torso-aligned human skeleton drawn in MuJoCo's visual-only scene."""
import numpy as np
import mujoco
import time
from .body import BODY_EDGES, rotation_matrix
from .retarget import TORSO_TO_ROBOT, vector
from .protocol import BODY_JOINTS
from .display_tracking import MetricSkeleton

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
              (13,17),(0,17),(17,18),(18,19),(19,20)]


class PoseOverlay:
    """No bodies, joints, contacts, or simulation qpos are added/modified."""
    def __init__(self, anchor):
        self.origin = anchor.position.copy()
        self.rotation = anchor.rotation.copy()
        self.basis = self.rotation @ TORSO_TO_ROBOT
        self.enabled = True
        self.status = 'waiting for measured torso'
        self.skeleton = MetricSkeleton('torso')

    def reset(self):
        self.skeleton.reset()
        self.status = 'display history reset; waiting for new measurements'

    def world(self, point):
        p = vector(point)
        return None if p is None else self.origin + self.basis @ p

    @staticmethod
    def sphere(scene, point, radius, color, label=''):
        if point is None or scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3,radius),
                           point, np.eye(3).ravel(), np.array(color,dtype=np.float32))
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.label = label
        scene.ngeom += 1

    @staticmethod
    def line(scene, start, end, radius, color):
        if start is None or end is None or np.linalg.norm(end-start)<1e-7 or scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.ones(3),
                           np.zeros(3), np.eye(3).ravel(), np.array(color,dtype=np.float32))
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,start,end)
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1

    def axes(self, scene, origin, rotation, length=.07):
        if origin is None:
            return
        for i,color in enumerate([[1,.15,.15,.9],[.15,1,.15,.9],[.2,.4,1,.9]]):
            self.line(scene,origin,origin+length*rotation[:,i],.0025,color)

    def draw(self, scene, pose, fresh=True, targets=None, clear=True, now=None, shoulder_targets=None, forearm_targets=None):
        now = time.monotonic() if now is None else now
        tracked = self.skeleton.update(pose,fresh=fresh,now=now)
        if clear:
            scene.ngeom = 0
        if not self.enabled:
            self.status = 'hidden (H to show)'
            return
        # Fixed marker is both origins. Amber means its human reference is held.
        amber = [1.,.65,.08,.95]
        self.sphere(scene,self.origin,.018,[1,.85,.1,1] if self.skeleton.torso_current else amber)
        self.axes(scene,self.origin,self.rotation,.12)
        body = (pose or {}).get('body') or {}
        info = self.skeleton.summary(now)
        self.status = (f"{info['observed']} measured / {info['held_or_inferred']} held or inferred; "
                       f"oldest {info['oldest_age_s']:.1f}s; torso "
                       + ('current' if info['torso_current'] else 'HELD'))
        cyan = [.1,.9,1.,.9]
        for name,p in tracked.items():
            self.sphere(scene,self.world(p.xyz),.008,cyan if p.state=='observed' else amber)
        for a,b in BODY_EDGES:
            p,q = tracked.get(BODY_JOINTS[a]),tracked.get(BODY_JOINTS[b])
            if p is not None and q is not None:
                color = cyan if p.state==q.state=='observed' else amber
                self.line(scene,self.world(p.xyz),self.world(q.xyz),.004,color)
        for side,arm in self.skeleton.arms.items():
            if arm['clamped'] and arm['target'] is not None:
                wrist = self.world(arm['shape'][2].xyz);target = self.world(arm['target'])
                self.sphere(scene,wrist,.013,amber,side[0].upper()+' CLAMPED')
                self.sphere(scene,target,.007,[1.,.15,.1,1])
                self.line(scene,wrist,target,.0015,[1.,.25,.1,.7])
        # Hands/orientation axes remain strictly current measurements. Persistent
        # body geometry above is display-only and never written into pose.
        if not fresh or not body.get('torso',{}).get('orientation_valid'):
            return
        head=body.get('head',{})
        if head.get('torso_relative_orientation_valid'):
            self.axes(scene,self.world(head.get('position_torso_m')),
                      self.basis@rotation_matrix(head['quaternion_torso_xyzw']))
        for hand in (pose or {}).get('hands',[]):
            color=[.8,.3,1.,.95] if hand.get('body_arm_side')=='left' else [1.,.5,.15,.95]
            points=[self.world(p.get('xyz_torso_m')) if p.get('valid') else None for p in hand.get('landmarks',[])]
            for p in points:self.sphere(scene,p,.0045,color)
            for a,b in HAND_EDGES:
                if max(a,b)<len(points):self.line(scene,points[a],points[b],.0025,color)
            if hand.get('torso_relative_orientation_valid'):
                self.axes(scene,self.world(hand.get('palm_position_torso_m')),
                          self.basis@rotation_matrix(hand['palm_quaternion_torso_xyzw']),.055)
        # Actual commanded, smoothed robot task points are a separate green layer.
        for name,target in (targets or {}).items():
            if 'wrist' in name or 'elbow' in name:
                self.sphere(scene,target.position,.012,[.2,1.,.25,.8])
        for side,(shoulder,elbow) in (shoulder_targets or {}).items():
            self.line(scene,shoulder,elbow,.006,[.2,1.,.25,.9])
            self.sphere(scene,elbow,.014,[.2,1.,.25,.9],side[0].upper()+' SHOULDER TARGET')
        for side,(elbow,wrist) in (forearm_targets or {}).items():
            self.line(scene,elbow,wrist,.006,[.2,1.,.25,.9])
            self.sphere(scene,wrist,.012,[.2,1.,.25,.9],side[0].upper()+' FOREARM TARGET')
