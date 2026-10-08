"""Lightweight CPU/OpenCV dashboard: no learned model, CUDA or plotting server."""
import time
from collections import deque, OrderedDict
import cv2
import numpy as np
from .protocol import sensor_hands, BODY_JOINTS
from .body import BODY_EDGES, rotation_matrix
from .display_tracking import ImageSkeleton, MetricSkeleton
from .geometry import Calibration, CalibrationError

EDGES = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]


def fit(image, w=480, h=360):
    out = np.zeros((h,w,3), np.uint8)
    if image is None:
        return out
    ih, iw = image.shape[:2]
    scale = min(w/iw, h/ih)
    resized = cv2.resize(image, (round(iw*scale), round(ih*scale)))
    rh, rw = resized.shape[:2]
    out[(h-rh)//2:(h-rh)//2+rh, (w-rw)//2:(w-rw)//2+rw] = resized
    return out

def hand_color(hand):
    return (255,210,70) if hand["chirality"] == "left" else ((70,240,90) if hand["chirality"] == "right" else (60,170,255))

def overlay(image, landmarks, color=(70,240,90), label="", edges=EDGES):
    out = image.copy()
    h, w = out.shape[:2]
    points = [(int(p["x"]*w), int(p["y"]*h)) if p["x"] is not None and p["confidence"] >= .4 else None for p in landmarks]
    for a,b in edges:
        if a < len(points) and b < len(points) and points[a] is not None and points[b] is not None:
            cv2.line(out, points[a], points[b], color, 2)
    for p in points:
        if p is not None:
            cv2.circle(out, p, 3, (0,180,255), -1)
    if points and points[0] is not None and label:
        cv2.putText(out,label,(points[0][0]+5,points[0][1]+16),0,.42,color,1,cv2.LINE_AA)
    return out

def dashed_line(image, start, end, color, width=2):
    h,w = image.shape[:2]
    visible,a,b = cv2.clipLine((0,0,w,h),tuple(np.asarray(start,dtype=int)),tuple(np.asarray(end,dtype=int)))
    if not visible:return
    x,y=np.array(a),np.array(b);length=np.linalg.norm(y-x)
    for offset in np.arange(0,length,10):
        p=x+(y-x)*offset/max(length,1e-9);q=x+(y-x)*min(offset+6,length)/max(length,1e-9)
        cv2.line(image,tuple(p.astype(int)),tuple(q.astype(int)),color,width)


def tracking_overlay(image, tracking, capture_time=None, held_all=False):
    """Same constraint points as iOS; amber dashes explicitly mean held/inferred."""
    out = image.copy()
    h,w = out.shape[:2]
    points = {p['name']:p for p in tracking['points']}
    capture_time = tracking.get('capture_time_s',0.) if capture_time is None else capture_time
    def xy(p):return np.array([p['x']*w,p['y']*h])
    for a,b in BODY_EDGES:
        p,q = points.get(BODY_JOINTS[a]),points.get(BODY_JOINTS[b])
        if p is None or q is None:continue
        x,y = xy(p),xy(q)
        held = held_all or not p['measurement_valid'] or not q['measurement_valid']
        verified=p.get('state')=='observed_rgbd' or q.get('state')=='observed_rgbd'
        color = (0,180,255) if held else ((255,220,40) if verified else (220,80,230))
        if held:
            dashed_line(out,x,y,color)
        else:cv2.line(out,tuple(x.astype(int)),tuple(y.astype(int)),color,2)
    for p in points.values():
        held = held_all or not p['measurement_valid']
        color=(0,180,255) if held else ((255,220,40) if p.get('state')=='observed_rgbd' else (220,80,230))
        cv2.circle(out,tuple(xy(p).astype(int)),3,color,-1)
        if held and p['name'].endswith(('Elbow','Wrist')):
            label = ('held' if held_all else p['state'])+f" {max(0.,capture_time-p['source_time_s']):.1f}s"
            if p.get('source')=='ubuntu_metric_arm_projection':label+=' 3D'
            pos=np.clip(xy(p)+[5,-6],[0,12],[max(0,w-115),h-2]).astype(int)
            cv2.putText(out,label,tuple(pos),0,.38,(0,180,255),1,cv2.LINE_AA)
    for side,arm in tracking.get('arms',{}).items():
        if arm.get('wrist_clamped') and side+'Wrist' in points:
            wrist=xy(points[side+'Wrist']);target=np.array(arm['target_wrist_xy'])*[w,h]
            dashed_line(out,wrist,target,(0,80,255),1)
            cv2.drawMarker(out,tuple(target.astype(int)),(0,80,255),cv2.MARKER_TILTED_CROSS,10,1)
            pos=np.clip(wrist+[5,14],[0,12],[max(0,w-145),h-2]).astype(int)
            cv2.putText(out,side+' wrist CLAMPED',tuple(pos),0,.38,(0,180,255),1)
    return out

def depth_image(depth):
    if depth is None:
        return None
    # Fixed 0.15-1.5 m range helps compare frames; zero remains black.
    gray = (255*(1-np.clip((depth.astype(float)/1000-.15)/1.35,0,1))).astype(np.uint8)
    out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    out[depth == 0] = 0
    return out

def jpeg_dimensions(data):
    """Inspect JPEG SOF before decoder allocation, including progressive JPEG."""
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("invalid JPEG")
    i = 2
    while i < len(data):
        if data[i] != 255:
            raise ValueError("invalid JPEG marker")
        while i < len(data) and data[i] == 255:
            i += 1
        if i >= len(data): break
        marker = data[i]; i += 1
        if marker in (0xD8,0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if i+2 > len(data): break
        n = int.from_bytes(data[i:i+2], "big")
        if n < 2 or i+n > len(data): break
        if marker in (0xC0,0xC1,0xC2):
            if n < 8: break
            h = int.from_bytes(data[i+3:i+5], "big")
            w = int.from_bytes(data[i+5:i+7], "big")
            if not 1 <= w <= 2048 or not 1 <= h <= 2048:
                raise ValueError("JPEG dimensions exceed limits")
            return w,h
        if marker == 0xDA: break
        i += n
    raise ValueError("missing JPEG dimensions")

class Dashboard:
    def __init__(self):
        self.rgb = None
        self.preview_header = None
        self.depth = None
        self.sensor_header = None
        self.pose = None
        self.received_s = 0
        self.preview_received_s = 0
        self.count = 0
        self.first_s = time.monotonic()
        self.source_times=deque(maxlen=60)
        self.source_session=None
        self.status = "Waiting for iPhone"
        self.image_skeleton = ImageSkeleton()
        self.metric_skeleton = MetricSkeleton('camera')
        self.projection = None
        self.projection_id = None
        self.sensor_frames = OrderedDict()
        self.preview_frames = OrderedDict()
        self.matched = None
        self.rgb_status = 'waiting for matched RGB/sensor frame'
        self.rgb_overlay_matched=False

    def reset_display(self):
        self.image_skeleton.reset();self.metric_skeleton.reset()
        self.sensor_frames.clear();self.preview_frames.clear();self.matched=None

    @staticmethod
    def _remember(cache, key, value):
        cache[key]=value
        while len(cache)>8:cache.popitem(last=False)

    def _match(self):
        common=self.sensor_frames.keys() & self.preview_frames.keys()
        common={k for k in common if all(self.sensor_frames[k][0][field]==self.preview_frames[k][2][field]
                    for field in ('calibration_id','capture_time_s','color_dimensions'))}
        if common:
            key=max(common,key=lambda k:self.sensor_frames[k][0]['capture_time_s'])
            header,tracking,arrival=self.sensor_frames[key]
            rgb,preview_arrival,_=self.preview_frames[key]
            if self.matched is None or (key[0]!=self.matched[0]['session_id'] or header['frame_id']>=self.matched[0]['frame_id']):
                self.matched=(header,tracking,rgb,arrival,preview_arrival)

    def sensor(self, header, depth, pose, now):
        identity=self.image_skeleton.source.identity;generation=self.image_skeleton.generation
        display_pose=dict(pose,received_monotonic_s=now)
        geometry=(pose['session_id'],pose.get('display_geometry_id',pose.get('calibration_id')),
                  tuple(header['color_dimensions']),tuple(header['depth_dimensions']))
        if geometry!=self.projection_id:
            self.projection=None;self.projection_id=geometry
        if self.projection is None and header.get('calibration') is not None:
            try:self.projection=Calibration(header['calibration'],header['color_dimensions'],header['depth_dimensions'])
            except (CalibrationError,np.linalg.LinAlgError):pass
        self.metric_skeleton.update(display_pose,now=now)
        tracking=self.image_skeleton.update(display_pose,header.get('body'),now,self.metric_skeleton,self.projection)
        if identity!=self.image_skeleton.source.identity or generation!=self.image_skeleton.generation:
            self.sensor_frames.clear();self.matched=None
            self.preview_frames=OrderedDict((k,v) for k,v in self.preview_frames.items()
                                           if k==(header['session_id'],header['frame_id']))
        self._remember(self.sensor_frames,(header['session_id'],header['frame_id']),(header,tracking,now))
        self._match()
        if self.source_session!=header['session_id']:
            self.source_times.clear();self.source_session=header['session_id']
        self.source_times.append(header['capture_time_s'])
        self.sensor_header = header; self.depth = depth; self.pose = pose
        self.received_s = now; self.count += 1
        self.status = "Live"

    @property
    def source_fps(self):
        return (len(self.source_times)-1)/max(self.source_times[-1]-self.source_times[0],.001) if len(self.source_times)>1 else 0

    def preview(self, header, payload, now):
        if jpeg_dimensions(payload) != tuple(header["color_dimensions"]):
            raise ValueError("JPEG dimensions do not match bounded header")
        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2][::-1] != tuple(header["color_dimensions"]):
            raise ValueError("JPEG dimensions do not match header")
        self.rgb = img; self.preview_header = header; self.preview_received_s = now
        self._remember(self.preview_frames,(header['session_id'],header['frame_id']),(img,now,header))
        self._match()

    def render(self, now=None):
        now = time.monotonic() if now is None else now
        fresh = now-self.received_s <= .5 and self.status == "Live"
        rgb = self.rgb if now-self.preview_received_s <= .5 else None
        # Keep the latest EXACT pair instead of flashing raw RGB between arrivals
        # on the independent sockets. A stale pair is frozen and labelled held.
        # Independently paced phone channels can permanently select alternating
        # camera frames. Keep video live without painting another frame's bones.
        use_raw=(rgb is not None and self.preview_header is not None and
                 (self.matched is None or (now-self.matched[3]>.5 and
                  (self.preview_header['session_id']!=self.matched[0]['session_id'] or
                   self.preview_header['frame_id']>self.matched[0]['frame_id']))))
        self.rgb_overlay_matched=self.matched is not None and not use_raw
        if use_raw:
            rgb=rgb.copy()
            self.rgb_status=f"unpaired RGB frame {self.preview_header['frame_id']} | no skeleton overlay"
            cv2.putText(rgb,self.rgb_status,(10,rgb.shape[0]-12),0,.43,(0,180,255),1)
        elif self.matched is not None:
            header,tracking,frame,arrival,preview_arrival=self.matched
            age=max(0.,now-arrival);pair_fresh=fresh and age<=.5
            rgb=tracking_overlay(frame,tracking,header['capture_time_s']+age,held_all=not pair_fresh)
            if pair_fresh:
                for hand in sensor_hands(header):
                    rgb=overlay(rgb,hand['landmarks'],hand_color(hand),hand['chirality']+' '+hand['hand_id'])
            self.rgb_status=f"{'matched' if pair_fresh else 'HELD matched'} RGB frame {header['frame_id']} / age {age:.1f}s"
            cv2.putText(rgb,self.rgb_status,(10,rgb.shape[0]-12),0,.48,(0,180,255) if not pair_fresh else (230,230,230),1)
        else:self.rgb_status='waiting for matched RGB/sensor frame'
        self.metric_skeleton.update(self.pose,fresh=fresh,now=now)
        three = self._three_d(fresh,now)
        view = np.hstack([fit(rgb), fit(depth_image(self.depth)), three])
        strip = np.zeros((255,1440,3), np.uint8)
        p = self.pose if fresh else None
        def fmt(value):
            return "invalid" if value is None else str(np.round(value, 4))
        latency = "clock alignment unavailable"
        if p and p.get("capture_to_receive_s") is not None:
            latency = f"capture-to-receive {p['capture_to_receive_s']*1000:.1f} +/- {p['clock_uncertainty_s']*1000:.1f} ms"
        hands = p.get("hands", []) if p else []
        rows = [f"HandDepth | {self.status if fresh else 'No current measurement'} | {len(hands)} hands | sensor source FPS {self.source_fps if fresh else 0:.1f}"]
        if p and p.get('pc_pose'):
            rows[0]+=f" | Ubuntu RGB pose {p['pc_pose']['inference_ms']:.0f} ms + TrueDepth fusion"
        elif p and p.get('processing_location')=='phone_rgbd_fallback_unpaired_or_slow_pc':
            rows[0]+=' | current phone RGB-D; waiting for matching PC RGB'
        for i in range(2):
            if i < len(hands):
                hand = hands[i]
                rows.append(f"{hand['chirality']} {hand['hand_id']} | XYZ m: {fmt(hand['palm_position_m'])} | Pinch m: {fmt(hand['pinch_distance_m'])} | Depth {hand['depth_valid']} | Orientation {hand['orientation_valid']}")
            else:
                rows.append("No additional hand detected")
        body = p.get("body") if p else None
        torso = body["torso"] if body else None
        head = body["head"] if body else None
        rows += [f"Torso XYZ m: {fmt(torso['position_m'] if torso else None)} | torso frame {bool(torso and torso['orientation_valid'])} | Head XYZ torso m: {fmt(head['position_torso_m'] if head else None)} | head frame {bool(head and head['orientation_valid'])}"]
        arms = body.get("arms",{}) if body else {}
        rows += [f"Elbow flexion rad: L {fmt(arms.get('left',{}).get('elbow_flexion_rad'))} / R {fmt(arms.get('right',{}).get('elbow_flexion_rad'))} | Shoulder/elbow/wrist + head + both hands; visible surfaces"]
        tracking = p.get("body_tracking") if p else None
        inferred = sum(a['elbow_inferred'] for a in tracking['arms'].values()) if tracking else 0
        recovered=len((body or {}).get('limb_tracking',{}).get('points',[]))
        rows[-1] += f" | Phone inferred elbows: {inferred} | Ubuntu RGB-D verified joints: {recovered} (cyan)"
        rows += [f"Reception age {(now-self.received_s)*1000:.0f} ms | {latency}",
                 "Cyan = RGB-D verified | Amber = held/inferred display ONLY | 3D label = projected metric arm | R resets display"]
        for i,row in enumerate(rows):
            cv2.putText(strip, row, (12,28+34*i), cv2.FONT_HERSHEY_SIMPLEX, .62, (220,230,235), 1, cv2.LINE_AA)
        return np.vstack([view,strip])

    def _three_d(self, fresh, now=None):
        out = np.zeros((360,480,3), np.uint8)
        # Center/fit only the display camera, never the public metric coordinates.
        # A fixed pixel origin clipped real hands near the sensor's image edges.
        pose = self.pose if fresh and self.pose else None
        hands = pose.get("hands",[]) if pose else []
        vertices = np.array([l["xyz_m"] for h in hands for l in h["landmarks"] if l["valid"]], dtype=float) if hands else np.empty((0,3))
        body = pose.get("body") if pose else None
        tracked = self.metric_skeleton.points
        body_points = list(tracked.values())
        if body_points:
            vertices = np.concatenate([vertices.reshape(-1,3),np.array([p.xyz for p in body_points])])
        center = np.median(vertices, axis=0) if len(vertices) else np.array([0.,0.,.5])
        basis = np.array([[1.,0.,.45], [0.,1.,-.35]])
        projected = (vertices-center) @ basis.T if len(vertices) else np.empty((0,2))
        extent = np.maximum(np.max(np.abs(projected),axis=0),.08) if len(vertices) else np.array([.15,.15])
        scale = min(210/extent[0], 110/extent[1], 1600)
        def project(v):
            p = basis @ (np.asarray(v)-center)
            return int(240+scale*p[0]), int(180+scale*p[1])
        # Fixed camera-frame orientation legend, independent of hand translation.
        for axis, color, label in [(np.array([.1,0,0]), (50,50,255), "X"), (np.array([0,.1,0]), (50,255,50), "Y"), (np.array([0,0,.1]), (255,100,50), "Z")]:
            a = (380,285)
            offset = basis @ axis*600
            b = (int(a[0]+offset[0]),int(a[1]+offset[1]))
            cv2.arrowedLine(out,a,b,color,2); cv2.putText(out,label,b,0,.5,color,1)
        cv2.putText(out,"Body/hands; camera XYZ unchanged",(10,345),0,.45,(170,180,190),1)
        def draw_frame(position, q, length):
            if position is None or q is None:return
            for axis,color in zip(rotation_matrix(q).T,[(50,50,255),(50,255,50),(255,100,50)]):
                cv2.arrowedLine(out,project(position),project(np.array(position)+length*axis),color,2)
        for a,b in BODY_EDGES:
            p,q = tracked.get(BODY_JOINTS[a]),tracked.get(BODY_JOINTS[b])
            if p is not None and q is not None:
                if p.state==q.state=='observed':cv2.line(out,project(p.xyz),project(q.xyz),(220,80,230),2)
                else:dashed_line(out,project(p.xyz),project(q.xyz),(0,180,255))
        for p in body_points:cv2.circle(out,project(p.xyz),3,(220,80,230) if p.state=='observed' else (0,180,255),-1)
        info=self.metric_skeleton.summary(now)
        cv2.putText(out,f"Held/inferred {info['held_or_inferred']} | oldest {info['oldest_age_s']:.1f}s",(10,20),0,.45,(0,180,255),1)
        for i,side in enumerate(('left','right')):
            cv2.putText(out,side[0].upper()+' '+self.metric_skeleton.arm_text(side,now),(10,38+16*i),0,.30,(0,180,255),1)
        for side,arm in self.metric_skeleton.arms.items():
            if arm['clamped'] and arm['target'] is not None:
                cv2.drawMarker(out,project(arm['target']),(0,70,255),cv2.MARKER_TILTED_CROSS,8,1)
                cv2.putText(out,side+' CLAMPED',project(arm['shape'][2].xyz),0,.4,(0,180,255),1)
        if body:
            draw_frame(body["torso"]["position_m"],body["torso"]["quaternion_xyzw"],.1)
            draw_frame(body["head"]["position_m"],body["head"]["quaternion_xyzw"],.05)
        for hand in hands:
            landmarks = hand["landmarks"]
            color = hand_color(hand)
            points = [project(l["xyz_m"]) if l["valid"] else None for l in landmarks]
            for a,b in EDGES:
                if a < len(points) and b < len(points) and points[a] is not None and points[b] is not None:
                    cv2.line(out,points[a],points[b],color,2)
            for p in points:
                if p is not None: cv2.circle(out,p,3,(0,200,255),-1)
            palm = hand["palm_position_m"]
            if palm is not None:
                cv2.circle(out,project(palm),7,color,2)
                cv2.putText(out,hand["chirality"]+" "+hand["hand_id"],project(palm),0,.4,color,1)
                q = hand.get("palm_quaternion_xyzw")
                if q is not None:
                    x,y,z,w = q
                    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                    for axis,color in zip(rotation.T,[(50,50,255),(50,255,50),(255,100,50)]):
                        cv2.arrowedLine(out,project(palm),project(np.array(palm)+.05*axis),color,2)
        return out
