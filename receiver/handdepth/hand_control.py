"""Current hand gesture evidence and forearm-relative orientation geometry."""
import numpy as np


def forearm_frame(upper, forearm):
    """Right-handed bend frame: x = bend normal, z = elbow-to-wrist.

    Nearly straight/folded limbs do not observe the bend plane. Do not substitute
    an inferred display elbow or an arbitrary camera axis in that case.
    """
    z=np.asarray(forearm,dtype=float)
    x=np.cross(upper,z);length=np.linalg.norm(x)
    if length<.12:return None
    x/=length
    return np.column_stack((x,np.cross(z,x),z))


def finger_curl(hand, color_dimensions):
    """Scale/rotation invariant RGB gesture, separate from measured metric XYZ.

    A folded finger's MCP-to-tip chord is shorter than its articulated path.
    Three confident fingers are required. Tiny/edge-on or incomplete detections
    remain unknown. Thumb/index pinch alone is not a fist.
    """
    points={}
    size=np.asarray(color_dimensions,dtype=float)
    for p in hand.get('landmarks',[]):
        xy=np.array([p.get('x'),p.get('y')],dtype=float)
        if p.get('confidence',0)>=.5 and np.isfinite(xy).all():
            points[p['name']]=xy*size
    out={'valid':False,'source':'current_vision_rgb_finger_curl','space':'color_pixels',
         'score':None,'fingers':{},'reason':'insufficient_current_finger_landmarks'}
    if not all(n in points for n in ('wrist','indexMCP','littleMCP')):return out
    span=np.linalg.norm(points['indexMCP']-points['littleMCP'])
    palm=np.linalg.norm((points['indexMCP']+points['littleMCP'])/2-points['wrist'])
    if min(span,palm)<max(4.,.008*min(size)):
        out['reason']='palm_too_small_or_edge_on';return out
    for finger in ('index','middle','ring','little'):
        names=[finger+n for n in ('MCP','PIP','DIP','Tip')]
        if not all(n in points for n in names):continue
        p=np.array([points[n] for n in names]);segments=np.linalg.norm(np.diff(p,axis=0),axis=1)
        path=float(sum(segments))
        if not max(6.,.35*span)<path<5*max(span,palm) or np.min(segments)<.5:continue
        ratio=np.linalg.norm(p[-1]-p[0])/path
        out['fingers'][finger]=float(np.clip((1-ratio)/.65,0,1))
    if len(out['fingers'])>=3:
        out.update(valid=True,score=float(np.mean(list(out['fingers'].values()))),reason='current_rgb_finger_shape')
    return out


class FistLatch:
    """Hysteresis plus two distinct source frames; missing input cancels intent."""
    close_threshold=.65
    open_threshold=.30

    def __init__(self):
        self.closed=None
        self.identity=None
        self.frame=-1
        self.capture=None
        self.invalidate()

    def invalidate(self):
        self.confirmed=False;self.candidate=None;self.count=0;self.first_capture=None

    def update(self, gesture, frame, capture, identity):
        valid=(gesture and gesture.get('valid') and gesture.get('source')=='current_vision_rgb_finger_curl'
               and gesture.get('source_frame_id')==frame and gesture.get('capture_time_s')==capture)
        if not valid:
            self.invalidate();return None
        if identity!=self.identity or self.capture is None or not 0<=capture-self.capture<=.5:
            self.invalidate();self.frame=-1
        self.identity=identity
        if frame<self.frame:
            self.invalidate();return None
        if frame!=self.frame:
            self.frame=frame;self.capture=capture
            values=np.array(list(gesture.get('fingers',{}).values()),dtype=float)
            if len(values)<3 or not np.isfinite(values).all():
                self.invalidate();return None
            close=np.count_nonzero(values>=self.close_threshold)>=3 and not np.any(values<self.open_threshold)
            opened=np.count_nonzero(values<=self.open_threshold)>=3
            desired=True if close else False if opened else None
            if desired is None:
                self.candidate=None;self.count=0;self.first_capture=None
            elif desired==self.candidate:
                self.count+=1
            else:
                self.candidate=desired;self.count=1;self.first_capture=capture
            if desired is not None and self.count>=2 and capture-self.first_capture>=.08:
                self.closed=desired;self.confirmed=True
        if not self.confirmed:return None
        return -.85 if self.closed else -.05
