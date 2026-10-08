"""Two observable shoulder angles, represented by a unit upper-arm direction."""
import numpy as np
import mujoco
from mink.tasks.task import Task


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if v.shape != (3,) or not np.isfinite(v).all() or length < 1e-9:
        raise ValueError('invalid upper-arm direction')
    return v/length


def angle(a, b):
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.clip(a @ b, -1., 1.)))


def approach_direction(previous, desired, dt, tau=.08, speed=1.4):
    """Geodesic low pass/rate limit; finite even for opposite directions."""
    a, b = unit(previous), unit(desired)
    theta = angle(a, b)
    if theta < 1e-9:
        return b
    tangent = b-(a @ b)*a
    if np.linalg.norm(tangent) < 1e-8:
        axis = np.eye(3)[np.argmin(np.abs(a))]
        tangent = axis-(axis @ a)*a
    tangent = unit(tangent)
    step = min(theta*(1-np.exp(-dt/tau)), speed*dt)
    return unit(np.cos(step)*a+np.sin(step)*tangent)


class UpperArmDirectionTask(Task):
    """Match direction, independent of translation and limb length.

    A vector has two angular DOFs. Shoulder-only control freezes axial twist;
    limb control observes its bend plane separately using the forearm task.
    """
    def __init__(self, model, prefix):
        super().__init__(cost=np.ones(3), gain=.7, lm_damping=1e-3)
        self.shoulder = model.site(prefix+'_shoulder').id
        self.elbow = model.site(prefix+'_elbow').id
        self.start,self.end = self.shoulder,self.elbow
        self.target = None

    def direction(self, configuration):
        d = configuration.data
        return unit(d.site_xpos[self.end]-d.site_xpos[self.start])

    def set_target(self, direction):
        self.target = unit(direction)

    def compute_error(self, configuration):
        if self.target is None:
            raise ValueError('upper-arm target is unset')
        return self.direction(configuration)-self.target

    def compute_jacobian(self, configuration):
        m, d = configuration.model, configuration.data
        upper = d.site_xpos[self.end]-d.site_xpos[self.start]
        u = unit(upper)
        js, je = np.zeros((3,m.nv)), np.zeros((3,m.nv))
        mujoco.mj_jacSite(m,d,js,None,self.start)
        mujoco.mj_jacSite(m,d,je,None,self.end)
        return (np.eye(3)-np.outer(u,u)) @ (je-js)/np.linalg.norm(upper)


class ForearmDirectionTask(UpperArmDirectionTask):
    """Elbow-to-wrist unit direction, using the G2's real sites and offsets.

    Together with the upper-arm task this observes elbow bend and bend-plane
    rotation. Wrist axial rotation remains unobserved and is held by control.
    """
    def __init__(self, model, prefix):
        super().__init__(model,prefix)
        self.wrist = model.site(prefix+'_wrist').id
        self.start,self.end = self.elbow,self.wrist
