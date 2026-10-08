import copy
import numpy as np
import pytest
from handdepth.retarget import FrameTarget,TORSO_TO_ROBOT
from handdepth.geometry import PoseEstimator
from generate_fixtures import body_header


def test_overlay_coincides_at_torso_origin_and_never_changes_model_state(header):
    mj=pytest.importorskip('mujoco')
    from handdepth.pose_overlay import PoseOverlay
    m=mj.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    d=mj.MjData(m);scene=mj.MjvScene(m,maxgeom=500);q=d.qpos.copy()
    p,_=PoseEstimator().estimate(body_header(header),np.full((24,32),500,dtype='<u2'))
    origin=np.array([.1,0,1.3]);overlay=PoseOverlay(FrameTarget(origin,np.eye(3)))
    assert np.allclose(overlay.world([0,0,0]),origin)
    assert np.allclose(overlay.world([.1,.2,.3]),origin+TORSO_TO_ROBOT@[.1,.2,.3])
    overlay.draw(scene,p)
    assert scene.ngeom>80
    assert np.allclose(scene.geoms[0].pos,origin)
    assert np.isfinite(np.array([g.pos for g in scene.geoms[:scene.ngeom]])).all()
    assert np.array_equal(d.qpos,q) and m.nbody==2
    previous=np.array([g.pos.copy() for g in scene.geoms[:scene.ngeom]])
    # Camera-relative person translation must not move the torso-aligned overlay.
    moved=copy.deepcopy(p);moved['body']['torso']['position_m']=[2,3,4]
    overlay.draw(scene,moved)
    assert np.allclose(previous,np.array([g.pos for g in scene.geoms[:scene.ngeom]]))
    overlay.draw(scene,p,fresh=False)
    assert scene.ngeom>4  # known body remains visible, explicitly amber/held
    assert all(p.state=='held' for p in overlay.skeleton.points.values())
    overlay.enabled=False;overlay.draw(scene,p)
    assert scene.ngeom==0
