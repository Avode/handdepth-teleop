import copy
import numpy as np
import pytest
from handdepth.geometry import Calibration,PoseEstimator
from handdepth.display_tracking import ImageSkeleton,MetricSkeleton
from handdepth.retarget import measured_forearm
from generate_fixtures import sensor_header,body_header
from test_display_tracking import measured_pose,point,next_frame,establish,hide,assert_lengths


def calibration():
    h=sensor_header(size=(640,480),depth_size=(320,240))
    return Calibration(h['calibration'],h['color_dimensions'],h['depth_dimensions'])


def phone_layer(p,c):
    p['body_tracking']={'points':[],'arms':{}}
    for name in ('rightShoulder','rightElbow','rightWrist','leftShoulder','leftElbow','leftWrist'):
        xyz=point(p,name)['xyz_m']
        xy=c.project_color(xyz) if xyz is not None else np.array([.5,.5])
        p['body_tracking']['points'].append(dict(name=name,x=float(xy[0]),y=float(xy[1]),
            confidence=.9,state='observed_body' if xyz is not None else 'inferred',measurement_valid=xyz is not None,
            source_frame_id=p['frame_id'] if xyz is not None else p['frame_id']-1,
            source_time_s=p['capture_time_s'] if xyz is not None else p['capture_time_s']-.1))
    return p


@pytest.mark.parametrize('side',['left','right'])
def test_rgb_uses_same_metric_solution_and_measured_hand_when_elbow_is_hidden(side):
    p=measured_pose();c=calibration();metric=MetricSkeleton('camera');establish(metric,p)
    p=hide(next_frame(p),side+'Elbow',side+'Wrist')
    wrist=np.array([.28 if side=='right' else -.28,.3,1.23])
    p['body']['arms'][side]={'hand_id':'observed-hand'}
    p['hands']=[dict(hand_id='observed-hand',body_arm_side=side,chirality=side,
        palm_position_m=None,orientation_valid=False,landmarks=[dict(name='wrist',valid=True,confidence=.9,xyz_m=wrist.tolist(),xyz_torso_m=(wrist-[0,0,1]).tolist())])]
    phone_layer(p,c);before=copy.deepcopy(p)
    metric.update(p);assert metric.arms[side]['mode']=='inferred';assert_lengths(metric,side)
    image=ImageSkeleton();tracking=image.update(p,None,p['received_monotonic_s'],metric,c)
    values={v['name']:v for v in tracking['points']}
    for joint in ('Shoulder','Elbow','Wrist'):
        n=side+joint;v=values[n];expected=c.project_color(metric.points[n].xyz)
        assert np.allclose([v['x'],v['y']],expected)
        assert v['source']=='ubuntu_metric_arm_projection' and v['constraint_frame_id']==p['frame_id']
    assert values[side+'Elbow']['state']=='inferred' and not values[side+'Elbow']['measurement_valid']
    assert values[side+'Wrist']['state']=='observed_hand' and values[side+'Wrist']['measurement_valid']
    assert np.allclose([values[side+'Wrist']['x'],values[side+'Wrist']['y']],c.project_color(wrist))
    assert p==before # no inferred coordinates enter measured pose/controller
    assert measured_forearm(p,side) is None
    assert tracking['arms'][side]['length_unit']=='metres'


@pytest.mark.parametrize('reset',['local','session','geometry'])
def test_projection_does_not_reintroduce_pre_reset_geometry(reset):
    p=measured_pose();c=calibration();s=MetricSkeleton('camera');establish(s,p);i=ImageSkeleton()
    i.update(p,None,1000,s,c)
    q=hide(next_frame(p),'rightElbow');phone_layer(q,c);s.update(q)
    if reset=='local':i.reset()
    elif reset=='session':q['session_id']='different'
    else:q['calibration_id']='different'
    result=i.update(q,None,1000.1,s,c)
    assert not any(v.get('source')=='ubuntu_metric_arm_projection' for v in result['points'])


def test_projected_clamp_retains_fixed_lengths_and_marks_actual_hand_target():
    p=measured_pose();c=calibration();s=MetricSkeleton('camera');establish(s,p)
    p=hide(next_frame(p),'rightElbow');point(p,'rightWrist')['xyz_m']=[1.5,.4,1.2]
    phone_layer(p,c);s.update(p);assert_lengths(s,'right')
    result=ImageSkeleton().update(p,None,1000.1,s,c)
    a=result['arms']['right'];w=next(v for v in result['points'] if v['name']=='rightWrist')
    assert a['wrist_clamped'] and not w['measurement_valid']
    assert np.allclose(a['target_wrist_xy'],c.project_color([1.5,.4,1.2]))
    assert not np.allclose([w['x'],w['y']],a['target_wrist_xy'])


def test_unseen_metric_geometry_keeps_phone_cue_without_inventing_lengths():
    p=hide(measured_pose(),'rightElbow');c=calibration();phone_layer(p,c)
    s=MetricSkeleton('camera');s.update(p)
    out=ImageSkeleton().update(p,None,1000,s,c)
    assert out['points']==p['body_tracking']['points']
    assert 'right' not in out['arms']


def test_projection_calibration_roundtrip_uses_lens_mapping():
    h=sensor_header(size=(640,480),depth_size=(320,240));raw=h['calibration']
    raw['lens_distortion_lut']=[.05]*len(raw['lens_distortion_lut'])
    raw['inverse_lens_distortion_lut']=[1/1.05-1]*len(raw['inverse_lens_distortion_lut'])
    c=Calibration(raw,h['color_dimensions'],h['depth_dimensions'])
    for xy in ([.1,.2],[.8,.9],[.5,.5]):
        xyz=c.unproject(c.landmark_pixel(*xy),1.2)
        assert np.allclose(c.project_color(xyz),xy,atol=1e-9)
    for xyz in ([0,0,0],[1,0,-1],[float('nan'),0,1]):assert c.project_color(xyz) is None


def test_missing_elbow_does_not_discard_independently_measured_hand_wrist(header):
    h=body_header(header)
    next(p for p in h['body']['landmarks'] if p['name']=='rightElbow').update(confidence=0.)
    next(p for p in h['body']['landmarks'] if p['name']=='rightWrist').update(confidence=.25)
    p,_=PoseEstimator().estimate(h,np.full((24,32),500,dtype='<u2'))
    arm=p['body']['arms']['right']
    assert arm['positions_camera_m']['rightElbow'] is None
    assert arm['positions_camera_m']['rightWrist'] is not None
    assert arm['wrist_source']=='associated_vision_hand_wrist_surface'
    assert not arm['measurement_valid'] and not arm['forearm_measurement_valid']
    assert measured_forearm(p,'right') is None


def test_old_metric_source_cannot_be_projected_on_new_rgb_frame():
    p=measured_pose();c=calibration();s=MetricSkeleton('camera');establish(s,p)
    p=hide(next_frame(p),'rightElbow');phone_layer(p,c);s.update(p)
    p=next_frame(p);phone_layer(p,c)
    out=ImageSkeleton().update(p,None,1000.2,s,c)
    assert not any(v.get('source')=='ubuntu_metric_arm_projection' for v in out['points'])
