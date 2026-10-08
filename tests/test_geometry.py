import copy
import numpy as np
import pytest
from handdepth.geometry import Calibration, CalibrationError, PoseEstimator, sample_surface, quaternion_xyzw

def calibration(header):
    return Calibration(header['calibration'],header['color_dimensions'],header['depth_dimensions'])

def test_scaling_pixel_centres_and_known_geometry(header):
    c = calibration(header)
    assert np.allclose(c.k_depth,[[40,0,16],[0,40,12],[0,0,1]])
    assert np.allclose(c.unproject([20,16],.5),[.05,.05,.5])
    assert np.allclose(c.landmark_pixel(.625,2/3),[20,16])
    depth = np.arange(32*24,dtype=np.uint16).reshape(24,32)
    assert np.array_equal(c.rectify(depth),depth)

def test_nonzero_lut_direction_and_resampling(header):
    header['calibration']['lens_distortion_lut'] = [.1,.1,.1]
    header['calibration']['inverse_lens_distortion_lut'] = [-1/11,-1/11,-1/11]
    c = calibration(header)
    assert np.allclose(c.map_points([42,24],c.forward),[43,24])
    assert np.allclose(c.map_points([43,24],c.inverse),[42,24])
    assert np.allclose(c.landmark_pixel(43/64,.5),[21,12])
    depth = np.arange(32*24,dtype=np.uint16).reshape(24,32)+1
    rect = c.rectify(depth)
    assert rect[12,24] == depth[c.map_y[12,24],c.map_x[12,24]]
    assert c.map_x[12,24] > 24
    assert np.any(rect == 0)  # rays at outer rectified pixels leave original field

@pytest.mark.parametrize('field,value', [('lens_distortion_lut',None),('inverse_lens_distortion_lut',[]),
    ('intrinsics_row_major',[[0,0,0],[0,0,0],[0,0,1]]),('reference_dimensions',[0,48]),
    ('color_from_reference',[[1,0,5],[0,1,0],[0,0,1]])])
def test_missing_or_unsupported_calibration(header,field,value):
    header['calibration'][field]=value
    with pytest.raises(CalibrationError): calibration(header)
    pose,_ = PoseEstimator().estimate(header,np.full((24,32),500,dtype=np.uint16))
    assert not pose['depth_valid'] and pose['palm_position_m'] is None

def test_surface_depth_invalid_and_edges(header):
    c = calibration(header); d=np.full((24,32),500,dtype=np.uint16)
    l = header['landmarks'][0]
    s = sample_surface(c,d,l)
    assert s['valid'] and s['xyz_m'][2] == .5 and s['sigma_z_m'] >= .0005
    x,y=np.floor(c.landmark_pixel(l['x'],l['y'])).astype(int)
    d[y,x]=0
    assert sample_surface(c,d,l)['reason'] == 'insufficient_depth'
    d[:]=500;d[:,x:]=1000
    assert sample_surface(c,d,l)['reason'] == 'depth_edge'
    d[:]=0
    assert not sample_surface(c,d,l)['valid']

def test_pose_quaternion_pinch_and_degenerate_palm(header):
    estimator = PoseEstimator();d=np.full((24,32),500,dtype=np.uint16)
    pose,_=estimator.estimate(header,d)
    expected = np.mean([[(p['x']*64-32)*.5/80,(p['y']*48-24)*.5/80,.5] for p in header['landmarks'] if p['name'] in ['wrist','indexMCP','middleMCP','ringMCP','littleMCP']],axis=0)
    assert np.allclose(pose['palm_position_m'],expected)
    assert pose['orientation_valid'] and np.isclose(np.linalg.norm(pose['palm_quaternion_xyzw']),1)
    assert np.allclose(pose['palm_quaternion_xyzw'],[0,0,1,0])
    assert pose['pinch_distance_m'] > 0
    estimator.hand_states["legacy-primary"].previous_q=-np.array(pose['palm_quaternion_xyzw'])
    pose2,_=estimator.estimate(header,d)
    assert np.dot(pose2['palm_quaternion_xyzw'],estimator.hand_states["legacy-primary"].previous_q) > .99
    for p in header['landmarks']: p.update(x=.5,y=.5)
    degenerate,_=estimator.estimate(header,d)
    assert not degenerate['orientation_valid'] and degenerate['palm_quaternion_xyzw'] is None

def test_relative_accuracy_and_absent_hand(header):
    d=np.full((24,32),500,dtype=np.uint16)
    header['depth_accuracy']='relative'
    pose,_=PoseEstimator().estimate(header,d)
    assert not pose['depth_valid'] and pose['reason']=='relative_depth_not_metric'
    header['depth_accuracy']='absolute';header['landmarks']=[]
    pose,_=PoseEstimator().estimate(header,d)
    assert not pose['tracking_valid'] and pose['palm_position_m'] is None

def test_quaternion_180_degree_axes():
    for r,expected in [(np.diag([1,-1,-1]),[1,0,0,0]),(np.diag([-1,1,-1]),[0,1,0,0])]:
        assert np.allclose(quaternion_xyzw(r),expected)

def test_orientation_flip_rejection_and_reacquisition(header):
    d=np.full((24,32),500,dtype=np.uint16); estimator=PoseEstimator()
    first,_=estimator.estimate(header,d)
    assert first['orientation_valid']
    header['frame_id']+=1;header['capture_time_s']+=.05
    for l in header['landmarks']: l['x']=1-l['x']
    flipped,_=estimator.estimate(header,d)
    assert not flipped['orientation_valid'] and flipped['orientation_rejection']=='abrupt_orientation_flip'
    header['capture_time_s']+=.05
    reacquired,_=estimator.estimate(header,d)
    assert reacquired['orientation_valid']

def test_calibration_missing_retains_detector_points(header):
    header['calibration']=None
    pose,_=PoseEstimator().estimate(header,np.full((24,32),500,dtype=np.uint16))
    assert len(pose['landmarks'])==21
    assert all(l['xyz_m'] is None and not l['valid'] for l in pose['landmarks'])
    assert pose['tracking_valid']

def test_full_sensor_reference_dimensions_exceed_stream_resolution(header):
    raw=header['calibration']
    raw['reference_dimensions']=[4032,3024]
    raw['intrinsics_row_major']=[[5040,0,2016],[0,5040,1512],[0,0,1]]
    raw['distortion_center']=[2016,1512]
    raw['color_from_reference']=[[1/63,0,0],[0,1/63,0],[0,0,1]]
    raw['depth_from_reference']=[[1/126,0,0],[0,1/126,0],[0,0,1]]
    c=calibration(header)
    assert np.allclose(c.k_depth,[[40,0,16],[0,40,12],[0,0,1]])
    assert np.allclose(c.unproject([20,16],.5),[.05,.05,.5])
