"""Isolated CPU RGB pose worker. JSON-lines in/out; diagnostic logs on stderr.

Runs in pose-venv so MediaPipe/OpenCV dependencies cannot replace viewer Qt/GL.
Only normalized RGB observations are returned. Model-predicted XYZ is unused.
"""
import base64
import json
import sys
import time
import cv2
import numpy as np
import mediapipe as mp

JOINTS={'leftShoulder':11,'rightShoulder':12,'leftElbow':13,'rightElbow':14,
        'leftWrist':15,'rightWrist':16}


class Detector:
    def __init__(self,path):
        self.path=path;self.model=None;self.identity=None;self.last_ms=-1

    def close(self):
        if self.model:self.model.close();self.model=None

    def detect(self,request):
        identity=(request['session_id'],request['calibration_id'])
        if self.identity!=identity:
            self.close()
            self.model=mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.path,delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1,
                min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5))
            self.identity=identity;self.last_ms=-1
        jpeg=base64.b64decode(request['jpeg'],validate=True)
        bgr=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        if bgr is None or list(bgr.shape[1::-1])!=request['color_dimensions']:
            raise ValueError('RGB dimensions mismatch')
        image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        timestamp=max(self.last_ms+1,round(request['capture_time_s']*1000));self.last_ms=timestamp
        start=time.perf_counter();result=self.model.detect_for_video(image,timestamp)
        out={k:request[k] for k in ['session_id','calibration_id','frame_id','capture_time_s','color_dimensions']}
        out.update(schema='handdepth.pc_pose.v1',detector='mediapipe_pose_landmarker',
                   inference_ms=(time.perf_counter()-start)*1000,landmarks={},predicted_xyz_used=False)
        if result.pose_landmarks:
            for name,index in JOINTS.items():
                p=result.pose_landmarks[0][index]
                score=min(float(p.visibility or 0),float(p.presence or 0))
                out['landmarks'][name]={'name':name,'x':float(p.x),'y':float(p.y),'confidence':score,
                                        'visibility':float(p.visibility or 0),'presence':float(p.presence or 0)}
        return out


def main():
    detector=Detector(sys.argv[1])
    try:
        for line in sys.stdin:
            try:
                if len(line)>4_000_000:raise ValueError('worker request too large')
                result=detector.detect(json.loads(line))
            except Exception as e:result={'error':str(e)}
            print(json.dumps(result,allow_nan=False),flush=True)
    finally:detector.close()


if __name__=='__main__':main()
