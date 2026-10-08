"""Bounded exact-frame RGB/depth pairing and independent Ubuntu RGB inference."""
import base64
from collections import OrderedDict
import json
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time

DEFAULT_MODEL=Path('/mnt/robotics-data/robotics/agibot-g2/models/handdepth/pose_landmarker_full.task')
APP=Path(__file__).resolve().parents[2]
MATCH_FIELDS=('session_id','calibration_id','frame_id','capture_time_s','color_dimensions')


def same_frame(a,b):
    return all(a.get(k)==b.get(k) for k in MATCH_FIELDS)


class PoseProcess:
    def __init__(self,model=DEFAULT_MODEL):
        from .recording import check_recording_root
        cache=check_recording_root(model.parent/'mpl-cache')
        env=dict(os.environ,MPLCONFIGDIR=str(cache),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        self.process=subprocess.Popen([str(APP/'pose-venv/bin/python'),'-I',
            str(APP/'tools/ubuntu/pose_worker.py'),str(model)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            text=True,bufsize=1,env=env)
        self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ)

    def detect(self,header,jpeg):
        request={k:header[k] for k in MATCH_FIELDS};request['jpeg']=base64.b64encode(jpeg).decode('ascii')
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        if not self.selector.select(timeout=3.):raise RuntimeError('PC pose worker timed out')
        line=self.process.stdout.readline(100_000)
        if not line or len(line)>=100_000:raise RuntimeError('PC pose worker stopped/invalid result')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        if not same_frame(header,result):raise RuntimeError('PC pose result frame mismatch')
        return result

    def close(self):
        self.selector.close()
        self.process.terminate()
        try:self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
        self.process.stdin.close();self.process.stdout.close()


class PCFusion:
    """One in-flight job, one replaceable pending job, at most eight frame pairs."""
    def __init__(self,callback,model=DEFAULT_MODEL,process=None):
        self.callback=callback;self.model=model;self.process=process
        self.sensor=OrderedDict();self.preview=OrderedDict();self.identity=None
        self.condition=threading.Condition();self.pending=None;self.stopped=False;self.epoch=0
        self.stats={'enabled':True,'model':model.name,'submitted':0,'completed':0,'replaced':0,
                    'discarded':0,'error':None,'inference_ms':None}
        self.thread=threading.Thread(target=self._run,name='Ubuntu RGB pose',daemon=True);self.thread.start()

    def reset(self):
        with self.condition:
            self.sensor.clear();self.preview.clear();self.pending=None;self.epoch+=1

    def add(self,channel,header,payload,received,pose=None):
        identity=(header['session_id'],header['calibration_id'],tuple(header['color_dimensions']))
        # Receiver invokes this only after its ordering/session gate accepts data.
        if self.identity!=identity:self.reset();self.identity=identity
        key=(header['session_id'],header['frame_id'])
        cache=self.sensor if channel=='sensor' else self.preview
        cache[key]=(header,payload,received,pose)
        while len(cache)>8:cache.popitem(last=False)
        if key not in self.sensor or key not in self.preview:return
        a=self.sensor.pop(key);b=self.preview.pop(key)
        if not same_frame(a[0],b[0]):self.stats['discarded']+=1;return
        with self.condition:
            if self.pending is not None:self.stats['replaced']+=1
            self.pending=(self.epoch,a,b[1]);self.stats['submitted']+=1;self.condition.notify()

    def _run(self):
        try:
            if self.process is None:self.process=PoseProcess(self.model)
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.stopped or self.pending is not None)
                    if self.stopped:break
                    epoch,sensor,jpeg=self.pending;self.pending=None
                header,depth,received,base_pose=sensor
                if time.monotonic()-received>.5:self.stats['discarded']+=1;continue
                result=self.process.detect(header,jpeg)
                if epoch!=self.epoch or self.stopped:self.stats['discarded']+=1;continue
                self.stats.update(completed=self.stats['completed']+1,inference_ms=result['inference_ms'])
                self.callback(header,depth,received,base_pose,result)
        except Exception as e:
            self.stats['error']=str(e)
        finally:
            if self.process:self.process.close()

    def close(self):
        with self.condition:self.stopped=True;self.pending=None;self.condition.notify()
        self.thread.join(timeout=4)
        if self.thread.is_alive() and isinstance(self.process,PoseProcess):
            self.process.process.terminate();self.thread.join(timeout=2)
