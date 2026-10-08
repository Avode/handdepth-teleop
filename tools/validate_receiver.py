"""Synthetic local integration/CPU benchmark/optional soak. Never a hardware claim."""
import asyncio
import copy
import io
import json
from pathlib import Path
import platform
import resource
import statistics
import tempfile
import time
import argparse
import cv2
import numpy as np
import websockets
from generate_fixtures import sensor_header,preview_packet,two_hand_header,body_header
from handdepth.protocol import encode,decode,MAX_MESSAGE,sensor_hands
from handdepth.server import Receiver
from handdepth.recording import Recorder,replay_packets

ROOT=Path(__file__).resolve().parents[1]

def benchmark(hands=1,body=False):
    r=Receiver();h=sensor_header(size=(640,480),depth_size=(320,240))
    if body:h=body_header(h)
    elif hands == 2: h=two_hand_header(h)
    depth=np.full((240,320),500,dtype='<u2').tobytes()
    times=[]
    for i in range(300):
        t=time.monotonic();h.update(frame_id=i,capture_time_s=t,processing_start_s=t,processing_end_s=t)
        raw=encode(h,depth)
        begin=time.perf_counter();assert r.process('sensor',raw,t);times.append((time.perf_counter()-begin)*1000)
    return {'frames':300,'hands':hands,'body':body,'color_dimensions':[640,480],'depth_dimensions':[320,240],
        'median_sensor_processing_ms':statistics.median(times),'p95_sensor_processing_ms':float(np.percentile(times,95))}

async def live(seconds,record,hands=1,body=False):
    output=io.StringIO() if record else None
    with tempfile.TemporaryDirectory(prefix='handdepth-validation-') as temp:
        recorder=Recorder(temp,fixture=True) if record else None
        r=Receiver(recorder,output);counts={'sensor':0,'preview':0};rtts=[]
        initial_rss=None;final_rss=None
        async with websockets.serve(r.handler,'127.0.0.1',0,max_size=MAX_MESSAGE,max_queue=1,compression=None) as server:
            port=server.sockets[0].getsockname()[1]
            h=sensor_header(size=(640,480),depth_size=(320,240))
            if body:h=body_header(h)
            elif hands == 2: h=two_hand_header(h)
            depth=np.full((240,320),500,dtype='<u2').tobytes();estimate=None
            async with websockets.connect(f'ws://127.0.0.1:{port}/sensor',proxy=None,compression=None) as sensor, websockets.connect(f'ws://127.0.0.1:{port}/preview',proxy=None,compression=None) as preview:
                start=time.monotonic();i=0;next_clock=0
                while time.monotonic()-start < seconds:
                    now=time.monotonic()
                    if now>=next_clock:
                        t0=now
                        await sensor.send(encode({'kind':'clock_ping','session_id':h['session_id'],'t0':t0}))
                        pong=decode(await sensor.recv()).header;t3=time.monotonic()
                        rtt=(t3-t0)-(pong['t2']-pong['t1'])
                        estimate={'offset_s':((pong['t1']-t0)+(pong['t2']-t3))/2,'rtt_s':rtt,'ios_sample_s':t3}
                        rtts.append(rtt*1000);next_clock=now+2
                    capture=time.monotonic()
                    # Small synthetic translation, fixed plane. Never labeled as live camera data.
                    hh=copy.deepcopy(h)
                    for hand in sensor_hands(hh):
                        for p in hand['landmarks']: p['x']+=.03*np.sin(i/30)
                    if body:
                        for p in hh['body']['landmarks']+[hh['body']['torso_surface']]:p['x']+=.03*np.sin(i/30)
                    hh.update(frame_id=i,capture_time_s=capture,processing_start_s=capture,processing_end_s=capture,
                              clock_estimate=estimate)
                    async def send_one(ws,raw,key):
                        await ws.send(raw);ack=decode(await ws.recv()).header
                        assert ack['accepted'];counts[key]+=1
                    await asyncio.gather(send_one(sensor,encode(hh,depth),'sensor'),send_one(preview,preview_packet(hh),'preview'))
                    if i==100: initial_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    i+=1
                    delay=start+i/15-time.monotonic()
                    if delay>0: await asyncio.sleep(delay)
                final_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                assert r.latest_pose['depth_valid'] and r.latest_pose['orientation_valid']
                assert len(r.latest_pose['hands'])==hands
                assert all(h['orientation_valid'] for h in r.latest_pose['hands'])
                if body:
                    assert r.latest_pose['body']['head']['quaternion_torso_xyzw'] is not None
                    assert all(h['torso_relative_orientation_valid'] for h in r.latest_pose['hands'])
                assert r.latest_pose['capture_to_receive_s'] is not None if seconds>=6 else True
                cv2.imwrite(str(ROOT/'docs/validation'/('receiver-body-synthetic.png' if body else 'receiver-two-hand-synthetic.png' if hands==2 else 'receiver-synthetic.png')),r.dashboard.render())
            await asyncio.sleep(.05)
            assert r.latest_pose['reason']=='sensor_disconnected'
        if recorder:
            recorder.close(); replay=Receiver();replayed=0
            for ch,t,raw in replay_packets(recorder.path):
                assert replay.process(ch,raw,t,replay=True);replayed+=1
            assert replayed==sum(counts.values())
            assert len(replay.latest_pose['hands'])==hands
            if body:assert replay.latest_pose['body']['torso']['orientation_valid']
            cv2.imwrite(str(ROOT/'docs/validation'/('replay-body-synthetic.png' if body else 'replay-two-hand-synthetic.png' if hands==2 else 'replay-synthetic.png')),replay.dashboard.render())
            # Exercise the installed CLI, including its headless snapshot/replay output.
            import subprocess,sys
            result=subprocess.run([str(ROOT/'.venv/bin/handdepth'),'replay',str(recorder.path),'--headless','--speed','0',
                '--pose-jsonl',str(Path(temp)/'replay-poses.jsonl'),'--snapshot',str(Path(temp)/'replay.png')],capture_output=True,text=True)
            assert result.returncode==0,result.stderr
        return {'duration_s':seconds,'hands':hands,'body':body,'acknowledged':counts,'clock_rtt_median_ms':statistics.median(rtts),
                'maximum_rss_platform_units':final_rss,'rss_at_frame_100_platform_units':initial_rss,
                'replay_verified':record,'hardware_input':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--soak-seconds',type=float,default=0);p.add_argument('--hands',type=int,choices=[1,2],default=1);p.add_argument('--body',action='store_true');a=p.parse_args()
    if a.body:a.hands=2
    result={'platform':platform.platform(),'python':platform.python_version(),'source':'synthetic','benchmark':benchmark(a.hands,a.body),
            'recording_integration':asyncio.run(live(10,True,a.hands,a.body))}
    if a.soak_seconds: result['soak']=asyncio.run(live(a.soak_seconds,False,a.hands,a.body))
    (ROOT/'docs/validation'/('receiver-body-results.json' if a.body else 'receiver-two-hand-results.json' if a.hands==2 else 'receiver-results.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
