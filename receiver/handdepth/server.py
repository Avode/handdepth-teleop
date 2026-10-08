import asyncio
import json
import logging
import time
import threading
from collections import OrderedDict,deque,Counter
import websockets
from .geometry import PoseEstimator
from .protocol import decode, encode, depth_array, finite, ProtocolError, MAX_MESSAGE
from .state import FrameGate, ClockAlignment
from .visualization import Dashboard

log = logging.getLogger("handdepth")

class Receiver:
    def __init__(self, recorder=None, pose_stream=None, max_age=.25):
        self.lock = threading.RLock()
        self.recorder = recorder
        self.pose_stream = pose_stream
        self.gate = FrameGate(max_age)
        self.estimator = PoseEstimator()
        self.dashboard = Dashboard()
        self.connections = {}
        self.latest_pose = None
        self.latest_received = 0
        self.last_expired = False
        self.pc_fusion = None
        self.fusion_estimator = PoseEstimator()
        self.fusion_identity = None
        self.fusion_input_live = False
        self.fusion_wait_s=.10
        self.pending_sensor=OrderedDict()
        self.publications=Counter();self.publication_times=deque(maxlen=120)
        self.transport={ch:{'received':0,'accepted':0,'rejected':Counter(),'arrivals':deque(maxlen=120),
            'captures':deque(maxlen=120),'last_frame':None,'processing_ms':None,'ack_ms':None}
            for ch in ('sensor','preview')}

    def _publish_sensor(self,header,depth,received,pose,source):
        if self.latest_pose and not self.latest_pose.get('receiver_event') and header['frame_id']<=self.latest_pose['frame_id']:return
        pose=dict(pose,processing_location=source)
        self.emit(pose)
        self.dashboard.sensor(header,depth,pose,received)
        for frame in list(self.pending_sensor):
            if frame<=header['frame_id']:self.pending_sensor.pop(frame)
        self.publications[source]+=1;self.publication_times.append(time.monotonic())
        if self.recorder and source!='phone_rgbd_direct':self.recorder.refinement(pose)

    def _flush_pending(self,now):
        if not self.fusion_input_live or self.last_expired:return
        ready=[v for v in self.pending_sensor.values() if now-v[2]>=self.fusion_wait_s]
        if ready:
            # A slow worker or permanently unpaired RGB must not freeze a healthy
            # sensor stream. Publish only the newest expired item, with ORIGINAL
            # measurement age. A late PC result cannot republish the same frame.
            h,d,received,pose=ready[-1]
            if now-received<=.5:self._publish_sensor(h,d,received,pose,'phone_rgbd_fallback_unpaired_or_slow_pc')

    def stream_status(self,now=None):
        now=time.monotonic() if now is None else now
        def hz(values):return (len(values)-1)/(values[-1]-values[0]) if len(values)>1 and values[-1]>values[0] else 0.
        with self.lock:
            channels={ch:{'received':s['received'],'accepted':s['accepted'],'rejected':dict(s['rejected']),
                'arrival_hz':hz(s['arrivals']),'source_hz':hz(s['captures']),'last_frame':s['last_frame'],
                'wire_age_s':now-s['arrivals'][-1] if s['arrivals'] else None,
                'phone_processing_ms':s['processing_ms'],'process_to_ack_ms':s['ack_ms']}
                for ch,s in self.transport.items()}
            return {'channels':channels,'publications':dict(self.publications),
                    'publish_hz':hz(self.publication_times),'fusion_wait_s':self.fusion_wait_s,
                    'pending_sensor_frames':len(self.pending_sensor),
                    'publication_source':(self.latest_pose or {}).get('processing_location')}

    def start_pc_pose(self,model):
        from .pc_pose import PCFusion
        self.pc_fusion=PCFusion(self._pc_result,model)

    def _pc_result(self,header,depth,received,base_pose,result):
        with self.lock:
            identity=(header['session_id'],header['calibration_id'],tuple(header['color_dimensions']))
            if identity!=self.fusion_identity or not self.fusion_input_live or time.monotonic()-received>.5:return
            if self.last_expired:return
            if self.latest_pose and header['frame_id']<=self.latest_pose['frame_id']:return
            pose,_=self.fusion_estimator.estimate(header,depth,pc_pose=result)
            for key in ('received_monotonic_s','capture_to_receive_s','clock_uncertainty_s','measurement_age_basis'):
                pose[key]=base_pose[key]
            pose['processing_location']='ubuntu_rgb_pose_plus_registered_depth_fusion'
            self._publish_sensor(header,depth,received,pose,'ubuntu_rgb_pose_plus_registered_depth_fusion')

    def close(self):
        if self.pc_fusion:self.pc_fusion.close()

    def emit(self, pose):
        self.latest_pose = pose
        if self.pose_stream:
            self.pose_stream.write(json.dumps(pose, allow_nan=False, separators=(",",":"))+"\n")
            self.pose_stream.flush()

    def invalidate(self, reason):
        with self.lock:
            self._invalidate(reason)

    def _invalidate(self, reason):
        self.fusion_input_live=False
        self.pending_sensor.clear()
        if self.pc_fusion:self.pc_fusion.reset()
        if self.latest_pose is None or self.last_expired:
            return
        pose = dict(self.latest_pose, tracking_valid=False, depth_valid=False, orientation_valid=False,
                    palm_position_m=None, palm_quaternion_xyzw=None, pinch_distance_m=None, landmarks=[], hands=[], body=None, body_tracking=None, primary_hand_id=None, reason=reason)
        pose["receiver_event"] = True
        pose["received_monotonic_s"] = time.monotonic()
        self.emit(pose)
        if self.recorder: self.recorder.event(pose)
        self.last_expired = True
        self.dashboard.status = reason
        self.estimator.reset_orientations()
        self.fusion_estimator.reset_orientations()

    def expire(self, now=None):
        with self.lock:
            now=time.monotonic() if now is None else now
            if now-self.latest_received > .5:
                self._invalidate("measurement_timeout")
            else:self._flush_pending(now)

    def process(self, channel, raw, received_s, replay=False):
        with self.lock:
            return self._process(channel, raw, received_s, replay)

    def _process(self, channel, raw, received_s, replay=False):
        packet = decode(raw)
        h = packet.header
        if h["kind"] != channel:
            raise ProtocolError("data on wrong channel")
        stats=self.transport[channel]
        stats['received']+=1;stats['arrivals'].append(received_s)
        stats['last_frame']=h['frame_id'];stats['processing_ms']=1000*(h['processing_end_s']-h['capture_time_s'])
        # replay uses recorded receiver timestamps for clock alignment/age.
        old_session = self.gate.session_id
        accepted, age, uncertainty = self.gate.accept(h, channel, received_s)
        if not accepted:
            stats['rejected'][self.gate.last_rejection or 'unknown']+=1
            return False
        stats['accepted']+=1;stats['captures'].append(h['capture_time_s'])
        if old_session != self.gate.session_id:
            self.estimator.reset()
            self.fusion_estimator.reset()
            self.latest_pose=None
            self.pending_sensor.clear()
        if channel == "sensor":
            depth = depth_array(packet)
            pose, rect = self.estimator.estimate(h, depth)
            pose.update(received_monotonic_s=received_s, capture_to_receive_s=age, clock_uncertainty_s=uncertainty,
                        measurement_age_basis="clock_aligned_capture" if age is not None else "receiver_arrival_only")
            identity=(h['session_id'],h['calibration_id'],tuple(h['color_dimensions']))
            if identity!=self.fusion_identity:
                self.fusion_estimator.reset();self.fusion_identity=identity;self.pending_sensor.clear()
            self.latest_received = time.monotonic() if replay else received_s
            self.last_expired = False
            self.fusion_input_live = True
            if self.pc_fusion and not self.pc_fusion.stats['error'] and not replay:
                # Publish one complete fused pose per frame, avoiding repeated
                # phone-invalid/PC-valid oscillation on the same control input.
                self.pc_fusion.add('sensor',h,depth,received_s,pose)
                self.pending_sensor[h['frame_id']]=(h,depth,received_s,pose)
                while len(self.pending_sensor)>8:self.pending_sensor.popitem(last=False)
            else:
                self._publish_sensor(h,depth,time.monotonic() if replay else received_s,pose,'phone_rgbd_direct')
        else:
            self.dashboard.preview(h, packet.payload, time.monotonic() if replay else received_s)
            if self.pc_fusion and not self.pc_fusion.stats['error']:
                self.pc_fusion.add('preview',h,packet.payload,received_s)
            pose = None
        if self.recorder:
            self.recorder.append(channel, raw, received_s, pose)
        return True

    async def handler(self, ws):
        path = ws.request.path
        if path not in ("/sensor", "/preview"):
            await ws.close(1008, "use /sensor or /preview"); return
        channel = path[1:]
        if channel in self.connections:
            await ws.close(1008, "one publisher per channel"); return
        self.connections[channel] = ws
        if channel == "sensor":
            with self.lock:
                self.gate.clock = ClockAlignment()
        sid = None
        try:
            async for raw in ws:
                received = time.monotonic()
                packet = decode(raw)
                h = packet.header
                if sid is not None and sid != h["session_id"]:
                    raise ProtocolError("session changed within connection")
                sid = h["session_id"]
                if h["kind"] == "clock_ping" and channel == "sensor":
                    t0 = finite(h.get("t0"), "clock t0")
                    reply = {"kind":"clock_pong", "session_id":sid, "t0":t0, "t1":received, "t2":time.monotonic(), "payload_encoding":"none"}
                    await ws.send(encode(reply))
                elif h["kind"] == channel:
                    accepted = await asyncio.to_thread(self.process, channel, raw, received)
                    with self.lock:self.transport[channel]['ack_ms']=1000*(time.monotonic()-received)
                    # End-to-end ACK prevents the iPhone from filling TCP buffers with frames.
                    await ws.send(encode({"kind":"ack", "session_id":sid, "frame_id":h["frame_id"], "accepted":accepted, "payload_encoding":"none"}))
                else:
                    raise ProtocolError("unexpected packet on channel")
        except (ProtocolError, ValueError, KeyError, OverflowError) as e:
            log.warning("rejected %s packet: %s",channel,e)
            await ws.close(1007, "malformed packet")
        except websockets.exceptions.ConnectionClosed:
            pass
        except OSError as e:
            log.error("storage/output error: %s",e)
            await ws.close(1011,"receiver storage failure")
        finally:
            self.connections.pop(channel, None)
            if channel == "sensor":
                self.invalidate("sensor_disconnected")
                log.info("sensor disconnected")

    async def run(self, host, port, headless=False, duration=None, stop_event=None):
        async with websockets.serve(self.handler, host, port, max_size=MAX_MESSAGE, max_queue=1,
                                    compression=None, ping_interval=2, ping_timeout=3, close_timeout=1,
                                    write_limit=64*1024):
            log.info("listening on %s:%s /sensor /preview",host,port)
            start = time.monotonic()
            while (duration is None or time.monotonic()-start < duration) and not (stop_event and stop_event.is_set()):
                self.expire()
                if not headless:
                    import cv2
                    with self.lock:
                        image = self.dashboard.render()
                    cv2.imshow("HandDepth", image)
                    key=cv2.waitKey(1) & 255
                    if key in (ord('r'),ord('R')):
                        with self.lock:self.dashboard.reset_display()
                    if key in (27, ord('q')):
                        break
                await asyncio.sleep(.033 if not headless else .01)
        if not headless:
            import cv2
            cv2.destroyAllWindows()
