import io
import json
import struct
import numpy as np
import pytest
from handdepth.protocol import encode, ProtocolError
from handdepth.state import FrameGate,ClockAlignment
from handdepth.recording import Recorder, replay_packets, check_recording_root
from handdepth.server import Receiver
from generate_fixtures import preview_packet

def test_clock_alignment_confidence_and_staleness():
    c=ClockAlignment()
    for i in range(3):
        c.update({'offset_s':900.,'rtt_s':.01,'ios_sample_s':100+i},1000+i+.02)
    age,error=c.age(102,1002.04)
    assert abs(age-.04)<1e-8 and error == pytest.approx(.005)
    assert c.age(102,1020)==(None,None)
    noisy=ClockAlignment()
    for i in range(3): noisy.update({'offset_s':900+i*.03,'rtt_s':.01,'ios_sample_s':100+i},1000+i+.1)
    assert noisy.age(102,1002.04)==(None,None)

def test_stale_duplicate_reconnect_and_preview_session(header):
    gate=FrameGate()
    assert gate.accept(header,'sensor',1000)[0]
    assert not gate.accept(header,'sensor',1000)[0]
    header.update(frame_id=43,capture_time_s=100.01,processing_start_s=100.02,processing_end_s=100.4)
    assert not gate.accept(header,'sensor',1000)[0]
    header.update(processing_end_s=100.03)
    assert gate.accept(header,'sensor',1000)[0]
    header['session_id']='next-session'
    assert not gate.accept(header,'preview',1000)[0]
    assert gate.accept(header,'sensor',1000)[0]

def test_record_replay_and_disconnect_invalidation(tmp_path,header):
    recorder=Recorder(tmp_path,fixture=True);poses=io.StringIO()
    r=Receiver(recorder,poses)
    raw=encode(header,np.full((24,32),500,dtype='<u2').tobytes())
    assert r.process('sensor',raw,1000)
    assert r.process('preview',preview_packet(header),1000.01)
    r.invalidate('sensor_disconnected')
    assert not r.latest_pose['depth_valid'] and r.latest_pose['palm_position_m'] is None
    recorder.close()
    records=list(replay_packets(recorder.path))
    assert [v[0] for v in records]==['sensor','preview']
    assert records[0][2]==raw
    replay=Receiver()
    for ch,t,data in records: assert replay.process(ch,data,t,replay=True)
    assert replay.latest_pose['palm_position_m'] is not None
    assert list((recorder.path/'calibration').glob('*.json'))
    assert json.loads((recorder.path/'metadata.json').read_text())['protocol_version']==1
    assert len(poses.getvalue().splitlines())==2

def test_no_ssd_fallback_or_mac_linux_paths(tmp_path):
    with pytest.raises(ValueError): check_recording_root('/mnt/robotics-data/robotics/agibot-g2/datasets/handdepth',fixture=True)
    with pytest.raises(ValueError): check_recording_root(tmp_path)
    assert check_recording_root(tmp_path,fixture=True)==tmp_path.resolve()

def test_truncated_recording(tmp_path):
    (tmp_path/'packets.hdr').write_bytes(b'HDREC1\n'+struct.pack('>BdI',0,100,30)+b'123')
    with pytest.raises(ProtocolError): list(replay_packets(tmp_path))

def test_dashboard_rejects_jpeg_dimensions(header):
    r=Receiver()
    raw=preview_packet(header)
    from handdepth.protocol import decode
    p=decode(raw);p.header['color_dimensions']=[20,20]
    with pytest.raises(ValueError): r.process('preview',encode(p.header,p.payload),1000)

def test_jpeg_preflight_blocks_large_allocation():
    from handdepth.visualization import jpeg_dimensions
    # SOF0 declares a massive raster despite a tiny binary packet.
    malicious=b'\xff\xd8\xff\xc0\x00\x0b\x08\xff\xff\xff\xff\x01\x01\x11\x00'
    with pytest.raises(ValueError): jpeg_dimensions(malicious)
