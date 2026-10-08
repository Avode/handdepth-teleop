from pathlib import Path
import copy
import json
import struct
import numpy as np
import pytest
from handdepth.protocol import decode, encode, depth_array, ProtocolError, MAX_HEADER, MAX_PAYLOAD
ROOT = Path(__file__).resolve().parents[1]

def test_shared_swift_python_packets():
    a = decode((ROOT/'fixtures/sensor-python.bin').read_bytes())
    b = decode((ROOT/'fixtures/sensor-swift.bin').read_bytes())
    assert a.header == b.header
    assert a.payload == b.payload == (ROOT/'fixtures/depth-u16le.bin').read_bytes()
    assert depth_array(a)[0,:4].tolist() == [0,1,256,65535]
    assert a.header['landmarks'][0]['x'] == .5
    assert len(decode((ROOT/'fixtures/preview.bin').read_bytes()).payload) > 100

def packet(header):
    return encode(header, np.full((24,32),500,dtype='<u2').tobytes())

@pytest.mark.parametrize('field,value', [('mirrored',True), ('orientation','portrait'), ('depth_dimensions',[640,480]),
    ('protocol_version',2), ('frame_id',-1), ('payload_length',-1), ('clock','wall'), ('processing_end_s',99)])
def test_bad_fields(header, field, value):
    data = packet(header)
    n, = struct.unpack('>I',data[:4]); h = json.loads(data[4:4+n]);h[field] = value
    raw = json.dumps(h).encode(); data=struct.pack('>I',len(raw))+raw+data[4+n:]
    with pytest.raises(ProtocolError): decode(data)

@pytest.mark.parametrize('bad', [b'',b'\x00\x00\x00\x00',struct.pack('>I',MAX_HEADER+1),b'\x00\x00\x00\x01{',b'\x00\x00\x00\x01\xff'])
def test_bad_frame(bad):
    with pytest.raises(ProtocolError): decode(bad)

def test_payload_limits_lengths_and_nan(header):
    data = packet(header)
    for bad in [data[:-1], data+b'x', b'x'*(MAX_PAYLOAD+MAX_HEADER+5)]:
        with pytest.raises(ProtocolError): decode(bad)
    for value in [float('nan'),float('inf')]:
        with pytest.raises(ProtocolError): encode({'value':value})
    for text in [b'{"x":NaN}',b'{"x":1e999}',b'{"x":1,"x":2}']:
        with pytest.raises(ProtocolError): decode(struct.pack('>I',len(text))+text)

def test_joint_confidence_and_nulls(header):
    header['landmarks'][0].update(x=None,y=None,confidence=0)
    assert decode(packet(header)).header['landmarks'][0]['x'] is None
    header['landmarks'][1]['confidence'] = 1.1
    with pytest.raises(ProtocolError): decode(packet(header))
