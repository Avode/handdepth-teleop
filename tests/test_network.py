import asyncio
import io
import json
import time
import numpy as np
import pytest
import websockets
from handdepth.server import Receiver
from handdepth.protocol import encode,decode,MAX_MESSAGE

@pytest.mark.asyncio
async def test_server_ping_packets_disconnect_and_reconnect(header):
    output=io.StringIO();r=Receiver(pose_stream=output)
    async with websockets.serve(r.handler,'127.0.0.1',0,max_size=MAX_MESSAGE,max_queue=1,compression=None) as server:
        port=server.sockets[0].getsockname()[1]
        async with websockets.connect(f'ws://127.0.0.1:{port}/sensor',proxy=None) as ws:
            await ws.send(encode({'kind':'clock_ping','session_id':header['session_id'],'t0':100,'payload_encoding':'none'}))
            pong=decode(await ws.recv()).header
            assert pong['t0']==100 and pong['t2']>=pong['t1']
            depth=np.full((24,32),500,dtype='<u2').tobytes()
            raw=encode(header,depth)
            await ws.send(raw)
            assert decode(await ws.recv()).header['accepted']
            await ws.send(raw)
            assert not decode(await ws.recv()).header['accepted']
        await asyncio.sleep(.05)
        assert r.latest_pose['reason']=='sensor_disconnected'
        async with websockets.connect(f'ws://127.0.0.1:{port}/sensor',proxy=None) as ws:
            await ws.send(raw)
            assert not decode(await ws.recv()).header['accepted']
            header.update(frame_id=43,capture_time_s=101,processing_start_s=101.001,processing_end_s=101.015)
            await ws.send(encode(header,depth))
            assert decode(await ws.recv()).header['accepted']
        async with websockets.connect(f'ws://127.0.0.1:{port}/sensor',proxy=None) as ws:
            await ws.send(b'bad')
            with pytest.raises(websockets.exceptions.ConnectionClosedError): await ws.recv()
        async with websockets.connect(f'ws://127.0.0.1:{port}/wrong',proxy=None) as ws:
            with pytest.raises(websockets.exceptions.ConnectionClosedError): await ws.recv()
