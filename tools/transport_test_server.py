"""Actual Swift URLSession integration harness. First sensor ACK stalls deliberately."""
import asyncio
import argparse
import time
from pathlib import Path
import websockets
from handdepth.protocol import encode,decode,MAX_MESSAGE

async def main(port_file):
    stalled=False
    highwater={}
    async def handler(ws):
        nonlocal stalled
        try:
            async for raw in ws:
                h=decode(raw).header; t1=time.monotonic()
                if h['kind']=='clock_ping':
                    await ws.send(encode({'kind':'clock_pong','session_id':h['session_id'],'t0':h['t0'],'t1':t1,'t2':time.monotonic()}))
                else:
                    channel=ws.request.path
                    assert h['frame_id'] > highwater.get(channel,-1), 'old frame replayed after reconnect'
                    highwater[channel]=h['frame_id']
                    if channel=='/sensor' and not stalled:
                        stalled=True
                        await asyncio.sleep(1.5)
                    await ws.send(encode({'kind':'ack','session_id':h['session_id'],'frame_id':h['frame_id'],'accepted':True}))
        except websockets.exceptions.ConnectionClosed:
            pass
    async with websockets.serve(handler,'127.0.0.1',0,max_size=MAX_MESSAGE,compression=None) as server:
        Path(port_file).write_text(str(server.sockets[0].getsockname()[1]))
        await asyncio.Future()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('port_file');a=p.parse_args()
    asyncio.run(main(a.port_file))
