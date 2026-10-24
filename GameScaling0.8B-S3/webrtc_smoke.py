"""Native FP16 model + two real WebRTC peers on loopback, not a game benchmark."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New private run directory')
    args=parser.parse_args();out=args.output.expanduser().resolve()
    if out.exists():parser.error('Choose a new output directory')
    out.mkdir(parents=True)
    # Test pairing never replaces the user's production credentials.
    from model_catalog import resolve_model,model_path
    spec=resolve_model();path=model_path(spec)
    os.environ['ANO_GAMESCALING_MODEL_PATH']=str(path)
    os.environ['ANO_GAMESCALING_DATA']=str(out/'endpoint-data')
    os.environ['ANO_GAMESCALING_PUBLIC_URL']='ws://127.0.0.1:8767/control'
    os.environ['ANO_GAMESCALING_PORT']='8767'
    os.environ['ANO_GAMESCALING_PROMPT_PROFILE']='general'
    import av
    from aiohttp import ClientSession
    from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription, VideoStreamTrack
    import numpy as np
    from PIL import Image,ImageDraw
    import webrtc_live as live
    from sdk import VERSION
    class Video(VideoStreamTrack):
        def __init__(self):super().__init__();self.counter=0
        async def recv(self):
            pts,base=await self.next_timestamp();self.counter+=1
            image=Image.new('RGB',(1280,720),'#f8fafc');draw=ImageDraw.Draw(image)
            draw.text((80,70),'WebRTC plumbing fixture: press W',fill='black',font_size=32)
            x=80+(self.counter*9)%1000;draw.ellipse((x,300,x+60,360),fill='#3865f5')
            frame=av.VideoFrame.from_ndarray(np.array(image),format='rgb24')
            frame.pts=pts;frame.time_base=base;return frame
    class Complete(Exception):pass
    async def check(Session,metrics):
        peer=RTCPeerConnection(RTCConfiguration(iceServers=[]))
        session=Session(lambda frame,controls:None)
        actions=peer.createDataChannel('actions',ordered=True)
        motion=peer.createDataChannel('motion',ordered=False,maxRetransmits=0)
        peer.addTransceiver(Video(),direction='sendonly')
        sid='native-loopback';observed=[];texts=[];ticker=None
        def send(kind,**values):actions.send(json.dumps(dict(v=VERSION,sessionId=sid,type=kind,**values)))
        @actions.on('message')
        def message(raw):
            row=json.loads(raw);observed.append(row)
            if row['type']=='ready':send('validated')
        motion.on('message',message)
        @peer.on('datachannel')
        def channel_ready(channel):
            if channel.label!='text':return
            channel.on('message',lambda raw:texts.append(json.loads(raw)))
            def instruction():channel.send(json.dumps(dict(v=VERSION,sessionId=sid,type='instruction',id=1,text='Press W using the visible controls.')))
            if channel.readyState=='open':instruction()
            else:channel.on('open',instruction)
        async def contexts():
            while True:
                if actions.readyState=='open':
                    now=time.monotonic()*1000
                    send('context',nonce='fixture',taskToken='fixture',generation=1,active=True,
                        nowMs=now,expiresAtMs=now+1000,features=['text'],
                        capabilities=['moveTo','keyDown','keyUp','mouseDown','mouseUp','scroll','releaseAll'])
                await asyncio.sleep(.1)
        try:
            await session.signal(dict(v=VERSION,sessionId=sid,type='hello'))
            await peer.setLocalDescription(await peer.createOffer())
            answer=await session.signal(dict(v=VERSION,sessionId=sid,type='offer',sdp=peer.localDescription.sdp))
            await peer.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'],type='answer'))
            ticker=asyncio.create_task(contexts());started=time.monotonic()
            while metrics.get('images_in_context')!=6 or session.decisions<6:
                if metrics.get('status')=='error':raise RuntimeError(metrics.get('error'))
                if session.closed:raise RuntimeError('Peer closed before six-image decision')
                if time.monotonic()-started>120:raise TimeoutError('Native loopback exceeded 120 seconds')
                await asyncio.sleep(.05)
            async with ClientSession() as http:
                for route in ('/live','/live/state','/live/received.jpg','/live/model.png'):
                    async with http.get('http://127.0.0.1:8767'+route) as response:
                        assert response.status==200,(route,response.status)
                        assert await response.read()
                async with http.get('http://127.0.0.1:8767/live',headers={'Host':'not-local.invalid'}) as response:
                    assert response.status==403
            assert session.validated and metrics['last_action_parsed']
            assert session.instruction and session.instruction['id']==1
            assert session.text_channel and session.text_channel.readyState=='open'
            report={'model':spec.name,'revision':spec.revision,'precision':metrics['precision'],
                'parameter_dtypes':metrics['parameter_dtypes'],'decoded_frames':session.frames,
                'decisions':session.decisions,'images_in_context':metrics['images_in_context'],
                'last_action_parsed':metrics['last_action_parsed'],'last_action_noop':metrics['last_action_noop'],
                'instruction_received':True,'text_channel_open':session.text_channel.readyState=='open',
                'native_text_messages_received':len(texts),'input_envelopes_received':sum(r['type']=='input' for r in observed),
                'input_types_received':sorted({r['action']['type'] for r in observed if r['type']=='input'}),
                'viewer_routes_ok':True,'nonlocal_host_rejected':True,'elapsed_seconds':time.monotonic()-started,
                'scope':'Native model + encoded synthetic video + real loopback WebRTC peers. No real game, browser, WAN or WSS proxy tested.'}
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        finally:
            if ticker:ticker.cancel();await asyncio.gather(ticker,return_exceptions=True)
            await session.close();await peer.close()
        raise Complete()
    try:asyncio.run(live.main(on_ready=check))
    except Complete:pass


if __name__=='__main__':main()
