"""Real FP16 model/image/native-action smoke check, not a gameplay evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from PIL import Image,ImageDraw
    from webrtc_live import NativePolicy,MODEL_SPEC
    if args.image:image=Image.open(args.image).convert('RGB')
    else:
        image=Image.new('RGB',(1280,720),'#f8fafc');draw=ImageDraw.Draw(image)
        draw.rectangle((300,200,900,500),outline='#3865f5',width=8)
        draw.text((440,320),'Transport fixture: press W',fill='#101114',font_size=32)
    policy=NativePolicy()
    action='<|action_start|>0 0 0 ; ; ; ; ; ; <|action_end|>'
    identity=('offline-smoke','fixture',1,0)
    policy.identity=identity[:2]
    policy.history=[(image.copy(),'current game screen',action) for _ in range(5)]
    emitted=[];started=time.monotonic()
    result,timing,_=policy.infer(image,identity,'Use the visible controls.',[.5,.5],
        lambda text,final:emitted.append({'chars':len(text),'final':final}))
    errors=[]
    if policy.parameter_dtypes!=['mlx.core.float16']:errors.append('Parameter dtype differs from FP16')
    if timing['images_in_context']!=6 or timing['history_frames']!=5:errors.append('Expected six images')
    if not result.parsed:errors.append('Native action did not parse')
    report={'model':MODEL_SPEC.name,'revision':MODEL_SPEC.revision,'precision':'fp16',
        'parameter_dtypes':policy.parameter_dtypes,'parsed':result.parsed,'noop':result.is_noop,'errors':errors,
        'native_output':timing['native_output'],'text_callbacks':emitted,
        'timing':{k:v for k,v in timing.items() if k not in ('native_output','input_message','reasoning')},
        'wall_ms':(time.monotonic()-started)*1000,
        'source':'user image' if args.image else 'synthetic plumbing fixture',
        'source_pixels_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
        'scope':'One real-model six-image FP16 call. No action applied; not game skill or sustained FPS.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('model','precision','parsed','noop','wall_ms','scope')}))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
