"""Paired native 1.0-temperature/2048-token tests of pixel-tensor reuse.

Only run with the live endpoint stopped. Operator prompts are identical per pair;
the checkpoint chooses whether/how much to reason. No forced empty thinking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_and_preprocessing import prompt_for


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--production-corpus',type=Path,required=True)
    ap.add_argument('--doom-frames',type=Path,nargs='+',required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[17,43])
    ap.add_argument('--instruction',default='Briefly explain what you see and your next move, then act.')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    import mlx.core as mx
    from PIL import Image
    from mlx_runtime import ReuseRuntime
    import webrtc_live as runner
    mx.set_memory_limit(10*1024**3)
    policy=runner.NativePolicy()
    corpus=json.loads(args.production_corpus.read_text())
    corpora=[('production-crossroad',[Path(f['path']) for f in corpus['frames']],runner.SYSTEM),
             ('recorded-doom',args.doom_frames,(ROOT/'upstream/prompts/example_doom_battle1.txt').read_text().rstrip('\n'))]
    saved_system,saved_out=runner.SYSTEM,runner.OUT
    runner.OUT=args.output.parent/'native-validation-input';runner.OUT.mkdir(exist_ok=True,parents=True)
    report={'precision':'fp16','images':6,'history':5,'temperature':1.0,'max_tokens':2048,'prefill_step_size':2048,'pairs':[],'all_outputs_match':True,'reasoning_seen':False}
    action='<|action_start|>0 0 0 ; W ; ; W ; ; W ; <|action_end|>'
    message='current game screen\nCurrent commanded pointer (unverified): [0.5, 0.5]'
    try:
        for corpus_name,paths,system in corpora:
            images=[Image.open(p).convert('RGB') for p in paths[:7]]
            runner.SYSTEM=system
            instruction=args.instruction
            for seed in args.seeds:
                variants=[]
                order=[False,True] if args.seeds.index(seed)%2==0 else [True,False]
                for enabled in order:
                    policy.runtime=ReuseRuntime(preprocess=enabled)
                    policy.runtime.prepare(policy.model,policy.processor,prompt_for(policy.processor,system),images[:6],system)
                    policy.identity=('validation',corpus_name)
                    policy.history=[(image,message,action) for image in images[1:6]]
                    mx.random.seed(seed)
                    text_events=[]
                    start=time.monotonic()
                    result,timing,_=policy.infer(images[6],('validation',corpus_name,1,0),instruction,[.5,.5],lambda text,final:text_events.append((text,final)))
                    reasoning=timing['reasoning'].strip()
                    report['reasoning_seen'] |= bool(reasoning)
                    row={'preprocess':enabled,'elapsed_ms':(time.monotonic()-start)*1000,'native_output':timing['native_output'],
                         'output_sha256':hashlib.sha256(timing['native_output'].encode()).hexdigest(),'parsed':result.parsed,
                         'reasoning_chars':len(reasoning),'final_text_event':bool(text_events and text_events[-1][1]),
                         'images_in_context':timing['images_in_context'],'history_frames':timing['history_frames'],
                         'preprocess_cache_hits':timing['preprocess_cache_hits'],'images_preprocessed':timing['images_preprocessed'],
                         'generation_tokens':timing.get('generation_tokens')}
                    assert row['images_in_context']==6 and row['history_frames']==5
                    if enabled:assert row['preprocess_cache_hits']>=5
                    if reasoning:assert row['final_text_event'],'Native text was not emitted'
                    variants.append(row)
                    print(json.dumps({'corpus':corpus_name,'seed':seed,**{k:v for k,v in row.items() if k!='native_output'}}),flush=True)
                identical=variants[0]['native_output']==variants[1]['native_output']
                report['all_outputs_match'] &= identical
                report['pairs'].append({'corpus':corpus_name,'seed':seed,'identical_output':identical,'variants':variants})
                args.output.write_text(json.dumps(report,indent=2)+'\n')
    finally:runner.SYSTEM=saved_system;runner.OUT=saved_out
    assert report['all_outputs_match'],'Seeded native outputs differ; do not enable by default'
    assert report['reasoning_seen'],'Optional reasoning was not exercised; no conversational validation claim'
    print(json.dumps({'native_validation':'passed','pairs':len(report['pairs']),'reasoning_seen':report['reasoning_seen']}),flush=True)


if __name__=='__main__':main()
