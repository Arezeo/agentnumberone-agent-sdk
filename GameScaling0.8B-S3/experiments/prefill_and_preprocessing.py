"""Paired FP16 experiments preserving six images and native optional reasoning.

Stop the live endpoint first. CPU image cache requires bitwise tensor parity.
Counterbalanced timing uses greedy sampling/48-token cap solely for replay;
live settings are never changed by this script. Failed candidates stay opt-in.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import os
import resource
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def prompt_for(processor, system, instruction=''):
    action='<|action_start|>0 0 0 ; W ; ; W ; ; W ; <|action_end|>'
    message='current game screen\nCurrent commanded pointer (unverified): [0.5, 0.5]'
    turns=[{'role':'system','content':system}]
    for _ in range(5):
        turns.extend([{'role':'user','content':[{'type':'text','text':message},{'type':'image'}]}, {'role':'assistant','content':action}])
    if instruction:message+='\nOperator instruction: '+instruction
    turns.extend([{'role':'user','content':[{'type':'text','text':message},{'type':'image'}]}, {'role':'assistant','content':'<think>'}])
    return processor.apply_chat_template(turns,tokenize=False,add_generation_prompt=False,continue_final_message=True)


def bootstrap(values, seed=37):
    import numpy as np
    data=np.asarray(values);rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(data,size=(5000,len(data)),replace=True),axis=1)
    return [float(v) for v in np.percentile(means,[2.5,97.5])]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--production-corpus',type=Path,required=True)
    ap.add_argument('--doom-frames',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rounds',type=int,default=2)
    ap.add_argument('--prefill-only',action='store_true',help='Exclude the pixel-cache variant')
    ap.add_argument('--baseline-only-cache',action='store_true',help='Compare corrected pixel cache with the unchanged 2048 baseline')
    ap.add_argument('--blocks',nargs='+',default=['512','1024','2048','4096','8192','none'],choices=['512','1024','2048','4096','8192','none'])
    args=ap.parse_args()
    if args.rounds<1:ap.error('--rounds must be positive')
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mlx_vlm import stream_generate
    from mlx_vlm.utils import prepare_inputs,should_add_special_tokens
    from mlx_runtime import ReuseRuntime
    from preprocess_cache import PreprocessedImages
    from webrtc_live import NativePolicy,SYSTEM,PRECISION,REVISIONS,MODEL_NAME
    from gamescaling import parse_action
    assert PRECISION=='fp16','Experiment requires FP16'
    mx.set_memory_limit(10*1024**3)
    policy=NativePolicy();model,processor=policy.model,policy.processor
    assert policy.parameter_dtypes==['mlx.core.float16']
    production=json.loads(args.production_corpus.read_text())
    paths=[Path(f['path']) for f in production['frames']]
    assert len(paths)>=10 and len(args.doom_frames)>=10
    corpora=[{'name':'production-crossroad','system':SYSTEM,'paths':paths[:10]},
             {'name':'recorded-doom','system':(ROOT/'upstream/prompts/example_doom_battle1.txt').read_text().rstrip('\n'),'paths':args.doom_frames[:10]}]
    for corpus in corpora:
        corpus['images']=[Image.open(p).convert('RGB') for p in corpus['paths']]
        corpus['prompt']=prompt_for(processor,corpus['system'])
    configs=[{'name':'baseline-2048','block':2048,'preprocess':False}, {'name':'preprocess-cache-2048','block':2048,'preprocess':True}]
    configs.extend({'name':'prefill-'+block,'block':None if block=='none' else int(block),'preprocess':False} for block in args.blocks if block!='2048')
    if args.baseline_only_cache:configs=configs[:2]
    if args.prefill_only:configs=[c for c in configs if not c['preprocess']]
    report={'protocol':{'images':6,'history':5,'precision':'fp16','temperature':0.0,'max_tokens':48,'reasoning_prefix':'<think>','rounds':args.rounds,'windows_per_corpus':4,'order':'case-dependent rotation; reversed on alternate rounds','uncertainty':'Exploratory paired and repeated-window-cluster bootstrap; not independent gameplay samples','memory_cap_gib':10,'selection_rule':'No changed greedy outputs, finite first-token logprobs, exact preprocessing tensors, and positive lower bound of paired bootstrap latency reduction. Timing is replay, not gameplay.'},
            'versions':{name:importlib.metadata.version(name) for name in ['mlx','mlx-vlm','transformers','numpy','Pillow']},
            'start_load_average':list(os.getloadavg()),'started_at':time.time(),'model_revision':REVISIONS[MODEL_NAME],'configurations':configs,'tensor_parity':[],'samples':[],'validation':[],'summary':{},'failures':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    def event(value):print(json.dumps(value),flush=True)
    def prepared(images,prompt):
        return prepare_inputs(processor,images=images,prompts=prompt,image_token_index=model.config.image_token_index,
                              add_special_tokens=should_add_special_tokens(model.config.model_type,processor))
    # Strict parity before any timing: every expanded token, mask, grid, and
    # normalized patch value must match the unmodified processor.
    cache=PreprocessedImages()
    for corpus in corpora:
        cache.clear()
        for index in range(5):
            images=corpus['images'][index:index+6]
            raw=prepared(images,corpus['prompt'])
            with cache.install(processor):cached=prepared(images,corpus['prompt'])
            assert set(raw)==set(cached),'Processor keys changed'
            differences={}
            for key in raw:
                a=np.asarray(raw[key]);b=np.asarray(cached[key])
                equal=a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
                differences[key]={'equal':equal,'shape':list(a.shape),'dtype':str(a.dtype)}
                assert equal,'Preprocess tensor mismatch: '+corpus['name']+' '+key
            assert cache.calls==6 and cache.hits+cache.misses==6,'Pixel cache not exercised'
            if index:assert cache.hits>=5,'Historical tensors were not reused'
            row={'corpus':corpus['name'],'window':index,'cache_hits':cache.hits,'images_preprocessed':cache.misses,'fields':differences}
            report['tensor_parity'].append(row)
        event({'phase':'tensor_parity','corpus':corpus['name'],'windows':5,'result':'bitwise_equal'})
    save()
    vectors={}
    def run(config,runtime,corpus,index,temperature=0.0,max_tokens=48,seed=201,instruction=''):
        images=corpus['images'][index:index+6]
        prompt=corpus['prompt'] if not instruction else prompt_for(processor,corpus['system'],instruction)
        mx.random.seed(seed)
        started=time.monotonic()
        options,stages=runtime.prepare(model,processor,prompt,images,corpus['system'])
        assert stages['images_in_context']==6
        first=None;first_vector=None;last=None;text='<think>';token_ids=[]
        gen=stream_generate(model,processor,prompt,image=None,temperature=temperature,max_tokens=max_tokens,
                            skip_special_tokens=False,prefill_step_size=config['block'],**options)
        try:
            for chunk in gen:
                if first is None:
                    first=(time.monotonic()-started)*1000
                    first_vector=np.asarray(chunk.logprobs,dtype=np.float32).copy()
                    assert np.isfinite(first_vector).all(),'Non-finite token logprobs'
                text+=chunk.text;last=chunk
                if chunk.token is not None:token_ids.append(int(chunk.token))
                if '<|action_end|>' in text and '</think>' in text:break
                if time.monotonic()-started>60:raise RuntimeError('Replay exceeded 60 seconds')
        finally:gen.close()
        elapsed=(time.monotonic()-started)*1000
        parsed=parse_action(text)
        return {'variant':config['name'],'corpus':corpus['name'],'window':index,'elapsed_ms':elapsed,'first_chunk_ms':first,'after_first_ms':elapsed-first,
                'generation_tokens':getattr(last,'generation_tokens',None),'output':text,'token_ids':token_ids,'parsed':parsed.parsed,
                'reasoning_chars':len(text.split('</think>')[0].removeprefix('<think>').strip()),'peak_mlx_bytes':mx.get_peak_memory(),**stages},first_vector
    for corpus in corpora:
        runtimes={c['name']:ReuseRuntime(preprocess=c['preprocess']) for c in configs}
        # Every block gets an excluded complete generation warm-up.
        for config in configs:
            try:run(config,runtimes[config['name']],corpus,0)
            except Exception as exc:
                report['failures'].append({'variant':config['name'],'corpus':corpus['name'],'phase':'warmup','error':str(exc)})
                event(report['failures'][-1])
        event({'phase':'warmup_done','corpus':corpus['name']})
        for round_index in range(args.rounds):
            for config in configs:
                runtimes[config['name']].clear()
                runtimes[config['name']].prepare(model,processor,corpus['prompt'],corpus['images'][:6],corpus['system'])
            for index in range(1,5):
                shift=(index+round_index)%len(configs)
                order=configs[shift:]+configs[:shift]
                if round_index%2:order=list(reversed(order))
                for config in order:
                    if any(f['variant']==config['name'] and f['corpus']==corpus['name'] for f in report['failures']):continue
                    try:
                        row,vector=run(config,runtimes[config['name']],corpus,index)
                        row['round']=round_index;row['order_index']=order.index(config)
                        report['samples'].append(row)
                        vectors[(corpus['name'],round_index,index,config['name'])]=vector
                        event({'phase':'timing','variant':config['name'],'corpus':corpus['name'],'round':round_index,'window':index,'ms':round(row['elapsed_ms'],1),'tokens':row['generation_tokens'],'preprocess_hits':row['preprocess_cache_hits']})
                    except Exception as exc:
                        report['failures'].append({'variant':config['name'],'corpus':corpus['name'],'phase':'timing','error':str(exc)})
                        event(report['failures'][-1])
                    save()
    baselines={(r['corpus'],r['round'],r['window']):r for r in report['samples'] if r['variant']=='baseline-2048'}
    for config in configs:
        rows=[r for r in report['samples'] if r['variant']==config['name']]
        comparisons=[]
        for row in rows:
            key=(row['corpus'],row['round'],row['window']);base=baselines[key]
            a=vectors[(*key,'baseline-2048')];b=vectors[(*key,config['name'])]
            comparisons.append({'corpus':key[0],'round':key[1],'window':key[2],
                                'reduction_percent':100*(1-row['elapsed_ms']/base['elapsed_ms']),
                                'identical_output':row['output']==base['output'],'identical_token_ids':row['token_ids']==base['token_ids'],
                                'same_parse_status':row['parsed']==base['parsed'],
                                'logprob_relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                                'logprob_max_abs':float(np.max(np.abs(a-b)))})
        if not rows:continue
        deltas=[r['reduction_percent'] for r in comparisons]
        ci=bootstrap(deltas)
        clusters=defaultdict(list)
        for comparison in comparisons:clusters[(comparison['corpus'],comparison['window'])].append(comparison['reduction_percent'])
        cluster_ci=bootstrap([statistics.mean(v) for v in clusters.values()])
        report['summary'][config['name']]={'unique_windows':len(clusters),'window_cluster_bootstrap_95ci_percent':cluster_ci,'n':len(rows),'mean_ms':statistics.mean(r['elapsed_ms'] for r in rows),
            'median_ms':statistics.median(r['elapsed_ms'] for r in rows),'mean_first_chunk_ms':statistics.mean(r['first_chunk_ms'] for r in rows),
            'mean_paired_reduction_percent':statistics.mean(deltas),'paired_reduction_bootstrap_95ci_percent':ci,
            'output_matches':sum(r['identical_output'] for r in comparisons),'token_matches':sum(r['identical_token_ids'] for r in comparisons),
            'max_logprob_relative_l2':max(r['logprob_relative_l2'] for r in comparisons),
            'candidate_pass':ci[0]>0 and cluster_ci[0]>0 and all(r['identical_output'] and r['same_parse_status'] for r in comparisons),
            'comparisons':comparisons}
    report['end_load_average']=list(os.getloadavg())
    report['ended_at']=time.time()
    report['process_cpu_seconds']=resource.getrusage(resource.RUSAGE_SELF).ru_utime
    save();event({'phase':'summary','results':{k:{name:v for name,v in row.items() if name!='comparisons'} for k,row in report['summary'].items()}})


if __name__=='__main__':main()
