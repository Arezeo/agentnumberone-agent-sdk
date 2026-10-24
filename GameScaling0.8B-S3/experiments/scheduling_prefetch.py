"""Offline scheduling experiment; never installed into the live runner.

Measures a full six-image A decision plus preparation/encoding of a selected
image B. Conditional hit/miss fixtures bound usefulness of a candidate C. They
are not live frame forecasts or decision-rate benchmarks. Every discarded GPU
job is drained and counted, and delay to A's action output is reported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_and_preprocessing import prompt_for,bootstrap
from prefetch_admission import Observation,admit


def pixels(image):return hashlib.sha256(image.tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--production-corpus',type=Path,required=True)
    ap.add_argument('--doom-frames',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rounds',type=int,default=4)
    args=ap.parse_args()
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mlx_vlm import stream_generate
    from mlx_runtime import ReuseRuntime
    import webrtc_live as runner
    assert runner.PRECISION=='fp16'
    mx.set_memory_limit(10*1024**3)
    policy=runner.NativePolicy();model,processor=policy.model,policy.processor
    stream=mx.new_stream(mx.gpu)  # Created and used on the one model-owner thread.
    clone=copy.deepcopy(processor.image_processor)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pixel-only-prefetch')
    paths=[Path(row['path']) for row in json.loads(args.production_corpus.read_text())['frames']]
    corpora=[('production-crossroad',paths,runner.SYSTEM),('recorded-doom',args.doom_frames,(ROOT/'upstream/prompts/example_doom_battle1.txt').read_text().rstrip('\n'))]
    report={'protocol':{'precision':'fp16','images_in_A':6,'history_in_A':5,'temperature':0.0,'max_tokens':48,'reasoning_prefix':'<think>','prefill_step_size':2048,'rounds':args.rounds,'variants':['serial','cpu-prefetch','gpu-prefetch'],'scenarios':['conditional-exact-hit','changed-pixels-miss'],'timing':'A decision plus selected-B image feature readiness; B language inference is not measured','first_action_metric':'A output completion; a throughput benefit cannot excuse delaying A','source_timing':'Explicit causal admission fixtures, not a live scheduling implementation','cpu_worker':'Only NumPy/PIL pixel processing; one cloned processor, no MLX calls','gpu_owner':'One model-owner thread; speculative vision submitted on a separate MLX GPU stream','drain':'All pending candidate GPU work is evaluated before completion, including discarded work'},'samples':[],'summary':{},'failures':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        clean={**report,'samples':[{k:v for k,v in row.items() if k!='_features'} for row in report['samples']]}
        args.output.write_text(json.dumps(clean,indent=2)+'\n')
    def event(row):print(json.dumps(row),flush=True)
    def cpu_prepare(image):
        started=time.monotonic()
        value=clone(images=[image])
        assert set(value)=={'pixel_values','image_grid_thw'}
        assert all(isinstance(v,np.ndarray) for v in value.values()),'CPU worker produced a device array'
        return value,started,time.monotonic()
    def encode(value,asynchronous=False):
        context=mx.stream(stream) if asynchronous else mx.stream(mx.default_stream(mx.gpu))
        with context:
            part,deep=model.vision_tower(mx.array(value['pixel_values']).astype(model.vision_tower.patch_embed.proj.weight.dtype),mx.array(value['image_grid_thw']))
            assert deep is None or len(deep)==0
            if asynchronous:mx.async_eval(part)
            else:mx.eval(part)
        return part
    feature_reference={};output_reference={}
    try:
        for corpus_name,paths,system in corpora:
            assert len(paths)>=10
            images=[Image.open(p).convert('RGB') for p in paths[:10]]
            prompt=prompt_for(processor,system)
            seed_runtime=ReuseRuntime(preprocess=True)
            seed_runtime.prepare(model,processor,prompt,images[:6],system)
            saved_features=dict(seed_runtime.features.entries)
            saved_pixels=dict(seed_runtime.pixels.entries)
            saved_ids=seed_runtime.prefix.ids
            saved_prefix=seed_runtime.prefix.snapshots
            def runtime():
                value=ReuseRuntime(preprocess=True)
                value.features.entries=dict(saved_features)
                value.pixels.entries=dict(saved_pixels)
                value.prefix.ids=saved_ids
                value.prefix.snapshots=saved_prefix
                return value
            # Excluded warm-up for GPU queues, clone, and every variant.
            baseline_value,_,_=cpu_prepare(images[7]);encode(baseline_value)
            future=pool.submit(cpu_prepare,images[8]);encode(future.result()[0],True);mx.synchronize(stream)
            variants=['serial','cpu-prefetch','gpu-prefetch']
            for round_index in range(args.rounds+1):
                for scenario_index,scenario in enumerate(['conditional-exact-hit','changed-pixels-miss']):
                    order=variants[round_index%3:]+variants[:round_index%3]
                    if (round_index+scenario_index)%2:order=list(reversed(order))
                    for variant in order:
                        actual=runtime()
                        candidate=images[7] if scenario_index==0 else images[8]
                        selected=images[7]
                        # The candidate source precedes an action boundary; the
                        # selected observation follows it and retains its own ID.
                        ticket=Observation(10,1.,('replay',1),pixels(candidate),'pinned-processor')
                        observation=Observation(11,2.1,('replay',1),pixels(selected),'pinned-processor')
                        allowed,reason=admit(ticket,observation,current_context=('replay',1),action_started_at=2.)
                        assert allowed==(scenario_index==0)
                        mx.random.seed(17)
                        start=time.monotonic();future=None;prepared=None;speculative=None;submitted=None
                        options,stages=actual.prepare(model,processor,prompt,images[1:7],system)
                        assert stages['images_in_context']==6 and stages['frames_encoded']==1
                        text='<think>';first=None;last=None;cpu_times=None
                        gen=stream_generate(model,processor,prompt,image=None,max_tokens=48,temperature=0.,skip_special_tokens=False,prefill_step_size=2048,**options)
                        try:
                            for chunk in gen:
                                if first is None:
                                    first=(time.monotonic()-start)*1000
                                    if variant!='serial':future=pool.submit(cpu_prepare,candidate)
                                text+=chunk.text;last=chunk
                                if variant=='gpu-prefetch' and speculative is None and future is not None and future.done():
                                    prepared,cpu_begin,cpu_end=future.result();cpu_times=(cpu_begin,cpu_end)
                                    submitted=time.monotonic();speculative=encode(prepared,True)
                                if '<|action_end|>' in text and '</think>' in text:break
                        finally:gen.close()
                        action_ready=time.monotonic()
                        if future is not None and prepared is None:
                            prepared,cpu_begin,cpu_end=future.result();cpu_times=(cpu_begin,cpu_end)
                        # Count all work, including speculation that misses.
                        if speculative is not None:mx.eval(speculative);mx.synchronize(stream)
                        if variant=='serial' or not allowed:
                            selected_value,_,_=cpu_prepare(selected)
                            selected_features=encode(selected_value)
                        elif speculative is not None:selected_features=speculative
                        else:selected_features=encode(prepared)
                        mx.eval(selected_features)
                        done=time.monotonic()
                        selected_array=np.asarray(selected_features).copy()
                        key=(corpus_name,scenario,round_index)
                        if variant=='serial':
                            feature_reference[key]=selected_array
                            output_reference[key]=text
                        row={'corpus':corpus_name,'scenario':scenario,'round':round_index,'warmup':round_index==0,'variant':variant,'total_ms':(done-start)*1000,'action_ready_ms':(action_ready-start)*1000,'first_chunk_ms':first,'generation_tokens':getattr(last,'generation_tokens',None),'native_output':text,'admitted':allowed,'admission_reason':reason,'candidate_source_id':ticket.frame_id,'selected_source_id':observation.frame_id,'discarded_candidates':int(variant!='serial' and not allowed),'candidate_cpu_overlaps_A':bool(cpu_times and cpu_times[0]<action_ready and cpu_times[1]>start),'gpu_submitted_before_A_output':bool(submitted and submitted<action_ready),'peak_mlx_bytes':mx.get_peak_memory(),'images_in_context':stages['images_in_context']}
                        # Keep vectors privately in process until paired refs exist.
                        row['_features']=selected_array
                        report['samples'].append(row)
                        event({k:v for k,v in row.items() if k not in ('native_output','_features')})
                        save()
            # Validate against matched serial outputs after counterbalanced order.
            for row in report['samples']:
                if row['corpus']!=corpus_name:continue
                key=(corpus_name,row['scenario'],row['round'])
                row['identical_output']=row['native_output']==output_reference[key]
                row['identical_B_features']=np.array_equal(row.pop('_features'),feature_reference[key])
                assert row['identical_output'] and row['identical_B_features'],'Scheduling changed a paired result'
            save()
        for scenario in ['conditional-exact-hit','changed-pixels-miss']:
            rows=[r for r in report['samples'] if r['scenario']==scenario and not r['warmup']]
            baselines={(r['corpus'],r['round']):r for r in rows if r['variant']=='serial'}
            report['summary'][scenario]={}
            for variant in ['serial','cpu-prefetch','gpu-prefetch']:
                subset=[r for r in rows if r['variant']==variant]
                gains=[100*(1-r['total_ms']/baselines[(r['corpus'],r['round'])]['total_ms']) for r in subset]
                action_deltas=[r['action_ready_ms']-baselines[(r['corpus'],r['round'])]['action_ready_ms'] for r in subset]
                report['summary'][scenario][variant]={'n':len(subset),'mean_total_ms':statistics.mean(r['total_ms'] for r in subset),'mean_action_ready_ms':statistics.mean(r['action_ready_ms'] for r in subset),'mean_paired_total_reduction_percent':statistics.mean(gains),'paired_total_reduction_bootstrap_95ci_percent':bootstrap(gains),'mean_first_action_delay_ms':statistics.mean(action_deltas),'paired_first_action_delay_bootstrap_95ci_ms':bootstrap(action_deltas),'output_matches':sum(r['identical_output'] for r in subset),'feature_matches':sum(r['identical_B_features'] for r in subset)}
        report['default_changed']=False
        report['limitation']='Conditional hit/miss compute experiment; separate GPU submission does not itself prove kernel overlap, and A+B image-feature work is not a measured B decision or gameplay rollout.'
        save();event({'phase':'summary','results':report['summary']})
    except Exception as error:
        report['failures'].append({'type':type(error).__name__,'message':str(error)})
        save()
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        mx.synchronize(stream)


if __name__=='__main__':main()
