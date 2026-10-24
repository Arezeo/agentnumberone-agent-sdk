"""Profile six unchanged images with and without reuse; verify last-token logits.

Run with the live model stopped, not alongside it. No training, game-state API,
quantization, history shortening, or forced empty reasoning is used. The bounded
latency microbenchmark uses greedy sampling and 48 tokens for comparability.
"""
import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args=parser.parse_args()
    if len(args.frames)<6:parser.error('Supply at least six frames')
    import mlx.core as mx
    from mlx_vlm import stream_generate
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
    from PIL import Image
    from mlx_runtime import ReuseRuntime
    from webrtc_live import NativePolicy, SYSTEM, MODEL, MODEL_SPEC, PRECISION, PREFILL_STEP_SIZE
    policy=NativePolicy()
    all_images=[Image.open(p).convert('RGB') for p in args.frames]
    images=all_images[:6]
    action='<|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>'
    messages=[{'role':'system','content':SYSTEM}]
    for image in images[:-1]:
        messages.extend([{'role':'user','content':[{'type':'text','text':'current game screen'},{'type':'image'}]},{'role':'assistant','content':action}])
    messages.extend([{'role':'user','content':[{'type':'text','text':'current game screen'},{'type':'image'}]},{'role':'assistant','content':'<think>'}])
    prompt=policy.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,continue_final_message=True)
    model,processor=policy.model,policy.processor
    full=prepare_inputs(processor,images=images,prompts=prompt,image_token_index=model.config.image_token_index,add_special_tokens=should_add_special_tokens(model.config.model_type,processor))
    def last_logits(options):
        options=dict(options)
        ids=options.pop('input_ids');pixels=options.pop('pixel_values',None)
        mask=options.pop('attention_mask',options.pop('mask',None))
        cache=options.pop('prompt_cache',None)
        if cache is None:cache=make_prompt_cache(model.language_model)
        explicit={k:options[k] for k in ('position_ids','rope_deltas') if options.get(k) is not None}
        embedded=model.get_input_embeddings(ids,pixels,mask=mask,**options)
        metadata={k:v for k,v in embedded.to_dict().items() if k!='inputs_embeds' and v is not None}
        metadata.update(explicit)
        output=model.language_model(ids,inputs_embeds=embedded.inputs_embeds,cache=cache,logits_to_keep=1,**metadata)
        result=output.logits[:,-1,:].astype(mx.float32);mx.eval(result)
        return result
    cold=last_logits(full)
    runtime=ReuseRuntime(preprocess=True)
    parity=[]
    # Repeat restoration to prove generation didn't mutate the saved prefix.
    for iteration in range(2):
        options,stages=runtime.prepare(model,processor,prompt,images,SYSTEM)
        warm=last_logits(options)
        diff=warm-cold
        relative=mx.sqrt(mx.sum(diff*diff)/mx.sum(cold*cold)).item()
        row={'iteration':iteration,'max_abs_diff':mx.max(mx.abs(diff)).item(),'relative_l2':relative,'same_top_token':mx.argmax(cold).item()==mx.argmax(warm).item(),**stages}
        assert relative < .003, 'Cached logits diverged: '+str(row)
        assert stages['images_in_context']==6
        assert stages['system_prefix_tokens']>0, 'System prefix not reused'
        if iteration:assert stages['vision_cache_hits']==6 and stages['system_prefix_hit']
        parity.append(row)
        print(json.dumps({'kind':'parity',**row}),flush=True)
    # Also verify a shifted multimodal history: restore only the system prefix,
    # and supply all six newly ordered image features and full mRoPE positions.
    shifted=all_images[1:7] if len(all_images)>6 else images[1:]+images[:1]
    shifted_full=prepare_inputs(processor,images=shifted,prompts=prompt,image_token_index=model.config.image_token_index,add_special_tokens=should_add_special_tokens(model.config.model_type,processor))
    shifted_cold=last_logits(shifted_full)
    shifted_options,shifted_stages=runtime.prepare(model,processor,prompt,shifted,SYSTEM)
    shifted_warm=last_logits(shifted_options)
    diff=shifted_warm-shifted_cold
    relative=mx.sqrt(mx.sum(diff*diff)/mx.sum(shifted_cold*shifted_cold)).item()
    shifted_parity={'relative_l2':relative,'max_abs_diff':mx.max(mx.abs(diff)).item(),'same_top_token':mx.argmax(shifted_cold).item()==mx.argmax(shifted_warm).item(),**shifted_stages}
    assert relative<.003,'Shifted history diverged'
    rows=[]
    variants=[('baseline',False,False,False),('vision',True,False,False),
              ('vision_and_system_prefix',True,True,False),('all_reuse',True,True,True)]
    for name,vision,prefix,preprocess in variants:
        runtime=ReuseRuntime(vision=vision,system_prefix=prefix,preprocess=preprocess)
        for repeat in range(args.repeats+1):
            start=time.monotonic()
            window=all_images[repeat:repeat+6] if len(all_images)>=args.repeats+6 else images
            options,stages=runtime.prepare(model,processor,prompt,window,SYSTEM)
            first=None;last=None;text=''
            gen=stream_generate(model,processor,prompt,image=None,max_tokens=48,temperature=0.0,skip_special_tokens=False,prefill_step_size=PREFILL_STEP_SIZE,**options)
            for chunk in gen:
                if first is None:first=(time.monotonic()-start)*1000
                last=chunk;text+=chunk.text
            elapsed=(time.monotonic()-start)*1000
            row={'variant':name,'warmup':repeat==0,'window':repeat if len(all_images)>=args.repeats+6 else 0,'elapsed_ms':elapsed,'first_chunk_ms':first,'generation_tokens':getattr(last,'generation_tokens',None),'output':text,**stages}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='output'}),flush=True)
    summary={}
    for name,_,_,_ in variants:
        subset=[r for r in rows if r['variant']==name and not r['warmup']]
        summary[name]={'n':len(subset),'mean_ms':statistics.mean(r['elapsed_ms'] for r in subset),'first_chunk_mean_ms':statistics.mean(r['first_chunk_ms'] for r in subset)}
    baseline_outputs={r['window']:r['output'] for r in rows if r['variant']=='baseline'}
    for row in rows:row['output_matches_baseline']=row['output']==baseline_outputs[row['window']]
    report={'model':MODEL_SPEC.name,'revision':MODEL_SPEC.revision,'precision':PRECISION,'parameter_dtypes':policy.parameter_dtypes,'images':6,'history':5,'source_sizes':[im.size for im in images],'peak_mlx_gib':mx.get_peak_memory()/1024**3,'benchmark_sampling':{'temperature':0.0,'max_tokens':48,'prefill_step_size':PREFILL_STEP_SIZE,'reasoning_prefill':'<think>','live_settings_changed':False},'parity':parity,'shifted_history_parity':shifted_parity,'summary':summary,'samples':rows,'sliding_windows':len(all_images)>=args.repeats+6,'note':'With at least repeats+6 frames, each measured window advances one frame, preserving six images while reusing five historical features. Six-frame mode repeats fixed images. This is not a gameplay score.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('SUMMARY '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
