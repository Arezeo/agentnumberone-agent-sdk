"""Native GameScaling model -> official action parser -> public WebRTC controls.
No learned Gemma action head, scripted solver, or hidden game state.
"""
import asyncio,json,sys,time,re,io,os,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'))
sys.path.insert(0,str(ROOT.parent/'scripts/remote-control'))
import agent as endpoint
from sdk import Controls
from aiohttp import web
from connection import pairing_token
from mlx_runtime import ReuseRuntime
from action_chunk import execute_chunk
from model_catalog import resolve_model, data_dir, model_path, validate_config
from gamescaling import build_system_prompt,parse_action
from gamescaling.keymaps import MOUSE_BUTTONS
from gamescaling.action import split_reasoning
MODEL_SPEC=resolve_model()
MODEL_NAME=MODEL_SPEC.name
REVISIONS={MODEL_NAME:MODEL_SPEC.revision}
DATA=data_dir()
OUT=DATA/'results'/('production-'+MODEL_SPEC.key)
PRECISION=os.environ.get('ANO_GAMESCALING_PRECISION','fp16')
if PRECISION not in ('bf16','fp16'):raise ValueError('Use bf16 or fp16; this example does not quantize weights')
MODEL=model_path(MODEL_SPEC)
RUN_ID=str(uuid.uuid4())
BACKEND='local MLX '+PRECISION.upper()+' (unquantized)'
PORT=int(os.environ.get('ANO_GAMESCALING_PORT','8767'))
ORIGIN=endpoint.origin(os.environ.get('ANO_GAMESCALING_ORIGIN','https://agentnumberone.io'))
PUBLIC_URL=os.environ.get('ANO_GAMESCALING_PUBLIC_URL','ws://127.0.0.1:'+str(PORT)+'/control')
VISION_REUSE=os.environ.get('ANO_GAMESCALING_VISION_REUSE','1')=='1'
PREFIX_REUSE=os.environ.get('ANO_GAMESCALING_PREFIX_REUSE','1')=='1'
PREPROCESS_REUSE=os.environ.get('ANO_GAMESCALING_PREPROCESS_REUSE','1')=='1'
PREFILL_STEP_SIZE=2048  # Paired experiments did not support changing this default.
OVERLAP=os.environ.get('ANO_GAMESCALING_OVERLAP','1')=='1'
SYSTEM=build_system_prompt('AgentNumberOne screen-controlled challenge or game',rules='Follow the task, rules, player assignment and controls visible on the screen and the operator instructions. In Pixel Duel use WASD to move, mouse to aim, LB to attack, Space to dash. Challenge tasks may require pointer positioning, clicks, held buttons, dragging or keyboard input. Mouse X/Y are relative screen displacements in per-mille units, applied to the current commanded pointer; they are not absolute coordinates. Your actions are executed for 200ms. Use observations to check progress. No hidden game state is available.',key_bindings={'W':'move up','A':'move left','S':'move down','D':'move right','LB':'left mouse button / attack','RB':'right mouse button','Space':'dash or task-specific space key','Shift':'hold shift','Ctrl':'hold control','Enter':'enter','Back':'backspace','Mouse X Y':'move pointer relatively (X>0 right, Y>0 down); units are 1/1000 of screen width/height'},mouse_aim=False)
# Preserve upstream sections exactly; all environment-specific text belongs in
# GAME_RULES and KEY_BINDINGS, as the published template instructs.
PROMPT_TEMPLATE=(ROOT/'upstream/prompts/system_prompt_template.txt').read_text().rstrip('\n')
_prompt_rules=SYSTEM.split('**Game Rules**\n',1)[1].split('\n**Common Keys in ',1)[0]
_prompt_keys=SYSTEM.split('**Common Keys in AgentNumberOne screen-controlled challenge or game**\n',1)[1].split('\n**Output Format**',1)[0]
assert SYSTEM==PROMPT_TEMPLATE.replace('{GAME_NAME}','AgentNumberOne screen-controlled challenge or game').replace('{GAME_RULES}',_prompt_rules).replace('{KEY_BINDINGS}',_prompt_keys)

PROMPT_PROFILE=os.environ.get('ANO_GAMESCALING_PROMPT_PROFILE','general')
if PROMPT_PROFILE=='pixel-duel':
 SYSTEM=(ROOT/'prompts/pixel-duel-system.txt').read_text().rstrip('\n')
elif PROMPT_PROFILE=='crossroad':
 SYSTEM=build_system_prompt('Crossroad',rules='You control the white chicken in a fixed third-person view. Increase the visible Score by hopping forward across lanes. Avoid moving cars and tree tiles; collect visible power-ups. When status says Cross the lanes, the game is already running: use movement keys, not Pause or Restart. The mouse does not steer the chicken or the camera. Each movement key press makes one tile hop, not continuous running. Release a movement key in an empty key group before pressing it again for another hop. Use actual movement key tokens in your six action groups when you intend to move. A collision ends the round; click Play again or Restart only when Round over is visible. Mouse X/Y are relative per-mille screen displacements from the current commanded pointer, not absolute coordinates. Follow operator instructions and visible feedback. No hidden game state is available.',key_bindings={'W':'one hop forward','A':'one hop left','S':'one hop backward','D':'one hop right','Up':'one hop forward','Left':'one hop left','Down':'one hop backward','Right':'one hop right','LB':'click a visible button, only when needed','Mouse X Y':'move pointer relatively; 1/1000 screen width/height; no camera control'},mouse_aim=False)
elif PROMPT_PROFILE=='doom':
 SYSTEM=(ROOT/'upstream/prompts/example_doom_battle1.txt').read_text().rstrip('\n')
elif PROMPT_PROFILE!='general':raise ValueError('Unsupported prompt profile')



def perception_ready(validated,context,latest,last,now=None):
 now=time.monotonic() if now is None else now
 return bool(validated and context and latest and latest[0]!=last and
             latest[3]==context.identity and now-latest[2]<2 and
             context.host_now(now)<context.message['expiresAtMs'])

def frame_is_black(image):
 # Reject an empty capture, not ordinary dark scenes with visible content.
 return max(high for low,high in image.convert('RGB').getextrema())<=2

def frame_image(frame):
 # VideoFrame.to_image() caches a mutable reformatter on the frame. The live
 # preview and inference can use that frame concurrently; keep conversion
 # private and single-threaded instead of sharing FFmpeg scaling workers.
 from PIL import Image
 if isinstance(frame,Image.Image):return frame.convert('RGB')
 from av.video.reformatter import VideoReformatter
 return VideoReformatter().reformat(frame,format='rgb24',threads=1).to_image()

def action_admissible(observed_active,observed_identity,current_identity,context,received,now=None):
 now=time.monotonic() if now is None else now
 return bool(observed_active and context and context.active(now) and
             observed_identity==current_identity and now-received<15)

def physical_code(token):
 if len(token)==1 and 'A'<=token<='Z':return 'Key'+token
 digits=['zero','one','two','three','four','five','six','seven','eight','nine']
 if token in digits:return 'Digit'+str(digits.index(token))
 return {'Up':'ArrowUp','Down':'ArrowDown','Left':'ArrowLeft','Right':'ArrowRight','Shift':'ShiftLeft','Ctrl':'ControlLeft','Alt':'AltLeft','Esc':'Escape','Back':'Backspace','Space':'Space','Enter':'Enter','Tab':'Tab','Delete':'Delete','Home':'Home','End':'End','Period':'Period','Slash':'Slash','Quote':'Quote'}.get(token)

def desired(tokens):
 keys=set();buttons=set();unsupported=[]
 for token in tokens:
  if token in MOUSE_BUTTONS:buttons.add(MOUSE_BUTTONS[token])
  elif physical_code(token):keys.add(physical_code(token))
  else:unsupported.append(token)
 return keys,buttons,unsupported

def held_inputs_to_send(requested,previous,first_slot):
 # Reassert on a fresh chunk: browser's 2s lease may have expired during inference.
 # Later slots only send edges; idempotent downs do not synthesize key presses.
 return requested if first_slot else requested-previous

def visible_reasoning(text):
 # Display native generated reasoning only; never the executable action tail.
 reasoning,_,closed=split_reasoning(text)
 if not closed and '<' in reasoning and '>' not in reasoning[reasoning.rfind('<'):]:
  reasoning=reasoning[:reasoning.rfind('<')]
 return re.sub(r'<\|[^>]*\|>', '', reasoning).strip()

class NativePolicy:
 def __init__(self):
  OUT.mkdir(parents=True,exist_ok=True)
  from mlx_vlm import load
  import mlx.core as mx
  mx.set_cache_limit(256*1024*1024)
  config=json.loads((MODEL/'config.json').read_text())
  validate_config(config,MODEL_SPEC)
  self.model,self.processor=load(str(MODEL),lazy=True)
  dtype=mx.float16 if PRECISION=='fp16' else mx.bfloat16
  self.model.set_dtype(dtype);mx.eval(self.model.parameters())
  from mlx.utils import tree_flatten
  parameters=tree_flatten(self.model.parameters())
  assert not any(value.dtype==mx.uint32 for _,value in parameters),'Packed weights found'
  assert all(value.dtype==dtype for _,value in parameters if mx.issubdtype(value.dtype,mx.floating)),'Parameter precision mismatch'
  self.parameter_dtypes=sorted({str(value.dtype) for _,value in parameters})
  self.history=[];self.identity=None
  self.runtime=ReuseRuntime(vision=VISION_REUSE,system_prefix=PREFIX_REUSE,preprocess=PREPROCESS_REUSE)
 def infer(self,frame,identity,instruction,pointer,emit_text=None,*,observation_text=None):
  from mlx_vlm import stream_generate
  memory_identity=identity[:2]
  if self.identity!=memory_identity:self.history=[];self.runtime.clear();self.identity=memory_identity
  full_start=time.monotonic()
  image=frame_image(frame)
  snapshot=OUT/'latest-model-input.tmp';image.save(snapshot,format='PNG');snapshot.replace(OUT/'latest-model-input.png')
  message='current game screen\nCurrent commanded pointer (unverified): '+json.dumps(pointer)
  if instruction:message+='\nOperator instruction: '+instruction
  if observation_text is not None:
   message=observation_text
  messages=[{'role':'system','content':SYSTEM}]
  for old_image,old_message,action in self.history:
   messages.extend([{'role':'user','content':[{'type':'text','text':old_message},{'type':'image'}]},{'role':'assistant','content':action}])
  messages.extend([{'role':'user','content':[{'type':'text','text':message},{'type':'image'}]},{'role':'assistant','content':'<think>'}])
  prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,continue_final_message=True)
  start=time.monotonic();text='<think>';first=None;last_visible='';last_chunk=None
  images=[x[0] for x in self.history]+[image]
  options,stages=self.runtime.prepare(self.model,self.processor,prompt,images,SYSTEM)
  generation_start=time.monotonic()
  generation=stream_generate(self.model,self.processor,prompt,image=None,max_tokens=2048,temperature=1.0,skip_special_tokens=False,prefill_step_size=PREFILL_STEP_SIZE,**options)
  for chunk in generation:
   last_chunk=chunk
   text+=chunk.text
   if first is None:first=(time.monotonic()-start)*1000
   visible=visible_reasoning(text)
   if emit_text and visible and visible!=last_visible:
    emit_text(visible,False);last_visible=visible
   if '<|action_end|>' in text and split_reasoning(text)[2]:break
  generation.close()
  if emit_text and last_visible:emit_text(visible_reasoning(text),True)
  action=parse_action(text)
  return action,{'native_output':text,'reasoning':split_reasoning(text)[0],'reasoning_closed':split_reasoning(text)[2],'inference_ms':(time.monotonic()-start)*1000,'first_chunk_ms':first,'history_frames':len(self.history),'source_size':list(image.size),'input_message':message,**stages,'end_to_end_ms':(time.monotonic()-full_start)*1000,'generation_ms':(time.monotonic()-generation_start)*1000,'language_prefill_to_first_chunk_ms':None if first is None else first-(generation_start-start)*1000,'after_first_chunk_ms':None if first is None else (time.monotonic()-start)*1000-first,'generation_tokens':getattr(last_chunk,'generation_tokens',None),'generation_tps':getattr(last_chunk,'generation_tps',None)},(image,message,action.history_text)
 def remember(self,identity,turn,applied):
  # Store attempted execution only after the transport loop has disposed the result.
  if identity[:2]==self.identity:
   image,message,action=turn
   if not applied:action='<|action_start|>0 0 0 ; ; ; ; ; ; <|action_end|>'
   self.history.append((image,message,action));self.history=self.history[-5:]

async def main(on_ready=None):
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'system-prompt.txt').write_text(SYSTEM+'\n')
 loop=asyncio.get_running_loop();executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='gamescaling-owner')
 policy=await loop.run_in_executor(executor,NativePolicy)
 metrics={'status':'waiting_for_browser','model':MODEL_NAME+' '+BACKEND,'prompt_profile':PROMPT_PROFILE,'source_revision':REVISIONS[MODEL_NAME],'run_id':RUN_ID,'native_action_generation':True,'gemma_action_head':False,'reasoning_enabled':True,'sampling_temperature':1.0,'max_output_tokens':2048,'native_text_streaming':True,'history_limit':5,'control_chunk_ms':200,'control_slots':6,'hold_watchdog_s':3,'application_receipts':False,'online_training':False,'pointer_initialization':[.5,.5],'perception_while_paused':True,'viewer_refresh_hz':5,'chat':'operator messages and image/action history condition native reasoning and actions; generated reasoning streams to bubble; optional reasoning may be empty'}
 metrics.update(preprocessed_image_reuse=PREPROCESS_REUSE,prefill_step_size=PREFILL_STEP_SIZE,vision_feature_reuse=VISION_REUSE,system_prefix_reuse=PREFIX_REUSE,inference_action_overlap=OVERLAP,precision=PRECISION,parameter_dtypes=getattr(policy,'parameter_dtypes',None),blank_frame_guard=True)
 live_session=None
 def record(row):
  with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'run_id':RUN_ID,'model':MODEL_NAME,**row})+'\n')
 class Session(endpoint.AgentSession):
  def __init__(self,*a,**kw):
   nonlocal live_session
   self.pointer=[.5,.5];self.keys=set();self.buttons=set();self.identity=None;self.decisions=0;self.sent=0;self.last_hold_update=0;self.context_revision=0;self.action_task=None;self.observe_after=0
   super().__init__(*a,**kw);metrics['status']='signaling';self.spawn(self.measure())
   live_session=self
   for key in ['last_model_frame','last_model_input_at','last_model_frame_received_at','last_policy_ms','last_reasoning_chars','last_commands_queued']:
    metrics.pop(key,None)
  def on_message(self,channel,raw):
   before=(self.context.identity,self.context.active()) if self.context else None
   super().on_message(channel,raw)
   after=(self.context.identity,self.context.active()) if self.context else None
   if before!=after:self.context_revision+=1
  def current_identity(self):
   return (self.session_id,self.context.identity if self.context else None,self.instruction['id'] if self.instruction else None,self.context_revision)
  def on_instruction(self,instruction):
   record({'kind':'instruction','session_id':self.session_id,'id':instruction['id'],'text':instruction['text']})
  def show_text(self,identity,text,final):
   current=self.current_identity()
   if not self.closed and current==identity:
    self.say(text,final)
    metrics.update(last_reasoning_chars=len(text),text_channel_state=self.text_channel.readyState if self.text_channel else 'unavailable')
    record({'kind':'native_text','instruction_id':identity[2],'text':text,'final':final})
  def send_text(self,utterance,text,final):
   sent=super().send_text(utterance,text,final)
   if sent:record({'kind':'text_sent','utterance':utterance,'chars':len(text),'final':final})
   return sent
  async def measure(self):
   old=time.monotonic();frames=0;decisions=0
   while not self.closed:
    await asyncio.sleep(1);now=time.monotonic()
    if (self.keys or self.buttons) and now-self.last_hold_update>3:
     self.release();record({'kind':'hold_watchdog_release','limit_s':3})
    metrics.update(received_fps=(self.frames-frames)/(now-old),decision_hz=(self.decisions-decisions)/(now-old),decoded_frames=self.frames,decisions=self.decisions,commands_queued=self.sent,validated=self.validated,peer_state=self.pc.connectionState,context_active=bool(self.context and self.context.active()))
    old=now;frames=self.frames;decisions=self.decisions
  def emit(self,kind,**fields):
   c=Controls(self.context.message['capabilities']);getattr(c,kind)(**fields)
   sequence=self.sequence+1;envelope=self.context.envelope(self.session_id,sequence,c.actions[0])
   ok=self.send('motion' if kind=='moveTo' else 'actions',{k:v for k,v in envelope.items() if k not in ('v','sessionId')})
   if not ok:raise RuntimeError('Command datachannel refused input')
   self.sequence=sequence;self.sent+=1
  def release(self):
   if self.context and self.context.active():
    if 'releaseAll' in self.context.message['capabilities']:self.emit('releaseAll')
    else:
     for key in sorted(self.keys):self.emit('keyUp',key=key)
     for button in sorted(self.buttons):self.emit('mouseUp',x=self.pointer[0],y=self.pointer[1],button=button)
   self.keys.clear();self.buttons.clear()
  def eligible(self,identity,received):
   return not self.closed and metrics.get('video_input_valid',True) and action_admissible(True,identity,self.current_identity(),self.context,received)
  async def inference(self):
   last=0
   try:
    while not self.closed:
     await asyncio.sleep(.01)
     context,latest=self.context,self.latest
     if not perception_ready(self.validated,context,latest,last) or latest[2]<=self.observe_after:continue
     image=await asyncio.to_thread(frame_image,latest[1])
     if frame_is_black(image):
      last=latest[0]
      metrics['video_input_valid']=False
      if self.action_task and not self.action_task.done():
       self.action_task.cancel();await asyncio.gather(self.action_task,return_exceptions=True)
      if metrics.get('status')!='blocked_black_video':
       self.release();record({'kind':'blank_video','session_id':self.session_id,'frame':last})
      metrics.update(status='blocked_black_video',video_input_valid=False)
      continue
     metrics['video_input_valid']=True
     identity=self.current_identity();observed_active=context.active()
     if identity!=self.identity:
      self.release();self.identity=identity;self.text_pending.clear();self.utterance+=1
     last=latest[0];metrics.update(status='generating' if observed_active else 'observing_paused',last_model_frame=last,last_model_input_at=time.time(),last_model_frame_received_at=time.time()-(time.monotonic()-latest[2]))
     submitted=time.monotonic()
     def emit_text(text,final):loop.call_soon_threadsafe(self.show_text,identity,text,final)
     action,timing,turn=await asyncio.wait_for(loop.run_in_executor(executor,policy.infer,image,identity,self.instruction_text,list(self.pointer),emit_text),60)
     self.decisions+=1
     if self.action_task:await self.action_task
     if observed_active and action.parsed and self.eligible(identity,latest[2]):
      started=loop.create_future()
      self.action_task=asyncio.create_task(self.apply_action(action,identity,latest,last,timing,submitted,observed_active,started))
      action_begin=await started
      accepted=action_begin is not None
      if accepted:self.observe_after=action_begin
      # History describes commanded controls, not acknowledged game events.
      # Context changes still invalidate every pending slot and model result.
      await loop.run_in_executor(executor,policy.remember,identity,turn,accepted)
      if not OVERLAP:await self.action_task
     else:
      self.release()
      await loop.run_in_executor(executor,policy.remember,identity,turn,False)
      self.finish_decision(action,identity,latest,last,timing,submitted,observed_active,0,[],False,None)
   except asyncio.CancelledError:raise
   except Exception as e:
    metrics.update(status='error',error=str(e));record({'kind':'error','error':str(e)});await self.close()
  async def apply_action(self,action,identity,latest,frame,timing,submitted,observed_active,started):
   queued=0;unsupported=[];begin=time.monotonic()
   try:
    if not self.eligible(identity,latest[2]):return
    dx,dy,scroll=action.mouse
    if dx or dy:
     self.pointer=[max(0,min(1,self.pointer[0]+dx/1000)),max(0,min(1,self.pointer[1]+dy/1000))]
     self.emit('moveTo',x=self.pointer[0],y=self.pointer[1]);queued+=1
    if scroll:
     if 'scroll' in self.context.message['capabilities']:self.emit('scroll',x=self.pointer[0],y=self.pointer[1],deltaY=-scroll*100);queued+=1
     else:unsupported.append('scroll')
    def apply_step(i,step):
     nonlocal queued
     keys,buttons,dropped=desired(step);unsupported.extend(dropped)
     for k in sorted(self.keys-keys):self.emit('keyUp',key=k);queued+=1
     for b in sorted(self.buttons-buttons):self.emit('mouseUp',x=self.pointer[0],y=self.pointer[1],button=b);queued+=1
     for k in sorted(held_inputs_to_send(keys,self.keys,i==0)):self.emit('keyDown',key=k);queued+=1
     for b in sorted(held_inputs_to_send(buttons,self.buttons,i==0)):self.emit('mouseDown',x=self.pointer[0],y=self.pointer[1],button=b);queued+=1
     self.keys=keys;self.buttons=buttons;self.last_hold_update=time.monotonic()
    executed=await execute_chunk(action.steps,apply_step,lambda:self.eligible(identity,latest[2]),started=started)
    if not executed:self.release()
    self.finish_decision(action,identity,latest,frame,timing,submitted,observed_active,queued,unsupported,executed,begin)
   except asyncio.CancelledError:
    self.release();raise
   except Exception as e:
    metrics.update(status='error',error=str(e));record({'kind':'error','error':str(e)});await self.close()
   finally:
    if not started.done():started.set_result(None)
  def finish_decision(self,action,identity,latest,frame,timing,submitted,observed_active,queued,unsupported,executed,begin):
   metrics.update(last_policy_ms=timing['inference_ms'],last_history_frames=timing['history_frames'],last_action_parsed=action.parsed,last_action_noop=action.is_noop,last_commands_queued=queued,last_receive_to_end_ms=(time.monotonic()-latest[2])*1000)
   metrics.update({key:timing.get(key) for key in ('preprocess_cache_hits','images_preprocessed','prompt_tokens','prepare_ms','vision_ms','system_prefix_ms','vision_cache_hits','frames_encoded','images_in_context','system_prefix_tokens','system_prefix_hit','generation_ms','language_prefill_to_first_chunk_ms','after_first_chunk_ms','generation_tokens','generation_tps','first_chunk_ms')})
   record({'kind':'decision','session_id':self.session_id,'frame':frame,'instruction_id':identity[2],'prompt_profile':PROMPT_PROFILE,'noop':action.is_noop,'observed_active':observed_active,'context_revision':identity[3],'parsed':action.parsed,'mouse':action.mouse,'steps':action.steps,'queued':queued,'unsupported_tokens':unsupported,'execution_complete':executed,'history_is_commanded':True,'inference_started_at_monotonic':submitted,'action_started_at_monotonic':begin,'inference_action_overlap':OVERLAP,'receive_to_end_ms':metrics['last_receive_to_end_ms'],'queue_and_inference_ms':((begin or time.monotonic())-submitted)*1000,**timing})
  async def close(self):
   if self.action_task and self.action_task is not asyncio.current_task() and not self.action_task.done():
    self.action_task.cancel()
    await asyncio.gather(self.action_task,return_exceptions=True)
   if not self.closed:
    try:self.release()
    except Exception:pass
   await super().close()
   if metrics['status']!='error':metrics['status']='disconnected'
   metrics.update(peer_state='closed',validated=False,received_fps=0,decision_hz=0)
 endpoint.AgentSession=Session
 token=pairing_token(DATA/'connection.txt',PUBLIC_URL)
 metrics['allowed_origins']=[ORIGIN]
 app=endpoint.make_app(ORIGIN,token,None)
 async def local_only(req):
  # The Tailscale proxy also routes here. Never expose frame pixels on its hostname.
  if req.host not in ('127.0.0.1:'+str(PORT),'localhost:'+str(PORT)) or req.remote not in ('127.0.0.1','::1'):
   raise web.HTTPForbidden(text='Live view is available on localhost only.')
 async def health(req):
  await local_only(req)
  return web.json_response(metrics)
 app.router.add_get('/health',health)
 async def live_view(req):
  await local_only(req)
  return web.Response(text=(ROOT/'live-view.html').read_text(),content_type='text/html',headers={'Cache-Control':'no-store','Content-Security-Policy':"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'"})
 async def live_state(req):
  await local_only(req)
  latest=live_session.latest if live_session else None
  return web.json_response({**metrics,'received_frame':latest[0] if latest else None,'received_frame_age_ms':(time.monotonic()-latest[2])*1000 if latest else None,'model_input_age_ms':(time.time()-metrics['last_model_input_at'])*1000 if metrics.get('last_model_input_at') else None,'server_time':time.time()},headers={'Cache-Control':'no-store'})
 async def received_frame(req):
  await local_only(req)
  latest=live_session.latest if live_session else None
  if not latest:raise web.HTTPNotFound(text='Waiting for video.')
  def encode():
   buffer=io.BytesIO();frame_image(latest[1]).save(buffer,format='JPEG',quality=85);return buffer.getvalue()
  body=await asyncio.to_thread(encode)
  return web.Response(body=body,content_type='image/jpeg',headers={'Cache-Control':'no-store','X-Frame-Id':str(latest[0])})
 async def model_frame(req):
  await local_only(req)
  if not metrics.get('last_model_input_at'):raise web.HTTPNotFound(text='Waiting for model input.')
  return web.Response(body=(OUT/'latest-model-input.png').read_bytes(),content_type='image/png',headers={'Cache-Control':'no-store','X-Frame-Id':str(metrics.get('last_model_frame'))})
 app.router.add_get('/live',live_view)
 app.router.add_get('/live/state',live_state)
 app.router.add_get('/live/received.jpg',received_frame)
 app.router.add_get('/live/model.png',model_frame)
 runner=web.AppRunner(app,access_log=None);await runner.setup();await web.TCPSite(runner,'127.0.0.1',PORT).start()
 print(json.dumps({'status':'ready','model':metrics['model'],'reasoning_enabled':True,'connection_file':str(DATA/'connection.txt'),'live_view':'http://127.0.0.1:'+str(PORT)+'/live'}),flush=True)
 try:
  if on_ready:await on_ready(Session,metrics)
  await asyncio.Event().wait()
 finally:await runner.cleanup();executor.shutdown(wait=False,cancel_futures=True)
if __name__=='__main__':asyncio.run(main())
