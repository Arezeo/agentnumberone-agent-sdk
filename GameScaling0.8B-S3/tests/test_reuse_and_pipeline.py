import asyncio
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_chunk import execute_chunk
from connection import pairing_token
from mlx_runtime import image_key
from webrtc_live import NativePolicy


class ReuseContractTests(unittest.TestCase):
    def test_feature_identity_includes_pixels_shape_and_grid(self):
        a = Image.new('RGB', (4, 4), 'red')
        b = a.copy(); b.putpixel((0, 0), (0, 0, 0))
        self.assertEqual(image_key(a, [1, 4, 4]), image_key(a.copy(), [1, 4, 4]))
        self.assertNotEqual(image_key(a, [1, 4, 4]), image_key(b, [1, 4, 4]))
        self.assertNotEqual(image_key(a, [1, 4, 4]), image_key(a, [1, 2, 8]))
        self.assertNotEqual(image_key(a, [1, 4, 4]), image_key(a.resize((2, 8)), [1, 4, 4]))

    def test_all_six_images_and_instruction_reach_generation(self):
        policy = NativePolicy.__new__(NativePolicy)
        images = [Image.new('RGB', (4, 4), (i, 0, 0)) for i in range(6)]
        action = '<|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>'
        policy.identity = ('session', 'game')
        policy.history = [(im, 'previous '+str(i), action) for i, im in enumerate(images[:5])]
        policy.model = object()
        captured = {}
        def template(messages, **kwargs):
            captured['messages'] = messages
            return 'rendered prompt'
        policy.processor = types.SimpleNamespace(apply_chat_template=template)
        def prepare(model, processor, prompt, frames, system):
            captured['frames'] = frames
            return {}, {'images_in_context':len(frames)}
        policy.runtime = types.SimpleNamespace(prepare=prepare)
        def generate(*args, **kwargs):
            captured['options'] = kwargs
            yield types.SimpleNamespace(text='hello</think>'+action, generation_tokens=10, generation_tps=100)
        output=[]
        with tempfile.TemporaryDirectory() as tmp, patch('webrtc_live.OUT', Path(tmp)), patch.dict(sys.modules, {'mlx_vlm':types.SimpleNamespace(stream_generate=generate)}):
            result, timing, turn = policy.infer(images[-1], ('session','game',1,0), 'Keep moving', [.5,.5], lambda text, final:output.append((text,final)))
        self.assertTrue(result.parsed)
        self.assertEqual(len(captured['frames']), 6)
        self.assertEqual([im.tobytes() for im in captured['frames']], [im.tobytes() for im in images])
        self.assertIn('Keep moving', captured['messages'][-2]['content'][0]['text'])
        self.assertEqual(captured['messages'][-1]['content'], '<think>')
        self.assertEqual(captured['options']['temperature'], 1.0)
        self.assertEqual(captured['options']['max_tokens'], 2048)
        self.assertEqual(captured['options']['prefill_step_size'], 2048)
        self.assertEqual(timing['history_frames'], 5)
        self.assertEqual(timing['images_in_context'], 6)
        self.assertIn(('hello', True), output)

    def test_pairing_url_private_stable_and_origin_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'connection.txt'
            a = pairing_token(path, 'wss://example.com/control')
            self.assertEqual(a, pairing_token(path, 'wss://example.com/control'))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotEqual(a, pairing_token(path, 'wss://other.example.com/control'))
            with self.assertRaises(ValueError):
                pairing_token(path, 'wss://example.com/control?token=private')


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_next_inference_can_start_before_chunk_finishes(self):
        started = asyncio.get_running_loop().create_future()
        slots = []
        task = asyncio.create_task(execute_chunk([['W']]*6, lambda i,s: slots.append(i), lambda: True, started=started))
        await started
        self.assertEqual(slots, [0])
        self.assertFalse(task.done())
        # Represents the next worker inference while the event loop drives slots.
        await asyncio.to_thread(lambda: None)
        self.assertTrue(await task)
        self.assertEqual(slots, list(range(6)))

    async def test_pause_invalidates_every_remaining_slot(self):
        active = True
        slots = []
        def apply(i, step):
            nonlocal active
            slots.append(i)
            active = False
        self.assertFalse(await execute_chunk([['W']]*6, apply, lambda: active))
        self.assertEqual(slots, [0])

    async def test_stale_context_never_starts_a_chunk(self):
        started = asyncio.get_running_loop().create_future()
        slots = []
        self.assertFalse(await execute_chunk([['W']]*6, lambda i,s:slots.append(i), lambda:False, started=started))
        self.assertEqual(slots, [])
        self.assertIsNone(await started)

    async def test_cancelled_chunk_does_not_emit_later_slots(self):
        started = asyncio.get_running_loop().create_future()
        slots=[]
        task=asyncio.create_task(execute_chunk([['W']]*6, lambda i,s:slots.append(i), lambda:True, started=started))
        await started
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        await asyncio.sleep(.04)
        self.assertEqual(slots,[0])



class LiveRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_runner_overlaps_and_cancels_context_changed_output(self):
        import time
        import webrtc_live as runner
        from sdk import Context
        calls=[];remembered=[];sent=[];instances=[]
        action=runner.parse_action('<think></think><|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>')
        class FakePolicy:
            parameter_dtypes=['test-only']
            def infer(self,frame,identity,instruction,pointer,emit_text):
                calls.append(time.monotonic())
                time.sleep(.04)
                return action,{'inference_ms':40,'history_frames':0},None
            def remember(self,identity,turn,accepted):remembered.append(accepted)
        class Channel:
            readyState='open';bufferedAmount=0
            def send(self,raw):sent.append((time.monotonic(),raw))
        original=runner.endpoint.AgentSession
        async def check(Session,metrics):
            session=Session(None);instances.append(session)
            session.session_id='session';session.validated=True
            session.channels={name:Channel() for name in ('motion','actions')}
            session.context=Context(dict(taskToken='task',generation=1,nowMs=0,expiresAtMs=10000,active=True,capabilities=['keyDown','keyUp','releaseAll']))
            try:
                for index in range(40):
                    session.latest=(index+1,Image.new('RGB',(8,8),'red'),time.monotonic(),session.context.identity)
                    await asyncio.sleep(.01)
                    if len(calls)>=2:break
                self.assertGreaterEqual(len(calls),2)
                # The next inference began before the previous 200ms chunk ended.
                self.assertLess(calls[1]-calls[0],.2)
                self.assertIsNotNone(session.action_task)
                self.assertFalse(session.action_task.done())
                # Change the context during that next inference. Neither the old
                # chunk's remaining key slots nor its pending output may execute.
                session.context_revision+=1
                cutoff=time.monotonic()
                await asyncio.sleep(.25)
                after=[json.loads(raw) for when,raw in sent if when>=cutoff]
                self.assertFalse(any(message.get('type')=='keyDown' for message in after))
                self.assertIn(False,remembered)
                self.assertNotEqual(metrics.get('status'),'error',metrics)
                # Resume fresh observations, then replace capture with black.
                for index in range(80):
                    session.latest=(100+index,Image.new('RGB',(8,8),'red'),time.monotonic(),session.context.identity)
                    await asyncio.sleep(.01)
                    if len(calls)>=4:break
                self.assertGreaterEqual(len(calls),4)
                session.latest=(999,Image.new('RGB',(8,8),'black'),time.monotonic(),session.context.identity)
                for index in range(40):
                    session.latest=(1000+index,Image.new('RGB',(8,8),'black'),time.monotonic(),session.context.identity)
                    await asyncio.sleep(.01)
                    if metrics.get('status')=='blocked_black_video':break
                self.assertEqual(metrics.get('status'),'blocked_black_video')
                cutoff=time.monotonic()
                await asyncio.sleep(.22)
                after=[json.loads(raw) for when,raw in sent if when>=cutoff]
                self.assertFalse(any(message.get('type')=='keyDown' for message in after))
            finally:
                await session.close()
        import json
        class RunnerChecked(Exception):pass
        async def ready(Session,metrics):
            await check(Session,metrics)
            raise RunnerChecked()
        # Use a real loopback HTTP listener and the actual Session implementation,
        # replacing only the model. No MLX dependency is imported in this test.
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner,'OUT',Path(tmp)/'results'), patch.object(runner,'DATA',Path(tmp)), patch.object(runner,'PORT',0), patch.object(runner,'NativePolicy',FakePolicy):
            try:
                with self.assertRaises(RunnerChecked):await runner.main(on_ready=ready)
            finally:
                runner.endpoint.AgentSession=original

if __name__ == '__main__':
    unittest.main()
