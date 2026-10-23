"""Paused observations must never become executable controls on a later resume."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
from webrtc_live import perception_ready,action_admissible,held_inputs_to_send,frame_is_black,frame_image
from PIL import Image
class BlankVideoTests(unittest.TestCase):
 def test_empty_black_capture_is_rejected(self):
  self.assertTrue(frame_is_black(Image.new('RGB',(64,32),(0,0,0))))
 def test_near_black_capture_is_rejected(self):
  self.assertTrue(frame_is_black(Image.new('RGB',(64,32),(1,2,1))))
 def test_dark_scene_with_visible_content_is_kept(self):
  image=Image.new('RGB',(64,32),(0,0,0));image.putpixel((32,16),(20,10,5))
  self.assertFalse(frame_is_black(image))
class FrameConversionTests(unittest.TestCase):
 def test_concurrent_preview_and_inference_keep_the_same_rgb_pixels(self):
  import av
  import numpy as np
  from concurrent.futures import ThreadPoolExecutor
  rgb=np.random.default_rng(42).integers(0,256,(72,128,3),dtype=np.uint8)
  frame=av.VideoFrame.from_ndarray(rgb,format='rgb24').reformat(format='yuv420p')
  expected=np.asarray(frame.to_image())
  with ThreadPoolExecutor(max_workers=4) as pool:
   images=list(pool.map(frame_image,[frame]*24))
  for image in images:np.testing.assert_array_equal(np.asarray(image),expected)
 def test_preconverted_image_is_supported(self):
  image=Image.new('RGB',(64,32),(10,20,30))
  self.assertEqual(frame_image(image).tobytes(),image.tobytes())
class Context:
 identity=('task',1)
 message={'expiresAtMs':5000}
 def __init__(self,active):self.is_active=active
 def host_now(self,now):return now*1000
 def active(self,now):return self.is_active and self.host_now(now)<self.message['expiresAtMs']
class PausedPerceptionTests(unittest.TestCase):
 def test_paused_frame_is_observed_but_not_executable(self):
  context=Context(False);frame=(12,object(),1.5,context.identity)
  self.assertTrue(perception_ready(True,context,frame,11,now=2))
  self.assertFalse(action_admissible(False,('s',1),('s',1),context,1.5,now=2))
 def test_resuming_does_not_admit_a_paused_frame_result(self):
  context=Context(True)
  self.assertFalse(action_admissible(False,('s',1),('s',1),context,1.5,now=2))
 def test_active_fresh_result_can_execute(self):
  self.assertTrue(action_admissible(True,('s',1),('s',1),Context(True),1.5,now=2))
 def test_pause_resume_epoch_rejects_old_active_result(self):
  self.assertFalse(action_admissible(True,('s',1),('s',3),Context(True),1.5,now=2))
 def test_observations_require_validation_freshness_and_context(self):
  c=Context(False);frame=(12,object(),1.5,c.identity)
  self.assertFalse(perception_ready(False,c,frame,11,now=2))
  self.assertFalse(perception_ready(True,c,frame,12,now=2))
  self.assertFalse(perception_ready(True,c,frame,11,now=4))
  self.assertFalse(perception_ready(True,c,(12,object(),1.5,('other',1)),11,now=2))
  self.assertFalse(perception_ready(True,c,(12,object(),5.5,c.identity),11,now=6))
class HeldControlLeaseTests(unittest.TestCase):
 def test_new_chunk_reasserts_held_keys_after_possible_browser_expiry(self):
  self.assertEqual(held_inputs_to_send({'KeyW'},{'KeyW'},True),{'KeyW'})
 def test_later_slot_does_not_duplicate_unchanged_holds(self):
  self.assertEqual(held_inputs_to_send({'KeyW'},{'KeyW'},False),set())
 def test_new_key_and_release_are_not_masked(self):
  self.assertEqual(held_inputs_to_send({'KeyD'},{'KeyW'},False),{'KeyD'})
  self.assertEqual(held_inputs_to_send(set(),{'KeyW'},True),set())
if __name__=='__main__':unittest.main()
