from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from preprocess_cache import PreprocessedImages


class FakeImageProcessor:
    def __init__(self):self.calls=[];self.rescale_factor=1
    def _process_one(self,image):
        self.calls.append(image.shape)
        return image.reshape(-1,3)*self.rescale_factor,[1,image.shape[0],image.shape[1]]
    def __call__(self,images):
        values=[self._process_one(np.asarray(im)) for im in images]
        return {'pixel_values':np.concatenate([value[0] for value in values]),'image_grid_thw':np.array([value[1] for value in values])}


class PreprocessCacheTests(unittest.TestCase):
    def setUp(self):
        self.images=[Image.new('RGB',(4,4),(i,0,0)) for i in range(7)]
        self.processor=SimpleNamespace(image_processor=FakeImageProcessor())
        self.cache=PreprocessedImages()
    def test_sliding_window_is_exact_with_five_hits_one_miss(self):
        raw=self.processor.image_processor(self.images[1:])
        with self.cache.install(self.processor):self.processor.image_processor(self.images[:6])
        with self.cache.install(self.processor):cached=self.processor.image_processor(self.images[1:])
        self.assertEqual((self.cache.hits,self.cache.misses),(5,1))
        for key in raw:np.testing.assert_array_equal(raw[key],cached[key])
        self.assertEqual(len(self.cache.entries),6)
        self.assertNotIn('_process_one',self.processor.image_processor.__dict__)
    def test_settings_and_changed_pixels_invalidate(self):
        with self.cache.install(self.processor):self.processor.image_processor(self.images[:6])
        self.processor.image_processor.rescale_factor=2
        with self.cache.install(self.processor):self.processor.image_processor(self.images[:6])
        self.assertEqual((self.cache.hits,self.cache.misses),(0,6))
        altered=[im.copy() for im in self.images[:6]];altered[0].putpixel((0,0),(255,0,0))
        with self.cache.install(self.processor):self.processor.image_processor(altered)
        self.assertEqual((self.cache.hits,self.cache.misses),(5,1))
    def test_unknown_processor_fails_before_install(self):
        with self.assertRaises(ValueError):
            with self.cache.install(SimpleNamespace(image_processor=object())):pass
    def test_exception_restores_processor(self):
        with self.assertRaises(RuntimeError):
            with self.cache.install(self.processor):raise RuntimeError('cancel')
        self.assertNotIn('_process_one',self.processor.image_processor.__dict__)
        self.assertEqual(set(self.processor.image_processor(self.images[:6])),{'pixel_values','image_grid_thw'})
    def test_existing_instance_override_is_restored(self):
        image_processor=self.processor.image_processor
        original=image_processor._process_one
        image_processor._process_one=original
        with self.cache.install(self.processor):image_processor(self.images[:6])
        self.assertIs(image_processor.__dict__['_process_one'],original)
    def test_bypassed_hook_is_not_reported_as_a_success(self):
        with self.assertRaisesRegex(RuntimeError,'bypassed'):
            with self.cache.install(self.processor):pass


if __name__=='__main__':unittest.main()
