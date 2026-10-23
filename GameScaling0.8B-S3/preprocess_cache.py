"""Bounded, opt-in pixel-tensor reuse for MLX's pinned Qwen3VL image processor.

The processor's original __call__ still handles all six images, patch assembly,
placeholder expansion, and text tokenization. Only its pure per-image operation
is cached. Install on the single model-owner thread, never concurrently.
"""
from contextlib import contextmanager
import hashlib
import json

SETTINGS = ('patch_size','temporal_patch_size','merge_size','min_pixels','max_pixels',
            'do_rescale','rescale_factor','do_normalize','image_mean','image_std','do_convert_rgb')


class PreprocessedImages:
    def __init__(self):
        self.entries = {}
        self.hits = self.misses = self.calls = 0

    def clear(self):
        self.entries.clear()
        self.hits = self.misses = self.calls = 0

    @contextmanager
    def install(self, processor):
        import numpy as np
        image_processor = processor.image_processor
        original = getattr(image_processor, '_process_one', None)
        if not callable(original):
            raise ValueError('Pixel reuse requires the pinned MLX Qwen3VL _process_one path')
        settings = json.dumps({key:getattr(image_processor,key,None) for key in SETTINGS}, sort_keys=True,default=repr)
        active = {}
        self.hits = self.misses = self.calls = 0

        def cached(image, *args, **kwargs):
            self.calls += 1
            array = np.asarray(image)
            digest = hashlib.sha256(settings.encode())
            digest.update(repr((array.shape,str(array.dtype),args,sorted(kwargs.items()))).encode())
            digest.update(array.tobytes())
            key = digest.hexdigest()
            part = self.entries.get(key, active.get(key))
            if part is None:
                patches, grid = original(image, *args, **kwargs)
                patches = np.asarray(patches).copy()
                grid = tuple(int(v) for v in grid)
                if len(grid)!=3 or int(np.prod(grid))!=len(patches):
                    raise ValueError('Unexpected Qwen image patch/grid layout')
                patches.flags.writeable=False
                part=(patches,grid)
                self.misses += 1
            else:
                self.hits += 1
            active[key]=part
            return part[0],list(part[1])

        had_instance_value = '_process_one' in image_processor.__dict__
        old_instance_value = image_processor.__dict__.get('_process_one')
        image_processor._process_one = cached
        try:
            yield
            if self.calls==0:
                raise RuntimeError('Pixel-cache hook was bypassed; refusing a silent no-op')
            self.entries=active  # only the active six-image input is retained
        finally:
            if had_instance_value:image_processor._process_one=old_instance_value
            else:del image_processor._process_one
