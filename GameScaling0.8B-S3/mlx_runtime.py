"""Exact-input reuse for the pinned Qwen3.5 GameScaling checkpoint.

Historical images remain in the prompt. Only image features and the unchanged
text-only system prefix are reused. Sliding multimodal/recurrent state is never
trimmed or substituted for a different history.
"""
import hashlib
import time


def image_key(image, grid):
    digest = hashlib.sha256()
    digest.update(repr((image.mode, image.size, tuple(grid))).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def detached_tree(value, mx):
    if isinstance(value, mx.array):
        return mx.array(value) + mx.zeros((), dtype=value.dtype)
    if isinstance(value, dict):
        return {k: detached_tree(v, mx) for k, v in value.items()}
    if isinstance(value, list):
        return [detached_tree(v, mx) for v in value]
    if isinstance(value, tuple):
        return tuple(detached_tree(v, mx) for v in value)
    return value


class VisionFeatures:
    def __init__(self):
        self.entries = {}

    def clear(self):
        self.entries.clear()

    def encode(self, model, inputs, images):
        import mlx.core as mx
        grids = inputs['image_grid_thw'].tolist()
        assert len(grids) == len(images), 'Image/grid count mismatch'
        # This checkpoint has no deepstack features. Do not silently drop them
        # if a different checkpoint is supplied.
        assert not model.config.vision_config.deepstack_visual_indexes
        pixels = inputs['pixel_values']
        offset = 0
        parts, active = [], {}
        hits = 0
        for image, grid in zip(images, grids):
            count = grid[0] * grid[1] * grid[2]
            key = image_key(image, grid)
            part = self.entries.get(key)
            if part is None:
                part, deepstack = model.vision_tower(
                    pixels[offset:offset + count].astype(model.vision_tower.patch_embed.proj.weight.dtype),
                    mx.array([grid]))
                assert deepstack is None or len(deepstack) == 0
                mx.eval(part)
            else:
                hits += 1
            active[key] = part
            parts.append(part)
            offset += count
        assert offset == pixels.shape[0], 'Pixel/grid layout mismatch'
        self.entries = active  # bounded to this decision's six images
        combined = mx.concatenate(parts, axis=0)
        mx.eval(combined)
        return combined, hits


class SystemPrefix:
    """Save a real prefix boundary, including recurrent and attention state.

    Never roll back an end-of-generation hybrid cache: recurrent state cannot
    generally be trimmed. Each decision restores a detached system-only snapshot.
    """
    def __init__(self):
        self.ids = None
        self.snapshots = None

    def clear(self):
        self.ids = self.snapshots = None

    def apply(self, model, processor, inputs, system):
        import mlx.core as mx
        from mlx_vlm.models.cache import make_prompt_cache
        # Match actual token IDs, not a guessed character offset. Stop before
        # the first user turn, so all six images remain in the uncached suffix.
        rendered = processor.apply_chat_template(
            [{'role': 'system', 'content': system}], tokenize=False,
            add_generation_prompt=False)
        ids = processor.tokenizer.encode(rendered, add_special_tokens=False)
        full = inputs['input_ids']
        if not ids or full[0, :len(ids)].tolist() != ids:
            return dict(inputs), 0, False
        if len(ids) >= full.shape[1]:
            return dict(inputs), 0, False
        hit = ids == self.ids and self.snapshots is not None
        if not hit:
            cache = make_prompt_cache(model.language_model)
            # Every value here precedes any image. No vision context is omitted.
            model.language_model(mx.array([ids]), cache=cache, logits_to_keep=1)
            mx.eval([c.state for c in cache])
            self.snapshots = [detached_tree(c.prefix_cache_snapshot(), mx) for c in cache]
            mx.eval(self.snapshots)
            self.ids = ids
        cache = make_prompt_cache(model.language_model)
        for entry, saved in zip(cache, self.snapshots):
            entry.prefix_cache_restore(detached_tree(saved, mx))
        mx.eval([c.state for c in cache])
        positions, deltas = model.language_model.get_rope_index(
            full, inputs.get('image_grid_thw'), inputs.get('video_grid_thw'),
            inputs.get('attention_mask'))
        options = {k: v for k, v in inputs.items() if k != 'attention_mask'}
        options.update(input_ids=full[:, len(ids):], mask=None,
                       prompt_cache=cache, position_ids=positions, rope_deltas=deltas)
        return options, len(ids), hit


class ReuseRuntime:
    def __init__(self, vision=True, system_prefix=True):
        self.vision_enabled = vision
        self.prefix_enabled = system_prefix
        self.features = VisionFeatures()
        self.prefix = SystemPrefix()

    def clear(self):
        self.features.clear()
        self.prefix.clear()

    def prepare(self, model, processor, prompt, images, system):
        import mlx.core as mx
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
        start = time.monotonic()
        inputs = prepare_inputs(
            processor, images=images, prompts=prompt,
            image_token_index=model.config.image_token_index,
            add_special_tokens=should_add_special_tokens(model.config.model_type, processor))
        mx.eval(*inputs.values())
        prepared = time.monotonic()
        combined, hits = (None, 0)
        if self.vision_enabled:
            combined, hits = self.features.encode(model, inputs, images)
        encoded = time.monotonic()
        options = {k: v for k, v in inputs.items() if k != 'attention_mask'}
        options['mask'] = inputs.get('attention_mask')
        prefix_tokens, prefix_hit = 0, False
        if self.prefix_enabled:
            prefix_options, prefix_tokens, prefix_hit = self.prefix.apply(model, processor, inputs, system)
            if prefix_tokens:
                options = prefix_options
        if combined is not None:
            options['cached_image_features'] = combined
        done = time.monotonic()
        return options, {
            'prepare_ms': (prepared-start)*1000,
            'vision_ms': (encoded-prepared)*1000 if self.vision_enabled else None,
            'system_prefix_ms': (done-encoded)*1000,
            'vision_cache_hits': hits,
            'frames_encoded': len(images)-hits,
            'images_in_context': len(images),
            'system_prefix_tokens': prefix_tokens,
            'system_prefix_hit': prefix_hit,
            'prompt_tokens': int(inputs['input_ids'].size),
            'vision_reuse': self.vision_enabled,
            'system_prefix_reuse': self.prefix_enabled,
        }
