"""Real SD3.5 same-seed replay with step-18 allocator replacement only."""
from dataclasses import asdict
import numpy as np
import torch
from cegwm.formal_ablation import _adaptive_allocation
from cegwm.method.content_adaptive import dino_last_layer_cls_patch_tiles, rgb_texture_tiles
from cegwm.method.content_iss import embed_content_iss, iss_beta, score_content_iss_image
from cegwm.method.survival_allocator import allocation_from_logits, macro_features
from cegwm.runtime.content_adaptive_sd35 import _decode_callback_latents
from cegwm.runtime.observation import require_ordinary_rgb_image


class AllocatorCallback:
    tensor_inputs = ('latents',)

    def __init__(self, key, assets, beta, variant, allocator=None, reference_cache=None):
        self.key, self.assets, self.beta = key, assets, beta
        self.variant, self.allocator = variant, allocator
        self.reference_cache = reference_cache if reference_cache is not None else {}
        self.features = None
        self.executed = False
        self.measurement = None
        self.remaining_steps = []

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        if step_index > 18:
            self.remaining_steps.append(step_index)
        if step_index != 18:
            return callback_kwargs
        if self.executed:
            raise RuntimeError('duplicate injection')
        latent = callback_kwargs['latents']
        embed = self.assets.embed_assets
        if 'allocation' not in self.reference_cache:
            self.reference_cache['allocation'] = _adaptive_allocation(latent, pipeline, self.assets)
        reference = self.reference_cache['allocation']
        if self.variant == 'original':
            allocation = reference
        else:
            image = _decode_callback_latents(pipeline, latent)
            semantic = dino_last_layer_cls_patch_tiles(image, embed.dino_processor, embed.dino_model)
            self.features = macro_features(semantic, rgb_texture_tiles(image), latent)
            if self.variant == 'survival':
                if self.allocator is None:
                    raise ValueError('survival requires frozen fitted allocator')
                allocation = self.allocator.predict(self.features, reference)
            else:
                logits = np.zeros(4)
                if self.variant.startswith('probe'):
                    index = int(self.variant.removeprefix('probe'))
                    if index not in range(4):
                        raise ValueError('macroblock probe outside 0..3')
                    logits[index] = .5
                elif self.variant != 'uniform':
                    raise ValueError('unknown allocator variant')
                allocation = allocation_from_logits(logits, reference, uniform=self.variant == 'uniform')
        embedded, self.measurement = embed_content_iss(latent, self.key, embed.hf_public_assets,
            embed.lf_public_assets, allocation, self.beta,
            hf_weight_interpolation="nearest" if self.variant == "original" else "bilinear")
        self.executed = True
        return {**callback_kwargs, 'latents': embedded}


def generate_variant(runtime, prompt, seed, primary_null, variant, allocator=None, reference_cache=None):
    from experiments.run_paper_main_worker_v2 import SYNCSEAL_RESIDUAL_MULTIPLIER
    assets = runtime['assets'].content_assets.iss_assets
    beta = iss_beta(score_content_iss_image(primary_null, runtime['key'], assets.lf_public_assets), assets.iss_asset)
    callback = AllocatorCallback(runtime['key'], assets, beta, variant, allocator, reference_cache)
    # Sequential independent replay resets scheduler each call. Never recurse from callback.
    result = runtime['pipeline'](prompt=prompt, num_inference_steps=20, height=512, width=512,
        generator=torch.Generator(device=runtime['device']).manual_seed(seed), output_type='pil',
        callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'])
    if not callback.executed or callback.remaining_steps != [19]:
        raise RuntimeError('injection must be followed by actual final denoising step 19')
    content = require_ordinary_rgb_image(result.images[0])
    marked = runtime['assets'].geometry_backend.embed_final_rgb(content, SYNCSEAL_RESIDUAL_MULTIPLIER)
    measurement = asdict(callback.measurement) if callback.measurement is not None else {}
    return require_ordinary_rgb_image(marked), callback.features, {'iss_beta':beta, 'embedding':measurement}
