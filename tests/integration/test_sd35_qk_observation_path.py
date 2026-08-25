from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from cegwm.geometry.qk_relation import keyed_qk_relation
from cegwm.runtime.sd35_qk_observation import SD35QKObservationSpec, observe_sd35_image_qk


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = type("Config", (), {"scaling_factor": 1.0, "shift_factor": 0.0})()

    def encode(self, pixels: torch.Tensor) -> object:
        latent = pixels.mean(1, keepdim=True).repeat(1, 4, 1, 1)
        return type("Encoded", (), {"latent_dist": type("Dist", (), {"mode": lambda self: latent})()})()


class _Scheduler:
    def set_timesteps(self, count: int) -> None:
        self.timesteps = torch.arange(count, 0, -1, dtype=torch.float32)

    def scale_noise(self, latent: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return latent + noise * 0.01 + timestep * 0.0


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q, self.to_k = torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)
        self.heads = 2

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.to_q(value)
        return self.to_k(value)


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"patch_size": 2})()
        self.layers = torch.nn.ModuleList([_Attention()])

    def forward(self, *, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        tokens = torch.nn.functional.avg_pool2d(hidden_states, 2, 2).permute(0, 2, 3, 1).reshape(1, -1, 4)
        return self.layers[0](tokens)


def test_image_only_observation_qk_enters_keyed_relation() -> None:
    pipeline = type("PublicRuntime", (), {})()
    pipeline.image_processor, pipeline.vae = _Processor(), _VAE()
    pipeline.scheduler, pipeline.transformer = _Scheduler(), _Transformer()
    spec = SD35QKObservationSpec(
        model_id="public/sd35",
        revision="frozen",
        attention_layer_paths=("layers.0",),
        inference_steps=2,
        schedule_index=0,
        public_noise_seed=5,
        max_grid=(4, 4),
        null_encoder_hidden_states=torch.zeros((1, 2, 4)),
        null_pooled_projections=torch.zeros((1, 4)),
    )
    observation = observe_sd35_image_qk(Image.new("RGB", (8, 8), "blue"), pipeline=pipeline, spec=spec)
    layer = observation.layers[0]
    relation = keyed_qk_relation(layer.query.numpy(), layer.key.numpy(), b"detector-key-0001")
    assert relation.relation.shape == (16, 16)
    assert np.isfinite(relation.projection)
