"""Deterministic CPU runtime fixtures for package-contained wiring checks."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from experiments.runners.internal import ResourceExecutionError
from main import key_schedule_sha256_counter
from runtime import (
    RuntimeBackendIdentity,
    RuntimeDetectionScheduleStep,
    RuntimeDeviceCapabilities,
    RuntimeQkForwardIdentity,
    RuntimeVaeFactors,
)


class _SyntheticPosterior:
    def __init__(self, mode_value: torch.Tensor) -> None:
        self._mode_value = mode_value

    def mode(self) -> torch.Tensor:
        return self._mode_value.detach().clone()


class _SyntheticProjection(torch.nn.Module):
    def __init__(
        self,
        *,
        layer_name: str,
        projection_role: str,
        root_key: str,
        model_revision: str,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.projection_role = projection_role
        self.root_key = root_key
        self.model_revision = model_revision

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        token_count = int(hidden_states.shape[1])
        grid_side = int(token_count**0.5)
        if (
            hidden_states.ndim != 3
            or hidden_states.shape[0] != 1
            or hidden_states.shape[2] != 3
            or grid_side * grid_side != token_count
        ):
            raise RuntimeError(
                "synthetic Q/K hidden states must be one square coordinate-mask grid"
            )
        decoded = hidden_states.detach().to(dtype=torch.float32)
        pooled_coordinates = decoded[:, :, :2]
        pooled_validity = decoded[:, :, 2] >= 0.5
        center_extent = 1.0 - 1.0 / grid_side
        sampling_grid = (
            (pooled_coordinates * 2.0 - 1.0) / center_extent
        ).reshape(1, grid_side, grid_side, 2)
        canonical_basis = torch.eye(
            token_count,
            dtype=torch.float32,
            device=hidden_states.device,
        ).reshape(1, token_count, grid_side, grid_side)
        sampling = functional.grid_sample(
            canonical_basis,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0].reshape(token_count, token_count).transpose(0, 1)
        sampling = sampling * pooled_validity.reshape(token_count, 1)
        stream = key_schedule_sha256_counter(
            self.root_key,
            {
                "candidate_id": "qk_relation_similarity",
                "operator": "attention_relation_signs",
                "responsibility_domain": "geometry_sync",
                "model_revision": self.model_revision,
                "layer_name": self.layer_name,
                "token_count": token_count,
                "tensor_role": "pair_uniform",
            },
            (token_count, token_count),
            distribution="uniform",
        )
        uniform = torch.tensor(
            stream.values,
            dtype=torch.float32,
            device=hidden_states.device,
        ).reshape(token_count, token_count)
        signs = torch.where(uniform >= 0.5, 1.0, -1.0)
        upper = torch.triu(signs, diagonal=1)
        symmetric = upper + upper.transpose(0, 1)
        if self.projection_role == "query":
            per_head = sampling
        elif self.projection_role == "key":
            per_head = (
                sampling @ symmetric.transpose(0, 1)
            ) * token_count**0.5
        else:
            raise RuntimeError("synthetic Q/K projection role is invalid")
        return torch.cat((per_head, per_head), dim=1).unsqueeze(0).to(
            dtype=torch.float16
        )


class _SyntheticNorm(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.tensor(
            0.5,
            dtype=value.dtype,
            device=value.device,
        )


class _SyntheticAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        layer_name: str,
        root_key: str,
        model_revision: str,
    ) -> None:
        super().__init__()
        self.heads = 2
        self.to_q = _SyntheticProjection(
            layer_name=layer_name,
            projection_role="query",
            root_key=root_key,
            model_revision=model_revision,
        )
        self.to_k = _SyntheticProjection(
            layer_name=layer_name,
            projection_role="key",
            root_key=root_key,
            model_revision=model_revision,
        )
        self.norm_q = _SyntheticNorm()
        self.norm_k = _SyntheticNorm()

    def project(self, hidden_states: torch.Tensor) -> None:
        self.to_q(hidden_states)
        self.to_k(hidden_states)


class SyntheticQkBackend:
    """Deterministic CPU backend for the existing public Q/K runtime path."""

    def __init__(self, *, root_key: str, model_revision: str) -> None:
        self.root_key = root_key
        self.model_revision = model_revision
        self.attentions = {
            layer_name: _SyntheticAttention(
                layer_name=layer_name,
                root_key=root_key,
                model_revision=model_revision,
            )
            for layer_name in (
                "transformer_blocks.0.attn",
                "transformer_blocks.23.attn",
            )
        }
        self.configuration = None

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=0,
        )

    def prepare(self, configuration, selected_device):
        self.configuration = configuration
        return RuntimeBackendIdentity(
            candidate_id=configuration.candidate_id,
            runtime_config_digest=configuration.runtime_config_digest,
            runtime_backend_name="synthetic-qk-cpu-backend",
            selected_device=selected_device,
            model_id=configuration.model_id,
            model_revision=configuration.model_revision,
            pipeline_class=configuration.pipeline_class,
            scheduler_class=configuration.scheduler_class,
            inference_steps=configuration.inference_steps,
            guidance_scale=configuration.guidance_scale,
            image_height=configuration.image_height,
            image_width=configuration.image_width,
            generation_seed_device=configuration.generation_seed_device,
            latent_dtype=configuration.latent_dtype,
            template_dtype=configuration.template_dtype,
            score_dtype=configuration.score_dtype,
            callback_index=configuration.callback_index,
            callback_hold_scheduler_intervals=(
                configuration.callback_hold_scheduler_intervals
            ),
            vae_decode_protocol=configuration.vae_decode_protocol,
            vae_encode_protocol=configuration.vae_encode_protocol,
            vae_scaling_factor_source=(
                configuration.vae_scaling_factor_source
            ),
            vae_shift_factor_source=configuration.vae_shift_factor_source,
            detection_schedule_index=(
                configuration.detection_schedule_index
            ),
            detection_conditioning_protocol=(
                configuration.detection_conditioning_protocol
            ),
            qk_layer_names=configuration.qk_layer_names,
            dependency_lock=configuration.dependency_lock,
        )

    def close(self) -> None:
        return None

    def vae_factors(self) -> RuntimeVaeFactors:
        return RuntimeVaeFactors(scaling_factor=1.0, shift_factor=0.0)

    def vae_encode(self, image: torch.Tensor) -> _SyntheticPosterior:
        coordinate_pixels = image[:, :2].to(dtype=torch.float32)
        valid = (
            (coordinate_pixels[:, :1] > 0.0)
            & (coordinate_pixels[:, 1:2] > 0.0)
        ).to(dtype=torch.float32)
        coordinate_minimum = 64.0
        coordinate_span = 190.0
        coordinates = (
            (coordinate_pixels - coordinate_minimum) / coordinate_span
        ).clamp(0.0, 1.0)
        pooled_validity = functional.adaptive_avg_pool2d(
            valid,
            (8, 8),
        )
        pooled_coordinates = functional.adaptive_avg_pool2d(
            coordinates * valid,
            (8, 8),
        ) / pooled_validity.clamp_min(1.0 / (512.0 * 512.0))
        mode = torch.cat(
            (pooled_coordinates, pooled_validity),
            dim=1,
        )
        return _SyntheticPosterior(mode)

    def create_detection_schedule(
        self,
        inference_steps: int,
    ) -> RuntimeDetectionScheduleStep:
        if self.configuration is None:
            raise RuntimeError("synthetic Q/K backend is not prepared")
        return RuntimeDetectionScheduleStep(
            scheduler_class=self.configuration.scheduler_class,
            inference_steps=inference_steps,
            detection_schedule_index=(
                self.configuration.detection_schedule_index
            ),
            detection_timestep=torch.tensor(731.0, dtype=torch.float32),
        )

    def scale_detection_noise(
        self,
        detection_latent: torch.Tensor,
        public_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        if detection_latent.shape != public_noise.shape:
            raise RuntimeError("synthetic scheduler noise shape drifted")
        return detection_latent.detach().clone()

    def attention_module(self, layer_name: str) -> torch.nn.Module:
        return self.attentions[layer_name]

    def run_qk_detection_forward(
        self,
        noisy_detection_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditioning,
    ) -> RuntimeQkForwardIdentity:
        del timestep
        if self.configuration is None:
            raise RuntimeError("synthetic Q/K backend is not prepared")
        hidden_states = noisy_detection_latent.flatten(2).transpose(1, 2)
        for layer_name in self.configuration.qk_layer_names:
            self.attentions[layer_name].project(hidden_states)
        return RuntimeQkForwardIdentity(
            runtime_config_digest=(
                self.configuration.runtime_config_digest
            ),
            model_id=self.configuration.model_id,
            model_revision=self.configuration.model_revision,
            scheduler_class=self.configuration.scheduler_class,
            inference_steps=self.configuration.inference_steps,
            detection_schedule_index=(
                self.configuration.detection_schedule_index
            ),
            detection_conditioning_protocol=(
                self.configuration.detection_conditioning_protocol
            ),
            prompt=conditioning.prompt,
            prompt_2=conditioning.prompt_2,
            prompt_3=conditioning.prompt_3,
            do_classifier_free_guidance=(
                conditioning.do_classifier_free_guidance
            ),
            qk_layer_names=self.configuration.qk_layer_names,
        )


class SyntheticDelegatingGeometryOperation:
    """Stable fault-injection wrapper around the existing formal operation."""

    def __init__(self, delegate, *, failure_mode: str) -> None:
        self._delegate = delegate
        self.failure_mode = failure_mode
        self.calls = 0

    def formal_runner_semantic_declaration(self) -> dict[str, object]:
        return {
            "delegate": self._delegate.formal_runner_semantic_declaration(),
            "failure_mode": self.failure_mode,
            "semantic_version": "synthetic_runner_path_exercise_v1",
        }

    def __call__(self, image: torch.Tensor, registered_key: str):
        del registered_key
        self.calls += 1
        if self.failure_mode == "resource_twice_then_qk_sync_failure":
            if self.calls <= 2:
                raise ResourceExecutionError(
                    "synthetic transient CPU resource interruption"
                )
            return self._delegate(image, "")
        if self.failure_mode == "qk_sync_failure":
            return self._delegate(image, "")
        if self.failure_mode == "execution_failure":
            raise ValueError("synthetic unexpected operation failure")
        raise RuntimeError("synthetic failure mode is invalid")


__all__ = [
    "SyntheticDelegatingGeometryOperation",
    "SyntheticQkBackend",
]
