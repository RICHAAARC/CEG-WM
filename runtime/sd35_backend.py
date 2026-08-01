"""Lazy real-model backend for the frozen SD3.5 runtime candidate.

Imports and model loading deliberately happen only in :meth:`prepare`, so the
default CPU test profile can validate the boundary without diffusers or model
weights.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from .backend import (
    GenerationCallback,
    RuntimeBackendError,
    RuntimeBackendIdentity,
    RuntimeDetectionConditioning,
    RuntimeDetectionScheduleStep,
    RuntimeDeviceCapabilities,
    RuntimeGenerationPromptIdentity,
    RuntimeQkForwardIdentity,
    RuntimeVaeFactors,
    RuntimeVaePosterior,
)
from .configuration import Sd35RuntimeConfiguration


class Sd35BackendError(RuntimeBackendError):
    """The real SD3.5 backend failed closed."""


def _explicit_absolute_root(value: str | Path, field_name: str) -> Path:
    root = Path(value)
    if not root.is_absolute():
        raise Sd35BackendError(f"{field_name} must be an explicit absolute path")
    return root.resolve()


def _roots_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


class Sd35PipelineBackend:
    """Diffusers SD3.5 backend connected to content_write_and_vae and qk_observation."""

    def __init__(
        self,
        *,
        cache_root: str | Path,
        persistent_root: str | Path,
        hf_token: str | None,
        prompt: str,
        negative_prompt: str = "",
    ) -> None:
        root = _explicit_absolute_root(cache_root, "cache_root")
        persistent = _explicit_absolute_root(
            persistent_root,
            "persistent_root",
        )
        if _roots_overlap(root, persistent):
            raise Sd35BackendError(
                "cache_root and persistent_root must be bidirectionally disjoint"
            )
        if not isinstance(hf_token, (str, type(None))):
            raise Sd35BackendError("hf_token must be text or None")
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise Sd35BackendError("generation prompts must be text")
        self._cache_root = root
        self._persistent_root = persistent
        self._hf_token = hf_token
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._configuration: Sd35RuntimeConfiguration | None = None
        self._device: torch.device | None = None
        self._pipeline: Any | None = None
        self._scheduler_type: type[Any] | None = None
        self._detection_scheduler: Any | None = None
        self._generation_running = False
        self._generation_prompt_identity: RuntimeGenerationPromptIdentity | None = None
        self._clear_prompt_after_generation = False
        self._requires_generation_prompt_selection = False

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(
            cpu_available=True,
            cuda_device_count=torch.cuda.device_count(),
        )

    def prepare(
        self,
        configuration: Sd35RuntimeConfiguration,
        selected_device: str,
    ) -> RuntimeBackendIdentity:
        if self._pipeline is not None:
            raise Sd35BackendError("SD3.5 backend may only be prepared once")
        if type(configuration) is not Sd35RuntimeConfiguration:
            raise Sd35BackendError("configuration must be Sd35RuntimeConfiguration")
        if selected_device != "cuda:0":
            raise Sd35BackendError("real SD3.5 qualification requires cuda:0")
        try:
            diffusers = importlib.import_module("diffusers")
            pipeline_type = getattr(diffusers, "StableDiffusion3Pipeline")
            scheduler_type = getattr(
                diffusers,
                "FlowMatchEulerDiscreteScheduler",
            )
        except (ImportError, AttributeError) as exc:
            raise Sd35BackendError(
                "registered diffusers SD3.5 classes are unavailable"
            ) from exc
        if f"diffusers.{pipeline_type.__name__}" != configuration.pipeline_class:
            raise Sd35BackendError("loaded pipeline class identity drifted")
        if f"diffusers.{scheduler_type.__name__}" != configuration.scheduler_class:
            raise Sd35BackendError("loaded scheduler class identity drifted")
        self._cache_root.mkdir(parents=True, exist_ok=True)
        try:
            pipeline = pipeline_type.from_pretrained(
                configuration.model_id,
                revision=configuration.model_revision,
                torch_dtype=torch.float16,
                token=self._hf_token,
                cache_dir=str(self._cache_root / "huggingface"),
            )
            pipeline = pipeline.to(selected_device)
        except Exception as exc:
            raise Sd35BackendError("SD3.5 model preparation failed") from exc
        if not isinstance(getattr(pipeline, "scheduler", None), scheduler_type):
            raise Sd35BackendError("prepared scheduler class identity drifted")
        if getattr(pipeline, "vae", None) is None or getattr(
            pipeline, "transformer", None
        ) is None:
            raise Sd35BackendError("prepared pipeline lacks VAE or transformer")
        self._configuration = configuration
        self._device = torch.device(selected_device)
        self._pipeline = pipeline
        self._scheduler_type = scheduler_type
        return RuntimeBackendIdentity(
            candidate_id=configuration.candidate_id,
            runtime_config_digest=configuration.runtime_config_digest,
            runtime_backend_name="diffusers_sd35_pipeline",
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
            vae_scaling_factor_source=configuration.vae_scaling_factor_source,
            vae_shift_factor_source=configuration.vae_shift_factor_source,
            detection_schedule_index=configuration.detection_schedule_index,
            detection_conditioning_protocol=(
                configuration.detection_conditioning_protocol
            ),
            qk_layer_names=configuration.qk_layer_names,
            dependency_lock=configuration.dependency_lock,
        )

    def _prepared(self) -> tuple[Sd35RuntimeConfiguration, torch.device, Any]:
        if (
            self._configuration is None
            or self._device is None
            or self._pipeline is None
        ):
            raise Sd35BackendError("SD3.5 backend is not prepared")
        return self._configuration, self._device, self._pipeline

    def set_generation_prompts(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> RuntimeGenerationPromptIdentity:
        """Bind one hf_only_reference_validation prompt after preparation without reloading the model."""

        self._prepared()
        if self._generation_running:
            raise Sd35BackendError("generation prompts cannot change while running")
        if type(prompt) is not str or not prompt or negative_prompt != "":
            raise Sd35BackendError(
                "hf_only_reference_validation generation requires nonempty prompt and exact empty negative prompt"
            )
        try:
            identity = RuntimeGenerationPromptIdentity.from_prompts(
                prompt,
                negative_prompt,
            )
        except RuntimeBackendError as exc:
            raise Sd35BackendError("generation prompt identity is invalid") from exc
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._generation_prompt_identity = identity
        self._clear_prompt_after_generation = True
        self._requires_generation_prompt_selection = False
        return identity

    def run_generation(
        self,
        initial_latent: torch.Tensor,
        callback: GenerationCallback,
    ) -> torch.Tensor:
        configuration, _device, pipeline = self._prepared()
        if self._generation_running:
            raise Sd35BackendError("overlapping generation is forbidden")
        if (
            self._requires_generation_prompt_selection
            and self._generation_prompt_identity is None
        ):
            raise Sd35BackendError(
                "next hf_only_reference_validation generation requires an explicit per-unit prompt"
            )
        try:
            prompt_identity = self._generation_prompt_identity or (
                RuntimeGenerationPromptIdentity.from_prompts(
                    self._prompt,
                    self._negative_prompt,
                )
            )
        except RuntimeBackendError as exc:
            raise Sd35BackendError("generation prompt snapshot is invalid") from exc
        prompt_snapshot = self._prompt
        negative_prompt_snapshot = self._negative_prompt
        if prompt_identity != RuntimeGenerationPromptIdentity.from_prompts(
            prompt_snapshot,
            negative_prompt_snapshot,
        ):
            raise Sd35BackendError("generation prompt snapshot identity drifted")
        self._generation_running = True

        def on_step_end(
            _pipeline: Any,
            step_index: int,
            _timestep: torch.Tensor,
            callback_kwargs: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            latent = callback_kwargs.get("latents")
            if not isinstance(latent, torch.Tensor):
                raise Sd35BackendError("generation callback did not expose latents")
            callback_kwargs["latents"] = callback(step_index, latent)
            return callback_kwargs

        try:
            with torch.inference_mode():
                output = pipeline(
                    prompt=prompt_snapshot,
                    negative_prompt=negative_prompt_snapshot,
                    latents=initial_latent,
                    num_inference_steps=configuration.inference_steps,
                    guidance_scale=configuration.guidance_scale,
                    height=configuration.image_height,
                    width=configuration.image_width,
                    output_type="latent",
                    return_dict=True,
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
        except Exception as exc:
            raise Sd35BackendError("SD3.5 generation failed") from exc
        finally:
            self._generation_running = False
            if self._clear_prompt_after_generation:
                self._prompt = ""
                self._negative_prompt = ""
                self._generation_prompt_identity = None
                self._clear_prompt_after_generation = False
                self._requires_generation_prompt_selection = True
        latent = getattr(output, "images", None)
        if not isinstance(latent, torch.Tensor):
            raise Sd35BackendError("SD3.5 generation did not return a latent tensor")
        return latent

    def vae_factors(self) -> RuntimeVaeFactors:
        _configuration, _device, pipeline = self._prepared()
        config = getattr(pipeline.vae, "config", None)
        try:
            return RuntimeVaeFactors(
                scaling_factor=float(config.scaling_factor),
                shift_factor=float(config.shift_factor),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise Sd35BackendError("prepared VAE factors are unavailable") from exc

    @staticmethod
    def _vae_execution_dtype(pipeline: Any) -> torch.dtype | None:
        vae = pipeline.vae
        vae_dtype = getattr(vae, "dtype", None)
        force_upcast = bool(
            getattr(getattr(vae, "config", None), "force_upcast", False)
        )
        if force_upcast and vae_dtype is torch.float16:
            try:
                vae.to(dtype=torch.float32)
            except Exception as exc:
                raise Sd35BackendError("prepared VAE upcast failed") from exc
            vae_dtype = getattr(vae, "dtype", torch.float32)
        return vae_dtype if isinstance(vae_dtype, torch.dtype) else None

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        _configuration, device, pipeline = self._prepared()
        try:
            with torch.inference_mode():
                vae_dtype = self._vae_execution_dtype(pipeline)
                decode_input = latent.to(
                    device=device,
                    dtype=vae_dtype or latent.dtype,
                )
                decoded = pipeline.vae.decode(
                    decode_input,
                    return_dict=True,
                ).sample
                image = pipeline.image_processor.postprocess(
                    decoded,
                    output_type="pt",
                )
        except Exception as exc:
            raise Sd35BackendError("prepared VAE decode failed") from exc
        if not isinstance(image, torch.Tensor):
            raise Sd35BackendError("prepared VAE decode returned a non-tensor")
        return image

    def vae_encode(self, image: torch.Tensor) -> RuntimeVaePosterior:
        configuration, device, pipeline = self._prepared()
        try:
            with torch.inference_mode():
                vae_dtype = self._vae_execution_dtype(pipeline)
                prepared_image = pipeline.image_processor.preprocess(
                    image,
                    height=configuration.image_height,
                    width=configuration.image_width,
                ).to(
                    device=device,
                    dtype=vae_dtype or image.dtype,
                )
                posterior = pipeline.vae.encode(
                    prepared_image,
                    return_dict=True,
                ).latent_dist
        except Exception as exc:
            raise Sd35BackendError("prepared VAE posterior encode failed") from exc
        if not isinstance(posterior, RuntimeVaePosterior):
            raise Sd35BackendError("prepared VAE did not expose posterior mode()")
        return posterior

    def create_detection_schedule(
        self,
        inference_steps: int,
    ) -> RuntimeDetectionScheduleStep:
        configuration, device, pipeline = self._prepared()
        if inference_steps != configuration.inference_steps:
            raise Sd35BackendError("detection inference-step identity drifted")
        assert self._scheduler_type is not None
        try:
            scheduler = self._scheduler_type.from_config(
                pipeline.scheduler.config
            )
            scheduler.set_timesteps(inference_steps, device=device)
            timestep = scheduler.timesteps[
                configuration.detection_schedule_index
            ].reshape(1)
        except Exception as exc:
            raise Sd35BackendError("detection schedule creation failed") from exc
        self._detection_scheduler = scheduler
        return RuntimeDetectionScheduleStep(
            scheduler_class=configuration.scheduler_class,
            inference_steps=inference_steps,
            detection_schedule_index=configuration.detection_schedule_index,
            detection_timestep=timestep,
        )

    def scale_detection_noise(
        self,
        detection_latent: torch.Tensor,
        public_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self._detection_scheduler is None:
            raise Sd35BackendError("detection schedule was not established")
        try:
            with torch.inference_mode():
                return self._detection_scheduler.scale_noise(
                    detection_latent,
                    timestep,
                    public_noise,
                )
        except Exception as exc:
            raise Sd35BackendError("detection scheduler scale_noise failed") from exc

    def attention_module(self, layer_name: str) -> torch.nn.Module:
        _configuration, _device, pipeline = self._prepared()
        value: object = pipeline.transformer
        for part in layer_name.split("."):
            if part.isdecimal():
                value = value[int(part)]  # type: ignore[index]
            else:
                value = getattr(value, part)
        if not isinstance(value, torch.nn.Module):
            raise Sd35BackendError("registered attention path is not a module")
        return value

    def run_qk_detection_forward(
        self,
        noisy_detection_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: RuntimeDetectionConditioning,
    ) -> RuntimeQkForwardIdentity:
        configuration, device, pipeline = self._prepared()
        if (
            conditioning.prompt
            or conditioning.prompt_2
            or conditioning.prompt_3
            or conditioning.do_classifier_free_guidance
        ):
            raise Sd35BackendError("Q/K detection requires empty text without CFG")
        try:
            with torch.inference_mode():
                encoded = pipeline.encode_prompt(
                    prompt="",
                    prompt_2="",
                    prompt_3="",
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                prompt_embeds = encoded[0]
                pooled_prompt_embeds = encoded[2]
                pipeline.transformer(
                    hidden_states=noisy_detection_latent,
                    timestep=timestep.expand(noisy_detection_latent.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )
        except Exception as exc:
            raise Sd35BackendError("image-only Q/K transformer forward failed") from exc
        return RuntimeQkForwardIdentity(
            runtime_config_digest=configuration.runtime_config_digest,
            model_id=configuration.model_id,
            model_revision=configuration.model_revision,
            scheduler_class=configuration.scheduler_class,
            inference_steps=configuration.inference_steps,
            detection_schedule_index=configuration.detection_schedule_index,
            detection_conditioning_protocol=(
                conditioning.detection_conditioning_protocol
            ),
            prompt=conditioning.prompt,
            prompt_2=conditioning.prompt_2,
            prompt_3=conditioning.prompt_3,
            do_classifier_free_guidance=(
                conditioning.do_classifier_free_guidance
            ),
            qk_layer_names=configuration.qk_layer_names,
        )

    def close(self) -> None:
        if self._generation_running:
            raise Sd35BackendError("backend cannot close during generation")
        self._detection_scheduler = None
        self._pipeline = None
        self._configuration = None
        self._device = None
        self._prompt = ""
        self._negative_prompt = ""
        self._generation_prompt_identity = None
        self._clear_prompt_after_generation = False
        self._requires_generation_prompt_selection = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
