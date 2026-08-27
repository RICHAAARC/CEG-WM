"""Image-only, public-runtime Q/K observation for Geometry-V1.

This adapter deliberately observes a supplied frozen runtime object directly.  It
does not invoke a generation pipeline or construct conditioning from text.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
from PIL import Image

from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image


@dataclass(frozen=True, slots=True)
class SD35QKObservationSpec:
    """Explicit public inputs for one detector-side Q/K observation."""

    model_id: str
    revision: str | None
    attention_layer_paths: tuple[str, ...]
    inference_steps: int
    schedule_index: int
    public_noise_seed: int
    max_grid: tuple[int, int]
    null_encoder_hidden_states: torch.Tensor
    null_pooled_projections: torch.Tensor


@dataclass(frozen=True, slots=True)
class SD35QKLayerObservation:
    """One image-derived attention Q/K pair and its public provenance."""

    layer_path: str
    query: torch.Tensor
    key: torch.Tensor
    source_dtype: torch.dtype
    source_device: torch.device
    source_shape: tuple[int, int, int]
    source_grid: tuple[int, int]
    sample_indices: torch.Tensor
    heads: int
    head_dim: int


@dataclass(frozen=True, slots=True)
class SD35QKObservation:
    """No-decision Q/K observation; it contains no key or embedding state."""

    layers: tuple[SD35QKLayerObservation, ...]
    latent_shape: tuple[int, int, int, int]
    schedule_index: int
    timestep: torch.Tensor
    public_noise_seed: int


@dataclass(frozen=True, slots=True)
class SD35QKAllLayerObservation:
    """Bounded all-layer observation for D0 discovery.

    Unlike the legacy explicit-path observer, projections are sampled inside
    hooks.  The returned mapping therefore never owns complete GPU projection
    outputs for the full transformer roster.
    """

    layers: tuple[SD35QKLayerObservation, ...]
    layer_failures: tuple[tuple[str, str], ...]
    latent_shape: tuple[int, int, int, int]
    schedule_index: int
    timestep: torch.Tensor
    public_noise_seed: int


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _validate_spec(spec: SD35QKObservationSpec) -> None:
    if not isinstance(spec, SD35QKObservationSpec):
        raise TypeError("spec must be an explicit SD35QKObservationSpec")
    if not isinstance(spec.model_id, str) or not spec.model_id.strip():
        raise ValueError("model_id must be explicit non-empty text")
    if spec.revision is not None and (not isinstance(spec.revision, str) or not spec.revision.strip()):
        raise ValueError("revision must be non-empty text or explicit None")
    if not isinstance(spec.attention_layer_paths, tuple) or not spec.attention_layer_paths:
        raise ValueError("attention_layer_paths must be a non-empty explicit tuple")
    if len(set(spec.attention_layer_paths)) != len(spec.attention_layer_paths):
        raise ValueError("attention_layer_paths must be unique")
    if any(not isinstance(path, str) or not path for path in spec.attention_layer_paths):
        raise ValueError("attention layer paths must be non-empty text")
    _require_int(spec.inference_steps, "inference_steps", minimum=1)
    _require_int(spec.schedule_index, "schedule_index", minimum=0)
    _require_int(spec.public_noise_seed, "public_noise_seed", minimum=0)
    if (
        not isinstance(spec.max_grid, tuple)
        or len(spec.max_grid) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1 for value in spec.max_grid)
    ):
        raise ValueError("max_grid must be an explicit positive (rows, columns) tuple")
    for name, value in (
        ("null_encoder_hidden_states", spec.null_encoder_hidden_states),
        ("null_pooled_projections", spec.null_pooled_projections),
    ):
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[0] != 1:
            raise ValueError(f"{name} must be a finite batch-one public tensor")
        if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite floating public tensor")


def _get_pipeline_member(pipeline: object, name: str) -> Any:
    value = getattr(pipeline, name, None)
    if value is None:
        raise TypeError(f"public runtime object must expose {name}")
    return value


def _patch_size(transformer: Any) -> int:
    config = getattr(transformer, "config", None)
    value = getattr(config, "patch_size", getattr(transformer, "patch_size", None))
    return _require_int(value, "transformer patch_size", minimum=1)


def _resolve_attention(transformer: Any, path: str) -> torch.nn.Module:
    get_submodule = getattr(transformer, "get_submodule", None)
    if not callable(get_submodule):
        raise TypeError("transformer must expose get_submodule for explicit attention paths")
    try:
        attention = get_submodule(path)
    except (AttributeError, KeyError) as error:
        raise ValueError(f"attention layer path was not found: {path}") from error
    if not isinstance(attention, torch.nn.Module):
        raise TypeError(f"attention layer is not a torch module: {path}")
    return attention


def _uniform_indices(rows: int, columns: int, max_grid: tuple[int, int]) -> torch.Tensor:
    sampled_rows = min(rows, int(max_grid[0]))
    sampled_columns = min(columns, int(max_grid[1]))
    row_indices = torch.linspace(0, rows - 1, sampled_rows, dtype=torch.float64).round().to(torch.int64)
    column_indices = torch.linspace(0, columns - 1, sampled_columns, dtype=torch.float64).round().to(torch.int64)
    return (row_indices[:, None] * columns + column_indices[None, :]).reshape(-1)


def _projection_output(value: Any, *, path: str, name: str, expected_tokens: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"{path}.{name} must emit a batch-one rank-3 tensor")
    if value.shape[1] != expected_tokens or value.shape[2] < 1:
        raise ValueError(f"{path}.{name} token shape is incompatible with the latent grid")
    if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{path}.{name} must emit finite floating values")
    return value


def observe_sd35_image_qk(
    image: Image.Image,
    *,
    pipeline: object,
    spec: SD35QKObservationSpec,
) -> SD35QKObservation:
    """Observe Q/K from the current RGB image using only explicit public inputs."""

    _validate_spec(spec)
    rgb_image = require_ordinary_rgb_image(image)
    image_processor = _get_pipeline_member(pipeline, "image_processor")
    vae = _get_pipeline_member(pipeline, "vae")
    scheduler = _get_pipeline_member(pipeline, "scheduler")
    transformer = _get_pipeline_member(pipeline, "transformer")
    if not isinstance(transformer, torch.nn.Module):
        raise TypeError("transformer must be a torch module")
    try:
        transformer_parameter = next(transformer.parameters())
    except StopIteration as error:
        raise TypeError("transformer must expose floating parameters for direct observation") from error
    if not transformer_parameter.dtype.is_floating_point:
        raise TypeError("transformer parameters must use a floating dtype")

    try:
        latent = encode_final_rgb_image(rgb_image, image_processor, vae)
    except BaseException as error:
        # The operational runner consumes this bounded stage tag without
        # inspecting exception strings.  It carries no model or image data.
        setattr(error, "geometry_failure_point", "vae_encode")
        raise
    if latent.ndim != 4 or latent.shape[0] != 1 or not latent.dtype.is_floating_point:
        raise ValueError("VAE observation must be a batch-one floating NCHW tensor")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("VAE observation must be finite")
    if latent.device != transformer_parameter.device or latent.dtype != transformer_parameter.dtype:
        raise ValueError("VAE latent must match the direct transformer device and dtype")
    latent_shape = tuple(int(value) for value in latent.shape)
    patch_size = _patch_size(transformer)
    if latent.shape[-2] % patch_size or latent.shape[-1] % patch_size:
        raise ValueError("latent spatial dimensions must divide exactly by transformer patch_size")
    source_grid = (latent.shape[-2] // patch_size, latent.shape[-1] // patch_size)
    expected_tokens = source_grid[0] * source_grid[1]
    if expected_tokens < 2:
        raise ValueError("latent grid must provide at least two tokens")

    set_timesteps = getattr(scheduler, "set_timesteps", None)
    scale_noise = getattr(scheduler, "scale_noise", None)
    if not callable(set_timesteps) or not callable(scale_noise):
        raise TypeError("scheduler must provide set_timesteps and scale_noise")
    try:
        set_timesteps(spec.inference_steps, device=latent.device)
        timesteps = getattr(scheduler, "timesteps", None)
        if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1:
            raise ValueError("scheduler must expose a rank-one timestep schedule")
        if timesteps.device != latent.device:
            raise ValueError("scheduler timesteps must be on the latent device")
        if not bool(torch.isfinite(timesteps).all()):
            raise ValueError("scheduler timesteps must be finite")
        if spec.schedule_index >= timesteps.numel():
            raise ValueError("schedule_index is outside the explicit scheduler schedule")
        # FlowMatchEulerDiscreteScheduler consumes a batch-shaped timestep.
        # Keep it rank one for both scale_noise and the transformer call.
        timestep = timesteps[spec.schedule_index : spec.schedule_index + 1]
        if timestep.shape != (1,):
            raise ValueError("scheduler timestep must preserve the batch-one shape")
        generator = torch.Generator(device=latent.device)
        generator.manual_seed(spec.public_noise_seed)
        noise = torch.randn(latent.shape, dtype=latent.dtype, device=latent.device, generator=generator)
        observed_latent = scale_noise(latent, timestep, noise)
        if not isinstance(observed_latent, torch.Tensor) or observed_latent.shape != latent.shape:
            raise ValueError("scheduler scale_noise must preserve latent shape")
        if not observed_latent.dtype.is_floating_point or not bool(torch.isfinite(observed_latent).all()):
            raise ValueError("scheduler observation latent must be finite")
    except BaseException as error:
        setattr(error, "geometry_failure_point", "scheduler")
        raise

    captured: dict[str, dict[str, list[torch.Tensor]]] = {}
    handles: list[Any] = []
    try:
        for path in spec.attention_layer_paths:
            attention = _resolve_attention(transformer, path)
            projections = {name: getattr(attention, name, None) for name in ("to_q", "to_k")}
            if not all(isinstance(projection, torch.nn.Module) for projection in projections.values()):
                raise TypeError(f"{path} must expose torch to_q and to_k projections")
            captured[path] = {"to_q": [], "to_k": []}
            for name, projection in projections.items():
                def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any, *, layer: str = path, projection_name: str = name) -> None:
                    captured[layer][projection_name].append(output)

                handles.append(projection.register_forward_hook(capture))
        try:
            with torch.no_grad():
                transformer(
                    hidden_states=observed_latent,
                    timestep=timestep,
                    # SD3 feeds both values directly to transformer linear layers;
                    # align this public null conditioning to those parameters.
                    encoder_hidden_states=spec.null_encoder_hidden_states.to(
                        device=transformer_parameter.device, dtype=transformer_parameter.dtype
                    ),
                    pooled_projections=spec.null_pooled_projections.to(
                        device=transformer_parameter.device, dtype=transformer_parameter.dtype
                    ),
                )
        except BaseException as error:
            setattr(error, "geometry_failure_point", "transformer_call")
            raise
    finally:
        for handle in handles:
            handle.remove()

    sample_indices = _uniform_indices(source_grid[0], source_grid[1], spec.max_grid)
    layers: list[SD35QKLayerObservation] = []
    try:
        for path in spec.attention_layer_paths:
            values = captured[path]
            if len(values["to_q"]) != 1 or len(values["to_k"]) != 1:
                raise ValueError(f"{path} projections must each be reached exactly once")
            query = _projection_output(values["to_q"][0], path=path, name="to_q", expected_tokens=expected_tokens)
            key = _projection_output(values["to_k"][0], path=path, name="to_k", expected_tokens=expected_tokens)
            if query.shape != key.shape:
                raise ValueError(f"{path} Q/K shapes must match")
            heads = _require_int(getattr(_resolve_attention(transformer, path), "heads", None), f"{path}.heads", minimum=1)
            if query.shape[-1] % heads:
                raise ValueError(f"{path} Q/K features must divide evenly into heads")
            source_indices = sample_indices.to(device=query.device)
            layers.append(
                SD35QKLayerObservation(
                    layer_path=path,
                    query=query[0, source_indices].detach().to(device="cpu", dtype=torch.float32),
                    key=key[0, source_indices].detach().to(device="cpu", dtype=torch.float32),
                    source_dtype=query.dtype,
                    source_device=query.device,
                    source_shape=tuple(int(value) for value in query.shape),
                    source_grid=source_grid,
                    sample_indices=sample_indices.cpu(),
                    heads=heads,
                    head_dim=query.shape[-1] // heads,
                )
            )
    except BaseException as error:
        setattr(error, "geometry_failure_point", "qk_capture")
        raise
    return SD35QKObservation(
        layers=tuple(layers),
        latent_shape=latent_shape,
        schedule_index=spec.schedule_index,
        timestep=timestep.detach().to(device="cpu"),
        public_noise_seed=spec.public_noise_seed,
    )


def observe_sd35_image_qk_sampled_all_layers(
    image: Image.Image,
    *,
    pipeline: object,
    spec: SD35QKObservationSpec,
) -> SD35QKAllLayerObservation:
    """Observe an explicit all-layer roster with hook-time bounded sampling.

    A malformed or unreached individual projection is recorded as a bounded
    layer failure while the remaining hooks continue.  Failures before the
    transformer call retain the usual fail-closed boundary tag.
    """
    _validate_spec(spec)
    rgb_image = require_ordinary_rgb_image(image)
    image_processor = _get_pipeline_member(pipeline, "image_processor")
    vae = _get_pipeline_member(pipeline, "vae")
    scheduler = _get_pipeline_member(pipeline, "scheduler")
    transformer = _get_pipeline_member(pipeline, "transformer")
    if not isinstance(transformer, torch.nn.Module):
        raise TypeError("transformer must be a torch module")
    try:
        parameter = next(transformer.parameters())
    except StopIteration as error:
        raise TypeError("transformer must expose floating parameters for direct observation") from error
    try:
        latent = encode_final_rgb_image(rgb_image, image_processor, vae)
    except BaseException as error:
        setattr(error, "geometry_failure_point", "vae_encode")
        raise
    if latent.ndim != 4 or latent.shape[0] != 1 or not latent.dtype.is_floating_point or not bool(torch.isfinite(latent).all()):
        raise ValueError("VAE observation must be finite batch-one floating NCHW")
    if latent.device != parameter.device or latent.dtype != parameter.dtype:
        raise ValueError("VAE latent must match the direct transformer device and dtype")
    patch_size = _patch_size(transformer)
    if latent.shape[-2] % patch_size or latent.shape[-1] % patch_size:
        raise ValueError("latent spatial dimensions must divide exactly by transformer patch_size")
    source_grid = (latent.shape[-2] // patch_size, latent.shape[-1] // patch_size)
    expected_tokens = source_grid[0] * source_grid[1]
    if expected_tokens < 2:
        raise ValueError("latent grid must provide at least two tokens")
    sample_indices = _uniform_indices(source_grid[0], source_grid[1], spec.max_grid)
    try:
        scheduler.set_timesteps(spec.inference_steps, device=latent.device)
        timesteps = scheduler.timesteps
        if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1 or spec.schedule_index >= timesteps.numel():
            raise ValueError("invalid explicit scheduler schedule")
        timestep = timesteps[spec.schedule_index : spec.schedule_index + 1]
        generator = torch.Generator(device=latent.device); generator.manual_seed(spec.public_noise_seed)
        noise = torch.randn(latent.shape, dtype=latent.dtype, device=latent.device, generator=generator)
        observed_latent = scheduler.scale_noise(latent, timestep, noise)
        if not isinstance(observed_latent, torch.Tensor) or observed_latent.shape != latent.shape or not bool(torch.isfinite(observed_latent).all()):
            raise ValueError("scheduler observation latent must be finite")
    except BaseException as error:
        setattr(error, "geometry_failure_point", "scheduler")
        raise
    captures: dict[str, dict[str, tuple[torch.Tensor, torch.dtype, torch.device, tuple[int, int, int]]]] = {}
    failures: dict[str, str] = {}
    handles: list[Any] = []
    try:
        for path in spec.attention_layer_paths:
            attention = _resolve_attention(transformer, path)
            heads = _require_int(getattr(attention, "heads", None), f"{path}.heads", minimum=1)
            for name in ("to_q", "to_k"):
                projection = getattr(attention, name, None)
                if not isinstance(projection, torch.nn.Module):
                    failures[path] = "projection_missing"; continue
                def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any, *, layer: str = path, projection_name: str = name, layer_heads: int = heads) -> None:
                    if layer in failures:
                        return
                    try:
                        value = _projection_output(output, path=layer, name=projection_name, expected_tokens=expected_tokens)
                        if value.shape[-1] % layer_heads:
                            raise ValueError("head_dimension_invalid")
                        bucket = captures.setdefault(layer, {})
                        if projection_name in bucket:
                            raise ValueError("projection_call_count")
                        selected = value[0, sample_indices.to(value.device)].detach().to(device="cpu", dtype=torch.float32)
                        bucket[projection_name] = (selected, value.dtype, value.device, tuple(int(v) for v in value.shape))
                    except (TypeError, ValueError):
                        failures[layer] = "projection_capture_invalid"
                handles.append(projection.register_forward_hook(capture))
        try:
            with torch.no_grad():
                transformer(hidden_states=observed_latent, timestep=timestep,
                            encoder_hidden_states=spec.null_encoder_hidden_states.to(device=parameter.device, dtype=parameter.dtype),
                            pooled_projections=spec.null_pooled_projections.to(device=parameter.device, dtype=parameter.dtype))
        except BaseException as error:
            setattr(error, "geometry_failure_point", "transformer_call")
            raise
    finally:
        for handle in handles:
            handle.remove()
    layers: list[SD35QKLayerObservation] = []
    for path in spec.attention_layer_paths:
        if path in failures:
            continue
        captured = captures.get(path, {})
        if set(captured) != {"to_q", "to_k"}:
            failures[path] = "projection_call_count"; continue
        query, qdtype, qdevice, qshape = captured["to_q"]
        key, kdtype, kdevice, kshape = captured["to_k"]
        if qshape != kshape or qdtype != kdtype or qdevice != kdevice:
            failures[path] = "qk_shape_mismatch"; continue
        heads = _require_int(getattr(_resolve_attention(transformer, path), "heads", None), f"{path}.heads", minimum=1)
        layers.append(SD35QKLayerObservation(path, query, key, qdtype, qdevice, qshape, source_grid, sample_indices.cpu(), heads, qshape[-1] // heads))
    return SD35QKAllLayerObservation(tuple(layers), tuple((path, failures[path]) for path in spec.attention_layer_paths if path in failures), tuple(int(v) for v in latent.shape), spec.schedule_index, timestep.detach().cpu(), spec.public_noise_seed)
