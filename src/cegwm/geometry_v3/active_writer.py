"""Active keyed Q/K writer primitives for Geometry-V3 P0.

Only transient tensors cross this module.  Public artifacts may retain the
scalar measurements returned here, never the key, anchor tensor, or Q/K.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import torch

from cegwm.geometry_v3.contracts import CanonicalRelationAnchor


P0_PROTOCOL_ID = "geometry-v3-keyed-qk-active-writer-p0-v1"
P0_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
P0_IMAGE_SIZE = 512
P0_INFERENCE_STEPS = 20
P0_SEED = 73
P0_WRITER_STEP_INDEX = 18
P0_ANCHOR_POINT_COUNT = 16
P0_PLACEMENT_BLOCKS = (4, 12, 20)
P0_RELATIVE_RMS_BUDGETS = (0.0025, 0.005)


@dataclass(frozen=True, slots=True)
class P0WriterConfig:
    block_index: int
    relative_rms_budget: float

    def __post_init__(self) -> None:
        if self.block_index not in P0_PLACEMENT_BLOCKS:
            raise ValueError("P0 block is not independently predeclared")
        if self.relative_rms_budget not in P0_RELATIVE_RMS_BUDGETS:
            raise ValueError("P0 relative RMS budget is not predeclared")

    @property
    def config_id(self) -> str:
        budget = f"{self.relative_rms_budget:.4f}".replace(".", "p")
        return f"block{self.block_index}-qk-rms{budget}"

    @property
    def layer_path(self) -> str:
        return f"transformer_blocks.{self.block_index}.attn"


P0_CONFIGS = tuple(
    P0WriterConfig(block, budget)
    for block in P0_PLACEMENT_BLOCKS
    for budget in P0_RELATIVE_RMS_BUDGETS
)


@dataclass(frozen=True, slots=True)
class WriterInjectionMeasurement:
    config_id: str
    feature_kind: str
    module_path: str
    relative_rms_budget: float
    actual_relative_rms: float
    call_count: int
    writer_step_index: int


def _token_channel_pattern(
    points: Sequence[Sequence[float]],
    public_digest: str,
    like: torch.Tensor,
    *,
    module_path: str,
) -> torch.Tensor:
    if not isinstance(like, torch.Tensor) or like.ndim < 3:
        raise TypeError("Q/K writer requires a tensor with token and channel axes")
    if not like.dtype.is_floating_point or not bool(torch.isfinite(like).all()):
        raise ValueError("Q/K writer input must be finite floating point")
    token_count, channel_count = int(like.shape[-2]), int(like.shape[-1])
    side = math.isqrt(token_count)
    if side * side != token_count or side < 2 or channel_count < 2:
        raise ValueError("Q/K token grid or channel dimension is incompatible")
    if len(public_digest) != 64:
        raise ValueError("canonical anchor digest is invalid")
    device = like.device
    coordinate_dtype = torch.float32
    axis = (torch.arange(side, device=device, dtype=coordinate_dtype) + 0.5) / side
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    spatial = torch.zeros((side, side), device=device, dtype=coordinate_dtype)
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError("canonical relation points must be two-dimensional")
        px, py = float(point[0]), float(point[1])
        if not math.isfinite(px) or not math.isfinite(py):
            raise ValueError("canonical relation points must be finite")
        sign = 1.0 if index % 2 == 0 else -1.0
        spatial = spatial + sign * torch.exp(
            -((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * 0.075**2)
        )
    spatial = spatial.reshape(token_count)
    spatial = spatial - spatial.mean()
    spatial_rms = torch.sqrt(torch.mean(spatial.square()))
    if not bool(torch.isfinite(spatial_rms)) or float(spatial_rms) <= 0.0:
        raise ValueError("canonical spatial pattern is degenerate")
    spatial = spatial / spatial_rms

    digest = hashlib.sha256((public_digest + "|" + module_path).encode("ascii")).digest()
    frequency = 1 + digest[0] % 11
    phase = 2.0 * math.pi * int.from_bytes(digest[1:5], "big") / 2**32
    channel_axis = (
        torch.arange(channel_count, device=device, dtype=coordinate_dtype) + 0.5
    ) / channel_count
    channel = torch.sin(2.0 * math.pi * frequency * channel_axis + phase)
    channel = channel - channel.mean()
    channel_rms = torch.sqrt(torch.mean(channel.square()))
    if not bool(torch.isfinite(channel_rms)) or float(channel_rms) <= 0.0:
        raise ValueError("canonical channel pattern is degenerate")
    channel = channel / channel_rms

    pattern = spatial[:, None] * channel[None, :]
    pattern = pattern - pattern.mean()
    pattern = pattern / torch.sqrt(torch.mean(pattern.square()))
    shape = (1,) * (like.ndim - 2) + (token_count, channel_count)
    return pattern.reshape(shape).expand_as(like).to(dtype=like.dtype)


def canonical_qk_pattern(
    anchor: CanonicalRelationAnchor,
    like: torch.Tensor,
    *,
    module_path: str,
    transformed_points: Sequence[Sequence[float]] | None = None,
) -> torch.Tensor:
    """Materialize a transient zero-mean unit-RMS low-rank token/channel pattern."""

    if not isinstance(anchor, CanonicalRelationAnchor):
        raise TypeError("writer anchor must come from keyed canonical derivation")
    points = anchor.points if transformed_points is None else transformed_points
    return _token_channel_pattern(points, anchor.public_digest, like, module_path=module_path)


def normalized_pattern_correlation(value: torch.Tensor, pattern: torch.Tensor) -> float:
    if value.shape != pattern.shape:
        raise ValueError("Q/K observation and anchor pattern shapes differ")
    left = value.detach().to(torch.float32)
    right = pattern.detach().to(torch.float32)
    left = left - left.mean(dim=(-2, -1), keepdim=True)
    right = right - right.mean(dim=(-2, -1), keepdim=True)
    denominator = torch.sqrt(torch.sum(left.square()) * torch.sum(right.square()))
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        raise ValueError("Q/K normalized correlation has no finite support")
    result = torch.sum(left * right) / denominator
    if not bool(torch.isfinite(result)):
        raise ValueError("Q/K normalized correlation is nonfinite")
    return float(result)


class ActiveQKWriterSession:
    """Install and remove the exact step-18 Q/K writer hooks."""

    def __init__(
        self,
        transformer: Any,
        config: P0WriterConfig,
        anchor: CanonicalRelationAnchor,
    ) -> None:
        self.transformer = transformer
        self.config = config
        self.anchor = anchor
        self._handles: list[Any] = []
        self._armed = False
        self._current_transformer_call = -1
        self._root_call_count = 0
        self._callback_steps: list[int] = []
        self._measurements: dict[str, WriterInjectionMeasurement] = {}

    @property
    def measurements(self) -> tuple[WriterInjectionMeasurement, ...]:
        return tuple(self._measurements[kind] for kind in ("q", "k") if kind in self._measurements)

    def _resolve_modules(self) -> tuple[Any, Any]:
        blocks = getattr(self.transformer, "transformer_blocks", None)
        if not isinstance(blocks, (torch.nn.ModuleList, list, tuple)):
            raise RuntimeError("SD3 transformer_blocks topology is unavailable")
        if len(blocks) <= self.config.block_index:
            raise RuntimeError("SD3 transformer block count differs from P0 declaration")
        attention = getattr(blocks[self.config.block_index], "attn", None)
        q_module, k_module = getattr(attention, "to_q", None), getattr(attention, "to_k", None)
        if not isinstance(q_module, torch.nn.Module) or not isinstance(k_module, torch.nn.Module):
            raise RuntimeError("SD3 sample-side attention Q/K modules are unavailable")
        return q_module, k_module

    def _root_pre_hook(self, module: Any, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        self._current_transformer_call = self._root_call_count
        self._root_call_count += 1

    def _feature_hook(self, kind: str, module_path: str):
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> torch.Tensor:
            del module, inputs
            if not self._armed:
                return output
            if self._current_transformer_call != P0_WRITER_STEP_INDEX:
                raise RuntimeError("P0 writer armed outside the frozen transformer call")
            if kind in self._measurements:
                raise RuntimeError("P0 writer attempted repeated Q/K injection")
            if not isinstance(output, torch.Tensor):
                raise TypeError("SD3 Q/K projection must return a tensor")
            pattern = canonical_qk_pattern(self.anchor, output, module_path=module_path)
            base = output.detach().to(torch.float32)
            base_rms = torch.sqrt(torch.mean(base.square()))
            if not bool(torch.isfinite(base_rms)) or float(base_rms) <= 0.0:
                raise ValueError("SD3 Q/K output has zero or nonfinite RMS")
            delta = pattern.to(torch.float32) * base_rms * self.config.relative_rms_budget
            injected = output + delta.to(dtype=output.dtype)
            actual_delta = injected.detach().to(torch.float32) - base
            ratio = torch.sqrt(torch.mean(actual_delta.square())) / base_rms
            if not bool(torch.isfinite(ratio)) or float(ratio) <= 0.0:
                raise ValueError("P0 writer actual relative RMS is invalid")
            if float(ratio) > self.config.relative_rms_budget * (1.0 + 1e-4):
                correction = self.config.relative_rms_budget / float(ratio)
                injected = output + (actual_delta * correction).to(dtype=output.dtype)
                actual_delta = injected.detach().to(torch.float32) - base
                ratio = torch.sqrt(torch.mean(actual_delta.square())) / base_rms
            actual = float(ratio)
            if actual > self.config.relative_rms_budget * (1.0 + 2e-4):
                raise RuntimeError("P0 writer exceeded its hard relative RMS budget")
            self._measurements[kind] = WriterInjectionMeasurement(
                config_id=self.config.config_id,
                feature_kind=kind,
                module_path=module_path,
                relative_rms_budget=self.config.relative_rms_budget,
                actual_relative_rms=actual,
                call_count=1,
                writer_step_index=P0_WRITER_STEP_INDEX,
            )
            return injected

        return hook

    def callback_on_step_end(
        self,
        pipeline: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        del pipeline, timestep
        if not isinstance(step_index, int) or not isinstance(callback_kwargs, dict):
            raise TypeError("diffusers step callback contract differs")
        if step_index != len(self._callback_steps):
            raise RuntimeError("diffusers denoising step order differs")
        self._callback_steps.append(step_index)
        if self._root_call_count != step_index + 1:
            raise RuntimeError("SD3 transformer call topology differs from one call per step")
        if step_index == P0_WRITER_STEP_INDEX - 1:
            if self._armed or self._measurements:
                raise RuntimeError("P0 writer step admission is inconsistent")
            self._armed = True
        elif step_index == P0_WRITER_STEP_INDEX:
            if set(self._measurements) != {"q", "k"}:
                raise RuntimeError("P0 writer did not inject both Q and K exactly once")
            self._armed = False
        return callback_kwargs

    def __enter__(self) -> "ActiveQKWriterSession":
        q_module, k_module = self._resolve_modules()
        if not isinstance(self.transformer, torch.nn.Module):
            raise TypeError("SD3 transformer must be a torch module")
        self._handles = [
            self.transformer.register_forward_pre_hook(self._root_pre_hook),
            q_module.register_forward_hook(
                self._feature_hook("q", f"{self.config.layer_path}.to_q")
            ),
            k_module.register_forward_hook(
                self._feature_hook("k", f"{self.config.layer_path}.to_k")
            ),
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._armed = False

    def assert_complete(self) -> tuple[WriterInjectionMeasurement, ...]:
        if self._callback_steps != list(range(P0_INFERENCE_STEPS)):
            raise RuntimeError("P0 generation did not complete the frozen 20-step schedule")
        if self._root_call_count != P0_INFERENCE_STEPS:
            raise RuntimeError("P0 generation transformer call count differs")
        if set(self._measurements) != {"q", "k"} or self._armed:
            raise RuntimeError("P0 writer injection did not close exactly once")
        return self.measurements
