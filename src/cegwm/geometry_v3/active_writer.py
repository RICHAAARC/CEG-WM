"""Active keyed Q/K writer primitives for Geometry-V3 P0.

Only transient tensors cross this module.  Public artifacts may retain the
scalar measurements returned here, never the key, anchor tensor, or Q/K.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Sequence

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
P0_FINAL_DTYPE_PROJECTION_ROUNDS = 24
P0_Q_DIAGNOSTIC_CHECKPOINTS = (
    "q_output_contract_pass",
    "q_pattern_materialized",
    "q_base_rms_validated",
    "q_delta_materialized",
    "q_ratio_validated",
    "q_budget_validated",
    "q_measurement_recorded",
)
P0D2_Q_DIAGNOSTIC_CHECKPOINTS = (
    "q_output_contract_pass",
    "q_pattern_materialized",
    "q_base_rms_validated",
    "q_delta_materialized",
    "q_ratio_validated",
    "q_initial_budget_comparison_completed",
    "q_correction_branch_entered",
    "q_corrected_output_materialized",
    "q_corrected_delta_materialized",
    "q_post_correction_ratio_computed",
    "q_hard_budget_rejected",
    "q_hard_budget_accepted",
    "q_budget_validated",
    "q_measurement_recorded",
)


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


@dataclass(frozen=True, slots=True)
class WriterScalarObservation:
    """Bounded public scalars from the transient production writer hook."""

    feature_kind: str
    module_path: str
    writer_step_index: int
    token_grid_side: int
    token_count: int
    channel_count: int
    spatial_axis: str
    normalization: str
    injection_sign: str
    contract_pass: bool
    axis_contract_pass: bool
    token_contract_pass: bool
    channel_contract_pass: bool
    normalization_contract_pass: bool
    pre_correct_correlation: float
    pre_wrong_correlation: float
    post_correct_correlation: float
    post_wrong_correlation: float
    actual_relative_rms: float


def _independent_pattern_contract(
    anchor: CanonicalRelationAnchor,
    pattern: torch.Tensor,
    *,
    module_path: str,
) -> tuple[bool, bool, bool, bool]:
    """Independently reconstruct the public row-major pattern contract."""

    if not isinstance(pattern, torch.Tensor) or pattern.ndim < 3:
        return False, False, False, False
    token_count, channel_count = int(pattern.shape[-2]), int(pattern.shape[-1])
    side = math.isqrt(token_count)
    token_contract = side >= 2 and side * side == token_count
    channel_contract = channel_count >= 2
    if not token_contract or not channel_contract or len(anchor.public_digest) != 64:
        return False, token_contract, channel_contract, False
    device = pattern.device
    flat = torch.arange(token_count, device=device, dtype=torch.long)
    rows = torch.div(flat, side, rounding_mode="floor").to(torch.float32)
    columns = torch.remainder(flat, side).to(torch.float32)
    y_centres = (rows + 0.5) / side
    x_centres = (columns + 0.5) / side
    spatial = torch.zeros(token_count, device=device, dtype=torch.float32)
    for index, point in enumerate(anchor.points):
        if len(point) != 2:
            return False, token_contract, channel_contract, False
        px, py = float(point[0]), float(point[1])
        if not math.isfinite(px) or not math.isfinite(py):
            return False, token_contract, channel_contract, False
        sign = 1.0 if index % 2 == 0 else -1.0
        spatial = spatial + sign * torch.exp(
            -((x_centres - px) ** 2 + (y_centres - py) ** 2) / (2.0 * 0.075**2)
        )
    spatial = spatial - spatial.mean()
    spatial_rms = torch.sqrt(torch.mean(spatial.square()))
    if not bool(torch.isfinite(spatial_rms)) or float(spatial_rms) <= 0.0:
        return False, token_contract, channel_contract, False
    spatial = spatial / spatial_rms

    digest = hashlib.sha256(
        (anchor.public_digest + "|" + module_path).encode("ascii")
    ).digest()
    frequency = 1 + digest[0] % 11
    phase = 2.0 * math.pi * int.from_bytes(digest[1:5], "big") / 2**32
    channel_indices = torch.arange(channel_count, device=device, dtype=torch.float32)
    channel_centres = (channel_indices + 0.5) / channel_count
    channel = torch.sin(2.0 * math.pi * frequency * channel_centres + phase)
    channel = channel - channel.mean()
    channel_rms = torch.sqrt(torch.mean(channel.square()))
    if not bool(torch.isfinite(channel_rms)) or float(channel_rms) <= 0.0:
        return False, token_contract, channel_contract, False
    channel = channel / channel_rms
    expected = spatial[:, None] * channel[None, :]
    expected = expected - expected.mean()
    expected = expected / torch.sqrt(torch.mean(expected.square()))
    expected = expected.reshape((1,) * (pattern.ndim - 2) + expected.shape).expand_as(pattern)
    observed = pattern.detach().to(torch.float32)
    normalization_contract = (
        bool(torch.isfinite(observed).all())
        and abs(float(observed.mean())) <= 1e-5
        and abs(float(torch.sqrt(torch.mean(observed.square()))) - 1.0) <= 1e-4
    )
    axis_contract = bool(torch.allclose(observed, expected, rtol=1e-5, atol=1e-5))
    expected_spatial = expected.reshape(-1, token_count, channel_count)[0]
    observed_spatial = observed.reshape(-1, token_count, channel_count)[0]
    token_contract = token_contract and bool(torch.allclose(
        observed_spatial @ channel,
        expected_spatial @ channel,
        rtol=1e-5,
        atol=1e-5,
    ))
    channel_contract = channel_contract and bool(torch.allclose(
        spatial @ observed_spatial,
        spatial @ expected_spatial,
        rtol=1e-5,
        atol=1e-5,
    ))
    return axis_contract, token_contract, channel_contract, normalization_contract


def _project_final_dtype_hard_budget(
    output: torch.Tensor,
    direction_delta: torch.Tensor,
    base: torch.Tensor,
    base_rms: torch.Tensor,
    hard_limit: float,
) -> torch.Tensor | None:
    """Return the largest sampled final-dtype perturbation inside the hard ball."""

    lower, upper = 0.0, 1.0
    best_scale = -1.0
    best: torch.Tensor | None = None
    candidate = (
        output + direction_delta.to(device=output.device, dtype=output.dtype)
    ).to(dtype=output.dtype)
    actual_delta = candidate.detach().to(torch.float32) - base
    ratio = torch.sqrt(torch.mean(actual_delta.square())) / base_rms
    if (
        bool(torch.isfinite(ratio))
        and float(ratio) > 0.0
        and float(ratio) <= hard_limit
    ):
        best_scale, best = 1.0, candidate
    for _ in range(P0_FINAL_DTYPE_PROJECTION_ROUNDS):
        scale = (lower + upper) * 0.5
        candidate = (
            output
            + (direction_delta * scale).to(
                device=output.device,
                dtype=output.dtype,
            )
        ).to(dtype=output.dtype)
        actual_delta = candidate.detach().to(torch.float32) - base
        ratio = torch.sqrt(torch.mean(actual_delta.square())) / base_rms
        finite = bool(torch.isfinite(ratio))
        actual = float(ratio) if finite else math.inf
        if finite and actual > 0.0 and actual <= hard_limit:
            if scale > best_scale:
                best_scale, best = scale, candidate
            lower = scale
        elif finite and actual <= 0.0:
            lower = scale
        else:
            upper = scale
    return best


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
        *,
        q_diagnostic_observer: Callable[[str], None] | None = None,
        q_diagnostic_checkpoints: Sequence[str] = P0_Q_DIAGNOSTIC_CHECKPOINTS,
        scalar_observer: Callable[[WriterScalarObservation], None] | None = None,
        scalar_wrong_anchor: CanonicalRelationAnchor | None = None,
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
        self._q_diagnostic_observer = q_diagnostic_observer
        self._q_diagnostic_checkpoints = tuple(q_diagnostic_checkpoints)
        self._scalar_observer = scalar_observer
        self._scalar_wrong_anchor = scalar_wrong_anchor
        if (scalar_observer is None) != (scalar_wrong_anchor is None):
            raise ValueError("P1M0 scalar observer and wrong anchor must be enabled together")
        if self._q_diagnostic_checkpoints not in {
            P0_Q_DIAGNOSTIC_CHECKPOINTS,
            P0D2_Q_DIAGNOSTIC_CHECKPOINTS,
        }:
            raise ValueError("P0 Q diagnostic checkpoint roster is not public")

    def _observe_q_checkpoint(self, kind: str, checkpoint: str) -> None:
        if kind != "q" or self._q_diagnostic_observer is None:
            return
        if checkpoint not in P0D2_Q_DIAGNOSTIC_CHECKPOINTS:
            raise RuntimeError("P0 Q diagnostic checkpoint is not public")
        if checkpoint not in self._q_diagnostic_checkpoints:
            return
        self._q_diagnostic_observer(checkpoint)

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
            self._observe_q_checkpoint(kind, "q_output_contract_pass")
            pattern = canonical_qk_pattern(self.anchor, output, module_path=module_path)
            self._observe_q_checkpoint(kind, "q_pattern_materialized")
            base = output.detach().to(torch.float32)
            base_rms = torch.sqrt(torch.mean(base.square()))
            if not bool(torch.isfinite(base_rms)) or float(base_rms) <= 0.0:
                raise ValueError("SD3 Q/K output has zero or nonfinite RMS")
            self._observe_q_checkpoint(kind, "q_base_rms_validated")
            delta = pattern.to(torch.float32) * base_rms * self.config.relative_rms_budget
            injected = (
                output + delta.to(device=output.device, dtype=output.dtype)
            ).to(dtype=output.dtype)
            actual_delta = injected.detach().to(torch.float32) - base
            self._observe_q_checkpoint(kind, "q_delta_materialized")
            ratio = torch.sqrt(torch.mean(actual_delta.square())) / base_rms
            if not bool(torch.isfinite(ratio)):
                raise ValueError("P0 writer actual relative RMS is invalid")
            if float(ratio) > 0.0:
                self._observe_q_checkpoint(kind, "q_ratio_validated")
            needs_correction = (
                float(ratio) <= 0.0
                or float(ratio) > self.config.relative_rms_budget * (1.0 + 1e-4)
            )
            self._observe_q_checkpoint(
                kind, "q_initial_budget_comparison_completed"
            )
            if needs_correction:
                self._observe_q_checkpoint(kind, "q_correction_branch_entered")
                hard_limit = self.config.relative_rms_budget * (1.0 + 2e-4)
                projected = _project_final_dtype_hard_budget(
                    output,
                    delta,
                    base,
                    base_rms,
                    hard_limit,
                )
                if projected is None:
                    self._observe_q_checkpoint(kind, "q_hard_budget_rejected")
                    raise RuntimeError(
                        "P0 writer has no representable positive hard-budget injection"
                    )
                injected = projected
                self._observe_q_checkpoint(kind, "q_corrected_output_materialized")
                actual_delta = injected.detach().to(torch.float32) - base
                self._observe_q_checkpoint(kind, "q_corrected_delta_materialized")
                ratio = torch.sqrt(torch.mean(actual_delta.square())) / base_rms
                self._observe_q_checkpoint(kind, "q_post_correction_ratio_computed")
            actual = float(ratio)
            hard_limit = self.config.relative_rms_budget * (1.0 + 2e-4)
            if not math.isfinite(actual) or actual <= 0.0 or actual > hard_limit:
                self._observe_q_checkpoint(kind, "q_hard_budget_rejected")
                raise RuntimeError("P0 writer exceeded its hard relative RMS budget")
            self._observe_q_checkpoint(kind, "q_hard_budget_accepted")
            self._observe_q_checkpoint(kind, "q_budget_validated")
            self._measurements[kind] = WriterInjectionMeasurement(
                config_id=self.config.config_id,
                feature_kind=kind,
                module_path=module_path,
                relative_rms_budget=self.config.relative_rms_budget,
                actual_relative_rms=actual,
                call_count=1,
                writer_step_index=P0_WRITER_STEP_INDEX,
            )
            if self._scalar_observer is not None:
                wrong_pattern = canonical_qk_pattern(
                    self._scalar_wrong_anchor,
                    output,
                    module_path=module_path,
                )
                token_count, channel_count = int(output.shape[-2]), int(output.shape[-1])
                token_grid_side = math.isqrt(token_count)
                pattern32 = pattern.detach().to(torch.float32)
                pattern_mean = float(pattern32.mean())
                pattern_rms = float(torch.sqrt(torch.mean(pattern32.square())))
                signed_projection = float(torch.sum(actual_delta * pattern32))
                (
                    axis_contract_pass,
                    token_contract_pass,
                    channel_contract_pass,
                    normalization_contract_pass,
                ) = _independent_pattern_contract(
                    self.anchor, pattern, module_path=module_path,
                )
                contract_pass = (
                    kind in {"q", "k"}
                    and module_path == f"{self.config.layer_path}.to_{kind}"
                    and token_grid_side * token_grid_side == token_count
                    and channel_count >= 2
                    and math.isfinite(pattern_mean)
                    and math.isfinite(pattern_rms)
                    and axis_contract_pass
                    and token_contract_pass
                    and channel_contract_pass
                    and normalization_contract_pass
                    and math.isfinite(signed_projection)
                    and signed_projection > 0.0
                )
                self._scalar_observer(WriterScalarObservation(
                    feature_kind=kind,
                    module_path=module_path,
                    writer_step_index=P0_WRITER_STEP_INDEX,
                    token_grid_side=token_grid_side,
                    token_count=token_count,
                    channel_count=channel_count,
                    spatial_axis="row_major_yx",
                    normalization="zero_mean_unit_rms",
                    injection_sign="positive",
                    contract_pass=contract_pass,
                    axis_contract_pass=axis_contract_pass,
                    token_contract_pass=token_contract_pass,
                    channel_contract_pass=channel_contract_pass,
                    normalization_contract_pass=normalization_contract_pass,
                    pre_correct_correlation=normalized_pattern_correlation(output, pattern),
                    pre_wrong_correlation=normalized_pattern_correlation(output, wrong_pattern),
                    post_correct_correlation=normalized_pattern_correlation(injected, pattern),
                    post_wrong_correlation=normalized_pattern_correlation(injected, wrong_pattern),
                    actual_relative_rms=actual,
                ))
            self._observe_q_checkpoint(kind, "q_measurement_recorded")
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
