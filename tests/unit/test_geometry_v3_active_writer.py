from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import cegwm.geometry_v3.active_writer as ACTIVE

from cegwm.geometry_v3.active_writer import (
    ActiveQKWriterSession,
    P0_CONFIGS,
    P0_FINAL_DTYPE_PROJECTION_ROUNDS,
    P0_INFERENCE_STEPS,
    P0_PLACEMENT_BLOCKS,
    P0_Q_DIAGNOSTIC_CHECKPOINTS,
    P0D2_Q_DIAGNOSTIC_CHECKPOINTS,
    P0_WRITER_STEP_INDEX,
    canonical_qk_pattern,
)
from cegwm.geometry_v3.contracts import derive_canonical_relation_anchor
from cegwm.geometry_v3.operational import generate_writer_config


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8, bias=False)
        self.to_k = torch.nn.Linear(8, 8, bias=False)
        torch.nn.init.eye_(self.to_q.weight)
        torch.nn.init.eye_(self.to_k.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return (self.to_q(hidden) + self.to_k(hidden)) * 0.5


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden)


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(_Block() for _ in range(21))
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden = block(hidden)
        return hidden


class _Pipeline:
    def __init__(
        self,
        *,
        extra_call_at: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.transformer = _Transformer().to(dtype=dtype)
        self.extra_call_at = extra_call_at
        self.dtype = dtype

    def __call__(
        self,
        *,
        callback_on_step_end: object,
        callback_on_step_end_tensor_inputs: list[str],
        **kwargs: object,
    ) -> SimpleNamespace:
        assert callback_on_step_end_tensor_inputs == ["latents"]
        assert kwargs["num_inference_steps"] == P0_INFERENCE_STEPS
        hidden = torch.linspace(-1.0, 1.0, 16 * 8).reshape(1, 16, 8).to(self.dtype)
        state = {"latents": torch.zeros((1, 4, 4, 4))}
        for step in range(P0_INFERENCE_STEPS):
            hidden = self.transformer(hidden)
            if step == self.extra_call_at:
                hidden = self.transformer(hidden)
            state = callback_on_step_end(self, step, torch.tensor(step), state)
        return SimpleNamespace(images=[Image.new("RGB", (512, 512), (12, 34, 56))])


@pytest.mark.unit
def test_fixed_p0_configs_exclude_passive_v1_layers() -> None:
    assert P0_PLACEMENT_BLOCKS == (4, 12, 20)
    assert len(P0_CONFIGS) == 6
    assert {config.block_index for config in P0_CONFIGS}.isdisjoint({14, 23})


@pytest.mark.unit
def test_keyed_patterns_are_zero_mean_unit_rms_and_key_separated() -> None:
    like = torch.ones((1, 16, 8), dtype=torch.float32)
    first = derive_canonical_relation_anchor("geometry-key-alpha", point_count=16)
    second = derive_canonical_relation_anchor("geometry-key-bravo", point_count=16)
    a = canonical_qk_pattern(first, like, module_path="transformer_blocks.4.attn.to_q")
    b = canonical_qk_pattern(second, like, module_path="transformer_blocks.4.attn.to_q")
    assert float(a.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(torch.sqrt(torch.mean(a.square()))) == pytest.approx(1.0, abs=1e-6)
    assert not torch.allclose(a, b)


@pytest.mark.unit
@pytest.mark.parametrize("config", P0_CONFIGS)
def test_real_torch_hooks_write_qk_once_at_step18_with_hard_budget_and_cleanup(config) -> None:
    pipeline = _Pipeline()
    anchor = derive_canonical_relation_anchor("geometry-key-0001", point_count=16)
    q_module = pipeline.transformer.transformer_blocks[config.block_index].attn.to_q
    k_module = pipeline.transformer.transformer_blocks[config.block_index].attn.to_k

    generated = generate_writer_config(pipeline, config, anchor)

    assert generated.image.mode == "RGB"
    assert {item.feature_kind for item in generated.measurements} == {"q", "k"}
    for item in generated.measurements:
        assert item.writer_step_index == P0_WRITER_STEP_INDEX
        assert item.call_count == 1
        assert 0.0 < item.actual_relative_rms <= config.relative_rms_budget * 1.0002
    assert not q_module._forward_hooks
    assert not k_module._forward_hooks
    assert not pipeline.transformer._forward_pre_hooks


def _independent_actual_relative_rms(
    base: torch.Tensor, injected: torch.Tensor
) -> float:
    base32 = base.detach().to(torch.float32)
    actual_delta = injected.detach().to(torch.float32) - base32
    assert torch.isfinite(actual_delta).all()
    return float(
        torch.sqrt(torch.mean(actual_delta.square()))
        / torch.sqrt(torch.mean(base32.square()))
    )


@pytest.mark.unit
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_registered_production_qk_hooks_complete_once_in_final_dtype(dtype) -> None:
    pipeline = _Pipeline(dtype=dtype)
    config = P0_CONFIGS[0]
    assert config.config_id == "block4-qk-rms0p0025"
    assert P0_WRITER_STEP_INDEX == 18
    session = ActiveQKWriterSession(
        pipeline.transformer,
        config,
        derive_canonical_relation_anchor("geometry-key-0001", point_count=16),
    )
    captures: dict[str, torch.Tensor] = {}

    def capture(kind: str):
        def hook(module, inputs, output) -> None:
            del module, inputs
            if session._current_transformer_call == P0_WRITER_STEP_INDEX:
                captures[kind] = output.detach().clone()

        return hook

    with session:
        q_module, k_module = session._resolve_modules()
        q_capture = q_module.register_forward_hook(capture("q"))
        k_capture = k_module.register_forward_hook(capture("k"))
        try:
            generated = pipeline(
                prompt="public prompt",
                num_inference_steps=P0_INFERENCE_STEPS,
                height=512,
                width=512,
                generator=torch.Generator().manual_seed(73),
                output_type="pil",
                callback_on_step_end=session.callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        finally:
            q_capture.remove()
            k_capture.remove()
    measurements = session.assert_complete()

    original = torch.linspace(-1.0, 1.0, 16 * 8).reshape(1, 16, 8).to(dtype)
    assert generated.images[0].mode == "RGB"
    assert set(captures) == {"q", "k"}
    assert {item.feature_kind for item in measurements} == {"q", "k"}
    for item in measurements:
        injected = captures[item.feature_kind]
        actual = _independent_actual_relative_rms(original, injected)
        assert injected.dtype == dtype
        assert injected.shape == original.shape
        assert torch.isfinite(injected).all()
        assert 0.0 < actual <= config.relative_rms_budget * (1.0 + 2e-4)
        assert item.actual_relative_rms == pytest.approx(actual, rel=0.0, abs=1e-12)
        assert item.call_count == 1


@pytest.mark.unit
def test_incompatible_transformer_call_topology_fails_closed_and_cleans_hooks() -> None:
    pipeline = _Pipeline(extra_call_at=5)
    config = P0_CONFIGS[0]
    anchor = derive_canonical_relation_anchor("geometry-key-0001", point_count=16)
    q_module = pipeline.transformer.transformer_blocks[config.block_index].attn.to_q
    with pytest.raises(RuntimeError, match="call topology"):
        generate_writer_config(pipeline, config, anchor)
    assert not q_module._forward_hooks
    assert not pipeline.transformer._forward_pre_hooks


@pytest.mark.unit
def test_writer_session_rejects_missing_declared_module_path() -> None:
    transformer = _Transformer()
    del transformer.transformer_blocks[4].attn.to_q
    session = ActiveQKWriterSession(
        transformer,
        P0_CONFIGS[0],
        derive_canonical_relation_anchor("geometry-key-0001", point_count=16),
    )
    with pytest.raises(RuntimeError, match="Q/K modules"):
        session.__enter__()


def _run_fixed_session(observer=None):
    pipeline = _Pipeline()
    session = ActiveQKWriterSession(
        pipeline.transformer,
        P0_CONFIGS[0],
        derive_canonical_relation_anchor("geometry-key-0001", point_count=16),
        q_diagnostic_observer=observer,
    )
    with session:
        output = pipeline(
            prompt="public prompt",
            num_inference_steps=P0_INFERENCE_STEPS,
            height=512,
            width=512,
            generator=torch.Generator().manual_seed(73),
            output_type="pil",
            callback_on_step_end=session.callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    return output, session.assert_complete()


@pytest.mark.unit
def test_q_diagnostic_observer_is_default_off_and_semantically_neutral() -> None:
    observed: list[str] = []
    default_output, default_measurements = _run_fixed_session()
    observed_output, observed_measurements = _run_fixed_session(observed.append)

    assert observed == [
        "q_output_contract_pass",
        "q_pattern_materialized",
        "q_base_rms_validated",
        "q_delta_materialized",
        "q_ratio_validated",
        "q_budget_validated",
        "q_measurement_recorded",
    ]
    assert default_output.images[0].tobytes() == observed_output.images[0].tobytes()
    assert default_measurements == observed_measurements


@pytest.mark.unit
@pytest.mark.parametrize(
    "stop_checkpoint",
    (
        "q_output_contract_pass",
        "q_pattern_materialized",
        "q_base_rms_validated",
        "q_delta_materialized",
        "q_ratio_validated",
        "q_budget_validated",
        "q_measurement_recorded",
    ),
)
def test_q_diagnostic_observer_reports_only_completed_production_checkpoints(
    stop_checkpoint: str,
) -> None:
    observed: list[str] = []

    def observer(checkpoint: str) -> None:
        observed.append(checkpoint)
        if checkpoint == stop_checkpoint:
            raise RuntimeError("diagnostic observer stop")

    with pytest.raises(RuntimeError, match="diagnostic observer stop"):
        _run_fixed_session(observer)

    expected = [
        "q_output_contract_pass",
        "q_pattern_materialized",
        "q_base_rms_validated",
        "q_delta_materialized",
        "q_ratio_validated",
        "q_budget_validated",
        "q_measurement_recorded",
    ]
    stop_index = expected.index(stop_checkpoint)
    assert observed == expected[: stop_index + 1]
    assert all(isinstance(item, str) for item in observed)


def _invoke_q_hook_with_output(output: torch.Tensor, observer):
    return _invoke_feature_hook_with_output("q", output, observer)


def _invoke_feature_hook_with_output(kind: str, output: torch.Tensor, observer=None):
    transformer = _Transformer()
    session = ActiveQKWriterSession(
        transformer,
        P0_CONFIGS[0],
        derive_canonical_relation_anchor("geometry-key-0001", point_count=16),
        q_diagnostic_observer=observer,
        q_diagnostic_checkpoints=P0D2_Q_DIAGNOSTIC_CHECKPOINTS,
    )
    session._armed = True
    session._current_transformer_call = P0_WRITER_STEP_INDEX
    hook = session._feature_hook(kind, f"transformer_blocks.4.attn.to_{kind}")
    return hook(None, (), output), session


@pytest.mark.unit
def test_q_correction_path_reports_accepted_checkpoints_without_math_change() -> None:
    observed: list[str] = []
    output = (
        torch.randn((1, 4, 4), generator=torch.Generator().manual_seed(28))
        * 1000.0
    ).to(torch.float16)

    injected, session = _invoke_q_hook_with_output(output, observed.append)

    assert injected.dtype == output.dtype
    assert {measurement.feature_kind for measurement in session.measurements} == {"q"}
    assert observed == [
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
        "q_hard_budget_accepted",
        "q_budget_validated",
        "q_measurement_recorded",
    ]


@pytest.mark.unit
def test_final_dtype_projection_repairs_previous_float16_hard_rejection() -> None:
    observed: list[str] = []
    output = torch.full((1, 4, 4), 0.1, dtype=torch.float16)

    injected, session = _invoke_q_hook_with_output(output, observed.append)

    actual = _independent_actual_relative_rms(output, injected)
    assert P0_FINAL_DTYPE_PROJECTION_ROUNDS == 24
    assert injected.dtype == output.dtype
    assert injected.device == output.device
    assert injected.shape == output.shape
    assert torch.isfinite(injected).all()
    assert 0.0 < actual <= P0_CONFIGS[0].relative_rms_budget * (1.0 + 2e-4)
    assert session.measurements[0].actual_relative_rms == pytest.approx(actual)
    assert "q_correction_branch_entered" in observed
    assert observed[-3:] == [
        "q_hard_budget_accepted",
        "q_budget_validated",
        "q_measurement_recorded",
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dtype", "magnitude", "offset"),
    (
        pytest.param(torch.float32, 1.0, 0.0, id="float32-normal"),
        pytest.param(torch.float16, 0.0, 0.1, id="float16-near-ulp"),
        pytest.param(torch.bfloat16, 0.0, 1.0, id="bfloat16-near-ulp"),
        pytest.param(torch.float32, 1e-10, 0.0, id="float32-tiny"),
        pytest.param(torch.float16, 1e-3, 0.0, id="float16-tiny"),
        pytest.param(torch.bfloat16, 1e-4, 0.0, id="bfloat16-tiny"),
        pytest.param(torch.float32, 1e10, 0.0, id="float32-large"),
        pytest.param(torch.float16, 1e3, 0.0, id="float16-large"),
        pytest.param(torch.bfloat16, 1e10, 0.0, id="bfloat16-large"),
    ),
)
@pytest.mark.parametrize("kind", ("q", "k"))
def test_final_dtype_projection_is_deterministic_and_independently_bounded(
    dtype: torch.dtype,
    magnitude: float,
    offset: float,
    kind: str,
) -> None:
    noise = torch.randn(
        (1, 16, 8), generator=torch.Generator().manual_seed(830)
    )
    output = (noise * magnitude + offset).to(dtype)

    first, first_session = _invoke_feature_hook_with_output(kind, output)
    second, second_session = _invoke_feature_hook_with_output(kind, output.clone())
    actual = _independent_actual_relative_rms(output, first)

    assert torch.equal(first, second)
    assert first.dtype == output.dtype
    assert first.device == output.device
    assert first.shape == output.shape
    assert torch.isfinite(first).all()
    assert math.isfinite(actual)
    assert 0.0 < actual <= P0_CONFIGS[0].relative_rms_budget * (1.0 + 2e-4)
    assert first_session.measurements[0].feature_kind == kind
    assert first_session.measurements == second_session.measurements


@pytest.mark.unit
@pytest.mark.parametrize("kind", ("q", "k"))
def test_final_dtype_projection_fails_closed_without_positive_representable_delta(
    kind: str,
) -> None:
    smallest_subnormal = torch.finfo(torch.float16).smallest_normal * 2**-10
    output = torch.full((1, 4, 4), smallest_subnormal, dtype=torch.float16)
    observed: list[str] = []

    with pytest.raises(RuntimeError, match="no representable positive"):
        _invoke_feature_hook_with_output(kind, output, observed.append)

    if kind == "q":
        assert observed[-1] == "q_hard_budget_rejected"
        assert "q_hard_budget_accepted" not in observed
        assert "q_measurement_recorded" not in observed


def _independent_normalized_correlation(value: torch.Tensor, pattern: torch.Tensor) -> float:
    left = value.detach().to(torch.float64)
    right = pattern.detach().to(torch.float64)
    left = left - left.mean(dim=(-2, -1), keepdim=True)
    right = right - right.mean(dim=(-2, -1), keepdim=True)
    return float(torch.sum(left * right) / torch.sqrt(torch.sum(left.square()) * torch.sum(right.square())))


def _handwritten_pattern(anchor, like: torch.Tensor, module_path: str) -> torch.Tensor:
    token_count, channel_count = like.shape[-2:]
    side = math.isqrt(token_count)
    flat = torch.arange(token_count, dtype=torch.long)
    row = torch.div(flat, side, rounding_mode="floor").to(torch.float64)
    column = torch.remainder(flat, side).to(torch.float64)
    y = (row + 0.5) / side
    x = (column + 0.5) / side
    spatial = torch.zeros(token_count, dtype=torch.float64)
    for index, (px, py) in enumerate(anchor.points):
        spatial += (1.0 if index % 2 == 0 else -1.0) * torch.exp(
            -((x - px) ** 2 + (y - py) ** 2) / (2.0 * 0.075**2)
        )
    spatial -= spatial.mean()
    spatial /= torch.sqrt(torch.mean(spatial.square()))
    digest = hashlib.sha256(
        (anchor.public_digest + "|" + module_path).encode("ascii")
    ).digest()
    frequency = 1 + digest[0] % 11
    phase = 2.0 * math.pi * int.from_bytes(digest[1:5], "big") / 2**32
    channel_index = torch.arange(channel_count, dtype=torch.float64)
    channel = torch.sin(
        2.0 * math.pi * frequency * ((channel_index + 0.5) / channel_count) + phase
    )
    channel -= channel.mean()
    channel /= torch.sqrt(torch.mean(channel.square()))
    expected = spatial[:, None] * channel[None, :]
    expected -= expected.mean()
    expected /= torch.sqrt(torch.mean(expected.square()))
    return expected.reshape((1,) * (like.ndim - 2) + expected.shape).expand_as(like).to(like.dtype)


@pytest.mark.unit
def test_default_disabled_scalar_observer_reports_only_bounded_public_contract(
) -> None:
    output = torch.linspace(-0.7, 0.9, 16 * 8).reshape(1, 16, 8)
    correct_anchor = derive_canonical_relation_anchor("geometry-key-0001", point_count=16)
    wrong_anchor = derive_canonical_relation_anchor("geometry-key-0002", point_count=16)
    module_path = "transformer_blocks.4.attn.to_q"
    assert any(abs(float(px) - float(py)) > 1e-6 for px, py in correct_anchor.points)
    correct = _handwritten_pattern(correct_anchor, output, module_path)
    wrong = _handwritten_pattern(wrong_anchor, output, module_path)
    observations = []
    transformer = _Transformer()
    session = ActiveQKWriterSession(
        transformer, P0_CONFIGS[0], correct_anchor,
        scalar_observer=observations.append, scalar_wrong_anchor=wrong_anchor,
    )
    session._armed = True
    session._current_transformer_call = P0_WRITER_STEP_INDEX
    injected = session._feature_hook("q", module_path)(None, (), output)

    assert len(observations) == 1
    observed = observations[0]
    assert observed.contract_pass is True
    assert observed.feature_kind == "q"
    assert observed.module_path == "transformer_blocks.4.attn.to_q"
    assert observed.spatial_axis == "row_major_yx"
    assert observed.normalization == "zero_mean_unit_rms"
    assert observed.injection_sign == "positive"
    assert observed.axis_contract_pass is True
    assert observed.token_contract_pass is True
    assert observed.channel_contract_pass is True
    assert observed.normalization_contract_pass is True
    assert (observed.token_grid_side, observed.token_count, observed.channel_count) == (4, 16, 8)
    assert observed.pre_correct_correlation == pytest.approx(
        _independent_normalized_correlation(output, correct), abs=1e-7,
    )
    assert observed.pre_wrong_correlation == pytest.approx(
        _independent_normalized_correlation(output, wrong), abs=1e-7,
    )
    assert observed.post_correct_correlation == pytest.approx(
        _independent_normalized_correlation(injected, correct), abs=1e-7,
    )
    assert observed.post_wrong_correlation == pytest.approx(
        _independent_normalized_correlation(injected, wrong), abs=1e-7,
    )
    assert 0.0 < observed.actual_relative_rms <= 0.0025 * 1.0002


@pytest.mark.unit
@pytest.mark.parametrize("permutation", ("axis", "token", "channel"))
def test_independent_scalar_contract_rejects_mean_rms_preserving_pattern_permutations(
    monkeypatch: pytest.MonkeyPatch, permutation: str,
) -> None:
    output = torch.linspace(-0.7, 0.9, 16 * 8).reshape(1, 16, 8)
    correct_anchor = derive_canonical_relation_anchor("geometry-key-0001", point_count=16)
    wrong_anchor = derive_canonical_relation_anchor("geometry-key-0002", point_count=16)
    original = ACTIVE.canonical_qk_pattern

    def permuted(anchor, like, *, module_path, transformed_points=None):
        value = original(
            anchor, like, module_path=module_path,
            transformed_points=transformed_points,
        )
        if permutation == "axis":
            return value.reshape(1, 4, 4, 8).transpose(1, 2).reshape_as(value)
        if permutation == "token":
            return torch.roll(value, shifts=1, dims=-2)
        return torch.roll(value, shifts=1, dims=-1)

    monkeypatch.setattr(ACTIVE, "canonical_qk_pattern", permuted)
    observed = []
    session = ActiveQKWriterSession(
        _Transformer(), P0_CONFIGS[0], correct_anchor,
        scalar_observer=observed.append, scalar_wrong_anchor=wrong_anchor,
    )
    session._armed = True
    session._current_transformer_call = P0_WRITER_STEP_INDEX
    session._feature_hook("q", "transformer_blocks.4.attn.to_q")(None, (), output)

    assert len(observed) == 1
    assert float(torch.abs(ACTIVE.canonical_qk_pattern(
        correct_anchor, output, module_path="transformer_blocks.4.attn.to_q",
    ).mean())) <= 1e-5
    assert float(torch.sqrt(torch.mean(ACTIVE.canonical_qk_pattern(
        correct_anchor, output, module_path="transformer_blocks.4.attn.to_q",
    ).square()))) == pytest.approx(1.0, abs=1e-4)
    assert observed[0].contract_pass is False
    assert observed[0].axis_contract_pass is False


@pytest.mark.unit
def test_scalar_observer_requires_paired_wrong_anchor() -> None:
    with pytest.raises(ValueError, match="enabled together"):
        ActiveQKWriterSession(
            _Transformer(), P0_CONFIGS[0],
            derive_canonical_relation_anchor("geometry-key-0001", point_count=16),
            scalar_observer=lambda _: None,
        )
