from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cegwm.geometry_v3.active_writer import (
    ActiveQKWriterSession,
    P0_CONFIGS,
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
    def __init__(self, *, extra_call_at: int | None = None) -> None:
        self.transformer = _Transformer()
        self.extra_call_at = extra_call_at

    def __call__(
        self,
        *,
        callback_on_step_end: object,
        callback_on_step_end_tensor_inputs: list[str],
        **kwargs: object,
    ) -> SimpleNamespace:
        assert callback_on_step_end_tensor_inputs == ["latents"]
        assert kwargs["num_inference_steps"] == P0_INFERENCE_STEPS
        hidden = torch.linspace(-1.0, 1.0, 16 * 8).reshape(1, 16, 8)
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
    hook = session._feature_hook("q", "transformer_blocks.4.attn.to_q")
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
def test_q_hard_budget_rejection_is_observed_before_existing_runtime_error() -> None:
    observed: list[str] = []
    output = torch.full((1, 4, 4), 0.1, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="hard relative RMS budget"):
        _invoke_q_hook_with_output(output, observed.append)

    assert observed[-1] == "q_hard_budget_rejected"
    assert "q_hard_budget_accepted" not in observed
    assert "q_budget_validated" not in observed
    assert "q_measurement_recorded" not in observed
