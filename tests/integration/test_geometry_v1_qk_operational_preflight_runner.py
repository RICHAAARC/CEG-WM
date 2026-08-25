from __future__ import annotations

import json
import pytest
import torch
from PIL import Image
from pathlib import Path

from experiments import run_geometry_v1_qk_operational_preflight as runner
from cegwm.runtime.sd35_qk_observation import SD35QKLayerObservation, SD35QKObservation


def test_layer_discovery_requires_distinct_real_attention_blocks() -> None:
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList()
    for _ in range(2):
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q, block.attn.to_k = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        transformer.transformer_blocks.append(block)
    assert runner._discover_layers(transformer) == ("transformer_blocks.0.attn", "transformer_blocks.1.attn")


def test_layer_discovery_fails_closed_for_one_block() -> None:
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList()
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.to_q, block.attn.to_k = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    transformer.transformer_blocks.append(block)
    with pytest.raises(ValueError, match="distinct"):
        runner._discover_layers(transformer)


def test_architecture_record_is_complete_for_only_sample_side_candidates() -> None:
    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = torch.nn.Linear(3, 4, bias=False)
            self.to_k = torch.nn.Linear(3, 4, bias=False)
            self.add_q_proj = torch.nn.Linear(3, 4, bias=False)
            self.add_k_proj = torch.nn.Linear(3, 4, bias=False)
            self.to_qkv = torch.nn.Linear(3, 12, bias=False)
            self.processor = object()

    transformer = torch.nn.Module()
    transformer.config = type("Config", (), {"num_layers": 2, "patch_size": 2, "in_channels": 4})()
    transformer.transformer_blocks = torch.nn.ModuleList()
    for _ in range(2):
        block = torch.nn.Module()
        block.attn = Attention()
        block.attn2 = torch.nn.Module()
        transformer.transformer_blocks.append(block)
    candidates = runner._discover_candidates(transformer)
    record = runner._architecture_record(transformer, candidates)
    assert record["config"]["num_layers"] == 2
    assert [item["path"] for item in record["attention_candidates"]] == [
        "transformer_blocks.0.attn", "transformer_blocks.1.attn",
    ]
    first = record["attention_candidates"][0]
    assert first["to_q"]["present"] and first["to_k"]["present"]
    assert first["to_q"]["weight_shape"] == [4, 3]
    assert first["other_routes"] == {"attn2": True, "add_q_proj": True, "add_k_proj": True, "to_qkv": True}


def test_observation_record_retains_runtime_source_shape_dtype_device_and_grids() -> None:
    source = torch.ones((1, 16, 4), dtype=torch.float32)
    layer = SD35QKLayerObservation(
        layer_path="transformer_blocks.0.attn",
        query=source[0, :2].detach().to(dtype=torch.float32),
        key=source[0, :2].detach().to(dtype=torch.float32),
        source_dtype=source.dtype,
        source_device=source.device,
        source_shape=(1, 16, 4),
        source_grid=(4, 4),
        sample_indices=torch.tensor([0, 1]),
        heads=2,
        head_dim=2,
    )
    record = runner._observation_record(
        SD35QKObservation(layers=(layer,), latent_shape=(1, 4, 8, 8), schedule_index=7, timestep=torch.tensor([1.0]), public_noise_seed=0),
        elapsed_seconds=0.0,
    )
    assert record["latent_grid"] == [8, 8]
    assert record["patch_grid"] == [4, 4] and record["token_count"] == 16
    assert record["layers"][0]["source_query_shape"] == [1, 16, 4]
    assert record["layers"][0]["source_key_shape"] == [1, 16, 4]
    assert record["layers"][0]["source_dtype"] == "torch.float32"
    assert record["layers"][0]["source_device"] == "cpu"


def test_unknown_public_revision_is_recorded_without_placeholder() -> None:
    pipeline = type("Pipeline", (), {})()
    pipeline.vae = torch.nn.Linear(1, 1)
    pipeline.transformer = torch.nn.Linear(1, 1)
    pipeline.scheduler = object()
    pipeline.image_processor = object()
    record = runner._runtime_record(pipeline)
    assert record["requested_revision"] is None
    assert record["resolved_revision"] is None
    assert record["resolution_status"] == "unavailable_from_public_runtime"


def test_snapshot_extraction_never_returns_a_path() -> None:
    assert runner._snapshot_hex("/cache/models/snapshots/abcdef0123456/component") == "abcdef0123456"
    assert runner._snapshot_hex("/cache/models/no-commit") is None


def test_component_identity_is_uniform_and_never_leaks_paths() -> None:
    class Config:
        _name_or_path = "/private/cache/snapshots/abcdef0123456/config"
        public_scalar = 1

    pipeline = type("Pipeline", (), {"_name_or_path": "/private/cache/snapshots/abcdef0123456/model", "config": Config()})()
    for name in ("vae", "transformer", "scheduler", "image_processor"):
        setattr(pipeline, name, type("Component", (), {"config": Config()})())
    components = runner._runtime_record(pipeline)["components"]
    assert set(components) == {"pipeline", "vae", "transformer", "scheduler", "image_processor"}
    expected = {"class", "config_class", "commit_candidate", "snapshot_candidate", "sanitized_config_digest", "public_name_or_path"}
    assert all(set(record) == expected for record in components.values())
    assert components["pipeline"]["snapshot_candidate"] is None
    assert "/private" not in repr(components)


def test_unique_public_snapshot_is_a_revision_candidate() -> None:
    pipeline = type("Pipeline", (), {"_name_or_path": "/x/snapshots/abcdef0123456/model", "config": object()})()
    assert runner._public_revision(pipeline) == ("abcdef0123456", "unique_public_snapshot")


def test_preflight_requires_cuda_before_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "_validate_execution_exact", lambda *_args: "geometry-v1-b2b-000000000000-operational-01")
    with pytest.raises(RuntimeError, match="cuda_required"):
        runner.operational_preflight([Image.new("RGB", (2, 2))], hf_token="token", root_key="key", expected_exact="0" * 40, repo_root=Path("."))


def test_null_conditioning_uses_only_sd3_hidden_and_pooled_tuple_slots() -> None:
    hidden = torch.full((1, 3, 2), 1.25, dtype=torch.float32)
    pooled = torch.full((1, 5), 9.5, dtype=torch.float64)

    class Pipeline:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def encode_prompt(
            self,
            prompt: str,
            prompt_2: str,
            prompt_3: str,
            *,
            do_classifier_free_guidance: bool,
        ) -> tuple[torch.Tensor, None, torch.Tensor, None]:
            self.calls.append({
                "prompt": prompt,
                "prompt_2": prompt_2,
                "prompt_3": prompt_3,
                "do_classifier_free_guidance": do_classifier_free_guidance,
            })
            return hidden, None, pooled, None

    pipeline = Pipeline()
    actual_hidden, actual_pooled, record = runner._null_conditioning(pipeline)
    assert pipeline.calls == [{"prompt": "", "prompt_2": "", "prompt_3": "", "do_classifier_free_guidance": False}]
    assert actual_hidden is not hidden and torch.equal(actual_hidden, hidden)
    assert actual_pooled is not pooled and torch.equal(actual_pooled, pooled)
    assert actual_hidden.shape != actual_pooled.shape
    assert record["hidden_shape"] == [1, 3, 2]
    assert record["pooled_shape"] == [1, 5]


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ((torch.ones((1, 2)), torch.ones((1, 2))), "four-item"),
        ((torch.ones((1, 2)), None, torch.ones((1, 2))), "four-item"),
        (("not-a-tensor", None, torch.ones((1, 2)), None), "tensors"),
        ((torch.ones((1, 2), dtype=torch.int64), None, torch.ones((1, 2)), None), "floating"),
        ((torch.ones(2), None, torch.ones((1, 2)), None), "rank"),
        ((torch.ones((2, 2)), None, torch.ones((1, 2)), None), "batch one"),
        ((torch.tensor([[float("nan")]]), None, torch.ones((1, 2)), None), "finite"),
    ],
)
def test_null_conditioning_rejects_non_sd3_or_invalid_selected_values(result: object, error: str) -> None:
    class Pipeline:
        def encode_prompt(
            self,
            prompt: str,
            prompt_2: str,
            prompt_3: str,
            *,
            do_classifier_free_guidance: bool,
        ) -> object:
            assert (prompt, prompt_2, prompt_3, do_classifier_free_guidance) == ("", "", "", False)
            return result

    with pytest.raises((TypeError, ValueError), match=error):
        runner._null_conditioning(Pipeline())


def test_null_conditioning_requires_all_three_sd3_prompt_arguments() -> None:
    class Pipeline:
        def encode_prompt(
            self,
            prompt: str,
            prompt_2: str,
            prompt_3: str,
            *,
            do_classifier_free_guidance: bool,
        ) -> tuple[torch.Tensor, None, torch.Tensor, None]:
            return torch.ones((1, 2)), None, torch.ones((1, 3)), None

    pipeline = Pipeline()
    with pytest.raises(TypeError):
        pipeline.encode_prompt(prompt="", do_classifier_free_guidance=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        pipeline.encode_prompt(prompt="", prompt_2="", do_classifier_free_guidance=False)  # type: ignore[call-arg]
    hidden, pooled, _record = runner._null_conditioning(pipeline)
    assert hidden.shape == (1, 2)
    assert pooled.shape == (1, 3)


def test_sanitized_failure_receipt_has_only_a_finite_failure_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = TypeError("not emitted")
    error.geometry_failure_point = "null_conditioning_call"  # type: ignore[attr-defined]
    monkeypatch.setattr(runner.Image, "open", lambda _path: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(runner, "operational_preflight", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    assert runner._main(["--repo-root", ".", "--expected-exact", "0" * 40, "image.png"]) == 1
    line = capsys.readouterr().out.strip()
    prefix, payload = line.split(" ", 1)
    assert prefix == "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE"
    receipt = json.loads(payload)
    assert receipt["failure_point"] == "null_conditioning_call"
    assert receipt["failure_point"] in runner._FAILURE_POINTS
    assert "not emitted" not in line


def test_late_failure_keeps_only_bounded_runtime_and_architecture_records(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = TypeError("not emitted")
    error.geometry_failure_point = "scheduler"  # type: ignore[attr-defined]
    error.geometry_runtime_record = {"requested_model_id": runner.MODEL_ID}  # type: ignore[attr-defined]
    error.geometry_architecture_record = {"attention_candidates": []}  # type: ignore[attr-defined]
    monkeypatch.setattr(runner.Image, "open", lambda _path: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(runner, "operational_preflight", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    assert runner._main(["--repo-root", ".", "--expected-exact", "0" * 40, "image.png"]) == 1
    _prefix, payload = capsys.readouterr().out.strip().split(" ", 1)
    receipt = json.loads(payload)
    assert receipt["failure_point"] == "scheduler"
    assert receipt["runtime"] == {"requested_model_id": runner.MODEL_ID}
    assert receipt["architecture"] == {"attention_candidates": []}
    assert "not emitted" not in payload


def test_input_open_failure_still_emits_a_bounded_sanitized_receipt(
    capsys: pytest.CaptureFixture[str]
) -> None:
    assert runner._main(["--repo-root", ".", "--expected-exact", "0" * 40, "missing.png"]) == 1
    prefix, payload = capsys.readouterr().out.strip().split(" ", 1)
    assert prefix == "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE"
    receipt = json.loads(payload)
    assert receipt["run_id"] == "geometry-v1-b2b-000000000000-operational-01"
    assert receipt["failure_point"] == "receipt_packaging"
