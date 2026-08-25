from __future__ import annotations

import pytest
import torch
from PIL import Image
from pathlib import Path

from experiments import run_geometry_v1_qk_operational_preflight as runner


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

        def encode_prompt(self, **kwargs: object) -> tuple[torch.Tensor, None, torch.Tensor, None]:
            self.calls.append(kwargs)
            return hidden, None, pooled, None

    pipeline = Pipeline()
    actual_hidden, actual_pooled, record = runner._null_conditioning(pipeline)
    assert pipeline.calls == [{"prompt": "", "do_classifier_free_guidance": False}]
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
        def encode_prompt(self, **kwargs: object) -> object:
            assert kwargs == {"prompt": "", "do_classifier_free_guidance": False}
            return result

    with pytest.raises((TypeError, ValueError), match=error):
        runner._null_conditioning(Pipeline())
