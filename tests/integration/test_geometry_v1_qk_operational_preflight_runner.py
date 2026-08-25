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


def test_preflight_requires_cuda_before_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "_validate_execution_exact", lambda *_args: "geometry-v1-b2b-000000000000-operational-01")
    with pytest.raises(RuntimeError, match="cuda_required"):
        runner.operational_preflight([Image.new("RGB", (2, 2))], hf_token="token", root_key="key", expected_exact="0" * 40, repo_root=Path("."))
