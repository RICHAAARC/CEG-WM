from __future__ import annotations

import json
import os
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


def _run_with_control(arguments: list[str], tmp_path: Path) -> tuple[int, str]:
    read_fd, write_fd = os.pipe()
    try:
        result = runner._main([*arguments, "--output-root", str(tmp_path / "package"), "--control-fd", str(write_fd)])
    finally:
        os.close(write_fd)
    line = os.read(read_fd, runner.MAX_CONTROL_BYTES + 1).decode("utf-8")
    os.close(read_fd)
    return result, line


def test_sanitized_failure_receipt_has_only_a_finite_failure_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    error = TypeError("not emitted")
    error.geometry_failure_point = "null_conditioning_call"  # type: ignore[attr-defined]
    monkeypatch.setattr(runner.Image, "open", lambda _path: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(runner, "operational_preflight", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    rc, line = _run_with_control(["--repo-root", ".", "--expected-exact", "0" * 40, "image.png"], tmp_path)
    assert rc == 1
    prefix, payload = line.split(" ", 1)
    assert prefix == "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE"
    receipt = json.loads(payload)
    assert receipt["failure_point"] == "null_conditioning_call"
    assert receipt["artifact_status"] == "complete"
    assert "not emitted" not in line


def test_late_failure_keeps_only_bounded_runtime_and_architecture_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    error = TypeError("not emitted")
    error.geometry_failure_point = "scheduler"  # type: ignore[attr-defined]
    error.geometry_runtime_record = {"requested_model_id": runner.MODEL_ID}  # type: ignore[attr-defined]
    error.geometry_architecture_record = {"attention_candidates": []}  # type: ignore[attr-defined]
    monkeypatch.setattr(runner.Image, "open", lambda _path: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(runner, "operational_preflight", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    rc, line = _run_with_control(["--repo-root", ".", "--expected-exact", "0" * 40, "image.png"], tmp_path)
    assert rc == 1 and len(line.encode()) <= runner.MAX_CONTROL_BYTES
    package = tmp_path / "package"
    receipt = json.loads((package / "receipt.json").read_text())
    assert receipt["failure_point"] == "scheduler"
    assert receipt["runtime"] == {"requested_model_id": runner.MODEL_ID}
    assert receipt["architecture"] == {"attention_candidates": []}
    assert "not emitted" not in (package / "receipt.json").read_text()


def test_input_open_failure_still_emits_a_bounded_sanitized_receipt(tmp_path: Path
) -> None:
    rc, line = _run_with_control(["--repo-root", ".", "--expected-exact", "0" * 40, "missing.png"], tmp_path)
    assert rc == 1
    prefix, payload = line.strip().split(" ", 1)
    assert prefix == "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE"
    receipt = json.loads(payload)
    assert receipt["run_id"] == "geometry-v1-b2b-000000000000-operational-01"
    assert receipt["failure_point"] == "receipt_packaging"


@pytest.mark.parametrize("blocks", [18, 24, 38, 64])
def test_architecture_receipt_size_is_package_bounded(blocks: int) -> None:
    receipt = {"status": "operational_preflight_complete", "run_id": "geometry-v1-b2b-" + "0" * 12 + "-operational-01", "science_denominator": 0,
               "architecture": {"attention_candidates": [{"path": f"transformer_blocks.{index}.attn", "metadata": "x" * 180} for index in range(blocks)]}}
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    assert len(encoded) <= runner.MAX_RECEIPT_BYTES
    if blocks >= 24:
        assert len(encoded) > 4096


def test_package_is_create_only_and_has_only_sanitized_terminal_members(tmp_path: Path) -> None:
    run_id = "geometry-v1-b2b-" + "0" * 12 + "-operational-01"
    receipt = {"status": "operational_preflight_complete", "run_id": run_id, "science_denominator": 0}
    package = runner._package_receipt(output_root=tmp_path / "drive-run", receipt=receipt, status_name="success.json", expected_exact="0" * 40, run_id=run_id)
    root = tmp_path / "drive-run"
    assert set(path.name for path in root.iterdir()) == {"receipt.json", "success.json", "checkpoint.json", "manifest.json", "SHA256SUMS", package["archive_filename"], package["sidecar_filename"]}
    with pytest.raises(FileExistsError):
        runner._package_receipt(output_root=root, receipt=receipt, status_name="success.json", expected_exact="0" * 40, run_id=run_id)
    public = "".join(path.read_text(errors="ignore") for path in root.iterdir() if path.suffix != ".zip")
    for forbidden in ("HF_TOKEN", "CEG_WM_ROOT_KEY", "/private/", "tensor("):
        assert forbidden not in public


def test_packaging_failure_emits_only_artifact_unavailable_control(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner.Image, "open", lambda _path: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(runner, "operational_preflight", lambda *_args, **_kwargs: {"status": "operational_preflight_complete", "run_id": "geometry-v1-b2b-" + "0" * 12 + "-operational-01", "science_denominator": 0})
    monkeypatch.setattr(runner, "_package_receipt", lambda **_kwargs: (_ for _ in ()).throw(OSError("private path")))
    rc, line = _run_with_control(["--repo-root", ".", "--expected-exact", "0" * 40, "image.png"], tmp_path)
    prefix, body = line.strip().split(" ", 1)
    assert rc == 1 and prefix == "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE"
    assert json.loads(body) == {"status": "failure", "underlying_status": "unknown", "artifact_status": "unavailable", "failure_point": "receipt_packaging", "run_id": "geometry-v1-b2b-" + "0" * 12 + "-operational-01"}


def test_late_operational_failure_followed_by_packaging_failure_preserves_known_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lost package must not erase an already-known operational failure."""
    error = RuntimeError("scheduler failure")
    error.geometry_failure_point = "scheduler"  # type: ignore[attr-defined]
    monkeypatch.setattr(runner.Image, "open", lambda _path: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(runner, "operational_preflight", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    monkeypatch.setattr(runner, "_package_receipt", lambda **_kwargs: (_ for _ in ()).throw(OSError("package unavailable")))
    rc, line = _run_with_control(["--repo-root", ".", "--expected-exact", "0" * 40, "image.png"], tmp_path)
    prefix, body = line.strip().split(" ", 1)
    assert rc == 1 and prefix == "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE"
    assert json.loads(body) == {
        "status": "failure", "underlying_status": "operational_failure", "artifact_status": "unavailable",
        "failure_point": "receipt_packaging", "run_id": "geometry-v1-b2b-" + "0" * 12 + "-operational-01",
    }


def test_failure_package_strips_injected_secrets_paths_and_tensor_like_values(tmp_path: Path) -> None:
    """The actual failure receipt/package path must not publish injected private data."""
    error = RuntimeError("HF_TOKEN=token-sentinel CEG_WM_ROOT_KEY=key-sentinel /private/input.png tensor([1])")
    error.geometry_failure_point = "scheduler"  # type: ignore[attr-defined]
    error.geometry_runtime_record = {  # type: ignore[attr-defined]
        "safe": "public", "HF_TOKEN": "token-sentinel", "private_path": "/private/input.png", "raw_tensor": "tensor([1])",
    }
    run_id = "geometry-v1-b2b-" + "0" * 12 + "-operational-01"
    receipt = runner._failure_receipt(error, run_id)
    runner._package_receipt(output_root=tmp_path / "package", receipt=receipt, status_name="failure.json", expected_exact="0" * 40, run_id=run_id)
    public = "".join(path.read_text(errors="ignore") for path in (tmp_path / "package").iterdir() if path.suffix != ".zip")
    for sentinel in ("token-sentinel", "key-sentinel", "/private/input.png", "tensor([1])", "HF_TOKEN", "CEG_WM_ROOT_KEY"):
        assert sentinel not in public


@pytest.mark.parametrize("status_name,status,rc,prefix", [
    ("success.json", "success", 0, "CEGWM_GEOMETRY_V1_OPERATIONAL_PREFLIGHT"),
    ("failure.json", "failure", 1, "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE"),
])
def test_control_mapping_is_one_line_bounded_and_stdio_cannot_pollute_fd(status_name: str, status: str, rc: int, prefix: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_id = "geometry-v1-b2b-" + "0" * 12 + "-operational-01"
    receipt = {"status": "operational_preflight_complete" if rc == 0 else "operational_failure", "run_id": run_id, "science_denominator": 0}
    package = runner._package_receipt(output_root=tmp_path / status, receipt=receipt, status_name=status_name, expected_exact="0" * 40, run_id=run_id)
    read_fd, write_fd = os.pipe()
    try:
        print("third-party stdout noise")
        runner._emit_control(write_fd, runner._SUCCESS_PREFIX if rc == 0 else runner._FAILURE_PREFIX, {"status": status, "run_id": run_id, "artifact_status": "complete", **package})
    finally:
        os.close(write_fd)
    line = os.read(read_fd, runner.MAX_CONTROL_BYTES + 1); os.close(read_fd)
    assert len(line) <= runner.MAX_CONTROL_BYTES and line.count(b"\n") == 1 and line.startswith(prefix.encode() + b" ")
    assert b"third-party stdout noise" not in line
    assert "third-party stdout noise" in capsys.readouterr().out


def test_exact_bound_acceptance_and_max_plus_one_rejection() -> None:
    accepted = {"x": "a" * (runner.MAX_RECEIPT_BYTES - len('{"x":""}'.encode()))}
    assert len(runner._bounded_json(accepted, runner.MAX_RECEIPT_BYTES)) == runner.MAX_RECEIPT_BYTES
    with pytest.raises(ValueError, match="bounded"):
        runner._bounded_json({"x": accepted["x"] + "a"}, runner.MAX_RECEIPT_BYTES)
    with pytest.raises(ValueError, match="bounded"):
        runner._bounded_json({"x": "a" * runner.MAX_CONTROL_BYTES}, runner.MAX_CONTROL_BYTES - 1)


def test_many_block_late_failure_is_package_bounded_and_preserves_failure_status(tmp_path: Path) -> None:
    run_id = "geometry-v1-b2b-" + "0" * 12 + "-operational-01"
    receipt = {"status": "operational_failure", "run_id": run_id, "science_denominator": 0, "failure_point": "scheduler", "architecture": {"attention_candidates": [{"block": index, "public": "x" * 180} for index in range(64)]}}
    assert len(json.dumps(receipt, separators=(",", ":")).encode()) > 4096
    package = runner._package_receipt(output_root=tmp_path / "failure", receipt=receipt, status_name="failure.json", expected_exact="0" * 40, run_id=run_id)
    assert package["receipt_bytes"] <= runner.MAX_RECEIPT_BYTES
    assert json.loads((tmp_path / "failure" / "receipt.json").read_text())["status"] == "operational_failure"
