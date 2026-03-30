"""
File purpose: weights_snapshot 构建语义一致性回归测试。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path

import pytest

from main.diffusion.sd3 import weights_snapshot


def test_resolve_effective_weights_snapshot_inputs_prefers_resolved_local_snapshot_binding(
    tmp_path: Path,
) -> None:
    """
    功能：resolved local snapshot 绑定必须优先于 raw HF 请求值。

    Verify the effective snapshot input resolver prefers resolved local-path
    bindings over raw HF request anchors.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "snapshots" / "0123456789abcdef"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    resolved_inputs = weights_snapshot.resolve_effective_weights_snapshot_inputs(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_source="hf",
        resolved_model_id=snapshot_dir.as_posix(),
        resolved_model_source="local_path",
        model_snapshot_path=snapshot_dir.as_posix(),
        hf_revision="main",
        local_files_only=True,
        cache_dir=None,
    )

    assert resolved_inputs["effective_model_id"] == snapshot_dir.as_posix()
    assert resolved_inputs["effective_model_source"] == "local_path"
    assert resolved_inputs["effective_resolution"] == "resolved_binding_priority"


def test_resolve_effective_weights_snapshot_inputs_falls_back_to_requested_when_no_local_binding() -> None:
    """
    功能：缺少 resolved 与本地 snapshot 绑定时必须回退到 raw 请求值。

    Verify the effective snapshot input resolver falls back to raw requested
    model anchors when no resolved or local snapshot binding exists.

    Returns:
        None.
    """
    resolved_inputs = weights_snapshot.resolve_effective_weights_snapshot_inputs(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_source="hf",
        resolved_model_id=None,
        resolved_model_source=None,
        model_snapshot_path=None,
        hf_revision="main",
        local_files_only=True,
        cache_dir=None,
    )

    assert resolved_inputs["effective_model_id"] == "stabilityai/stable-diffusion-3.5-medium"
    assert resolved_inputs["effective_model_source"] == "hf"
    assert resolved_inputs["effective_resolution"] == "requested_fallback"


def test_compute_weights_snapshot_preserves_hf_build_local_files_only_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：snapshot 阶段必须接受并复用 HF build 的 local_files_only 语义。

    Verify weights_snapshot no longer rejects local_files_only=False when the
    real HF build path allows downloads.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    huggingface_hub = pytest.importorskip("huggingface_hub")

    snapshot_dir = tmp_path / "snapshots" / "0123456789abcdef"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "model_index.json").write_text('{"ok": true}', encoding="utf-8")

    captured_kwargs = {}

    def _fake_snapshot_download(**kwargs):
        captured_kwargs.update(kwargs)
        return str(snapshot_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

    digest_value, snapshot_meta, snapshot_error = weights_snapshot.compute_weights_snapshot_sha256(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        model_source="hf",
        hf_revision="main",
        local_files_only=False,
        cache_dir=None,
    )

    assert snapshot_error is None
    assert isinstance(digest_value, str) and digest_value not in {"", "<absent>"}
    assert captured_kwargs.get("local_files_only") is False
    assert snapshot_meta.get("local_files_only") is False