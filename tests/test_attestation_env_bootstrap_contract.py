"""
文件目的：验证 attestation env bootstrap 与 restore helper 的合同语义。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.notebook_runtime_common import ensure_attestation_env_bootstrap


def _build_cfg() -> Dict[str, Any]:
    """
    功能：构造最小 attestation 配置。 

    Build a minimal runtime config carrying the attestation env-var bindings.

    Args:
        None.

    Returns:
        Runtime config mapping.
    """
    return {
        "attestation": {
            "enabled": True,
            "k_master_env_var": "CEG_WM_K_MASTER",
            "k_prompt_env_var": "CEG_WM_K_PROMPT",
            "k_seed_env_var": "CEG_WM_K_SEED",
        }
    }


def _clear_attestation_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：清理测试中的 attestation 环境变量。 

    Clear the attestation env vars used by the helper contract tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    for env_name in ["CEG_WM_K_MASTER", "CEG_WM_K_PROMPT", "CEG_WM_K_SEED"]:
        monkeypatch.delenv(env_name, raising=False)


def test_ensure_attestation_env_bootstrap_generates_missing_secret_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：缺失 secrets 文件时 helper 必须生成当前会话可用的临时 secret。 

    Verify the helper generates a fresh session secret file, injects os.environ,
    and writes a masked info file when the secret file is absent.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    drive_root = tmp_path / "drive"
    cfg_obj = _build_cfg()
    _clear_attestation_env(monkeypatch)

    summary = ensure_attestation_env_bootstrap(
        cfg_obj,
        drive_root,
        allow_generate=True,
        allow_missing=False,
    )

    secrets_path = drive_root / "secrets" / "attestation_env.json"
    info_path = drive_root / "secrets" / "attestation_env_info.json"
    secrets_payload = json.loads(secrets_path.read_text(encoding="utf-8"))
    info_payload = json.loads(info_path.read_text(encoding="utf-8"))
    info_text = info_path.read_text(encoding="utf-8")

    assert summary["status"] == "generated"
    assert summary["generated"] is True
    assert summary["reused_existing"] is False
    assert secrets_path.exists()
    assert info_path.exists()
    assert sorted(summary["present_env_vars"]) == sorted(summary["required_env_vars"])
    assert info_payload["status"] == "generated"
    assert set(info_payload["required_env_vars"]) == {
        "CEG_WM_K_MASTER",
        "CEG_WM_K_PROMPT",
        "CEG_WM_K_SEED",
    }

    assert len(secrets_payload["CEG_WM_K_MASTER"]) == 64
    assert len(secrets_payload["CEG_WM_K_PROMPT"]) == 32
    assert len(secrets_payload["CEG_WM_K_SEED"]) == 32
    assert info_payload["masked_values"]["CEG_WM_K_MASTER"] != secrets_payload["CEG_WM_K_MASTER"]
    assert secrets_payload["CEG_WM_K_MASTER"] not in info_text
    assert secrets_payload["CEG_WM_K_PROMPT"] not in info_text
    assert secrets_payload["CEG_WM_K_SEED"] not in info_text

    assert summary["masked_values"]["CEG_WM_K_MASTER"] == info_payload["masked_values"]["CEG_WM_K_MASTER"]
    assert summary["masked_values"]["CEG_WM_K_PROMPT"] == info_payload["masked_values"]["CEG_WM_K_PROMPT"]
    assert summary["masked_values"]["CEG_WM_K_SEED"] == info_payload["masked_values"]["CEG_WM_K_SEED"]


def test_ensure_attestation_env_bootstrap_reuses_valid_secret_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：已有合法 secrets 文件时 helper 必须复用，不得静默重写。 

    Verify the helper reuses an existing valid secret file and restores the
    attestation env vars into os.environ without replacing the file contents.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    drive_root = tmp_path / "drive"
    secrets_root = drive_root / "secrets"
    secrets_root.mkdir(parents=True, exist_ok=True)
    cfg_obj = _build_cfg()
    _clear_attestation_env(monkeypatch)

    existing_payload = {
        "CEG_WM_K_MASTER": "a" * 64,
        "CEG_WM_K_PROMPT": "b" * 32,
        "CEG_WM_K_SEED": "c" * 32,
    }
    secrets_path = secrets_root / "attestation_env.json"
    secrets_path.write_text(json.dumps(existing_payload, indent=2), encoding="utf-8")

    summary = ensure_attestation_env_bootstrap(
        cfg_obj,
        drive_root,
        allow_generate=True,
        allow_missing=False,
    )

    reused_payload = json.loads(secrets_path.read_text(encoding="utf-8"))
    assert summary["status"] == "reused"
    assert summary["generated"] is False
    assert summary["reused_existing"] is True
    assert reused_payload == existing_payload
    assert summary["present_env_vars"] == summary["required_env_vars"]
    assert summary["masked_values"]["CEG_WM_K_MASTER"] == "aaaa...aaaa"
    assert summary["masked_values"]["CEG_WM_K_PROMPT"] == "bbbb...bbbb"
    assert summary["masked_values"]["CEG_WM_K_SEED"] == "cccc...cccc"


def test_ensure_attestation_env_bootstrap_rejects_invalid_secret_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：非法 secrets 文件必须被拒绝，且不得被自动覆盖。 

    Verify the helper rejects malformed secret files and does not silently
    overwrite them even when generation is allowed.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    drive_root = tmp_path / "drive"
    secrets_root = drive_root / "secrets"
    secrets_root.mkdir(parents=True, exist_ok=True)
    cfg_obj = _build_cfg()
    _clear_attestation_env(monkeypatch)

    secrets_path = secrets_root / "attestation_env.json"
    invalid_text = json.dumps(
        {
            "CEG_WM_K_MASTER": "not_hex",
            "CEG_WM_K_PROMPT": "b" * 32,
            "CEG_WM_K_SEED": "c" * 32,
        },
        indent=2,
    )
    secrets_path.write_text(invalid_text, encoding="utf-8")

    with pytest.raises(ValueError):
        ensure_attestation_env_bootstrap(
            cfg_obj,
            drive_root,
            allow_generate=True,
            allow_missing=False,
        )

    assert secrets_path.read_text(encoding="utf-8") == invalid_text
