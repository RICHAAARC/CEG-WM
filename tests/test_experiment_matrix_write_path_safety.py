"""
文件目的：experiment matrix 写盘路径安全回归测试。
Module type: General module

覆盖点：
1. output_summary 目录逃逸必须被拒绝。
2. summary 写盘必须落在 run_root_base/artifacts 下并通过受控写盘入口。
3. write_bypass_scan 对 scripts 层 open(..., "w") 必须阻断。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.audits.audit_write_bypass_scan import run_audit
from scripts import run_experiment_matrix as run_experiment_matrix_module
from scripts.run_experiment_matrix import main as run_experiment_matrix_main


def test_experiment_matrix_rejects_escape_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：拒绝 output_summary 目录逃逸路径。

    Verify that traversal-style output_summary path is rejected before any file write.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)
    escape_rel = "../../_matrix_escape_out.json"
    escaped_target = (tmp_path / escape_rel).resolve()
    batch_root = tmp_path / "matrix_batch"

    fake_summary: Dict[str, Any] = {
        "total": 1,
        "executed": 1,
        "succeeded": 1,
        "failed": 0,
        "batch_root": str(batch_root),
    }

    def _fake_runner(
        config_path: str,
        strict: bool = False,
        validate_protocol: bool = True,
        batch_root: str | None = None,
    ) -> Dict[str, Any]:
        return fake_summary

    if escaped_target.exists():
        escaped_target.unlink()

    monkeypatch.setattr("scripts.run_experiment_matrix.run_experiment_matrix_batch", _fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment_matrix.py",
            "--config",
            "configs/default.yaml",
            "--output-summary",
            escape_rel,
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        run_experiment_matrix_main()
    assert exit_info.value.code == 1

    assert not escaped_target.exists(), "escape target must not be created"


def test_experiment_matrix_summary_written_via_controlled_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：summary 必须通过受控写盘入口写入 artifacts 目录。

    Verify summary output path is constrained under batch_root/artifacts/experiment_matrix.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    batch_root = tmp_path / "matrix_batch"
    fake_summary: Dict[str, Any] = {
        "total": 1,
        "executed": 1,
        "succeeded": 1,
        "failed": 0,
        "batch_root": str(batch_root),
        "aggregate_report_path": str(batch_root / "artifacts" / "aggregate_report.json"),
        "grid_manifest_path": str(batch_root / "artifacts" / "grid_manifest.json"),
        "grid_summary_path": str(batch_root / "artifacts" / "grid_summary.json"),
        "attack_coverage_manifest_path": str(batch_root / "artifacts" / "attack_coverage_manifest.json"),
    }

    def _fake_runner(
        config_path: str,
        strict: bool = False,
        validate_protocol: bool = True,
        batch_root: str | None = None,
    ) -> Dict[str, Any]:
        return fake_summary

    monkeypatch.setattr("scripts.run_experiment_matrix.run_experiment_matrix_batch", _fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment_matrix.py",
            "--config",
            "configs/default.yaml",
            "--output-summary",
            "summary.json",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        run_experiment_matrix_main()
    assert exit_info.value.code == 0

    summary_path = batch_root / "artifacts" / "experiment_matrix" / "summary.json"
    assert summary_path.exists(), "summary file must exist under artifacts/experiment_matrix"

    artifacts_dir = (batch_root / "artifacts").resolve()
    summary_resolved = summary_path.resolve()
    summary_resolved.relative_to(artifacts_dir)

    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload.get("batch_root") == str(batch_root)
    assert not list(artifacts_dir.rglob("*.writing")), "artifacts 目录不应残留 .writing 临时文件"


def test_write_bypass_scan_blocks_scripts_open_write(tmp_path: Path) -> None:
    """
    功能：静态审计必须阻断 scripts 层 open(..., "w") 旁路写盘。

    Verify run_audit returns FAIL when key script uses open with write mode.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.
    """
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    bad_script = scripts_dir / "run_experiment_matrix.py"
    bad_script.write_text(
        "def bad_write():\n"
        "    with open('x.json', 'w', encoding='utf-8') as f:\n"
        "        f.write('{}')\n",
        encoding="utf-8",
    )

    audit_result = run_audit(tmp_path)

    assert audit_result["result"] == "FAIL"
    evidence = audit_result.get("evidence", {})
    assert evidence.get("fail_count", 0) >= 1
    fail_paths = [
        Path(item["path"]).name
        for item in evidence.get("matches", [])
        if item.get("classification") == "FAIL"
    ]
    assert "run_experiment_matrix.py" in fail_paths


def test_run_experiment_matrix_batch_disables_paper_faithfulness_for_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：run_experiment_matrix_batch 应在矩阵子运行中关闭 paper faithfulness。 

    Verify run_experiment_matrix_batch normalizes paper_faithfulness.enabled to False
    before building experiment grid.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    captured_cfg: Dict[str, Any] = {}

    def _fake_load_yaml_with_provenance(_path: Path):
        return ({"paper_faithfulness": {"enabled": True, "alignment_check": True}}, {})

    def _fake_normalize_ablation_flags(_cfg: Dict[str, Any]) -> None:
        return None

    def _fake_build_grid(cfg: Dict[str, Any]):
        captured_cfg.update(cfg)
        return [{"dummy": True}]

    def _fake_run_grid(_grid, strict: bool = False):
        return {
            "total": 1,
            "executed": 1,
            "succeeded": 1,
            "failed": 0,
            "batch_root": "outputs/experiment_matrix",
        }

    monkeypatch.setattr(run_experiment_matrix_module.config_loader, "load_yaml_with_provenance", _fake_load_yaml_with_provenance)
    monkeypatch.setattr(run_experiment_matrix_module.config_loader, "normalize_ablation_flags", _fake_normalize_ablation_flags)
    monkeypatch.setattr(run_experiment_matrix_module.experiment_matrix, "build_experiment_grid", _fake_build_grid)
    monkeypatch.setattr(run_experiment_matrix_module.experiment_matrix, "run_experiment_grid", _fake_run_grid)

    summary = run_experiment_matrix_module.run_experiment_matrix_batch(
        config_path="configs/paper_full_cuda.yaml",
        strict=False,
        validate_protocol=False,
        batch_root=None,
    )

    assert summary.get("failed") == 0
    paper_cfg = captured_cfg.get("paper_faithfulness")
    assert isinstance(paper_cfg, dict)
    assert paper_cfg.get("enabled") is False
    assert paper_cfg.get("alignment_check") is False
