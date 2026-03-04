"""
文件目的：repro bundle manifest 与 pointers 回归测试。
Module type: Core innovation module
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_repro_pipeline_module(repo_root: Path):
    """
    功能：动态加载 run_repro_pipeline 脚本模块。

    Load repro pipeline module for direct function testing.

    Args:
        repo_root: Repository root path.

    Returns:
        Imported module object.
    """
    module_path = repo_root / "scripts" / "run_repro_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_repro_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_yaml_files(tmp_path: Path) -> tuple[Path, Path]:
    """
    功能：写入最小 cfg 与 attack protocol YAML。

    Write minimal YAML inputs.

    Args:
        tmp_path: Temporary directory.

    Returns:
        Tuple of config and attack protocol paths.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
policy_path: policy/default
evaluate:
  thresholds_path: artifacts/thresholds/thresholds_artifact.json
  detect_records_glob: records/detect_record.json
""".strip()
        + "\n",
        encoding="utf-8",
    )

    attack_protocol_path = tmp_path / "attack_protocol.yaml"
    attack_protocol_path.write_text(
        """
version: attack_protocol_v1
families:
  rotate:
    params_versions:
      v1: {}
params_versions:
  rotate::v1:
    family: rotate
    params: {}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path, attack_protocol_path


def _make_stage_runner() -> Any:
    """
    功能：构造离线 stage runner。

    Build minimal offline stage runner.

    Args:
        None.

    Returns:
        Callable stage runner.
    """
    def _runner(
        stage_name: str,
        run_root: Path,
        config_path: Path,
        attack_protocol_path: Path,
        seeds: str | None,
        max_samples: int | None,
        repo_root: Path,
    ) -> str:
        _ = config_path
        _ = attack_protocol_path
        _ = seeds
        _ = max_samples
        _ = repo_root

        records_dir = run_root / "records"
        artifacts_dir = run_root / "artifacts"
        records_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if stage_name == "detect":
            # _prepare_minimal_gt_detect_records 需要该文件存在（分数可缺失，会走 0.25 回退）
            (records_dir / "detect_record.json").write_text(
                json.dumps(
                    {
                        "operation": "detect",
                        "content_evidence_payload": {
                            "score": 0.8,
                            "status": "ok",
                        },
                    }
                ),
                encoding="utf-8",
            )

        if stage_name == "calibrate":
            thresholds_dir = artifacts_dir / "thresholds"
            thresholds_dir.mkdir(parents=True, exist_ok=True)
            (thresholds_dir / "thresholds_artifact.json").write_text(
                json.dumps(
                    {
                        "threshold_id": "content_score_np_fpr_0_01",
                        "score_name": "content_score",
                        "target_fpr": 0.01,
                        "threshold_value": 0.5,
                        "threshold_key_used": "fpr_0_01",
                        "rule_version": "np_v1",
                    }
                ),
                encoding="utf-8",
            )
            (thresholds_dir / "threshold_metadata_artifact.json").write_text(
                json.dumps({"method": "neyman_pearson_v1", "n_samples": 8}),
                encoding="utf-8",
            )

        if stage_name == "evaluate":
            report = {
                "cfg_digest": "cfg_digest_test",
                "plan_digest": "plan_digest_test",
                "thresholds_digest": "thresholds_digest_test",
                "threshold_metadata_digest": "threshold_metadata_digest_test",
                "impl_digest": "impl_digest_test",
                "fusion_rule_version": "fusion_v1",
                "attack_protocol_version": "attack_protocol_v1",
                "attack_protocol_digest": "attack_protocol_digest_test",
                "policy_path": "policy/default",
                "metrics": {
                    "n_total": 1,
                    "n_accepted": 1,
                    "n_pos": 1,
                    "n_neg": 0,
                    "confusion": {"tp": 1, "fp": 0},
                    "tpr_at_fpr_primary": 1.0,
                    "fpr_empirical": 0.0,
                    "geo_available_rate": 1.0,
                    "rescue_rate": 0.0,
                    "reject_rate": 0.0,
                },
                "metrics_by_attack_condition": [
                    {
                        "group_key": "rotate::v1",
                        "n_total": 1,
                        "n_accepted": 1,
                        "n_pos": 1,
                        "n_neg": 0,
                        "tp": 1,
                        "fp": 0,
                        "tpr_at_fpr_primary": 1.0,
                        "fpr_empirical": 0.0,
                        "geo_available_rate": 1.0,
                        "rescue_rate": 0.0,
                        "reject_rate_by_reason": {},
                    }
                ],
            }
            (records_dir / "evaluate_record.json").write_text(
                json.dumps({"operation": "evaluate", "evaluation_report": report}),
                encoding="utf-8",
            )

        return f"stub_stage:{stage_name}"

    return _runner


def _make_signoff_runner() -> Any:
    """
    功能：构造离线 signoff runner。

    Build minimal signoff runner for repro bundle tests.

    Args:
        None.

    Returns:
        Callable signoff runner.
    """
    def _runner(run_root: Path, repo_root: Path) -> str:
        _ = repo_root
        artifacts_dir = run_root / "artifacts"

        (artifacts_dir / "run_closure.json").write_text(
            json.dumps({"status": {"ok": True}}),
            encoding="utf-8",
        )
        (artifacts_dir / "records_manifest.json").write_text(
            json.dumps({"schema_version": "v1.0"}),
            encoding="utf-8",
        )

        signoff_dir = artifacts_dir / "signoff"
        snapshot_dir = signoff_dir / "frozen_constraints_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        (signoff_dir / "signoff_report.json").write_text(
            json.dumps({"freeze_signoff_decision": "ALLOW_FREEZE"}),
            encoding="utf-8",
        )
        (snapshot_dir / "frozen_contracts.yaml").write_text("contract_version: v1\n", encoding="utf-8")
        (snapshot_dir / "runtime_whitelist.yaml").write_text("whitelist_version: v1\n", encoding="utf-8")
        (snapshot_dir / "policy_path_semantics.yaml").write_text("policy_path_semantics_version: v1\n", encoding="utf-8")

        return "stub_signoff"

    return _runner


def _sha256_file(file_path: Path) -> str:
    """
    功能：计算文件 SHA256。

    Compute SHA256 digest for file.

    Args:
        file_path: Target file path.

    Returns:
        SHA256 hex string.
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_repro_bundle_manifest_contains_required_anchor_fields(tmp_path: Path) -> None:
    """
    功能：验证 repro bundle manifest 含必备锚点字段。

    Verify repro bundle manifest contains required anchor fields.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_repro_pipeline_module(repo_root)

    run_root = tmp_path / "run_root"
    config_path, attack_protocol_path = _write_minimal_yaml_files(tmp_path)

    module.run_repro_pipeline(
        run_root=run_root,
        config_path=config_path,
        attack_protocol_path=attack_protocol_path,
        seeds="seed_test",
        max_samples=4,
        repo_root=repo_root,
        stage_runner=_make_stage_runner(),
        signoff_runner=_make_signoff_runner(),
    )

    manifest_path = run_root / "artifacts" / "repro_bundle" / "manifest.json"
    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))

    required_fields = [
        "cfg_digest",
        "plan_digest",
        "thresholds_digest",
        "threshold_metadata_digest",
        "attack_protocol_version",
        "policy_path",
        "impl_digest",
        "fusion_rule_version",
    ]

    for field_name in required_fields:
        assert field_name in manifest_obj, f"missing required field: {field_name}"
        assert isinstance(manifest_obj[field_name], str), f"field should be str: {field_name}"
        assert manifest_obj[field_name] != "<absent>", f"field should not be <absent>: {field_name}"


def test_repro_bundle_pointers_sha256_match_files(tmp_path: Path) -> None:
    """
    功能：验证 pointers.json 中 sha256 与文件实际一致。

    Verify pointers sha256 values match actual files.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_repro_pipeline_module(repo_root)

    run_root = tmp_path / "run_root"
    config_path, attack_protocol_path = _write_minimal_yaml_files(tmp_path)

    module.run_repro_pipeline(
        run_root=run_root,
        config_path=config_path,
        attack_protocol_path=attack_protocol_path,
        seeds="seed_test",
        max_samples=4,
        repo_root=repo_root,
        stage_runner=_make_stage_runner(),
        signoff_runner=_make_signoff_runner(),
    )

    pointers_path = run_root / "artifacts" / "repro_bundle" / "pointers.json"
    pointers_obj = json.loads(pointers_path.read_text(encoding="utf-8"))

    assert isinstance(pointers_obj.get("files"), list)
    for item in pointers_obj["files"]:
        file_path = run_root / item["path"]
        assert file_path.exists(), f"pointer file missing: {item['path']}"
        assert _sha256_file(file_path) == item["sha256"], f"sha256 mismatch: {item['path']}"


def test_repro_bundle_manifest_and_pointers_sha256_match(tmp_path: Path) -> None:
    """
    功能：验证 repro bundle manifest 与 pointers 的哈希一致性。

    Verify repro bundle manifest exists and pointers SHA256 values match actual files.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        None.
    """
    test_repro_bundle_manifest_contains_required_anchor_fields(tmp_path)
    test_repro_bundle_pointers_sha256_match_files(tmp_path)


def test_audit_repro_bundle_integrity_passes_on_generated_bundle(tmp_path: Path) -> None:
    """
    功能：验证 repro bundle 完整性审计在最小产物上通过。

    Verify audit_repro_bundle_integrity passes on generated minimal bundle.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_repro_pipeline_module(repo_root)

    run_root = tmp_path / "run_root"
    config_path, attack_protocol_path = _write_minimal_yaml_files(tmp_path)

    module.run_repro_pipeline(
        run_root=run_root,
        config_path=config_path,
        attack_protocol_path=attack_protocol_path,
        seeds="seed_test",
        max_samples=4,
        repo_root=repo_root,
        stage_runner=_make_stage_runner(),
        signoff_runner=_make_signoff_runner(),
    )

    audit_script = repo_root / "scripts" / "audits" / "audit_repro_bundle_integrity.py"
    result = subprocess.run(
        [sys.executable, str(audit_script), str(repo_root), str(run_root)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    audit_obj = json.loads(result.stdout)
    assert audit_obj.get("result") == "PASS"
