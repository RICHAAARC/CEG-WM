"""
文件目的：最小复现流水线回归测试。
Module type: Core innovation module
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict


def _load_repro_pipeline_module(repo_root: Path):
    """
    功能：动态加载 run_repro_pipeline 脚本模块。

    Load repro pipeline module from scripts directory.

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

    Write minimal YAML files for pipeline input.

    Args:
        tmp_path: Temporary directory.

    Returns:
        Tuple of (config_path, attack_protocol_path).
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
policy_path: policy/default
model:
  model_id: test_model
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
    功能：构造最小阶段执行 stub。

    Build stage runner stub for offline fast tests.

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
            evaluation_report = {
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
                    "n_total": 2,
                    "n_accepted": 2,
                    "n_pos": 1,
                    "n_neg": 1,
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
                        "n_total": 2,
                        "n_accepted": 2,
                        "n_pos": 1,
                        "n_neg": 1,
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
            evaluate_record = {
                "operation": "evaluate",
                "evaluation_report": evaluation_report,
            }
            (records_dir / "evaluate_record.json").write_text(
                json.dumps(evaluate_record),
                encoding="utf-8",
            )

        return f"stub_stage:{stage_name}"

    return _runner


def _make_signoff_runner() -> Any:
    """
    功能：构造最小 signoff 执行 stub。

    Build signoff runner stub to create required signoff artifacts.

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


def test_repro_pipeline_generates_manifest_and_outputs_under_run_root(tmp_path: Path) -> None:
    """
    功能：验证最小复现流水线生成 run_manifest 与 repro 输出路径。

    Verify minimal repro pipeline generates expected outputs under run_root.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_repro_pipeline_module(repo_root)

    run_root = tmp_path / "run_root"
    config_path, attack_protocol_path = _write_minimal_yaml_files(tmp_path)

    result = module.run_repro_pipeline(
        run_root=run_root,
        config_path=config_path,
        attack_protocol_path=attack_protocol_path,
        seeds="seed_test",
        max_samples=4,
        repo_root=repo_root,
        stage_runner=_make_stage_runner(),
        signoff_runner=_make_signoff_runner(),
    )

    assert isinstance(result, dict)

    manifest_path = run_root / "artifacts" / "repro" / "run_manifest.json"
    report_path = run_root / "artifacts" / "repro" / "evaluation_report.json"
    table_path = run_root / "artifacts" / "repro" / "tables" / "metrics.csv"

    assert manifest_path.exists(), "run_manifest.json should be generated"
    assert report_path.exists(), "evaluation_report.json should be generated"
    assert table_path.exists(), "metrics.csv should be generated"

    manifest_path.resolve().relative_to(run_root.resolve())
    report_path.resolve().relative_to(run_root.resolve())
    table_path.resolve().relative_to(run_root.resolve())


def test_repro_pipeline_outputs_under_run_root(tmp_path: Path) -> None:
    """
    功能：验证复现流水线输出不越界 run_root。

    Verify repro pipeline outputs are strictly under run_root.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        None.
    """
    test_repro_pipeline_generates_manifest_and_outputs_under_run_root(tmp_path)


def test_repro_calibrate_gt_negative_uses_repro_marker(tmp_path: Path) -> None:
    """
    功能：验证 repro 负样本 usage marker 不会命中 synthetic closure 过滤。 

    Verify repro GT negative sample uses a repro-specific usage marker.

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

    negative_path = (
        run_root
        / "artifacts"
        / "repro"
        / "gt_calibrate"
        / "detect_record_calibrate_gt_negative.json"
    )
    negative_obj = json.loads(negative_path.read_text(encoding="utf-8"))
    content_payload = negative_obj.get("content_evidence_payload", {})

    assert content_payload.get("calibration_sample_usage") == "repro_ground_truth_closure_negative_marker"
    assert content_payload.get("calibration_sample_usage") != "synthetic_negative_for_ground_truth_closure"
