"""
File purpose: evaluation 协议工程化回归测试。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from main.evaluation import protocol_loader
from main.evaluation import attack_plan
from main.evaluation import metrics
from main.evaluation import report_builder
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.detect.orchestrator import run_evaluate_orchestrator


class _ExtractorRaiser:
    def extract(self, cfg: Dict[str, Any]) -> None:
        _ = cfg
        raise AssertionError("extract should not be called in evaluate readonly mode")


class _FusionRaiser:
    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> None:
        _ = cfg
        _ = content_evidence
        _ = geometry_evidence
        raise AssertionError("fuse should not be called in evaluate readonly mode")


def _build_impl_set() -> BuiltImplSet:
    return BuiltImplSet(
        content_extractor=_ExtractorRaiser(),
        geometry_extractor=_ExtractorRaiser(),
        fusion_rule=_FusionRaiser(),
        subspace_planner=object(),
        sync_module=object(),
    )


def test_attack_protocol_loaded_from_configs_fact_source(tmp_path: Path) -> None:
    """
    功能：验证协议从 configs 事实源加载。

    Verify that attack protocol is loaded from configs fact source and versioned.
    """
    # 准备：创建临时协议文件（复用仓库 YAML 格式）。
    protocol_yaml = """
version: "attack_protocol_v1"
families:
  test_family:
    description: "test family"
    params_versions:
      v1:
        param1: 100
params_versions:
  test_family::v1:
    family: "test_family"
    params: {param1: 100}
"""
    protocol_path = tmp_path / "test_protocol.yaml"
    protocol_path.write_text(protocol_yaml, encoding="utf-8")

    # 加载协议（指定临时路径）。
    cfg = {
        "evaluate": {
            "attack_protocol_path": str(protocol_path),
        }
    }
    spec = protocol_loader.load_attack_protocol_spec(cfg)

    # 断言：version 非空且 digest 已计算。
    assert spec.get("version") == "attack_protocol_v1"
    assert isinstance(spec.get("attack_protocol_digest"), str)
    assert spec.get("attack_protocol_digest") != "<absent>"
    assert len(spec.get("attack_protocol_digest", "")) == 64  # SHA256 hex


def test_attack_plan_is_deterministic_and_digestable(tmp_path: Path) -> None:
    """
    功能：验证攻击计划的确定性与可序列化性。

    Verify attack plan is deterministic (same protocol → same plan digest).
    """
    # 准备：创建协议规范。
    protocol_spec = {
        "version": "attack_protocol_v1",
        "families": {
            "rotate": {"params_versions": {"v1": {}}},
            "resize": {"params_versions": {"v1": {}}},
        },
        "params_versions": {
            "rotate::v1": {"family": "rotate"},
            "resize::v1": {"family": "resize"},
        },
    }

    # 生成计划（两次）。
    plan1 = attack_plan.generate_attack_plan(protocol_spec)
    plan2 = attack_plan.generate_attack_plan(protocol_spec)

    # 断言：计划内容相同。
    assert plan1.protocol_version == plan2.protocol_version
    assert plan1.conditions == plan2.conditions
    
    # 断言：两个计划的 digest 相同（确定性）。
    digest1 = plan1.compute_digest()
    digest2 = plan2.compute_digest()
    assert digest1 == digest2
    
    # 断言：conditions 已排序。
    assert plan1.conditions == sorted(plan1.conditions)


def test_evaluate_report_contains_required_anchor_fields(tmp_path: Path) -> None:
    """
    功能：验证评测报告包含所有必需的锚点字段。

    Verify evaluation report contains all required anchor fields for auditability.
    """
    # 构造最小评测流程。
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps({
            "threshold_id": "test_threshold",
            "score_name": "content_score",
            "target_fpr": 0.01,
            "threshold_value": 0.5,
            "threshold_key_used": "fpr_0_01",
            "rule_version": "np_v1",
        }),
        encoding="utf-8",
    )
    (tmp_path / "threshold_metadata_artifact.json").write_text(
        json.dumps({
            "rule_id": "np_v1",
            "target_fpr": 0.01,
            "n_null": 1,
        }),
        encoding="utf-8",
    )

    detect_path = tmp_path / "detect.json"
    detect_path.write_text(
        json.dumps({
            "content_evidence_payload": {"status": "ok", "score": 0.8},
            "label": True,
            "attack": {"family": "rotate", "params_version": "v1"},
            "plan_digest": "test_plan_digest_from_detect",
            "impl_set_capabilities_digest": "test_impl_set_digest_from_detect",
        }),
        encoding="utf-8",
    )

    cfg = {
        "evaluate": {
            "thresholds_path": str(thresholds_path),
            "detect_records_glob": str(detect_path),
        },
        "__evaluate_cfg_digest__": "test_cfg_digest",
        "__policy_path__": "test_policy_path",
    }

    # 运行 evaluate。
    result = run_evaluate_orchestrator(cfg, _build_impl_set())

    # 提取报告对象。
    report = result.get("evaluation_report")
    assert isinstance(report, dict)

    # 断言：包含所有必需的锚点字段。
    required_anchors = [
        "cfg_digest",
        "plan_digest",
        "thresholds_digest",
        "threshold_metadata_digest",
        "attack_protocol_version",
        "policy_path",
        "impl_digest",
        "fusion_rule_version",
    ]
    for anchor_key in required_anchors:
        assert anchor_key in report, f"Missing anchor field: {anchor_key}"
        assert isinstance(report[anchor_key], str), f"Anchor {anchor_key} should be string"

    non_absent_anchors = [
        "cfg_digest",
        "plan_digest",
        "thresholds_digest",
        "threshold_metadata_digest",
        "policy_path",
        "impl_digest",
    ]
    for anchor_key in non_absent_anchors:
        assert report[anchor_key] != "<absent>", f"Anchor {anchor_key} should not be <absent>"

    # append-only: 新增路由摘要与实现锚点容器。
    assert isinstance(report.get("routing_decisions"), dict)
    assert isinstance(report.get("routing_digest"), str)
    assert isinstance(report.get("impl_anchors"), dict)
    assert isinstance(report["impl_anchors"].get("content"), dict)
    assert isinstance(report["impl_anchors"].get("geometry"), dict)
    assert isinstance(report["impl_anchors"].get("fusion"), dict)


def test_metrics_by_attack_condition_present_and_grouped(tmp_path: Path) -> None:
    """
    功能：验证分组指标的存在性与格式。

    Verify metrics are grouped by attack condition and include required fields.
    """
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps({
            "threshold_id": "test_threshold",
            "score_name": "content_score",
            "target_fpr": 0.01,
            "threshold_value": 0.5,
            "threshold_key_used": "fpr_0_01",
            "rule_version": "np_v1",
        }),
        encoding="utf-8",
    )

    # 创建两个不同条件的 detect 记录。
    detect_rotate = tmp_path / "detect_rotate.json"
    detect_rotate.write_text(
        json.dumps({
            "content_evidence_payload": {"status": "ok", "score": 0.8},
            "label": True,
            "attack": {"family": "rotate", "params_version": "v1"},
            "geometry_evidence_payload": {"status": "ok", "geo_score": 0.6},
            "decision": {"routing_decisions": {"rescue_triggered": False}},
        }),
        encoding="utf-8",
    )

    detect_resize = tmp_path / "detect_resize.json"
    detect_resize.write_text(
        json.dumps({
            "content_evidence_payload": {"status": "ok", "score": 0.6},
            "label": False,
            "attack": {"family": "resize", "params_version": "v1"},
        }),
        encoding="utf-8",
    )

    cfg = {
        "evaluate": {
            "thresholds_path": str(thresholds_path),
            "detect_records_glob": str(tmp_path / "detect_*.json"),
        },
        "__evaluate_cfg_digest__": "test_cfg_digest",
    }

    # 运行 evaluate。
    result = run_evaluate_orchestrator(cfg, _build_impl_set())

    # 提取条件指标。
    conditional_metrics = result.get("conditional_metrics", {})
    assert isinstance(conditional_metrics, dict)
    
    attack_group_metrics = conditional_metrics.get("attack_group_metrics", [])
    assert isinstance(attack_group_metrics, list)
    assert len(attack_group_metrics) >= 2, "Should have at least 2 attack groups (rotate, resize)"

    # 断言：分组键正确。
    group_keys = {item.get("group_key") for item in attack_group_metrics}
    assert "rotate::v1" in group_keys
    assert "resize::v1" in group_keys

    # 断言：每个分组包含必需的指标字段。
    required_fields = [
        "group_key",
        "n_total",
        "n_accepted",
        "tpr_at_fpr_primary",
        "geo_available_rate",
        "reject_rate_by_reason",
    ]
    for group_metrics in attack_group_metrics:
        for field in required_fields:
            assert field in group_metrics, f"Missing metric field {field} in {group_metrics.get('group_key')}"


def test_protocol_loader_handles_missing_file_gracefully() -> None:
    """
    功能：验证协议加载器在缺失文件时的容错行为。

    Verify protocol loader returns default spec when file is missing.
    """
    cfg = {
        "evaluate": {
            "attack_protocol_path": "/nonexistent/path/protocol.yaml",
        }
    }

    spec = protocol_loader.load_attack_protocol_spec(cfg)

    # 断言：返回默认规范（不抛出异常）。
    assert isinstance(spec, dict)
    assert spec.get("version") in ("protocol_version_not_loaded", "<absent>") or spec.get("version") is None


def test_attack_plan_validates_input() -> None:
    """
    功能：验证 attack_plan 对输入的验证。

    Verify attack plan validates conditions format.
    """
    # 构造有效计划。
    valid_plan = attack_plan.AttackPlan(
        protocol_version="v1",
        conditions=["rotate::v1", "resize::v1"],
    )
    assert attack_plan.validate_attack_plan(valid_plan) is True

    # 断言：无效条件列表应该抛出异常。
    invalid_plan = attack_plan.AttackPlan(
        protocol_version="v1",
        conditions=["invalid_no_separator"],  # 缺少 ::
    )
    with pytest.raises(ValueError):
        attack_plan.validate_attack_plan(invalid_plan)


def test_metrics_compute_overall_and_grouped() -> None:
    """
    功能：验证 metrics 模块正确计算 overall 和 grouped 指标。

    Verify metrics computation for both overall and per-condition aggregation.
    """
    # 准备记录。
    records = [
        {
            "content_evidence_payload": {"status": "ok", "score": 0.9},
            "label": True,
            "attack": {"family": "rotate", "params_version": "v1"},
        },
        {
            "content_evidence_payload": {"status": "ok", "score": 0.3},
            "label": False,
            "attack": {"family": "rotate", "params_version": "v1"},
        },
        {
            "content_evidence_payload": {"status": "ok", "score": 0.7},
            "label": True,
            "attack": {"family": "resize", "params_version": "v1"},
        },
    ]

    protocol_spec = {
        "family_field_candidates": ["attack.family"],
        "params_version_field_candidates": ["attack.params_version"],
    }

    # 计算指标。
    overall_metrics, breakdown = metrics.compute_overall_metrics(records, 0.5)
    grouped_metrics = metrics.compute_attack_group_metrics(records, 0.5, protocol_spec)

    # 断言：overall 指标。
    assert overall_metrics["n_total"] == 3
    assert overall_metrics["n_accepted"] == 3
    assert overall_metrics["tpr_at_fpr_primary"] == pytest.approx(1.0)  # 2 TP out of 2 pos
    assert overall_metrics["fpr_empirical"] == 0.0  # 0 FP out of 1 neg

    # 断言：grouped 指标。
    assert len(grouped_metrics) == 2
    group_keys = {g["group_key"] for g in grouped_metrics}
    assert "rotate::v1" in group_keys
    assert "resize::v1" in group_keys
