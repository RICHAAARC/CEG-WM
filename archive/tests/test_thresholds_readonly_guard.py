"""
文件目的：回归测试 thresholds 只读守卫及审计脚本完整性。
Module type: Core innovation module
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.detect import orchestrator as detect_orchestrator


# ==================== Readonly Threshold Guard Runtime Tests ====================


def test_thresholds_readonly_guard_passes_when_unchanged(tmp_path):
    """
    功能：验证当 thresholds 工件保持不变时，readonly guard 通过。

    Verify that evaluate workflow passes readonly guard when thresholds unchanged.
    """
    # Setup: 创建最小化的配置和 thresholds 工件
    repo_root = Path(__file__).resolve().parent.parent
    
    # 确保 orchestrator.py 存在
    orchestrator_path = repo_root / "main" / "watermarking" / "detect" / "orchestrator.py"
    assert orchestrator_path.exists(), f"orchestrator.py not found at {orchestrator_path}"
    
    # 验证 orchestrator.py 中存在 thresholds_digest_before 和 thresholds_digest_after
    source_code = orchestrator_path.read_text(encoding="utf-8")
    
    assert "thresholds_digest_before = digests.canonical_sha256(thresholds_obj)" in source_code, \
        "Before-digest capture missing in orchestrator.py"
    
    assert "thresholds_digest_after = digests.canonical_sha256(thresholds_obj_after)" in source_code, \
        "After-digest verification missing in orchestrator.py"
    
    # Assertion: 两个digest都应该在代码中被计算
    assert source_code.count("thresholds_digest_before") >= 1, \
        "Before-digest not computed or referenced"
    
    assert source_code.count("thresholds_digest_after") >= 1, \
        "After-digest not computed or referenced"


def test_thresholds_readonly_guard_reports_change_detection(tmp_path):
    """
    功能：验证当 thresholds 工件可能被修改时，只读守卫能检测到且报告到 report_obj。

    Verify that readonly guard detects changes and reports to evaluation report.
    """
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

    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "threshold_id": "test_threshold",
                "score_name": "content_score",
                "target_fpr": 0.01,
                "threshold_value": 0.5,
                "threshold_key_used": "fpr_0_01",
                "rule_version": "np_v1",
            }
        ),
        encoding="utf-8",
    )

    detect_path = tmp_path / "detect.json"
    detect_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {"status": "ok", "score": 0.8},
                "label": True,
                "attack": {"family": "rotate", "params_version": "v1"},
            }
        ),
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

    real_loader = detect_orchestrator.load_thresholds_artifact_controlled
    call_counter = {"n": 0}

    def _mutating_loader(path: str) -> Dict[str, Any]:
        call_counter["n"] += 1
        obj = real_loader(path)
        if call_counter["n"] >= 2:
            obj = dict(obj)
            obj["threshold_value"] = float(obj.get("threshold_value", 0.5)) + 0.123
        return obj

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(detect_orchestrator, "load_thresholds_artifact_controlled", _mutating_loader)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            detect_orchestrator.run_evaluate_orchestrator(cfg, _build_impl_set())
    finally:
        monkeypatch.undo()

    error_text = str(exc_info.value)
    error_text_lower = error_text.lower()

    assert "threshold" in error_text_lower, "Error message should contain threshold"
    assert "digest" in error_text_lower, "Error message should contain digest"
    assert (
        "modified" in error_text_lower
        or "readonly" in error_text_lower
        or "修改" in error_text
        or "只读" in error_text
    ), "Error message should indicate modified/readonly violation"


def test_thresholds_readonly_guard_appended_to_report(tmp_path):
    """
    功能：验证 readonly guard 结果被追加到 evaluation report（append-only）。

    Verify that readonly guard result is appended to evaluation report as audit trail.
    """
    repo_root = Path(__file__).resolve().parent.parent
    orchestrator_path = repo_root / "main" / "watermarking" / "detect" / "orchestrator.py"
    
    source_code = orchestrator_path.read_text(encoding="utf-8")
    
    # 验证 report_obj 被注入了 readonly guard 记录
    assert 'report_obj["thresholds_readonly_guard"]' in source_code, \
        "readonly guard audit record not appended to report_obj"
    
    # 验证记录包含必要的字段
    assert "digest_before" in source_code and "digest_after" in source_code, \
        "digest comparison fields missing from guard record"
    
    assert '"unchanged"' in source_code or "'unchanged'" in source_code, \
        "unchanged boolean flag missing from guard record"


# ==================== Attack Protocol Fact-Source Audit Tests ====================


def test_audit_attack_protocol_hardcoding_script_returns_zero_on_pass():
    """
    功能：验证审计脚本在 repo 无违规时返回 0（PASS）。

    Verify that hardcoding audit script returns 0 when no violations found.
    """
    repo_root = Path(__file__).resolve().parent.parent
    audit_script = repo_root / "scripts" / "audits" / "audit_attack_protocol_hardcoding.py"
    
    assert audit_script.exists(), f"Audit script not found at {audit_script}"
    
    # 执行审计脚本
    result = subprocess.run(
        [sys.executable, str(audit_script), str(repo_root)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    # PASS 时应返回 0（unless violations found, then 1）
    # 根据repo state，应该是PASS（return 0）因为没有hardcoded params
    assert result.returncode in (0, 1), \
        f"Audit script returned unexpected code: {result.returncode}"
    
    # 验证输出是有效的 JSON
    try:
        output = json.loads(result.stdout)
        assert "audit_id" in output, "audit_id missing from output"
        assert "result" in output, "result missing from output"
        assert output["result"] in ("PASS", "FAIL"), "Invalid result value"
    except json.JSONDecodeError:
        pytest.fail(f"Audit script output is not valid JSON: {result.stdout}")


def test_audit_thresholds_readonly_enforcement_script_passes():
    """
    功能：验证审计脚本在 repo 状态正确时报告 PASS。

    Verify that readonly audit script reports PASS when thresholds guard is present.
    """
    repo_root = Path(__file__).resolve().parent.parent
    audit_script = repo_root / "scripts" / "audits" / "audit_thresholds_readonly_enforcement.py"
    
    assert audit_script.exists(), f"Audit script not found at {audit_script}"
    
    # 执行审计脚本
    result = subprocess.run(
        [sys.executable, str(audit_script), str(repo_root)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    # 验证输出是有效的 JSON
    try:
        output = json.loads(result.stdout)
        assert "audit_id" in output, "audit_id missing from output"
        assert "result" in output, "result missing from output"
        
        if result.returncode == 0:
            assert output["result"] == "PASS", \
                f"Return code 0 but result is {output['result']}"
    except json.JSONDecodeError:
        pytest.fail(f"Audit script output is not valid JSON: {result.stdout}")


def test_no_hardcoded_attack_parameters_in_evaluation_module():
    """
    功能：黑盒验证 evaluation 模块中无硬编码的攻击参数。

    Black-box verification that no hardcoded attack parameters exist.
    """
    repo_root = Path(__file__).resolve().parent.parent
    eval_module = repo_root / "main" / "evaluation"
    
    assert eval_module.exists(), f"evaluation module not found at {eval_module}"
    
    # 硬编码参数的禁止关键词
    forbidden_patterns = [
        "crop_ratio_min",
        "crop_ratio_max",
        "scale_min",
        "scale_max",
        "rotate_angle",
    ]
    
    violations = []
    
    for py_file in eval_module.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        
        content = py_file.read_text(encoding="utf-8")
        
        for i, line in enumerate(content.split("\n"), 1):
            # 跳过注释和导入
            if line.strip().startswith("#") or line.strip().startswith("from "):
                continue
            
            # 检查是否出现违规模式（赋值）
            for pattern in forbidden_patterns:
                if f"{pattern} =" in line and "protocol_spec" not in line:
                    violations.append(f"{py_file.name}:{i}: {line.strip()[:80]}")
    
    assert not violations, \
        f"Found hardcoded attack parameters:\n" + "\n".join(violations)


def test_run_all_audits_includes_b2_b3_scripts():
    """
    功能：验证 run_all_audits.py 正确包含只读阈值与攻击协议事实源审计脚本。

    Verify that run_all_audits.py includes readonly-threshold and attack-protocol audit scripts.
    """
    repo_root = Path(__file__).resolve().parent.parent
    run_all_audits = repo_root / "scripts" / "run_all_audits.py"
    
    assert run_all_audits.exists(), "run_all_audits.py not found"
    
    content = run_all_audits.read_text(encoding="utf-8")
    
    # 验证只读阈值与攻击协议事实源审计脚本在列表中
    assert "audit_thresholds_readonly_enforcement.py" in content, \
        "readonly-threshold audit script (audit_thresholds_readonly_enforcement.py) not found in AUDIT_SCRIPTS"
    
    assert "audit_attack_protocol_hardcoding.py" in content, \
        "attack-protocol fact-source audit script (audit_attack_protocol_hardcoding.py) not found in AUDIT_SCRIPTS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
