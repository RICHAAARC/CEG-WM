"""
测试 spec 和 audit 闭合验证

Module type: General module

Test paper_faithfulness_spec and run_all_audits closure verification.
Verifies that audit scripts declared in spec are present and registered.
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any
import yaml

# 添加 scripts 目录到路径以导入 run_all_audits 函数
import sys
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from run_all_audits import (
    _load_spec_audit_requirements,
    _validate_spec_audit_closure,
)


class TestSpecAuditRequirementsLoading:
    """测试从 spec 加载审计需求。"""

    def test_load_spec_without_audit_requirements(self, tmp_path: Path):
        """测试加载没有 audit_gate_requirements 的 spec。"""
        repo_root = tmp_path
        configs_dir = repo_root / "configs"
        configs_dir.mkdir()

        spec_cfg = {
            "paper_faithfulness_spec_version": "v1.0",
            "authority": {"location": "configs/paper_faithfulness_spec.yaml"},
            "target_method_families": {}
            # 没有 audit_gate_requirements
        }
        spec_path = configs_dir / "paper_faithfulness_spec.yaml"
        spec_path.write_text(yaml.dump(spec_cfg), encoding="utf-8")

        result = _load_spec_audit_requirements(repo_root)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_load_spec_with_audit_requirements(self, tmp_path: Path):
        """测试加载包含 audit_gate_requirements 的 spec。"""
        repo_root = tmp_path
        configs_dir = repo_root / "configs"
        configs_dir.mkdir()

        spec_cfg = {
            "paper_faithfulness_spec_version": "v1.0",
            "authority": {"location": "configs/paper_faithfulness_spec.yaml"},
            "audit_gate_requirements": {
                "audit_1": {
                    "audit_script": "scripts/audits/audit_test_1.py",
                    "failure_level": "BLOCK"
                },
                "audit_2": {
                    "audit_script": "scripts/audits/audit_test_2.py",
                    "failure_level": "BLOCK"
                }
            }
        }
        spec_path = configs_dir / "paper_faithfulness_spec.yaml"
        spec_path.write_text(yaml.dump(spec_cfg), encoding="utf-8")

        result = _load_spec_audit_requirements(repo_root)
        assert len(result) == 2
        assert result["audit_1"] == "scripts/audits/audit_test_1.py"
        assert result["audit_2"] == "scripts/audits/audit_test_2.py"

    def test_load_spec_file_not_found(self, tmp_path: Path):
        """测试 spec 文件不存在时返回空。"""
        repo_root = tmp_path
        result = _load_spec_audit_requirements(repo_root)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_load_spec_with_invalid_yaml(self, tmp_path: Path):
        """测试 spec 文件 YAML 格式错误。"""
        repo_root = tmp_path
        configs_dir = repo_root / "configs"
        configs_dir.mkdir()

        spec_path = configs_dir / "paper_faithfulness_spec.yaml"
        spec_path.write_text("invalid: yaml: content:", encoding="utf-8")

        with pytest.raises(ValueError, match="Failed to load spec"):
            _load_spec_audit_requirements(repo_root)


class TestSpecAuditClosure:
    """测试 spec 和 audit 闭合验证。"""

    def test_closure_validation_with_all_scripts_present(self, tmp_path: Path):
        """测试所有脚本都存在时通过。"""
        repo_root = tmp_path
        scripts_dir = repo_root / "scripts"
        audits_dir = scripts_dir / "audits"
        audits_dir.mkdir(parents=True)

        (audits_dir / "audit_test_1.py").write_text("# test", encoding="utf-8")
        (audits_dir / "audit_test_2.py").write_text("# test", encoding="utf-8")

        spec_requirements = {
            "req_1": "scripts/audits/audit_test_1.py",
            "req_2": "scripts/audits/audit_test_2.py"
        }
        configured_scripts = [
            "audits/audit_test_1.py",
            "audits/audit_test_2.py"
        ]

        results = _validate_spec_audit_closure(
            spec_requirements, configured_scripts, repo_root, strict=False
        )
        assert all(r.get("result") != "FAIL" for r in results)

    def test_closure_validation_with_missing_script(self, tmp_path: Path):
        """测试脚本缺失时失败。"""
        repo_root = tmp_path
        scripts_dir = repo_root / "scripts"
        audits_dir = scripts_dir / "audits"
        audits_dir.mkdir(parents=True)

        (audits_dir / "audit_test_1.py").write_text("# test", encoding="utf-8")

        spec_requirements = {
            "req_1": "scripts/audits/audit_test_1.py",
            "req_2": "scripts/audits/audit_test_2.py"
        }
        configured_scripts = [
            "audits/audit_test_1.py",
            "audits/audit_test_2.py"
        ]

        results = _validate_spec_audit_closure(
            spec_requirements, configured_scripts, repo_root, strict=True
        )
        assert len(results) > 0
        assert any(r.get("result") == "FAIL" for r in results)
        evidence = results[0].get("evidence", {})
        assert len(evidence.get("missing_scripts", [])) > 0

    def test_closure_validation_with_unregistered_script(self, tmp_path: Path):
        """测试脚本未注册时失败。"""
        repo_root = tmp_path
        scripts_dir = repo_root / "scripts"
        audits_dir = scripts_dir / "audits"
        audits_dir.mkdir(parents=True)

        (audits_dir / "audit_test_1.py").write_text("# test", encoding="utf-8")
        (audits_dir / "audit_test_2.py").write_text("# test", encoding="utf-8")

        spec_requirements = {
            "req_1": "scripts/audits/audit_test_1.py",
            "req_2": "scripts/audits/audit_test_2.py"
        }
        configured_scripts = [
            "audits/audit_test_1.py"
        ]

        results = _validate_spec_audit_closure(
            spec_requirements, configured_scripts, repo_root, strict=True
        )
        assert len(results) > 0
        assert any(r.get("result") == "FAIL" for r in results)
        evidence = results[0].get("evidence", {})
        assert len(evidence.get("unregistered_scripts", [])) > 0

    def test_closure_validation_empty_spec_requirements(self):
        """测试空的 spec 需求列表。"""
        spec_requirements = {}
        configured_scripts = ["audits/audit_test_1.py"]

        results = _validate_spec_audit_closure(
            spec_requirements, configured_scripts, Path("/tmp"), strict=False
        )
        # 空的需求列表应该返回空结果列表或 PASS
        assert len(results) == 0 or all(r.get("result") != "FAIL" for r in results)

    def test_closure_validation_type_errors(self):
        """测试参数类型错误。"""
        with pytest.raises(TypeError, match="spec_audit_requirements must be dict"):
            _validate_spec_audit_closure(
                "not dict", ["scripts"], Path("/tmp"), strict=False
            )

        with pytest.raises(TypeError, match="configured_scripts must be list"):
            _validate_spec_audit_closure(
                {}, "not list", Path("/tmp"), strict=False
            )

        with pytest.raises(TypeError, match="repo_root must be Path"):
            _validate_spec_audit_closure(
                {}, [], "/string/path", strict=False
            )

    def test_closure_validation_strict_mode(self, tmp_path: Path):
        """测试严格模式下缺失脚本被标记为 BLOCK。"""
        repo_root = tmp_path
        scripts_dir = repo_root / "scripts"
        audits_dir = scripts_dir / "audits"
        audits_dir.mkdir(parents=True)

        spec_requirements = {
            "missing_audit": "scripts/audits/missing_audit.py"
        }
        configured_scripts = []

        results = _validate_spec_audit_closure(
            spec_requirements, configured_scripts, repo_root, strict=True
        )
        assert len(results) > 0
        result = results[0]
        assert result.get("result") == "FAIL"
        assert result.get("severity") == "BLOCK"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
