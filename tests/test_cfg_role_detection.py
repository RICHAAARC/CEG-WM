"""
测试 cfg role 检测与入口一致性验证

Module type: General module

Test cfg role detection and paper_full_cuda profile cfg misuse prevention.
Verifies that spec configs cannot be passed to paper_full_cuda profile.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

# 添加 scripts 目录到路径以导入 run_onefile_workflow 函数
import sys
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from run_onefile_workflow import (
    _detect_cfg_role,
    _validate_cfg_role_for_profile,
    CFG_ROLE_SPEC,
    CFG_ROLE_RUNTIME,
    PROFILE_PAPER_FULL_CUDA,
    PROFILE_CPU_SMOKE,
)


class TestCfgRoleDetection:
    """测试 cfg role 检测。"""

    def test_detect_spec_config(self):
        """测试识别规范配置文件。"""
        spec_cfg = {
            "paper_faithfulness_spec_version": "v1.0",
            "authority": {
                "location": "configs/paper_faithfulness_spec.yaml",
                "uniqueness": "唯一权威",
            },
            "audit_gate_requirements": {
                "some_audit": {
                    "audit_script": "scripts/audits/some_audit.py"
                }
            }
        }
        role = _detect_cfg_role(spec_cfg)
        assert role == CFG_ROLE_SPEC

    def test_detect_runtime_config(self):
        """测试识别运行期配置文件。"""
        runtime_cfg = {
            "policy_path": "content_only",
            "device": "cuda",
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
        }
        role = _detect_cfg_role(runtime_cfg)
        assert role == CFG_ROLE_RUNTIME

    def test_detect_runtime_config_without_authority(self):
        """测试没有 authority 字段的配置识别为 runtime。"""
        cfg = {
            "paper_faithfulness_spec_version": "v1.0",
            "target_method_families": {}
            # 缺少 authority 和 audit_gate_requirements
        }
        role = _detect_cfg_role(cfg)
        assert role == CFG_ROLE_RUNTIME

    def test_detect_cfg_role_with_partial_spec_fields(self):
        """测试仅有 authority 但无 audit_gate_requirements 识别为 runtime。"""
        cfg = {
            "authority": {"location": "test"},
            # 缺少 audit_gate_requirements
        }
        role = _detect_cfg_role(cfg)
        assert role == CFG_ROLE_RUNTIME

    def test_detect_cfg_role_type_error(self):
        """测试输入非 dict 时的类型错误。"""
        with pytest.raises(TypeError, match="cfg_obj must be dict"):
            _detect_cfg_role("not a dict")


class TestCfgRoleValidation:
    """测试 cfg role 与 profile 的兼容性验证。"""

    def test_spec_cfg_with_paper_full_cuda_profile_fails(self):
        """测试 spec cfg 传递给 paper_full_cuda profile 时失败。"""
        spec_cfg = {
            "authority": {"location": "test"},
            "audit_gate_requirements": {}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(spec_cfg, f)
            cfg_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="requires runtime config.*spec config"):
                _validate_cfg_role_for_profile(spec_cfg, cfg_path, PROFILE_PAPER_FULL_CUDA)
        finally:
            if cfg_path.exists():
                cfg_path.unlink()

    def test_runtime_cfg_with_paper_full_cuda_profile_passes(self):
        """测试 runtime cfg 传递给 paper_full_cuda profile 时通过。"""
        runtime_cfg = {
            "policy_path": "content_only",
            "device": "cuda",
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(runtime_cfg, f)
            cfg_path = Path(f.name)

        try:
            # 应该不抛出异常
            _validate_cfg_role_for_profile(runtime_cfg, cfg_path, PROFILE_PAPER_FULL_CUDA)
        finally:
            if cfg_path.exists():
                cfg_path.unlink()

    def test_spec_cfg_with_cpu_smoke_profile_fails(self):
        """测试 spec cfg 传递给 cpu_smoke profile 时失败。"""
        spec_cfg = {
            "authority": {"location": "test"},
            "audit_gate_requirements": {}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(spec_cfg, f)
            cfg_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="requires runtime config"):
                _validate_cfg_role_for_profile(spec_cfg, cfg_path, PROFILE_CPU_SMOKE)
        finally:
            if cfg_path.exists():
                cfg_path.unlink()

    def test_validation_type_errors(self):
        """测试参数类型错误。"""
        cfg = {"test": "value"}
        cfg_path = Path("/test/path.yaml")

        with pytest.raises(TypeError, match="cfg_obj must be dict"):
            _validate_cfg_role_for_profile("not dict", cfg_path, PROFILE_PAPER_FULL_CUDA)

        with pytest.raises(TypeError, match="cfg_path must be Path"):
            _validate_cfg_role_for_profile(cfg, "/string/path", PROFILE_PAPER_FULL_CUDA)

        with pytest.raises(TypeError, match="profile must be non-empty str"):
            _validate_cfg_role_for_profile(cfg, cfg_path, None)

    def test_validation_error_message_includes_guidance(self):
        """测试错误消息包含修正指导。"""
        spec_cfg = {
            "authority": {"location": "configs/paper_faithfulness_spec.yaml"},
            "audit_gate_requirements": {}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(spec_cfg, f)
            cfg_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                _validate_cfg_role_for_profile(spec_cfg, cfg_path, PROFILE_PAPER_FULL_CUDA)
            
            error_msg = str(exc_info.value)
            assert "paper_full_cuda.yaml" in error_msg
            assert "paper_faithfulness_spec.yaml" in error_msg
        finally:
            if cfg_path.exists():
                cfg_path.unlink()


def test_actual_runtime_configs_are_detected_as_runtime() -> None:
    """测试实际运行期配置文件被识别为 runtime。"""
    repo_root = Path(__file__).resolve().parent.parent
    runtime_cfg_paths = [
        repo_root / "configs" / "default.yaml",
        repo_root / "configs" / "smoke_cpu.yaml",
        repo_root / "configs" / "paper_full_cuda.yaml",
    ]

    for cfg_path in runtime_cfg_paths:
        cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert isinstance(cfg_obj, dict)
        assert _detect_cfg_role(cfg_obj) == CFG_ROLE_RUNTIME


def test_actual_spec_config_is_detected_as_spec() -> None:
    """测试实际 paper faithfulness spec 被识别为 spec。"""
    repo_root = Path(__file__).resolve().parent.parent
    spec_cfg_path = repo_root / "configs" / "paper_faithfulness_spec.yaml"
    spec_cfg_obj = yaml.safe_load(spec_cfg_path.read_text(encoding="utf-8"))

    assert isinstance(spec_cfg_obj, dict)
    assert _detect_cfg_role(spec_cfg_obj) == CFG_ROLE_SPEC


def test_smoke_cpu_profile_accepts_actual_smoke_runtime_config() -> None:
    """测试 cpu_smoke profile 接受真实 smoke_cpu 运行期配置。"""
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "configs" / "smoke_cpu.yaml"
    cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    assert isinstance(cfg_obj, dict)
    _validate_cfg_role_for_profile(cfg_obj, cfg_path, PROFILE_CPU_SMOKE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
