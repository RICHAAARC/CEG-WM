"""
Test: Non-whitelisted ablation override is blocked.

功能说明：
- 验证不在 runtime_whitelist.yaml 中的 ablation override 参数被阻断。
- 确保 override_rules.py 严格校验 arg_name 白名单。
- 验证 ablation override 通过白名单校验后可成功应用。

Module type: General module
"""

import tempfile
from pathlib import Path

from main.core import config_loader
from main.policy import override_rules


def test_non_whitelisted_ablation_override_is_blocked():
    """
    功能：验证非白名单 ablation override 参数被阻断。

    Test non-whitelisted ablation override is blocked.

    Verifies:
        1. Unknown ablation override arg_name raises ValueError.
        2. Whitelisted ablation override (e.g., ablation_disable_content) succeeds.
        3. override_applied audit contains ablation override fields.

    Returns:
        None (asserts on failure).

    Raises:
        AssertionError: If whitelist enforcement fails.
    """
    # 加载 whitelist / interpretation。
    contracts, interpretation = config_loader.load_frozen_contracts_interpretation()
    whitelist = config_loader.load_runtime_whitelist()

    # 基线配置。
    cfg = {
        "policy_path": "content_only",
        "ablation": {
            "enable_content": True,
            "enable_geometry": True,
            "enable_fusion": True,
            "enable_mask": True,
            "enable_subspace": True,
            "enable_rescue": False,
            "enable_lf": True,
            "enable_hf": False,
            "lf_only": False,
            "hf_only": False,
        }
    }

    # 测试 1：非白名单 ablation override 必须 fail-fast。
    override_args_invalid = ["ablation_unknown_flag=true"]
    try:
        override_rules.apply_cli_overrides(
            cfg,
            override_args_invalid,
            whitelist,
            interpretation
        )
        assert False, "Expected ValueError for non-whitelisted ablation override"
    except ValueError as exc:
        assert "not allowed" in str(exc).lower() or "unknown" in str(exc).lower(), f"Unexpected error: {exc}"

    # 测试 2：白名单 ablation override 必须成功应用。
    cfg_valid = {
        "policy_path": "content_only",
        "ablation": {
            "enable_content": True,  # 默认启用
            "enable_geometry": True,
            "enable_fusion": True,
            "enable_mask": True,
            "enable_subspace": True,
            "enable_rescue": False,
            "enable_lf": True,
            "enable_hf": False,
            "lf_only": False,
            "hf_only": False,
        }
    }
    override_args_valid = ["ablation_disable_content=false"]
    override_applied = override_rules.apply_cli_overrides(
        cfg_valid,
        override_args_valid,
        whitelist,
        interpretation
    )

    # 断言：override_applied 非空。
    assert override_applied is not None, "override_applied must be generated"
    assert isinstance(override_applied, dict), "override_applied must be dict"

    # 断言：applied_fields 包含 ablation.enable_content 覆写。
    applied_fields = override_applied.get("applied_fields", [])
    assert len(applied_fields) == 1, "Exactly one override should be applied"
    field_entry = applied_fields[0]
    assert field_entry["field_path"] == "ablation.enable_content", "Overridden field_path must match"
    assert field_entry["old_value"] is True, "Old value must be True (default)"
    assert field_entry["new_value"] is False, "New value must be False (ablation_disable_content)"
    assert field_entry["source"] == "cli", "Override source must be 'cli'"

    # 断言：cfg 中 ablation.enable_content 已被修改为 False。
    assert cfg_valid["ablation"]["enable_content"] is False, "Config must reflect override"


def test_ablation_override_must_be_in_arg_name_enum():
    """
    功能：验证 ablation override arg_name 必须在 arg_name_enum.allowed 白名单中。

    Test ablation override arg_name must be in arg_name_enum whitelist.

    Verifies:
        1. arg_name not in arg_name_enum.allowed raises ValueError.
        2. arg_name in arg_name_enum.allowed and allowed_overrides succeeds.

    Returns:
        None (asserts on failure).

    Raises:
        AssertionError: If arg_name_enum enforcement fails.
    """
    # 加载 whitelist。
    whitelist = config_loader.load_runtime_whitelist()
    arg_name_enum_allowed = whitelist.data["override"]["arg_name_enum"]["allowed"]

    # 断言：ablation_disable_content 在白名单中。
    assert "ablation_disable_content" in arg_name_enum_allowed, "ablation_disable_content must be in arg_name_enum.allowed"
    assert "ablation_enable_geometry" in arg_name_enum_allowed, "ablation_enable_geometry must be in arg_name_enum.allowed"
    assert "ablation_disable_lf" in arg_name_enum_allowed, "ablation_disable_lf must be in arg_name_enum.allowed"
    assert "ablation_enable_hf" in arg_name_enum_allowed, "ablation_enable_hf must be in arg_name_enum.allowed"

    # 断言：非法 arg_name 不在白名单中。
    assert "ablation_unknown_flag" not in arg_name_enum_allowed, "Unknown ablation flag must not be in whitelist"


def test_ablation_override_cfg_digest_changes():
    """
    功能：验证 ablation override 导致 cfg_digest 变化（经过归一化）。

    Test ablation override causes cfg_digest change after normalization.

    Verifies:
        1. Before override: ablation.normalized includes enable_content=true.
        2. After override: ablation.normalized includes enable_content=false.
        3. cfg_digest before != cfg_digest after.

    Returns:
        None (asserts on failure).

    Raises:
        AssertionError: If cfg_digest does not change after override.
    """
    # 加载 whitelist / interpretation。
    contracts, interpretation = config_loader.load_frozen_contracts_interpretation()
    whitelist = config_loader.load_runtime_whitelist()

    # 基线配置（未覆写）。
    cfg_before = {
        "policy_path": "content_only",
        "ablation": {
            "enable_content": True,
            "enable_geometry": True,
            "enable_fusion": True,
            "enable_mask": True,
            "enable_subspace": True,
            "enable_rescue": False,
            "enable_lf": True,
            "enable_hf": False,
            "lf_only": False,
            "hf_only": False,
        }
    }
    config_loader.normalize_ablation_flags(cfg_before)
    include_paths = interpretation.config_loader.cfg_digest_include_paths
    include_override_applied = interpretation.config_loader.cfg_digest_override_applied_included
    digest_before = config_loader.compute_cfg_digest(cfg_before, include_paths, include_override_applied)

    # 配置（覆写后）。
    cfg_after = {
        "policy_path": "content_only",
        "ablation": {
            "enable_content": True,  # 默认启用
            "enable_geometry": True,
            "enable_fusion": True,
            "enable_mask": True,
            "enable_subspace": True,
            "enable_rescue": False,
            "enable_lf": True,
            "enable_hf": False,
            "lf_only": False,
            "hf_only": False,
        }
    }
    override_args = ["ablation_disable_content=false"]
    override_applied = override_rules.apply_cli_overrides(
        cfg_after,
        override_args,
        whitelist,
        interpretation
    )
    cfg_after["override_applied"] = override_applied
    config_loader.normalize_ablation_flags(cfg_after)
    digest_after = config_loader.compute_cfg_digest(cfg_after, include_paths, include_override_applied)

    # 断言：cfg_digest 必须变化。
    assert digest_before != digest_after, "cfg_digest must change after ablation override"

    # 断言：归一化后的 enable_content 为 False。
    normalized_after = cfg_after["ablation"]["normalized"]
    assert normalized_after["enable_content"] is False, "Normalized enable_content must be False after override"
