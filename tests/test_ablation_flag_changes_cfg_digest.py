"""
验证 消融实验标志的更改必须改变 cfg_digest
"""

from pathlib import Path
import tempfile

from main.core import config_loader
from main.core import digests


def test_ablation_flag_changes_cfg_digest():
    """
    功能：验证 ablation flags 变化导致 cfg_digest 变化。

    Test ablation flag changes alter cfg_digest.

    Verifies:
        1. ablation.normalized is included in cfg_digest computation.
        2. enable_content=true vs enable_content=false yields different cfg_digest.
        3. Multiple flag changes accumulate into different cfg_digest values.

    Returns:
        None (asserts on failure).

    Raises:
        AssertionError: If ablation flags do not alter cfg_digest.
    """
    # 加载冻结契约与 whitelist/semantics。
    contracts, interpretation = config_loader.load_frozen_contracts_interpretation()
    whitelist = config_loader.load_runtime_whitelist()
    semantics = config_loader.load_policy_path_semantics()

    # 基线配置：全启用。
    cfg_baseline = {
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

    # 归一化 ablation flags。
    config_loader.normalize_ablation_flags(cfg_baseline)

    # 计算 baseline cfg_digest。
    include_paths = interpretation.config_loader.cfg_digest_include_paths
    include_override_applied = interpretation.config_loader.cfg_digest_override_applied_included
    baseline_digest = config_loader.compute_cfg_digest(cfg_baseline, include_paths, include_override_applied)

    # 配置 1：禁用 content 链。
    cfg_content_off = {
        "policy_path": "content_only",
        "ablation": {
            "enable_content": False,  # 变化点
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
    config_loader.normalize_ablation_flags(cfg_content_off)
    digest_content_off = config_loader.compute_cfg_digest(cfg_content_off, include_paths, include_override_applied)

    # 配置 2：禁用 geometry 链。
    cfg_geometry_off = {
        "policy_path": "content_only",
        "ablation": {
            "enable_content": True,
            "enable_geometry": False,  # 变化点
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
    config_loader.normalize_ablation_flags(cfg_geometry_off)
    digest_geometry_off = config_loader.compute_cfg_digest(cfg_geometry_off, include_paths, include_override_applied)

    # 配置 3：LF/HF 互换（lf_only=False, enable_lf=False, enable_hf=True）。
    cfg_hf_mode = {
        "policy_path": "content_only",
        "ablation": {
            "enable_content": True,
            "enable_geometry": True,
            "enable_fusion": True,
            "enable_mask": True,
            "enable_subspace": True,
            "enable_rescue": False,
            "enable_lf": False,  # 变化点
            "enable_hf": True,   # 变化点
            "lf_only": False,
            "hf_only": False,
        }
    }
    config_loader.normalize_ablation_flags(cfg_hf_mode)
    digest_hf_mode = config_loader.compute_cfg_digest(cfg_hf_mode, include_paths, include_override_applied)

    # 断言：所有 digest 必须互不相同。
    assert baseline_digest != digest_content_off, "cfg_digest must change when enable_content changes"
    assert baseline_digest != digest_geometry_off, "cfg_digest must change when enable_geometry changes"
    assert baseline_digest != digest_hf_mode, "cfg_digest must change when LF/HF flags change"
    assert digest_content_off != digest_geometry_off, "Different ablation flags must yield different digests"
    assert digest_content_off != digest_hf_mode, "Different ablation flags must yield different digests"
    assert digest_geometry_off != digest_hf_mode, "Different ablation flags must yield different digests"

    # 验证：归一化后的字段存在且可序列化。
    for cfg in [cfg_baseline, cfg_content_off, cfg_geometry_off, cfg_hf_mode]:
        assert "normalized" in cfg["ablation"], "normalized field must be generated"
        normalized = cfg["ablation"]["normalized"]
        assert isinstance(normalized, dict), "normalized must be dict"
        # 确保可稳定序列化（digests.normalize_for_digest 不抛异常）。
        digests.normalize_for_digest(normalized)


def test_ablation_mutual_exclusion_enforced():
    """
    功能：验证 lf_only / hf_only 互斥约束在归一化时被强制执行。

    Test lf_only / hf_only mutual exclusion is enforced.

    Verifies:
        1. lf_only=true, hf_only=true raises ValueError.
        2. lf_only=true sets enable_lf=true, enable_hf=false.
        3. hf_only=true sets enable_hf=true, enable_lf=false.

    Returns:
        None (asserts on failure).

    Raises:
        AssertionError: If mutual exclusion is not enforced.
    """
    # 测试 1：lf_only=true, hf_only=true 必须 fail-fast。
    cfg_conflict = {
        "policy_path": "content_only",
        "ablation": {
            "lf_only": True,
            "hf_only": True,
        }
    }
    try:
        config_loader.normalize_ablation_flags(cfg_conflict)
        assert False, "Expected ValueError for lf_only=true, hf_only=true"
    except ValueError as exc:
        assert "lf_only and ablation.hf_only cannot both be true" in str(exc).lower(), f"Unexpected error: {exc}"

    # 测试 2：lf_only=true → enable_lf=true, enable_hf=false。
    cfg_lf_only = {
        "policy_path": "content_only",
        "ablation": {
            "lf_only": True,
            "hf_only": False,
        }
    }
    config_loader.normalize_ablation_flags(cfg_lf_only)
    normalized_lf = cfg_lf_only["ablation"]["normalized"]
    assert normalized_lf["enable_lf"] is True, "lf_only=true must set enable_lf=true"
    assert normalized_lf["enable_hf"] is False, "lf_only=true must set enable_hf=false"

    # 测试 3：hf_only=true → enable_hf=true, enable_lf=false。
    cfg_hf_only = {
        "policy_path": "content_only",
        "ablation": {
            "lf_only": False,
            "hf_only": True,
        }
    }
    config_loader.normalize_ablation_flags(cfg_hf_only)
    normalized_hf = cfg_hf_only["ablation"]["normalized"]
    assert normalized_hf["enable_hf"] is True, "hf_only=true must set enable_hf=true"
    assert normalized_hf["enable_lf"] is False, "hf_only=true must set enable_lf=false"


def test_ablation_normalized_manual_set_blocked():
    """
    功能：禁止用户手动设置 ablation.normalized 字段（必须由归一化函数生成）。

    Test manual ablation.normalized setting is blocked.

    Verifies:
        1. Pre-existing ablation.normalized (non-None) raises ValueError.
        2. ablation.normalized=null is allowed (auto-generated).

    Returns:
        None (asserts on failure).

    Raises:
        AssertionError: If manual normalized setting is not blocked.
    """
    # 测试 1：ablation.normalized 已存在且非 None 必须 fail-fast。
    cfg_manual = {
        "policy_path": "content_only",
        "ablation": {
            "normalized": {"enable_content": True},  # 手动设置
        }
    }
    try:
        config_loader.normalize_ablation_flags(cfg_manual)
        assert False, "Expected ValueError for manually set ablation.normalized"
    except ValueError as exc:
        assert "must not be manually set" in str(exc).lower(), f"Unexpected error: {exc}"

    # 测试 2：ablation.normalized=null 允许（自动生成）。
    cfg_null = {
        "policy_path": "content_only",
        "ablation": {
            "normalized": None,  # 允许
        }
    }
    config_loader.normalize_ablation_flags(cfg_null)
    assert "normalized" in cfg_null["ablation"], "normalized must be generated"
    assert isinstance(cfg_null["ablation"]["normalized"], dict), "normalized must be dict after normalization"
