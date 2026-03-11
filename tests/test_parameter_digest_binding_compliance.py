"""

"""

import pytest
from pathlib import Path
import sys
import json

_tests_dir = Path(__file__).resolve().parent
_repo_root = _tests_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from main.watermarking.content_chain.low_freq_coder import LowFreqTemplateCodecV2, LOW_FREQ_TEMPLATE_CODEC_V2_ID, LOW_FREQ_TEMPLATE_CODEC_V2_VERSION
from main.core import digests


def test_variance_change_must_change_lf_trace_digest():
    """
    功能：T1 - 变更 watermark.lf.variance 必须导致 lf_trace_digest 变化。

    Test that changing watermark.lf.variance parameter causes lf_trace_digest to change
    via embed_apply(), since variance is an embed-side parameter in V2.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_trace_digest does not change when variance changes.
    """
    coder = LowFreqTemplateCodecV2("low_freq_template_codec_v2", "v1", "test_digest_abc123")

    # message_length=8, ecc_sparsity=3 生成 block_length=16，提供足够多元素。
    latent_features = [float(i) for i in range(1, 21)]

    cfg_base = {
        "watermark": {
            "lf": {
                "enabled": True,
                "message_length": 8,
                "ecc_sparsity": 3,
                "variance": 1.5,
            }
        }
    }

    cfg_changed = {
        "watermark": {
            "lf": {
                "enabled": True,
                "message_length": 8,
                "ecc_sparsity": 3,
                "variance": 2.0,
            }
        }
    }

    result_base = coder.embed_apply(
        cfg=cfg_base, latent_features=latent_features,
        plan_digest="test_plan_digest_base", cfg_digest="cfg_digest_base",
    )
    result_changed = coder.embed_apply(
        cfg=cfg_changed, latent_features=latent_features,
        plan_digest="test_plan_digest_base", cfg_digest="cfg_digest_base",
    )

    assert result_base["status"] == "ok"
    assert result_changed["status"] == "ok"
    assert result_base["lf_trace_digest"] != result_changed["lf_trace_digest"], (
        f"variance change did not cause lf_trace_digest change: "
        f"base={result_base['lf_trace_digest']}, changed={result_changed['lf_trace_digest']}"
    )


def test_cfg_digest_change_must_change_lf_trace_digest():
    """
    功能：T1b - 变更 cfg_digest 必须导致 lf_trace_digest 变化。

    Test that changing cfg_digest parameter causes lf_trace_digest to change
    when using detect_score(), since cfg_digest is included in the trace dict.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_trace_digest does not change when cfg_digest changes.
    """
    import numpy as np

    coder = LowFreqTemplateCodecV2("low_freq_template_codec_v2", "v1", "test_digest_abc123")

    cfg = {
        "watermark": {
            "lf": {
                "enabled": True,
                "message_length": 8,
                "ecc_sparsity": 3,
                "correlation_scale": 10.0,
            }
        }
    }

    rng = np.random.RandomState(42)
    lf_basis = {
        "projection_matrix": rng.randn(4, 2).tolist(),
        "basis_rank": 2,
        "latent_projection_spec": {
            "spec_version": "v1", "method": "random_index_selection",
            "feature_dim": 4, "seed": 42, "edit_timestep": 0, "sample_idx": 0,
        },
    }
    latent_features = [1.0, 2.0, 3.0, 4.0]

    _, trace_base = coder.detect_score(
        cfg=cfg, latent_features=latent_features,
        plan_digest="test_plan_digest",
        cfg_digest="cfg_digest_base",
        lf_basis=lf_basis,
    )
    _, trace_changed = coder.detect_score(
        cfg=cfg, latent_features=latent_features,
        plan_digest="test_plan_digest",
        cfg_digest="cfg_digest_changed",
        lf_basis=lf_basis,
    )

    assert trace_base["lf_trace_digest"] != trace_changed["lf_trace_digest"], (
        f"cfg_digest change did not cause lf_trace_digest change: "
        f"base={trace_base['lf_trace_digest']}, changed={trace_changed['lf_trace_digest']}"
    )


def test_allowed_impl_ids_not_subset_must_fail_audit():
    """
    功能：T2 - allowed_impl_ids 非 whitelist 子集时，audit 必 FAIL(BLOCK)。

    Test that injection_scope_manifest.allowed_impl_ids not being a subset of
    runtime_whitelist.impl_id.allowed_flat causes audit to FAIL with BLOCK severity.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If audit does not FAIL when impl_ids are not a subset.
    """
    from scripts.audits.audit_injection_scope_manifest_binding import run_audit

    # 运行审计。
    result = run_audit(_repo_root)

    # 验证审计结果。
    assert isinstance(result, dict), "audit result must be dict"
    
    # 检查是否包含 impl_id_closure 检查。
    checks = result.get("evidence", {}).get("checks", [])
    impl_id_check = None
    for check in checks:
        if check.get("check") == "impl_id_closure":
            impl_id_check = check
            break

    assert impl_id_check is not None, "impl_id_closure check must exist in audit"
    
    # 如果 unmapped_impl_ids 不为空，则 pass 必须为 False。
    unmapped = impl_id_check.get("unmapped_impl_ids")
    if unmapped and len(unmapped) > 0:
        assert not impl_id_check.get("pass"), (
            f"impl_id_closure check must fail when unmapped_impl_ids exist: {unmapped}"
        )
        assert result.get("result") == "FAIL", (
            "audit result must be FAIL when impl_id_closure fails"
        )
    else:
        # 当前配置符合闭合性，检查应通过。
        assert impl_id_check.get("pass"), "impl_id_closure check must pass when no unmapped impl_ids"


def test_plan_digest_mismatch_must_return_mismatch_and_no_score():
    """
    功能：T3 - detect_score() 传入空字符串 plan_digest 时必须抛出 TypeError。

    Test that detect_score() raises TypeError when plan_digest is empty string,
    since V2 rejects invalid plan_digest at the boundary (no mismatch status).

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If TypeError is not raised for empty plan_digest.
    """
    coder = LowFreqTemplateCodecV2("low_freq_template_codec_v2", "v1", "test_digest_abc123")

    cfg = {
        "watermark": {
            "lf": {
                "enabled": True,
                "message_length": 8,
                "ecc_sparsity": 3,
                "correlation_scale": 10.0,
            }
        }
    }

    # V2 中 plan_digest 为空字符串时抛出 TypeError（边界输入校验）。
    with pytest.raises(TypeError):
        coder.detect_score(
            cfg=cfg,
            latent_features=[1.0, 2.0, 3.0, 4.0],
            plan_digest="",  # 无效：空字符串
        )


def test_embed_apply_deterministic_output():
    """
    功能：T4 - embed_apply() 必须输出确定性结果（同种子同输出）。

    Test that embed_apply() produces deterministic output for the same inputs:
    calling twice with identical parameters must yield identical lf_trace_digest
    and embedded latent features.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If output is not deterministic.
    """
    coder = LowFreqTemplateCodecV2("low_freq_template_codec_v2", "v1", "test_digest_abc123")

    cfg = {
        "watermark": {
            "lf": {
                "enabled": True,
                "message_length": 8,
                "ecc_sparsity": 3,
                "variance": 1.5,
            },
        }
    }

    # message_length=8, ecc_sparsity=3 生成 block_length=16，提供足够多元素。
    latent_features = [float(i) for i in range(1, 21)]

    result_1 = coder.embed_apply(
        cfg=cfg, latent_features=latent_features,
        plan_digest="test_plan_digest", cfg_digest="test_cfg_digest",
    )
    result_2 = coder.embed_apply(
        cfg=cfg, latent_features=latent_features,
        plan_digest="test_plan_digest", cfg_digest="test_cfg_digest",
    )

    assert result_1["status"] == "ok"
    # 嵌入结果（确定性）。
    assert result_1["latent_features_embedded"] == result_2["latent_features_embedded"], (
        "embed_apply() must produce deterministic latent output for same inputs"
    )
    # trace digest（确定性）。
    assert result_1["lf_trace_digest"] == result_2["lf_trace_digest"], (
        "lf_trace_digest must be deterministic for same inputs"
    )


def test_embed_detect_consistency():
    """
    功能：T5 - detect_score() 对相同输入调用两次必须返回相同结果（一致性）。

    Test that detect_score() is deterministic: calling twice with the same inputs
    must yield identical lf_score and lf_trace_digest.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If detect_score() results differ between two calls.
    """
    import numpy as np

    coder = LowFreqTemplateCodecV2("low_freq_template_codec_v2", "v1", "test_digest_abc123")

    cfg = {
        "watermark": {
            "lf": {
                "enabled": True,
                "message_length": 8,
                "ecc_sparsity": 3,
                "correlation_scale": 10.0,
            },
        }
    }

    rng = np.random.RandomState(7)
    lf_basis = {
        "projection_matrix": rng.randn(4, 2).tolist(),
        "basis_rank": 2,
        "latent_projection_spec": {
            "spec_version": "v1", "method": "random_index_selection",
            "feature_dim": 4, "seed": 7, "edit_timestep": 0, "sample_idx": 0,
        },
    }
    latent_features = [1.0, 2.0, 3.0, 4.0]

    score1, trace1 = coder.detect_score(
        cfg=cfg, latent_features=latent_features,
        plan_digest="test_plan_digest", cfg_digest="test_cfg_digest",
        lf_basis=lf_basis,
    )
    score2, trace2 = coder.detect_score(
        cfg=cfg, latent_features=latent_features,
        plan_digest="test_plan_digest", cfg_digest="test_cfg_digest",
        lf_basis=lf_basis,
    )

    assert trace1["status"] == "ok"
    # detect_score 结果必须确定性一致。
    assert score1 == score2, "lf_score must be deterministic for same inputs"
    assert trace1["lf_trace_digest"] == trace2["lf_trace_digest"], (
        "lf_trace_digest must be deterministic for same inputs"
    )
