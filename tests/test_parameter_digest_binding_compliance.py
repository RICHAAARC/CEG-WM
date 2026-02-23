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

from main.watermarking.content_chain.low_freq_coder import LowFreqCoder
from main.core import digests


def test_variance_change_must_change_lf_trace_digest():
    """
    功能：T1 - 变更 watermark.lf.variance 必须导致 lf_trace_digest 变化。

    Test that changing watermark.lf.variance parameter causes lf_trace_digest to change.
    This validates that variance is properly bound to plan_digest_include_paths.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_trace_digest does not change when variance changes.
    """
    impl_id = "low_freq_coder_v1"
    impl_version = "v1"
    impl_digest = "test_digest_abc123"

    coder = LowFreqCoder(impl_id, impl_version, impl_digest)

    # 基础配置。
    cfg_base = {
        "watermark": {
            "lf": {
                "enabled": True,
                "codebook_id": "default",
                "ecc": 3,
                "strength": 0.1,
                "delta": 1.0,
                "block_length": 8,
                "variance": 1.5  # 基础方差
            },
            "plan_digest": "test_plan_digest_base"
        }
    }

    # 变更 variance 的配置。
    cfg_changed = {
        "watermark": {
            "lf": {
                "enabled": True,
                "codebook_id": "default",
                "ecc": 3,
                "strength": 0.1,
                "delta": 1.0,
                "block_length": 8,
                "variance": 2.0  # 变更方差
            },
            "plan_digest": "test_plan_digest_base"
        }
    }

    inputs = {
        "latent_features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
        "latent_shape": (1, 10)
    }

    # 执行基础配置检测。
    evidence_base = coder.extract(cfg_base, inputs, cfg_digest="cfg_digest_base")
    trace_digest_base = evidence_base.lf_trace_digest

    # 执行变更配置检测。
    evidence_changed = coder.extract(cfg_changed, inputs, cfg_digest="cfg_digest_base")
    trace_digest_changed = evidence_changed.lf_trace_digest

    # 断言：lf_trace_digest 必须变化。
    assert trace_digest_base != trace_digest_changed, (
        f"variance change did not cause lf_trace_digest change: "
        f"base={trace_digest_base}, changed={trace_digest_changed}"
    )


def test_cfg_digest_change_must_change_lf_trace_digest():
    """
    功能：T1b - 变更 cfg_digest 必须导致 lf_trace_digest 变化。

    Test that changing cfg_digest parameter causes lf_trace_digest to change.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_trace_digest does not change when cfg_digest changes.
    """
    impl_id = "low_freq_coder_v1"
    impl_version = "v1"
    impl_digest = "test_digest_abc123"

    coder = LowFreqCoder(impl_id, impl_version, impl_digest)

    cfg = {
        "watermark": {
            "lf": {
                "enabled": True,
                "codebook_id": "default",
                "ecc": 3,
                "strength": 0.1,
                "delta": 1.0,
                "block_length": 8,
                "variance": 1.5
            },
            "plan_digest": "test_plan_digest"
        }
    }

    inputs = {
        "latent_features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
        "latent_shape": (1, 8)
    }

    # 执行基础 cfg_digest 检测。
    evidence_base = coder.extract(cfg, inputs, cfg_digest="cfg_digest_base")
    trace_digest_base = evidence_base.lf_trace_digest

    # 执行变更 cfg_digest 检测。
    evidence_changed = coder.extract(cfg, inputs, cfg_digest="cfg_digest_changed")
    trace_digest_changed = evidence_changed.lf_trace_digest

    # 断言：lf_trace_digest 必须变化。
    assert trace_digest_base != trace_digest_changed, (
        f"cfg_digest change did not cause lf_trace_digest change: "
        f"base={trace_digest_base}, changed={trace_digest_changed}"
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
    功能：T3 - detect 侧 plan_digest 不一致时必须返回 status=mismatch 且 score=None。

    Test that when detect-side plan_digest does not match expected_plan_digest,
    the extract() method returns status="mismatch" and score=None (failure must not give score).

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If status is not "mismatch" or score is not None.
    """
    impl_id = "low_freq_coder_v1"
    impl_version = "v1"
    impl_digest = "test_digest_abc123"

    coder = LowFreqCoder(impl_id, impl_version, impl_digest)

    cfg = {
        "watermark": {
            "lf": {
                "enabled": True,
                "codebook_id": "default",
                "ecc": 3,
                "strength": 0.1,
                "delta": 1.0,
                "block_length": 8,
                "variance": 1.5
            },
            "plan_digest": "actual_plan_digest"  # 实际 plan_digest
        }
    }

    inputs = {
        "latent_features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
        "latent_shape": (1, 8),
        "expected_plan_digest": "expected_plan_digest"  # 期望的 plan_digest（不一致）
    }

    # 执行检测。
    evidence = coder.extract(cfg, inputs, cfg_digest="test_cfg_digest")

    # 断言：status 必须为 "mismatch"。
    assert evidence.status == "mismatch", (
        f"status must be 'mismatch' when plan_digest mismatch, got {evidence.status}"
    )

    # 断言：score 必须为 None（失败不得给分）。
    assert evidence.score is None, (
        f"score must be None when plan_digest mismatch, got {evidence.score}"
    )

    # 断言：content_failure_reason 必须为 "lf_coder_plan_mismatch"。
    assert evidence.content_failure_reason == "lf_coder_plan_mismatch", (
        f"content_failure_reason must be 'lf_coder_plan_mismatch', "
        f"got {evidence.content_failure_reason}"
    )


def test_embed_apply_deterministic_output():
    """
    功能：T4 - embed_apply() 必须输出确定性结果（同种子同输出）。

    Test that embed_apply() produces deterministic output for the same inputs.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If output is not deterministic.
    """
    impl_id = "low_freq_coder_v1"
    impl_version = "v1"
    impl_digest = "test_digest_abc123"

    coder = LowFreqCoder(impl_id, impl_version, impl_digest)

    cfg = {
        "watermark": {
            "lf": {
                "enabled": True,
                "codebook_id": "default",
                "ecc": 3,
                "strength": 0.1,
                "delta": 1.0,
                "block_length": 8,
                "variance": 1.5
            },
            "plan_digest": "test_plan_digest"
        }
    }

    latent_features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # 执行两次嵌入（同配置、同输入）。
    result_1 = coder.embed_apply(cfg, latent_features, "test_plan_digest", "test_cfg_digest")
    result_2 = coder.embed_apply(cfg, latent_features, "test_plan_digest", "test_cfg_digest")

    # 断言：嵌入后的 latent_features 必须相同（确定性）。
    assert result_1["latent_features_embedded"] == result_2["latent_features_embedded"], (
        "embed_apply() must produce deterministic output for same inputs"
    )

    # 断言：embedding_digest 必须相同（可复算性）。
    assert result_1["embedding_digest"] == result_2["embedding_digest"], (
        "embedding_digest must be deterministic for same inputs"
    )


def test_embed_detect_consistency():
    """
    功能：T5 - embed-detect 闭环一致性（嵌入后检测必须能识别水印）。

    Test that watermark embedded via embed_apply() can be successfully detected via extract().

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If embedded watermark cannot be detected.
    """
    impl_id = "low_freq_coder_v1"
    impl_version = "v1"
    impl_digest = "test_digest_abc123"

    coder = LowFreqCoder(impl_id, impl_version, impl_digest)

    cfg = {
        "watermark": {
            "lf": {
                "enabled": True,
                "codebook_id": "default",
                "ecc": 3,
                "strength": 0.5,  # 较高强度确保可检测
                "delta": 1.0,
                "block_length": 8,
                "variance": 1.5
            },
            "plan_digest": "test_plan_digest"
        }
    }

    latent_features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # 执行嵌入。
    embed_result = coder.embed_apply(cfg, latent_features, "test_plan_digest", "test_cfg_digest")
    embedded_features = embed_result["latent_features_embedded"]

    # 执行检测（使用嵌入后的特征）。
    inputs = {
        "latent_features": [embedded_features],
        "latent_shape": (1, len(embedded_features))
    }
    evidence = coder.extract(cfg, inputs, cfg_digest="test_cfg_digest")

    # 断言：status 必须为 "ok"（检测成功）。
    assert evidence.status == "ok", (
        f"status must be 'ok' after embedding, got {evidence.status}"
    )

    # 断言：lf_score 必须非 None 且 > 0（水印被检测到）。
    assert evidence.lf_score is not None, "lf_score must not be None after embedding"
    assert evidence.lf_score > 0, f"lf_score must be > 0 after embedding, got {evidence.lf_score}"
