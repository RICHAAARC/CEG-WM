"""
真实张量级水印注入系统集成测试

功能说明：
- 验证 LF/HF 子空间投影与编码的基本功能。
- 验证 latent_modifier 与 hook 的集成。
- 验证 plan_digest 一致性校验。
- 验证 content_score 计算与 mismatch/absent 语义。

Module type: Test module
"""

import sys
import pytest
import numpy as np
from typing import Dict, Any, Optional


def test_lf_projection_and_encoding():
    """
    测试 LF 子空间投影与编码。
    """
    from main.watermarking.content_chain import channel_lf
    
    # 创建模拟的 latent 和 basis。
    latent_dim = 64
    latent = np.random.randn(latent_dim).astype(np.float32)
    
    # 创建简单的 LF basis（随机正交投影）。
    lf_rank = 8
    projection_matrix = np.random.randn(latent_dim, lf_rank).astype(np.float32)
    # 正交化。
    projection_matrix, _ = np.linalg.qr(projection_matrix)
    
    basis = {"projection_matrix": projection_matrix}
    cfg = {"lf_enabled": True, "lf_strength": 1.5}
    
    # 测试投影。
    coeffs = channel_lf.compute_lf_basis_projection(latent, basis)
    assert coeffs.shape == (lf_rank,), f"Expected shape {(lf_rank,)}, got {coeffs.shape}"
    assert np.all(np.isfinite(coeffs)), "Coefficients should be finite"
    
    # 测试编码。
    encoded_coeffs, encoding_evidence = channel_lf.apply_low_freq_encoding(
        coeffs, key=42, cfg=cfg
    )
    assert encoded_coeffs.shape == coeffs.shape, "Encoded coeffs shape mismatch"
    assert isinstance(encoding_evidence, dict), "Evidence should be dict"
    assert "strength_applied" in encoding_evidence, "Missing strength_applied"
    
    # 测试恢复。
    latent_recovered = channel_lf.reconstruct_from_lf_coeffs(
        encoded_coeffs, basis, latent.shape
    )
    assert latent_recovered.shape == latent.shape, "Recovered shape mismatch"
    
    # 测试分数提取。
    score = channel_lf.extract_lf_score(latent_recovered, basis, expected_pattern_seed=42, cfg=cfg)
    assert isinstance(score, float), "Score should be float"
    assert score >= 0, "Score should be non-negative"
    
    print("✓ test_lf_projection_and_encoding passed")


def test_hf_projection_and_constraint():
    """
    测试 HF 子空间投影与约束。
    """
    from main.watermarking.content_chain import channel_hf
    
    # 创建模拟的 latent 和 HF basis。
    latent_dim = 64
    latent = np.random.randn(latent_dim).astype(np.float32)
    
    hf_rank = 8
    projection_matrix = np.random.randn(latent_dim, hf_rank).astype(np.float32)
    projection_matrix, _ = np.linalg.qr(projection_matrix)
    
    basis = {"hf_projection_matrix": projection_matrix}
    cfg = {"hf_enabled": True, "hf_threshold_percentile": 75.0}
    
    # 测试投影。
    coeffs = channel_hf.compute_hf_basis_projection(latent, basis)
    assert coeffs.shape == (hf_rank,), f"Expected shape {(hf_rank,)}, got {coeffs.shape}"
    
    # 测试约束。
    constrained_coeffs, constraint_evidence = channel_hf.apply_hf_truncation_constraint(
        coeffs, cfg
    )
    assert constrained_coeffs.shape == coeffs.shape, "Constrained shape mismatch"
    assert isinstance(constraint_evidence, dict), "Evidence should be dict"
    assert "threshold_percentile_applied" in constraint_evidence, "Missing threshold"
    
    # 测试恢复。
    latent_recovered = channel_hf.reconstruct_from_hf_coeffs(
        constrained_coeffs, basis, latent.shape
    )
    assert latent_recovered.shape == latent.shape, "Recovered shape mismatch"
    
    # 测试分数提取。
    score = channel_hf.extract_hf_score(latent_recovered, basis, cfg=cfg)
    assert isinstance(score, float), "Score should be float"
    assert score >= 0, "Score should be non-negative"
    
    print("✓ test_hf_projection_and_constraint passed")


def test_latent_modifier():
    """
    测试统一 latent 修改器。
    """
    from main.watermarking.content_chain.latent_modifier import LatentModifier
    
    # 创建修改器。
    modifier = LatentModifier(
        impl_id="unified_latent_modifier_v1",
        impl_version="v1"
    )
    
    # 创建模拟 latent 和 plan。
    latent_dim = 64
    latent = np.random.randn(latent_dim).astype(np.float32)
    
    lf_rank = 8
    hf_rank = 8
    
    lf_basis_matrix = np.random.randn(latent_dim, lf_rank).astype(np.float32)
    lf_basis_matrix, _ = np.linalg.qr(lf_basis_matrix)
    
    hf_basis_matrix = np.random.randn(latent_dim, hf_rank).astype(np.float32)
    hf_basis_matrix, _ = np.linalg.qr(hf_basis_matrix)
    
    plan = {
        "lf_basis": {"projection_matrix": lf_basis_matrix},
        "hf_basis": {"hf_projection_matrix": hf_basis_matrix}
    }
    
    cfg = {
        "lf_enabled": True,
        "hf_enabled": True,
        "lf_strength": 1.5,
        "hf_threshold_percentile": 75.0,
        "watermark_seed": 42
    }
    
    # 运行修改。
    latent_modified, step_evidence = modifier.apply_latent_update(
        latents=latent,
        plan=plan,
        cfg=cfg,
        step_index=0,
        key=None
    )
    
    assert latent_modified.shape == latent.shape, "Modified shape mismatch"
    assert isinstance(step_evidence, dict), "Step evidence should be dict"
    assert "lf_evidence" in step_evidence, "Missing lf_evidence"
    assert "hf_evidence" in step_evidence, "Missing hf_evidence"
    
    lf_ev = step_evidence["lf_evidence"]
    hf_ev = step_evidence["hf_evidence"]
    assert lf_ev.get("status") == "ok", f"LF status should be ok, got {lf_ev.get('status')}"
    assert hf_ev.get("status") == "ok", f"HF status should be ok, got {hf_ev.get('status')}"
    
    print("✓ test_latent_modifier passed")


def test_lf_evidence_aggregation():
    """
    测试 LF 证据聚合与摘要生成。
    """
    from main.watermarking.content_chain import channel_lf
    
    latent_before = np.random.randn(64).astype(np.float32)
    latent_after = latent_before + np.random.randn(64).astype(np.float32) * 0.1
    
    trace_components = [
        {"step_index": 0, "lf_metrics": {"norm_delta": 0.1}},
        {"step_index": 1, "lf_metrics": {"norm_delta": 0.05}}
    ]
    
    cfg = {"lf_enabled": True, "lf_strength": 1.5}
    
    evidence = channel_lf.build_lf_embed_evidence(
        latents_before=latent_before,
        latents_after=latent_after,
        trace_components=trace_components,
        encoding_evidence={"pattern_seed": 42},
        cfg=cfg,
        plan_digest="abc123"
    )
    
    assert evidence["status"] == "ok", f"Status should be ok, got {evidence['status']}"
    assert "lf_trace_digest" in evidence, "Missing lf_trace_digest"
    assert "lf_metrics" in evidence, "Missing lf_metrics"
    assert len(evidence["lf_trace_digest"]) == 64, "lf_trace_digest should be 64-char hex"
    
    print("✓ test_lf_evidence_aggregation passed")


def test_plan_digest_consistency_validation():
    """
    测试 plan_digest 一致性校验。
    """
    from main.watermarking.content_chain.detector_scoring import validate_plan_digest_consistency
    
    # 测试一致。
    is_consistent, reason = validate_plan_digest_consistency(
        embed_time_plan_digest="abc123",
        detect_time_plan_digest="abc123"
    )
    assert is_consistent, f"Should be consistent, got {reason}"
    
    # 测试不一致。
    is_consistent, reason = validate_plan_digest_consistency(
        embed_time_plan_digest="abc123",
        detect_time_plan_digest="xyz789"
    )
    assert not is_consistent, "Should be mismatch"
    assert "mismatch" in reason, f"Reason should mention mismatch, got {reason}"
    
    # 测试 embed 缺失。
    is_consistent, reason = validate_plan_digest_consistency(
        embed_time_plan_digest=None,
        detect_time_plan_digest="abc123"
    )
    assert not is_consistent, "Embed absent should be mismatch"
    assert "absent" in reason, f"Reason should mention absent, got {reason}"
    
    print("✓ test_plan_digest_consistency_validation passed")


def test_content_score_computation():
    """
    测试 content_score 组合规则。
    """
    from main.watermarking.content_chain.detector_scoring import compute_content_score
    
    # LF ok, HF ok → 组合分数。
    content_score, rule = compute_content_score(
        lf_score=0.8,
        hf_score=0.5,
        lf_status="ok",
        hf_status="ok"
    )
    assert content_score is not None, "Score should not be None"
    assert content_score >= 0.8, "Score should be LF-primary"
    
    # LF ok, HF absent → LF only。
    content_score, rule = compute_content_score(
        lf_score=0.8,
        hf_score=None,
        lf_status="ok",
        hf_status="absent"
    )
    assert content_score == 0.8, f"Score should be 0.8, got {content_score}"
    
    # LF mismatch → score=None。
    content_score, rule = compute_content_score(
        lf_score=0.8,
        hf_score=0.5,
        lf_status="mismatch",
        hf_status="ok"
    )
    assert content_score is None, "Score should be None on LF mismatch"
    
    # LF failed → score=None。
    content_score, rule = compute_content_score(
        lf_score=None,
        hf_score=0.5,
        lf_status="failed",
        hf_status="ok"
    )
    assert content_score is None, "Score should be None on LF failed"
    
    print("✓ test_content_score_computation passed")


def run_all_integration_tests():
    """
    运行所有集成测试。
    """
    tests = [
        test_lf_projection_and_encoding,
        test_hf_projection_and_constraint,
        test_latent_modifier,
        test_lf_evidence_aggregation,
        test_plan_digest_consistency_validation,
        test_content_score_computation
    ]
    
    print("\n" + "="*60)
    print("Running Real Tensor-Level Watermarking Integration Tests")
    print("="*60 + "\n")
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
