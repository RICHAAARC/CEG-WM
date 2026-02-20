"""
File purpose: 测试 detect 侧 paper_faithfulness 在同一 run_root 下一致性为 ok。
Module type: Core innovation module

Test that detect-side paper_faithfulness consistency is 'ok' when using
the same run_root and valid embed record.
"""

import pytest
from main.watermarking.detect.orchestrator import _evaluate_paper_faithfulness_consistency


def test_detect_paper_faithfulness_consistency_ok_on_same_run_root():
    """
    功能：测试在同一 run_root 下 embed+detect 的 consistency 为 ok。

    Test paper faithfulness consistency is 'ok' when embed and detect
    share the same run_root with valid paper_spec and evidence digests.

    Args:
        None.

    Returns:
        None.
    """
    # 构造最小 fixture：模拟一个完整的 embed_record。
    # 所有必达字段都存在且有效（非 <absent> 或 <failed>）。
    embed_record = {
        "paper_faithfulness": {
            "spec_version": "v1.0",
            "spec_digest": "abc123def456789012345678901234567890123456789012345678901234"
        },
        "content_evidence": {
            "pipeline_fingerprint": {
                "transformer_num_blocks": 24,
                "scheduler_class_name": "FlowMatchEulerDiscreteScheduler",
                "vae_latent_channels": 16
            },
            "pipeline_fingerprint_digest": "fedcba98765432109876543210fedcba98765432109876543210fedcba987654",
            "injection_site_spec": {
                "hook_type": "callback_on_step_end",
                "target_module_name": "StableDiffusion3Pipeline",
                "target_tensor_name": "latents"
            },
            "injection_site_digest": "123456abcdef7890abcdef1234567890abcdef1234567890abcdef123456",
            "alignment_report": {
                "overall_status": "PASS",
                "total_checks": 4,
                "pass_count": 4,
                "fail_count": 0,
                "na_count": 0
            },
            "alignment_digest": "654321fedcba098765432109876543210fedcba098765432109876543210fed"
        }
    }
    
    # 调用 _evaluate_paper_faithfulness_consistency
    status, absent_reasons, mismatch_reasons, fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=embed_record
    )
    
    # 断言：status 必须为 ok
    assert status == "ok", f"Expected status='ok', got '{status}'"
    
    # 断言：所有 reasons 列表必须为空
    assert absent_reasons == [], f"Expected empty absent_reasons, got {absent_reasons}"
    assert mismatch_reasons == [], f"Expected empty mismatch_reasons, got {mismatch_reasons}"
    assert fail_reasons == [], f"Expected empty fail_reasons, got {fail_reasons}"


def test_detect_paper_faithfulness_consistency_ok_with_trajectory_digest():
    """
    功能：测试包含 trajectory_digest 的完整对齐证据链一致性为 ok。

    Test that consistency is 'ok' when trajectory_digest is also present
    and valid (not <absent> or <failed>).

    Args:
        None.

    Returns:
        None.
    """
    embed_record = {
        "paper_faithfulness": {
            "spec_version": "v1.0",
            "spec_digest": "abc123def456789012345678901234567890123456789012345678901234"
        },
        "content_evidence": {
            "pipeline_fingerprint_digest": "fedcba98765432109876543210fedcba98765432109876543210fedcba987654",
            "injection_site_digest": "123456abcdef7890abcdef1234567890abcdef1234567890abcdef123456",
            "alignment_digest": "654321fedcba098765432109876543210fedcba098765432109876543210fed",
            "trajectory_evidence": {
                "status": "ok",
                "trajectory_spec_digest": "aabbccdd11223344556677889900aabbccdd11223344556677889900aabb",
                "trajectory_digest": "ddeeff00998877665544332211ffddeeff00998877665544332211ffdd"
            }
        }
    }
    
    status, absent_reasons, mismatch_reasons, fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=embed_record
    )
    
    # 断言：status 必须为 ok（trajectory_digest 不影响一致性判断，仅在 alignment_report 中评估）
    assert status == "ok", f"Expected status='ok', got '{status}'"
    assert absent_reasons == [], f"Expected empty absent_reasons, got {absent_reasons}"
    assert mismatch_reasons == [], f"Expected empty mismatch_reasons, got {mismatch_reasons}"
    assert fail_reasons == [], f"Expected empty fail_reasons, got {fail_reasons}"

