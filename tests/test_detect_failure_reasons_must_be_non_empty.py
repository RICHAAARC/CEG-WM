"""
File purpose: 测试 detect 侧 failure reasons 必须非空（门禁约束）。
Module type: Core innovation module

Test that detect-side paper_faithfulness consistency validation enforces
non-empty reasons for any non-ok status (absent/mismatch/fail).
"""

import pytest
from main.watermarking.detect.orchestrator import _evaluate_paper_faithfulness_consistency


def test_paper_faithfulness_consistency_absent_reasons_non_empty():
    """
    功能：测试 input_record 缺失时 absent_reasons 必须非空。

    Test that absent_reasons is non-empty when input_record is None.

    Args:
        None.

    Returns:
        None.
    """
    status, absent_reasons, mismatch_reasons, fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=None
    )
    
    # 断言：status 必须为 absent
    assert status == "absent", f"Expected status='absent', got '{status}'"
    
    # 断言：absent_reasons 必须非空（冻结约束）
    assert isinstance(absent_reasons, list), "absent_reasons must be list"
    assert len(absent_reasons) > 0, "absent_reasons must be non-empty when status='absent'"
    
    # 断言：mismatch_reasons 和 fail_reasons 必须为空列表
    assert mismatch_reasons == [], f"Expected empty mismatch_reasons, got {mismatch_reasons}"
    assert fail_reasons == [], f"Expected empty fail_reasons, got {fail_reasons}"


def test_paper_faithfulness_consistency_mismatch_digest_absent_reasons_non_empty():
    """
    功能：测试 paper_spec_digest 标记为 <absent> 时 absent_reasons 必须非空。

    Test that absent_reasons is non-empty when paper_spec_digest is marked as <absent>.

    Args:
        None.

    Returns:
        None.
    """
    input_record = {
        "paper_faithfulness": {
            "spec_digest": "<absent>"
        },
        "content_evidence": {}
    }
    
    status, absent_reasons, mismatch_reasons, fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=input_record
    )
    
    # 断言：status 必须为 absent
    assert status == "absent", f"Expected status='absent', got '{status}'"
    
    # 断言：absent_reasons 必须非空（必须包含 paper_spec_digest_marked_absent）
    assert isinstance(absent_reasons, list), "absent_reasons must be list"
    assert len(absent_reasons) > 0, "absent_reasons must be non-empty when status='absent'"
    assert "paper_spec_digest_marked_absent" in absent_reasons, \
        f"Expected 'paper_spec_digest_marked_absent' in absent_reasons, got {absent_reasons}"


def test_paper_faithfulness_consistency_fail_digest_fail_reasons_non_empty():
    """
    功能：测试 pipeline_fingerprint_digest 标记为 <failed> 时 fail_reasons 必须非空。

    Test that fail_reasons is non-empty when pipeline_fingerprint_digest is marked as <failed>.

    Args:
        None.

    Returns:
        None.
    """
    input_record = {
        "paper_faithfulness": {
            "spec_digest": "abc123"
        },
        "content_evidence": {
            "pipeline_fingerprint_digest": "<failed>",
            "injection_site_digest": "def456",
            "alignment_digest": "ghi789"
        }
    }
    
    status, absent_reasons, mismatch_reasons, fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=input_record
    )
    
    # 断言：status 必须为 fail
    assert status == "fail", f"Expected status='fail', got '{status}'"
    
    # 断言：fail_reasons 必须非空（必须包含 pipeline_fingerprint_digest_marked_failed）
    assert isinstance(fail_reasons, list), "fail_reasons must be list"
    assert len(fail_reasons) > 0, "fail_reasons must be non-empty when status='fail'"
    assert "pipeline_fingerprint_digest_marked_failed" in fail_reasons, \
        f"Expected 'pipeline_fingerprint_digest_marked_failed' in fail_reasons, got {fail_reasons}"


def test_paper_faithfulness_consistency_ok_all_reasons_empty():
    """
    功能：测试所有字段有效时 status 为 ok 且所有 reasons 为空。

    Test that when all fields are valid, status='ok' and all reasons lists are empty.

    Args:
        None.

    Returns:
        None.
    """
    input_record = {
        "paper_faithfulness": {
            "spec_digest": "abc123def456"
        },
        "content_evidence": {
            "pipeline_fingerprint_digest": "fedcba987654",
            "injection_site_digest": "123456abcdef",
            "alignment_digest": "654321fedcba"
        }
    }
    
    status, absent_reasons, mismatch_reasons, fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=input_record
    )
    
    # 断言：status 必须为 ok
    assert status == "ok", f"Expected status='ok', got '{status}'"
    
    # 断言：所有 reasons 必须为空列表
    assert absent_reasons == [], f"Expected empty absent_reasons, got {absent_reasons}"
    assert mismatch_reasons == [], f"Expected empty mismatch_reasons, got {mismatch_reasons}"
    assert fail_reasons == [], f"Expected empty fail_reasons, got {fail_reasons}"


def test_paper_faithfulness_consistency_missing_content_evidence_absent_reasons_non_empty():
    """
    功能：测试 content_evidence 缺失时 absent_reasons 必须非空。

    Test that absent_reasons is non-empty when content_evidence is missing.

    Args:
        None.

    Returns:
        None.
    """
    input_record = {
        "paper_faithfulness": {
            "spec_digest": "abc123"
        }
        # content_evidence 缺失
    }
    
    status, absent_reasons, mismatch_reasons, fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=input_record
    )
    
    # 断言：status 必须为 absent
    assert status == "absent", f"Expected status='absent', got '{status}'"
    
    # 断言：absent_reasons 必须非空（必须包含 content_evidence_absent）
    assert isinstance(absent_reasons, list), "absent_reasons must be list"
    assert len(absent_reasons) > 0, "absent_reasons must be non-empty when status='absent'"
    assert "content_evidence_absent" in absent_reasons, \
        f"Expected 'content_evidence_absent' in absent_reasons, got {absent_reasons}"

