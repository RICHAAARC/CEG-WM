"""
功能：当启用 paper_faithfulness 时，pipeline_fingerprint 必须为 ok（不允许 absent）

File purpose: Test that pipeline fingerprint is ok when paper faithfulness is enabled.
Module type: General module
"""

import pytest
from unittest.mock import MagicMock

from main.watermarking.paper_faithfulness import alignment_evaluator


class MockPipeline:
    """Mock SD3 pipeline with minimal structure for testing."""
    def __init__(self):
        self.transformer = MagicMock()
        self.transformer.config = MagicMock()
        self.transformer.config.num_blocks = 24
        self.transformer.config.attention_head_dim = 64
        self.transformer.config.num_attention_heads = 16
        self.scheduler = MagicMock()
        self.vae = MagicMock()


def test_pipeline_fingerprint_must_not_be_absent_when_paper_faithfulness_enabled():
    """
    功能：当 enable_paper_faithfulness=true 时，
          pipeline_fingerprint 状态为 absent 必须导致检查返回 FAIL（不是 NA）。

    Test that pipeline fingerprint with status=absent returns FAIL when paper_faithfulness enabled.
    """
    pipeline_fingerprint = {
        "status": "absent",
        "reason": "pipeline_obj_is_none",
        "transformer_num_blocks": "<absent>",
        "scheduler_class_name": "<absent>",
        "vae_latent_channels": "<absent>"
    }

    # 调用检查函数，enable_paper_faithfulness=true
    check_result = alignment_evaluator._check_pipeline_fingerprint_presence(
        pipeline_fingerprint,
        enable_paper_faithfulness=True  # 关键：启用 paper_faithfulness
    )

    # 断言：result 必须为 FAIL，不能是 NA
    assert check_result["result"] == "FAIL", \
        f"Expected FAIL when paper_faithfulness enabled and fingerprint is absent, " \
        f"but got {check_result['result']}"
    assert "paper_faithfulness enabled" in check_result["failure_message"]


def test_pipeline_fingerprint_can_be_na_when_paper_faithfulness_disabled():
    """
    功能：当 enable_paper_faithfulness=false 时，
          pipeline_fingerprint 状态为 absent 返回 NA 是合法的。

    Test that pipeline fingerprint with status=absent returns NA when paper_faithfulness disabled.
    """
    pipeline_fingerprint = {
        "status": "absent",
        "reason": "pipeline_obj_is_none",
        "transformer_num_blocks": "<absent>",
        "scheduler_class_name": "<absent>",
        "vae_latent_channels": "<absent>"
    }

    # 调用检查函数，enable_paper_faithfulness=false
    check_result = alignment_evaluator._check_pipeline_fingerprint_presence(
        pipeline_fingerprint,
        enable_paper_faithfulness=False  # 禁用 paper_faithfulness
    )

    # 断言：result 可以是 NA（传统行为）
    assert check_result["result"] == "NA", \
        f"Expected NA when paper_faithfulness disabled and fingerprint is absent, " \
        f"but got {check_result['result']}"


def test_pipeline_fingerprint_pass_when_ok():
    """
    功能：当 pipeline_fingerprint 状态为 ok 且有合法结构时，检查返回 PASS。

    Test that pipeline fingerprint with valid structure returns PASS.
    """
    pipeline_fingerprint = {
        "status": "ok",
        "transformer_num_blocks": 24,
        "scheduler_class_name": "FlowMatchEulerDiscreteScheduler",
        "vae_latent_channels": 16
    }

    check_result = alignment_evaluator._check_pipeline_fingerprint_presence(
        pipeline_fingerprint,
        enable_paper_faithfulness=True
    )

    assert check_result["result"] == "PASS", \
        f"Expected PASS for valid pipeline_fingerprint, " \
        f"but got {check_result['result']}: {check_result.get('failure_message', '')}"
