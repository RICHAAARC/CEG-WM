"""论文研究项目核心包。"""

from .content_chain.embedder import (
    ContentEmbeddingResult,
    ContentEmbedderError,
    content_embedder,
)
from .joint_decision import (
    ConditionalRecoveryError,
    ConditionalRecoveryResult,
    ContentDetectorBinding,
    JointDecisionThresholds,
    conditional_recovery_decision,
    validate_conditional_recovery_result,
)

__all__ = [
    "ConditionalRecoveryError",
    "ConditionalRecoveryResult",
    "ContentEmbeddingResult",
    "ContentEmbedderError",
    "ContentDetectorBinding",
    "JointDecisionThresholds",
    "conditional_recovery_decision",
    "content_embedder",
    "validate_conditional_recovery_result",
]
