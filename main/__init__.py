"""论文研究项目核心包。"""

from .content_chain.embedder import (
    ContentEmbeddingResult,
    ContentEmbedderError,
    ContentMaterializationObservation,
    ContentMaterializationResult,
    ContentMaterializer,
    content_actual_budget_accepts,
    content_materialization_replay_identity,
    content_embedder,
    reconcile_content_materialization_budget,
    scale_content_delta_binary32,
)
from .geometry_chain import QkLayerObservation
from .joint_decision import (
    ConditionalRecoveryError,
    ConditionalRecoveryResult,
    ContentDetectorBinding,
    JointDecisionThresholds,
    conditional_recovery_decision,
    validate_conditional_recovery_result,
)
from .shared import derive_public_noise_stream

__all__ = [
    "ConditionalRecoveryError",
    "ConditionalRecoveryResult",
    "ContentEmbeddingResult",
    "ContentEmbedderError",
    "ContentMaterializationObservation",
    "ContentMaterializationResult",
    "ContentMaterializer",
    "ContentDetectorBinding",
    "JointDecisionThresholds",
    "QkLayerObservation",
    "conditional_recovery_decision",
    "content_actual_budget_accepts",
    "content_materialization_replay_identity",
    "content_embedder",
    "derive_public_noise_stream",
    "reconcile_content_materialization_budget",
    "scale_content_delta_binary32",
    "validate_conditional_recovery_result",
]
