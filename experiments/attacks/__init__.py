"""Deterministic, method-independent experiment attacks."""

from .geometric import (
    AttackArtifact,
    AttackRegistry,
    GeometricAttackError,
    GeometricAttackResult,
    GeometricAttackSpec,
    apply_geometric_attack,
    load_attack_registry,
    validate_attack_artifact,
    validate_attack_registry,
    validate_geometric_attack_spec,
)

__all__ = [
    "AttackArtifact",
    "AttackRegistry",
    "GeometricAttackError",
    "GeometricAttackResult",
    "GeometricAttackSpec",
    "apply_geometric_attack",
    "load_attack_registry",
    "validate_attack_artifact",
    "validate_attack_registry",
    "validate_geometric_attack_spec",
    "CONTRASTIVE_LF_ATTACK_IDS",
    "ContrastiveLfAttackError",
    "ContrastiveLfAttackResult",
    "apply_contrastive_lf_attack",
    "validate_jpeg_capability",
]
from .contrastive_lf_branch_attribution import (
    ATTACK_IDS as CONTRASTIVE_LF_ATTACK_IDS,
    ContrastiveLfAttackError,
    ContrastiveLfAttackResult,
    apply_contrastive_lf_attack,
    validate_jpeg_capability,
)
