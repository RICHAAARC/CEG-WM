"""
融合规则注册表与基线实现

功能说明：
- 定义了一个融合规则注册表，用于管理不同的融合规则实现。
- 提供了一个基线实现，用于在没有具体融合规则时返回确定性的决策结果。
- 实现了输入验证和错误处理，确保接口的健壮性。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests
from main.watermarking.fusion import neyman_pearson
from main.watermarking.fusion.decision import NeumanPearsonFusionRule
from main.watermarking.fusion.interfaces import FusionDecision

from .registry_base import FactoryType, RegistryBase
from .capabilities import ImplCapabilities


FUSION_BASELINE_IDENTITY_ID = "fusion_baseline_identity_v1"


class FusionBaselineIdentity:
    """
    功能：融合规则基线实现。

    Baseline fusion rule that returns deterministic decision values.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If any input is invalid.
    """

    def __init__(self, impl_id: str, impl_version: str, impl_digest: str) -> None:
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 输入不合法，必须 fail-fast。
            raise ValueError("impl_id must be non-empty str")
        if not isinstance(impl_version, str) or not impl_version:
            # impl_version 输入不合法，必须 fail-fast。
            raise ValueError("impl_version must be non-empty str")
        if not isinstance(impl_digest, str) or not impl_digest:
            # impl_digest 输入不合法，必须 fail-fast。
            raise ValueError("impl_digest must be non-empty str")
        self.impl_id = impl_id
        self.impl_version = impl_version
        self.impl_digest = impl_digest

    def fuse(
        self,
        cfg: Dict[str, Any],
        content_evidence: Dict[str, Any],
        geometry_evidence: Dict[str, Any]
    ) -> FusionDecision:
        """
        功能：输出基线融合决策。

        Return deterministic fusion decision for baseline registry rule.

        Args:
            cfg: Config mapping.
            content_evidence: Content evidence mapping.
            geometry_evidence: Geometry evidence mapping.

        Returns:
            FusionDecision instance.

        Raises:
            TypeError: If inputs are not dict.
            ValueError: If evidence fields are invalid.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if not isinstance(content_evidence, dict):
            # content_evidence 类型不合法，必须 fail-fast。
            raise TypeError("content_evidence must be dict")
        if not isinstance(geometry_evidence, dict):
            # geometry_evidence 类型不合法，必须 fail-fast。
            raise TypeError("geometry_evidence must be dict")

        thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
        thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)

        content_status = _extract_status_field(
            content_evidence,
            "content_evidence",
            "absent"
        )
        geometry_status = _extract_status_field(
            geometry_evidence,
            "geometry_evidence",
            "absent"
        )
        content_score = _extract_optional_score(content_evidence, "content_signal")
        geometry_score = _extract_optional_score(geometry_evidence, "geometry_signal")

        decision_status = "decided" if content_status != "absent" else "abstain"
        is_watermarked = False if decision_status == "decided" else None

        evidence_summary = {
            "content_score": content_score,
            "geometry_score": geometry_score,
            "content_status": content_status,
            "geometry_status": geometry_status,
            "fusion_rule_id": self.impl_id
        }
        audit = {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "decision_status": decision_status
        }
        return FusionDecision(
            is_watermarked=is_watermarked,
            decision_status=decision_status,
            thresholds_digest=thresholds_digest,
            evidence_summary=evidence_summary,
            audit=audit
        )


def _extract_status_field(payload: Dict[str, Any], field_name: str, fallback: str) -> str:
    """
    功能：提取证据状态字段。

    Extract evidence status field with fallback behavior.

    Args:
        payload: Evidence mapping.
        field_name: Field name to read.
        fallback: Fallback status string.

    Returns:
        Status string.

    Raises:
        TypeError: If payload is invalid.
        ValueError: If status field is invalid.
    """
    if not isinstance(payload, dict):
        # payload 类型不合法，必须 fail-fast。
        raise TypeError("payload must be dict")
    if not isinstance(field_name, str) or not field_name:
        # field_name 类型不合法，必须 fail-fast。
        raise TypeError("field_name must be non-empty str")
    if not isinstance(fallback, str) or not fallback:
        # fallback 类型不合法，必须 fail-fast。
        raise TypeError("fallback must be non-empty str")

    value = payload.get(field_name, fallback)
    if not isinstance(value, str) or not value:
        # 状态字段不合法，必须 fail-fast。
        raise ValueError(f"{field_name} must be non-empty str")
    return value


def _extract_optional_score(payload: Dict[str, Any], field_name: str) -> float | None:
    """
    功能：提取可选数值分数。

    Extract an optional numeric score from evidence mapping.

    Args:
        payload: Evidence mapping.
        field_name: Field name to read.

    Returns:
        Optional numeric score.

    Raises:
        TypeError: If payload is invalid.
        ValueError: If score type is invalid.
    """
    if not isinstance(payload, dict):
        # payload 类型不合法，必须 fail-fast。
        raise TypeError("payload must be dict")
    if not isinstance(field_name, str) or not field_name:
        # field_name 类型不合法，必须 fail-fast。
        raise TypeError("field_name must be non-empty str")
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        # score 类型不合法，必须 fail-fast。
        raise ValueError(f"{field_name} must be number or None")
    return float(value)


_FUSION_REGISTRY = RegistryBase("fusion_rule")


def _derive_impl_digest(impl_id: str, impl_version: str) -> str:
    """
    功能：计算 impl_digest。

    Compute impl digest from impl_id and impl_version.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version string.

    Returns:
        Canonical digest string.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(impl_id, str) or not impl_id:
        # impl_id 输入不合法，必须 fail-fast。
        raise ValueError("impl_id must be non-empty str")
    if not isinstance(impl_version, str) or not impl_version:
        # impl_version 输入不合法，必须 fail-fast。
        raise ValueError("impl_version must be non-empty str")
    return digests.canonical_sha256({
        "impl_id": impl_id,
        "impl_version": impl_version
    })


def _build_fusion_baseline_identity(cfg: Dict[str, Any]) -> FusionBaselineIdentity:
    """
    功能：构造融合规则占位实现。

    Build baseline fusion rule.

    Args:
        cfg: Config mapping.

    Returns:
        FusionBaselineIdentity instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(FUSION_BASELINE_IDENTITY_ID, impl_version)
    return FusionBaselineIdentity(FUSION_BASELINE_IDENTITY_ID, impl_version, impl_digest)


_FUSION_REGISTRY.register_factory(
    FUSION_BASELINE_IDENTITY_ID,
    _build_fusion_baseline_identity,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)


def _build_fusion_neyman_pearson(cfg: Dict[str, Any]) -> NeumanPearsonFusionRule:
    """
    功能：构造融合规则 S-10 真实实现。

    Build Neyman-Pearson fusion rule with NP primary + geometry auxiliary.

    Args:
        cfg: Config mapping.

    Returns:
        NeumanPearsonFusionRule instance.

    Raises:
        TypeError: If cfg is not dict.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_id = "fusion_neyman_pearson_v1"
    impl_digest = _derive_impl_digest(impl_id, impl_version)
    return NeumanPearsonFusionRule(impl_id, impl_version, impl_digest)


_FUSION_REGISTRY.register_factory(
    "fusion_neyman_pearson_v1",
    _build_fusion_neyman_pearson,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3-medium", "stabilityai/stable-diffusion-3-large"]
    )
)

# 静态注册完成后立即冻结，禁止运行期修改。
_FUSION_REGISTRY.seal()


def resolve_fusion_rule(impl_id: str) -> FactoryType:
    """
    功能：解析融合规则 impl_id。

    Resolve fusion rule factory.

    Args:
        impl_id: Implementation identifier.

    Returns:
        Factory callable.

    Raises:
        ValueError: If impl_id is invalid or unknown.
    """
    return _FUSION_REGISTRY.resolve_factory(impl_id)


def list_fusion_impl_ids() -> list[str]:
    """
    功能：列出融合规则 impl_id。

    List fusion rule impl_id values.

    Args:
        None.

    Returns:
        List of impl_id values.
    """
    return _FUSION_REGISTRY.list_impl_ids()
