"""
统一内容链提取器（Embed + Detect 双模式）

功能说明：
- 根据 detect.content.enabled 配置自动切换 embed/detect 模式。
- Embed 模式：提取语义掩码并返回 mask_digest等结构证据。
- Detect 模式：执行完整的 LF/HF 检测并返回 content_score。
- 严格遵循冻结语义：absent/failed/mismatch/ok 的触发条件冻结。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from main.core import digests

from .semantic_mask_provider import SemanticMaskProvider, SEMANTIC_MASK_PROVIDER_ID, SEMANTIC_MASK_PROVIDER_VERSION
from .content_detector import ContentDetector, CONTENT_DETECTOR_ID, CONTENT_DETECTOR_VERSION
from .interfaces import ContentEvidence


UNIFIED_CONTENT_EXTRACTOR_ID = "unified_content_extractor_v1"
UNIFIED_CONTENT_EXTRACTOR_VERSION = "v1"

UNIFIED_CONTENT_EXTRACTOR_V2_ID = "unified_content_extractor_v2"
UNIFIED_CONTENT_EXTRACTOR_V2_VERSION = "v2"


class UnifiedContentExtractor:
    """
    功能：统一内容链提取器（Embed + Detect 双模式）。

    Unified content extractor supporting both embed and detect modes.
    Automatically delegates to SemanticMaskProvider (embed) or ContentDetector (detect)
    based on detect.content.enabled configuration.

    Embed mode (detect.content.enabled=False):
      - Extracts semantic mask via SemanticMaskProvider.
      - Returns ContentEvidence with mask_digest, mask_stats (structural evidence, score=None).

    Detect mode (detect.content.enabled=True):
      - Executes LF/HF detection via ContentDetector.
      - Returns ContentEvidence with content_score (valid score when status=ok).

    Args:
        impl_id: Implementation identifier string.
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If constructor arguments are invalid.
    """

    def __init__(self, impl_id: str, impl_version: str, impl_digest: str) -> None:
        if not isinstance(impl_id, str) or not impl_id:
            raise ValueError("impl_id must be non-empty str")
        if not isinstance(impl_version, str) or not impl_version:
            raise ValueError("impl_version must be non-empty str")
        if not isinstance(impl_digest, str) or not impl_digest:
            raise ValueError("impl_digest must be non-empty str")

        self.impl_id = impl_id
        self.impl_version = impl_version
        self.impl_digest = impl_digest

        # 初始化子模块实例（延迟构造以保持确定性）
        mask_provider_digest = digests.canonical_sha256({
            "impl_id": SEMANTIC_MASK_PROVIDER_ID,
            "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
        })
        self._mask_provider = SemanticMaskProvider(
            SEMANTIC_MASK_PROVIDER_ID,
            SEMANTIC_MASK_PROVIDER_VERSION,
            mask_provider_digest
        )

        detector_digest = digests.canonical_sha256({
            "impl_id": CONTENT_DETECTOR_ID,
            "impl_version": CONTENT_DETECTOR_VERSION
        })
        self._detector = ContentDetector(
            CONTENT_DETECTOR_ID,
            CONTENT_DETECTOR_VERSION,
            detector_digest
        )

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        cfg_digest: Optional[str] = None
    ) -> ContentEvidence:
        """
        功能：提取内容链证据（embed 或 detect 模式自动选择）。

        Extract content evidence in embed or detect mode based on configuration.

        Mode selection:
          - If detect.content.enabled=False -> Embed mode (SemanticMaskProvider).
          - If detect.content.enabled=True -> Detect mode (ContentDetector).

        Args:
            cfg: Configuration mapping.
            inputs: Optional inputs mapping.
            cfg_digest: Optional canonical config digest.

        Returns:
            ContentEvidence instance.

        Raises:
            TypeError: If input types are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            raise TypeError("inputs must be dict or None")
        if cfg_digest is not None and not isinstance(cfg_digest, str):
            raise TypeError("cfg_digest must be str or None")

        # 判断模式
        detect_content_enabled = cfg.get("detect", {}).get("content", {}).get("enabled", False)
        if not isinstance(detect_content_enabled, bool):
            raise TypeError("detect.content.enabled must be bool")

        if detect_content_enabled:
            # Detect 模式：调用 ContentDetector
            return self._detector.extract(cfg, inputs=inputs, cfg_digest=cfg_digest)
        else:
            # Embed 模式：调用 SemanticMaskProvider
            return self._mask_provider.extract(cfg, inputs=inputs, cfg_digest=cfg_digest)


class UnifiedContentExtractorV2(UnifiedContentExtractor):
    """
    功能：统一内容链提取器 v2，与 v1 逻辑相同但绑定 v2 impl_id。

    Upgraded content extractor binding to unified_content_extractor_v2 impl_id.
    All logic is identical to UnifiedContentExtractor; only the impl identity differs.

    Args:
        impl_id: Implementation identifier string (must be unified_content_extractor_v2).
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If constructor arguments are invalid.
    """

    pass

