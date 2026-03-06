"""
低频水印编码器

功能说明：
- 基于 PRC-Watermark 的核心机制实现 LF 子空间水印编码。
- 实现伪高斯采样（Pseudogaussian sampling）：codeword * |randn()|。
- 实现 erf-based 后验概率恢复用于检测。
- 生成可复算的 lf_trace_digest，绑定配置、计划与编码参数。
- 严格区分 absent（未启用）、failed（异常）、mismatch（参数不一致）语义。

Module type: Core innovation module
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from main.core import digests

from .interfaces import ContentEvidence
from .ldpc_codec import build_ldpc_spec, decode_soft_llr, encode_message_bits


def erf(x: float) -> float:
    """
    功能：误差函数（Error Function）近似实现。

    Approximation of error function erf(x) for posterior recovery.
    Uses standard approximation with max error < 1.5e-7.

    Args:
        x: Input value.

    Returns:
        erf(x) approximation.

    Raises:
        None.
    """
    # 标准 erf 近似公式（Abramowitz and Stegun）。
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y


def sample_pseudogaussian(codeword: list, seed: int) -> list:
    """
    功能：伪高斯采样（Pseudogaussian sampling）。

    Implements pseudogaussian sampling: codeword * |randn()|.
    Uses deterministic pseudo-random generation for reproducibility.

    Args:
        codeword: Binary codeword (±1 values).
        seed: Random seed for reproducibility.

    Returns:
        Pseudogaussian samples (codeword weighted by absolute Gaussian).

    Raises:
        None.
    """
    import random
    random.seed(seed)

    pseudogaussian = []
    for c in codeword:
        # 使用 Box-Muller 变换生成伪高斯随机数。
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        # 取绝对值并乘以码字。
        pseudogaussian.append(c * abs(z))

    return pseudogaussian


def recover_posteriors_erf(latents: list, variance: float = 1.5) -> list:
    """
    功能：基于 erf 从 latents 恢复后验概率。

    Recover posterior probabilities from latents using error function.
    Implements: erf(z / sqrt(2 * variance * (1 + variance))).

    Args:
        latents: Latent features (flattened vector).
        variance: Variance parameter (default 1.5, matching PRC-Watermark).

    Returns:
        Posterior probabilities for each latent.

    Raises:
        None.
    """
    denominator = math.sqrt(2 * variance * (1 + variance))
    posteriors = []
    for z in latents:
        posterior = erf(z / denominator)
        posteriors.append(posterior)

    return posteriors


LOW_FREQ_CODER_ID = "low_freq_coder_v1"
LOW_FREQ_CODER_VERSION = "v1"
LOW_FREQ_CODER_TRACE_VERSION = "v1"

# Paper-faithful PRC LF coder impl
LF_CODER_PRC_ID = "lf_coder_prc_v1"
LF_CODER_PRC_VERSION = "v1"

# 允许的失败原因枚举。
ALLOWED_LF_CODER_FAILURE_REASONS = {
    "lf_coder_disabled",                  # enable_lf 配置为 false，低频编码未启用
    "lf_coder_no_plan",                   # 缺失 plan_digest，无法确定编码子空间
    "lf_coder_invalid_input",             # 输入缺失或形状不符
    "lf_coder_plan_mismatch",             # plan_digest 不一致，编码参数与计划脱离
    "lf_coder_encoding_failed",           # 编码计算异常
    "lf_coder_trace_digest_mismatch",     # lf_trace_digest 与预期不符
}


def _normalize_lf_ecc_mode(ecc_value: Any) -> Tuple[str, Optional[int]]:
    """
    功能：归一化 LF ECC 双语义输入。

    Normalize LF ECC dual semantics for compatibility period.

    Args:
        ecc_value: ECC value from config, supports int or "sparse_ldpc".

    Returns:
        Tuple of (ecc_mode, legacy_redundancy).

    Raises:
        ValueError: If ecc_value is invalid.
    """
    if isinstance(ecc_value, str):
        if ecc_value == "sparse_ldpc":
            return "sparse_ldpc", None
        raise ValueError(f"ecc string must be 'sparse_ldpc', got {ecc_value}")
    if isinstance(ecc_value, int) and 2 <= ecc_value <= 8:
        return "legacy_int", int(ecc_value)
    raise ValueError(f"ecc must be int in [2, 8] or 'sparse_ldpc', got {ecc_value}")


class LowFreqCoder:
    """
    功能：低频水印编码器，在 LF 子空间施加隐蔽编码。

    Low-frequency watermark coder implementing reproducible, auditable encoding
    with plan digest binding and failure semantics strictness.

    Implements the ContentExtractor protocol frozen in interfaces.py.
    Emits ContentEvidence with lf_score, lf_trace_digest, and strict failure
    semantics (absent, failed, mismatch) matching frozen enumeration.

    Encoding applies only in LF subspace; outputs coding trace and score.
    Failure to produce valid score (e.g., plan mismatch) must set score=None
    and status!=ok to prevent false watermark evidence.

    Args:
        impl_id: Implementation identifier string (frozen to low_freq_coder_v1).
        impl_version: Implementation version string (frozen to v1).
        impl_digest: Implementation digest computed from source code.

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

    def embed_apply(
        self,
        cfg: Dict[str, Any],
        latent_features: Any,
        plan_digest: str,
        cfg_digest: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        功能：PRC-lite embed 侧注入闭环，确定性嵌入水印到 LF 子空间。

        Apply deterministic PRC-lite watermark embedding to low-frequency subspace.
        Returns modified latent features and embedding trace for audit.

        必须满足：
        1. embed 侧必须是确定性的（种子派生绑定 plan_digest + LF 参数）
        2. 不允许写入大矩阵到 records；只能写摘要（注入前后统计摘要、seed 派生信息摘要）
        3. 输出包含可复算摘要（embedding_digest、n、Δ、strength、seed_digest）

        Args:
            cfg: Configuration dict with watermark.lf.* parameters.
            latent_features: Input latent features (flattened or nested list/array).
            plan_digest: Plan digest binding (must match cfg plan_digest).
            cfg_digest: Optional cfg canonical digest.

        Returns:
            Dict with keys:
            - "latent_features_embedded": Modified latent features with watermark.
            - "embedding_trace": Audit trace with embedding_digest, seed_digest, statistics.
            - "embedding_digest": Canonical SHA256 digest of embedding operation.

        Raises:
            ValueError: If inputs are invalid or plan_digest mismatch.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if plan_digest is None or not isinstance(plan_digest, str):
            raise ValueError("plan_digest must be non-empty str")

        # 验证 plan_digest 一致性。
        cfg_plan_digest = cfg.get("watermark", {}).get("plan_digest")
        if cfg_plan_digest != plan_digest:
            raise ValueError(
                f"plan_digest mismatch: cfg={cfg_plan_digest}, input={plan_digest}. "
                "Embed and detect must use same plan_digest."
            )

        lf_cfg = cfg.get("watermark", {}).get("lf", {})
        enabled = lf_cfg.get("enabled", False)
        if not enabled:
            # LF 未启用，返回原始 latent_features（无嵌入）。
            return {
                "latent_features_embedded": latent_features,
                "embedding_trace": None,
                "embedding_digest": None
            }

        # 读取 LF 参数（与 plan_digest_include_paths 对齐）。
        codebook_id = lf_cfg.get("codebook_id", "default")
        ecc = lf_cfg.get("ecc", 3)
        strength = lf_cfg.get("strength", 0.1)
        delta = lf_cfg.get("delta", 1.0)
        block_length = lf_cfg.get("block_length", 8)
        variance = lf_cfg.get("variance", 1.5)

        # 参数验证（与 extract() 保持一致）。
        ecc_mode, _legacy_redundancy = _normalize_lf_ecc_mode(ecc)
        if not isinstance(strength, (int, float)) or strength <= 0 or strength > 1:
            raise ValueError(f"strength must be float in (0, 1], got {strength}")
        if not isinstance(delta, (int, float)) or delta <= 0:
            raise ValueError(f"delta must be positive number, got {delta}")
        if not isinstance(block_length, int) or block_length < 1 or block_length > 64:
            raise ValueError(f"block_length must be int in [1, 64], got {block_length}")
        if not isinstance(variance, (int, float)) or variance <= 0:
            raise ValueError(f"variance must be positive number, got {variance}")

        # 扁平化 latent_features。
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, (list, tuple)):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result

        if isinstance(latent_features, (list, tuple)):
            lf_vector = flatten(latent_features)
        else:
            lf_vector = list(latent_features)

        # 派生确定性种子（绑定 plan_digest + 所有 LF 参数）。
        import hashlib
        param_seed_input = f"{plan_digest}:{codebook_id}:{ecc_mode}:{ecc}:{strength}:{delta}:{block_length}:{variance}"
        param_seed = hashlib.sha256(param_seed_input.encode()).hexdigest()
        seed_int = int(param_seed[:16], 16) % 100000
        seed_digest = hashlib.sha256(str(seed_int).encode()).hexdigest()[:16]

        # 生成 ±1 codeword（确定性，可复算）。
        codeword = []
        for i in range(block_length):
            bit = ((seed_int * (i + 1) + 17) % 2) * 2 - 1
            codeword.append(bit)

        # 提取前 n 个 LF 系数。
        n = min(block_length, len(lf_vector))
        if n == 0:
            raise ValueError(f"lf_vector length {len(lf_vector)} < block_length {block_length}")

        lf_coeffs = [float(lf_vector[i]) for i in range(n)]

        # PRC-lite 真实嵌入机制：伪高斯嵌入。
        # 对每个系数 coeff_i：
        #   z_i = codeword_i * |randn()| * strength * delta
        #   coeff_embedded_i = coeff_i + z_i
        embedded_coeffs = []
        import random
        random.seed(seed_int)
        for i in range(n):
            # 生成伪高斯随机数（Box-Muller 变换）。
            u1 = random.random()
            u2 = random.random()
            z_gaussian = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            # 伪高斯嵌入：codeword * |randn()| * strength * delta。
            z_embed = codeword[i] * abs(z_gaussian) * strength * delta
            coeff_embedded = lf_coeffs[i] + z_embed
            embedded_coeffs.append(coeff_embedded)

        # 计算嵌入前后统计摘要（不写大矩阵，只写摘要）。
        mean_before = sum(lf_coeffs) / n if n > 0 else 0.0
        mean_after = sum(embedded_coeffs) / n if n > 0 else 0.0
        energy_before = sum(c * c for c in lf_coeffs) / n if n > 0 else 0.0
        energy_after = sum(c * c for c in embedded_coeffs) / n if n > 0 else 0.0

        # 构造嵌入 trace（只保存摘要）。
        embedding_trace = {
            "codebook_id": codebook_id,
            "ecc": ecc,
            "ecc_mode": ecc_mode,
            "strength": strength,
            "delta": delta,
            "block_length": block_length,
            "variance": variance,
            "n_embedded": n,
            "seed_digest": seed_digest,  # 种子摘要（可复算性依据）
            "mean_before": float(mean_before),
            "mean_after": float(mean_after),
            "energy_before": float(energy_before),
            "energy_after": float(energy_after),
            "codeword_snapshot": codeword[:min(8, len(codeword))],  # 只保存前 8 位用于审计
            "detection_method": "erf_posterior_recovery",
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest
        }

        # 计算 embedding_digest（可复算摘要）。
        embedding_digest = digests.canonical_sha256(embedding_trace)

        # 将嵌入后的系数写回 lf_vector。
        for i in range(n):
            lf_vector[i] = embedded_coeffs[i]

        return {
            "latent_features_embedded": lf_vector,
            "embedding_trace": embedding_trace,
            "embedding_digest": embedding_digest
        }

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        cfg_digest: Optional[str] = None
    ) -> ContentEvidence:
        """
        功能：在 LF 子空间进行水印编码与检测评分。

        Perform low-frequency watermark coding and extract score.

        Three semantic modes:
        1. enable_lf=false: Returns absent status (abstinence, not error).
        2. enable_lf=true, plan available: Attempts encoding and score extraction.
        3. enable_lf=true, plan missing or mismatch: Returns status="mismatch" or "failed".

        When status="ok": score is non-None float, lf_score is populated.
        When status!="ok": score must be None, content_failure_reason populated.

        Args:
            cfg: Configuration dict with optional keys:
                - "enable_lf" (bool, default False): Whether to enable LF coding.
                - "lf_codebook_id" (str, optional): Codebook identifier.
                - "lf_redundancy" (int, default 3): Redundancy level (2-8).
                - "lf_power" (float, default 0.1): Encoding power level.
                - "watermark" (dict, optional): Sub-configuration containing plan_digest.
                  - "plan_digest" (str, optional): Expected plan digest binding.

            inputs: Optional input dict with keys:
                - "latent_features" (array-like, optional): Latent space features.
                - "latent_shape" (tuple, optional): Expected shape of latent.

            cfg_digest: Optional canonical SHA256 digest of cfg.
                       Enables mask_digest binding without full cfg dependency.

        Returns:
            ContentEvidence instance with frozen structure.
                - status: "ok" (valid score), "absent" (disabled), "failed" (error),
                  "mismatch" (plan inconsistency).
                - score: Non-None float when status="ok"; None otherwise.
                - lf_score: Optional low-frequency score (populated when ok).
                - lf_trace_digest: Audit digest of encoding trace.
                - plan_digest: Echoed from inputs for consistency audit.
                - audit: Dict with impl_identity, impl_version, impl_digest, trace_digest.
                - content_failure_reason: Enumeration string when status!="ok".

        Raises:
            TypeError: If cfg or inputs types are invalid.
            ValueError: If critical fields are malformed.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            # inputs 类型不合法，必须 fail-fast。
            raise TypeError("inputs must be dict or None")

        # (1) 解析启用状态。对齐 injection_scope_manifest.yaml 中 plan_digest_include_paths 的声明。
        lf_cfg = cfg.get("watermark", {}).get("lf", {})
        enabled = lf_cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            # enabled 类型不合法，必须 fail-fast。
            raise TypeError("watermark.lf.enabled must be bool")

        # 若禁用，返回 absent 语义（非错误）。
        if not enabled:
            trace_payload = _build_lf_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=False,
                plan_digest=None,
                encoding_trace=None,
                cfg_digest=cfg_digest
            )
            trace_digest = digests.canonical_sha256(trace_payload)

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest
            }

            return ContentEvidence(
                status="absent",
                score=None,
                audit=audit,
                lf_trace_digest=trace_digest,
                lf_score=None,
                plan_digest=None,
                content_failure_reason=None
            )

        # (2) 解析计划摘要与编码参数。
        plan_digest = cfg.get("watermark", {}).get("plan_digest")
        if plan_digest is None:
            # 无计划，无法确定编码子空间。
            trace_payload = _build_lf_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=None,
                encoding_trace=None,
                cfg_digest=cfg_digest
            )
            trace_digest = digests.canonical_sha256(trace_payload)

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest
            }

            return ContentEvidence(
                status="mismatch",
                score=None,
                audit=audit,
                lf_trace_digest=trace_digest,
                lf_score=None,
                plan_digest=None,
                content_failure_reason="lf_coder_no_plan"
            )

        # 若 inputs 中提供了 expected_plan_digest，必须验证一致性。
        # 若 plan_digest 不一致，必须返回 status="mismatch" 且 score=None（失败不得给分）。
        if inputs is None:
            inputs = {}

        expected_plan_digest = inputs.get("expected_plan_digest")
        if expected_plan_digest is not None and expected_plan_digest != plan_digest:
            # plan_digest 不一致，编码参数与计划脱离。
            trace_payload = _build_lf_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=plan_digest,
                encoding_trace=None,
                cfg_digest=cfg_digest
            )
            trace_digest = digests.canonical_sha256(trace_payload)

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest,
                "plan_digest_expected": expected_plan_digest,
                "plan_digest_actual": plan_digest
            }

            return ContentEvidence(
                status="mismatch",
                score=None,
                audit=audit,
                lf_trace_digest=trace_digest,
                lf_score=None,
                plan_digest=plan_digest,
                content_failure_reason="lf_coder_plan_mismatch"
            )

        # (3) 验证输入有效性。
        latent_features = inputs.get("latent_features")
        latent_shape = inputs.get("latent_shape")

        if latent_features is None or latent_shape is None:
            # 输入缺失。
            trace_payload = _build_lf_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=plan_digest,
                encoding_trace=None,
                cfg_digest=cfg_digest
            )
            trace_digest = digests.canonical_sha256(trace_payload)

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest
            }

            return ContentEvidence(
                status="failed",
                score=None,
                audit=audit,
                lf_trace_digest=trace_digest,
                lf_score=None,
                plan_digest=plan_digest,
                content_failure_reason="lf_coder_invalid_input"
            )

        # (4) 执行编码与评分（PRC-lite 真实 LF 编码实现）。
        try:
            # 读取 LF 参数（这些参数已在 plan_digest_include_paths 中声明）。
            codebook_id = lf_cfg.get("codebook_id", "default")
            ecc = lf_cfg.get("ecc", 3)  # 纠错编码强度（redundancy level）
            strength = lf_cfg.get("strength", 0.1)  # 编码强度（embedding power）
            delta = lf_cfg.get("delta", 1.0)  # 量化步长 Δ（quantization step）
            block_length = lf_cfg.get("block_length", 8)  # 码字块长度 n（codeword block size）
            variance = lf_cfg.get("variance", 1.5)  # 伪高斯方差（pseudogaussian variance）

            # 验证参数范围。
            ecc_mode, _legacy_redundancy = _normalize_lf_ecc_mode(ecc)
            if not isinstance(strength, (int, float)) or strength <= 0 or strength > 1:
                # strength 参数不合法，必须 fail-fast。
                raise ValueError(f"strength must be float in (0, 1], got {strength}")
            if not isinstance(delta, (int, float)) or delta <= 0:
                # delta 参数不合法，必须 fail-fast。
                raise ValueError(f"delta must be positive number, got {delta}")
            if not isinstance(block_length, int) or block_length < 1 or block_length > 64:
                # block_length 参数不合法，必须 fail-fast。
                raise ValueError(f"block_length must be int in [1, 64], got {block_length}")
            if not isinstance(variance, (int, float)) or variance <= 0:
                # variance 参数不合法，必须 fail-fast。
                raise ValueError(f"variance must be positive number, got {variance}")

            # 真实 LF 编码实现：PRC-lite 低频子空间 rounding 编码。
            # (1) 将 latent_features 扁平化为 1D 向量（处理嵌套列表）。
            lf_vector_raw = latent_features
            
            # 扁平化处理：支持嵌套列表、数组等格式。
            def flatten(lst):
                """递归扁平化嵌套列表。"""
                result = []
                for item in lst:
                    if isinstance(item, (list, tuple)):
                        result.extend(flatten(item))
                    else:
                        result.append(item)
                return result
            
            try:
                if isinstance(lf_vector_raw, (list, tuple)):
                    lf_vector = flatten(lf_vector_raw)
                else:
                    # 假设是数组或可迭代对象。
                    lf_vector = list(lf_vector_raw)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Failed to flatten latent_features: {e}")
            
            # (2) 使用 plan_digest + codebook_id 派生确定性伪随机种子。
            # 关键：种子必须绑定 plan_digest，确保同一计划下可复算。
            try:
                import hashlib
                # (2a) 派生种子：绑定所有影响编码的参数。
                # 设计：plan_digest 已包含 delta/block_length/strength/ecc（来自 plan_digest_include_paths），
                #       此处再绑定 codebook_id 确保码字唯一性。
                param_seed_input = f"{plan_digest}:{codebook_id}:{ecc_mode}:{ecc}:{strength}:{delta}:{block_length}"
                param_seed = hashlib.sha256(param_seed_input.encode()).hexdigest()
                seed_int = int(param_seed[:16], 16) % 100000
                
                # (2b) 生成 ±1 codeword（长度 n = block_length）。
                # 使用 seed_int 生成确定性伪随机 codeword。
                codeword = []
                for i in range(block_length):
                    # 伪随机生成 ±1（确定性，可复算）。
                    bit = ((seed_int * (i + 1) + 17) % 2) * 2 - 1  # 映射 {0,1} → {-1,+1}
                    codeword.append(bit)
                
                # (2c) 提取前 n 个低频系数执行量化 rounding。
                # 真实 PRC 编码机制：对每个系数 coeff_i，执行：
                #   q_i = round(coeff_i / Δ)
                #   coeff_i → q_i * Δ + sign(codeword_i) * strength * Δ
                # 检测时计算相关统计：
                #   lf_score = mean(sign(coeff_i - round(coeff_i/Δ)*Δ) * codeword_i)
                # 得分方向：越大越是水印（符合 NP 校准需要）。
                n = min(block_length, len(lf_vector) if isinstance(lf_vector, (list, tuple)) else 0)
                if n == 0:
                    # 向量长度不足，无法编码。
                    raise ValueError(f"lf_vector length {len(lf_vector) if isinstance(lf_vector, (list, tuple)) else 0} < block_length {block_length}")
                
                # 提取前 n 个系数。
                try:
                    lf_coeffs = [float(lf_vector[i]) for i in range(n)] if isinstance(lf_vector, (list, tuple)) else list(lf_vector)[:n]
                except (TypeError, IndexError, ValueError) as e:
                    raise ValueError(f"Failed to extract LF coefficients: {e}")
                
                # (2d) PRC-lite 真实检测机制。
                # 步骤 1：从 latents 恢复后验概率（使用 erf 函数）。
                # 步骤 2：计算后验概率与码字的乘积。
                # 步骤 3：基于乘积统计生成检测分数。
                
                # 恢复后验概率：posteriors = erf(latents / sqrt(2 * var * (1 + var)))。
                posteriors = recover_posteriors_erf(lf_coeffs[:n], variance)
                
                # 计算后验概率与码字的逐元素乘积。
                # PRC 检测统计：Π_i = posterior_i * codeword_i。
                posterior_product_sum = 0.0
                posterior_abs_sum = 0.0
                for i in range(n):
                    posterior_weighted = posteriors[i] * codeword[i]
                    posterior_product_sum += posterior_weighted
                    posterior_abs_sum += abs(posteriors[i])
                
                # (2e) 计算归一化 lf_score。
                # PRC-lite 检测分数设计：
                # - 基础分数：后验概率加权和 / n（范围 [-1, 1]）
                # - 归一化到 [0, 1]：(score + 1) / 2
                # - 调制因子：strength（编码强度）
                # - 置信度因子：posterior_abs_sum / n（高置信度后验的占比）
                
                if n > 0:
                    # 基础后验加权分数。
                    posterior_score_raw = posterior_product_sum / n  # ∈ [-1, +1]
                    # 映射到 [0, 1]。
                    posterior_score_normalized = (posterior_score_raw + 1.0) / 2.0
                    # 后验置信度（高置信度后验越多，分数越可靠）。
                    posterior_confidence = posterior_abs_sum / n  # ∈ [0, 1]
                    
                    # 最终检测分数：融合后验分数与置信度。
                    # 权重：后验分数 0.8，置信度 0.2。
                    lf_score_value = (posterior_score_normalized * 0.8 + posterior_confidence * 0.2) * strength
                    lf_score_value = min(1.0, max(0.0, lf_score_value))
                else:
                    lf_score_value = 0.0
                
                # 保存中间统计用于审计。
                lf_correlation = posterior_score_raw if n > 0 else 0.0
                latent_energy_normalized = posterior_confidence if n > 0 else 0.0
                
            except Exception as encoding_calc_error:
                # 编码计算异常回退。
                raise ValueError(f"LF encoding score calculation failed: {encoding_calc_error}")

            # 计算编码痕迹（规范化可复算）。
            # 痕迹中记录所有影响分数的因素，确保可重放。
            encoding_trace = {
                "codebook_id": codebook_id,
                "ecc": ecc,
                "strength": strength,
                "delta": delta,
                "block_length": block_length,
                "variance": variance,  # PRC 伪高斯方差（新增供应链审计字段）
                "detection_method": "erf_posterior_recovery",  # 基于 erf 的后验概率恢复检测方法
                "plan_digest": plan_digest,
                "cfg_digest": cfg_digest,
                "lf_score": lf_score_value,  # 审计：记录计算得到的分数
                "seed_int": seed_int,  # 审计：伪随机种子（可复算性）
                "n_coeffs_used": n,  # 审计：实际使用的系数数量
                "posterior_score_raw": lf_correlation,  # 审计：原始后验加权分数 ∈ [-1, +1]
                "posterior_confidence": latent_energy_normalized  # 审计：后验置信度 ∈ [0, 1]
            }

        except (ValueError, TypeError, KeyError) as e:
            # 编码异常，单一主因上报。
            failure_reason = "lf_coder_encoding_failed"

            trace_payload = _build_lf_trace_payload(
                cfg=cfg,
                impl_id=self.impl_id,
                impl_version=self.impl_version,
                impl_digest=self.impl_digest,
                enabled=True,
                plan_digest=plan_digest,
                encoding_trace=None,
                cfg_digest=cfg_digest
            )
            trace_digest = digests.canonical_sha256(trace_payload)

            audit = {
                "impl_identity": self.impl_id,
                "impl_version": self.impl_version,
                "impl_digest": self.impl_digest,
                "trace_digest": trace_digest
            }

            return ContentEvidence(
                status="failed",
                score=None,
                audit=audit,
                lf_trace_digest=trace_digest,
                lf_score=None,
                plan_digest=plan_digest,
                content_failure_reason=failure_reason
            )

        # (5) 构造成功路径审计字段。
        trace_payload = _build_lf_trace_payload(
            cfg=cfg,
            impl_id=self.impl_id,
            impl_version=self.impl_version,
            impl_digest=self.impl_digest,
            enabled=True,
            plan_digest=plan_digest,
            encoding_trace=encoding_trace,
            cfg_digest=cfg_digest
        )
        lf_trace_digest = digests.canonical_sha256(trace_payload)

        audit = {
            "impl_identity": self.impl_id,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": lf_trace_digest
        }

        # (6) 返回成功证据。
        return ContentEvidence(
            status="ok",
            score=lf_score_value,
            audit=audit,
            lf_trace_digest=lf_trace_digest,
            lf_score=lf_score_value,
            plan_digest=plan_digest,
            content_failure_reason=None
        )


def _build_lf_trace_payload(
    cfg: Dict[str, Any],
    impl_id: str,
    impl_version: str,
    impl_digest: str,
    enabled: bool,
    plan_digest: Optional[str],
    encoding_trace: Optional[Dict[str, Any]],
    cfg_digest: Optional[str] = None
) -> Dict[str, Any]:
    """
    功能：构造可复算的低频编码追踪有效负载。

    Build deterministic trace payload for LF coder digest computation.

    Args:
        cfg: Configuration mapping containing 'watermark.lf.*' parameters.
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.
        enabled: Whether LF coding is enabled.
        plan_digest: Optional plan digest binding.
        encoding_trace: Optional encoding parameters and trace.
        cfg_digest: Optional canonical SHA256 digest of cfg.

    Returns:
        JSON-like dict for canonical SHA256 computation.
        Keys are named to match plan_digest_include_paths (codebook_id, ecc, strength, delta, block_length, variance).

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(impl_id, str) or not impl_id:
        # impl_id 类型不合法，必须 fail-fast。
        raise TypeError("impl_id must be non-empty str")
    if not isinstance(enabled, bool):
        # enabled 类型不合法，必须 fail-fast。
        raise TypeError("enabled must be bool")
    if plan_digest is not None and not isinstance(plan_digest, str):
        # plan_digest 类型不合法，必须 fail-fast。
        raise TypeError("plan_digest must be str or None")
    if encoding_trace is not None and not isinstance(encoding_trace, dict):
        # encoding_trace 类型不合法，必须 fail-fast。
        raise TypeError("encoding_trace must be dict or None")
    if cfg_digest is not None and not isinstance(cfg_digest, str):
        # cfg_digest 类型不合法，必须 fail-fast。
        raise TypeError("cfg_digest must be str or None")

    # 统一键空间为 watermark.lf，与 plan_digest_include_paths 对齐。
    lf_cfg = cfg.get("watermark", {}).get("lf", {})

    # 当 enabled=true 时主 payload 必须包含全部 plan_digest_include_paths 字段。
    # 当 enabled=false 时，这些字段为 None，但键必须存在，确保四种返回路径使用同一键空间。
    payload = {
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "trace_version": LOW_FREQ_CODER_TRACE_VERSION,
        "enabled": enabled,
        "plan_digest": plan_digest,
        "cfg_digest": cfg_digest,  # 新增：cfg_digest 必须写入主 payload
        # 包含所有 plan_digest_include_paths 中的 watermark.lf.* 字段。
        "codebook_id": lf_cfg.get("codebook_id", "default") if enabled else None,
        "ecc": lf_cfg.get("ecc", 3) if enabled else None,
        "strength": lf_cfg.get("strength", 0.1) if enabled else None,
        "delta": lf_cfg.get("delta", 1.0) if enabled else None,
        "block_length": lf_cfg.get("block_length", 8) if enabled else None,
        "variance": lf_cfg.get("variance", 1.5) if enabled else None,  # 新增：variance 必须写入主 payload
    }

    if encoding_trace is not None:
        payload["encoding_trace"] = encoding_trace

    return payload


def encode_low_freq_dct(
    image_array: np.ndarray,
    band_spec: Dict[str, Any],
    key_material: str,
    params: Dict[str, Any]
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    功能：在 LF 子域执行块 DCT 的最小可行嵌入。

    Apply minimal viable LF embedding using block DCT coefficient modulation.

    Args:
        image_array: Input uint8 image array in HWC format.
        band_spec: Planner band specification mapping.
        key_material: Deterministic key material string.
        params: LF embedding parameters.

    Returns:
        Tuple of (watermarked_image_array, lf_trace_summary).

    Raises:
        TypeError: If input types are invalid.
        ValueError: If parameter values are invalid.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be np.ndarray")
    if image_array.ndim != 3:
        raise ValueError("image_array must be HWC array")
    if not isinstance(band_spec, dict):
        raise TypeError("band_spec must be dict")
    if not isinstance(key_material, str) or not key_material:
        raise TypeError("key_material must be non-empty str")
    if not isinstance(params, dict):
        raise TypeError("params must be dict")

    block_size = int(params.get("dct_block_size", 8))
    alpha = float(params.get("alpha", 1.5))
    redundancy = int(params.get("redundancy", 1))
    coeff_indices_raw = params.get("lf_coeff_indices", [(1, 1), (1, 2), (2, 1)])
    if block_size <= 1:
        raise ValueError("dct_block_size must be > 1")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if redundancy <= 0:
        raise ValueError("redundancy must be positive")
    if not isinstance(coeff_indices_raw, list) or len(coeff_indices_raw) == 0:
        raise ValueError("lf_coeff_indices must be non-empty list")

    coeff_indices = []
    for item in coeff_indices_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each lf_coeff_indices item must be 2-length sequence")
        ci = int(item[0])
        cj = int(item[1])
        if ci < 0 or cj < 0 or ci >= block_size or cj >= block_size:
            raise ValueError("lf_coeff_indices values out of block range")
        coeff_indices.append((ci, cj))

    height, width, channels = image_array.shape
    blocks_h = height // block_size
    blocks_w = width // block_size
    total_blocks = blocks_h * blocks_w
    if total_blocks <= 0:
        return image_array.copy(), {
            "lf_status": "absent",
            "lf_absent_reason": "image_too_small_for_lf_blocks",
        }

    lf_selector = band_spec.get("lf_selector_summary") if isinstance(band_spec.get("lf_selector_summary"), dict) else {}
    region_ratio = float(lf_selector.get("region_ratio", 1.0))
    region_ratio = max(0.0, min(1.0, region_ratio))

    import random
    rng_seed = int(digests.canonical_sha256({"key_material": key_material, "tag": "lf_dct"})[:16], 16)
    rng = random.Random(rng_seed)

    all_block_ids = list(range(total_blocks))
    rng.shuffle(all_block_ids)
    selected_count = max(1, int(round(total_blocks * region_ratio)))
    selected_count = min(total_blocks, selected_count * redundancy)
    selected_blocks = all_block_ids[:selected_count]

    work = image_array.astype(np.float32).copy()
    dct_matrix = _build_dct_matrix(block_size)
    idct_matrix = dct_matrix.T

    coeff_ops = 0
    bit_values = []
    for block_id in selected_blocks:
        block_row = block_id // blocks_w
        block_col = block_id % blocks_w
        r0 = block_row * block_size
        c0 = block_col * block_size
        bit = -1.0 if rng.random() < 0.5 else 1.0
        bit_values.append(bit)
        for channel_idx in range(channels):
            # np.ascontiguousarray 确保切片内存连续，避免非连续视图传入 BLAS 矩阵乘法时触发进程崩溃。
            block = np.ascontiguousarray(
                work[r0:r0 + block_size, c0:c0 + block_size, channel_idx]
            )
            coeff = _dct2(block, dct_matrix)
            for ci, cj in coeff_indices:
                coeff[ci, cj] = coeff[ci, cj] + bit * alpha
                coeff_ops += 1
            restored = _idct2(coeff, idct_matrix)
            work[r0:r0 + block_size, c0:c0 + block_size, channel_idx] = restored

    watermarked = np.clip(np.rint(work), 0, 255).astype(np.uint8)
    lf_trace_summary = {
        "lf_status": "ok",
        "method": "block_dct_lf_modulation",
        "block_size": block_size,
        "coeff_indices": [[ci, cj] for ci, cj in coeff_indices],
        "alpha": alpha,
        "redundancy": redundancy,
        "lf_region_ratio": region_ratio,
        "selected_block_count": selected_count,
        "total_block_count": total_blocks,
        "coeff_operation_count": coeff_ops,
        "bit_mean": float(sum(bit_values) / len(bit_values)) if bit_values else 0.0,
        "band_spec_digest": digests.canonical_sha256(band_spec),
    }
    return watermarked, lf_trace_summary


def detect_low_freq_score(
    image_array: np.ndarray,
    band_spec: Dict[str, Any],
    key_material: str,
    params: Dict[str, Any]
) -> tuple[Optional[float], Dict[str, Any]]:
    """
    功能：提取 LF 通道原始分数，用于后续 NP 校准。

    Extract LF raw score from block DCT coefficients for calibration readiness.

    Args:
        image_array: Input uint8 image array in HWC format.
        band_spec: Planner band specification mapping.
        key_material: Deterministic key material string.
        params: LF detection parameters.

    Returns:
        Tuple of (lf_score, lf_detect_trace).

    Raises:
        TypeError: If input types are invalid.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be np.ndarray")
    if image_array.ndim != 3:
        return None, {"lf_status": "fail", "lf_failure_reason": "lf_invalid_input"}
    if not isinstance(band_spec, dict):
        raise TypeError("band_spec must be dict")
    if not isinstance(key_material, str) or not key_material:
        raise TypeError("key_material must be non-empty str")
    if not isinstance(params, dict):
        raise TypeError("params must be dict")

    block_size = int(params.get("dct_block_size", 8))
    coeff_indices_raw = params.get("lf_coeff_indices", [(1, 1), (1, 2), (2, 1)])
    coeff_indices = [(int(v[0]), int(v[1])) for v in coeff_indices_raw if isinstance(v, (list, tuple)) and len(v) == 2]
    if block_size <= 1 or len(coeff_indices) == 0:
        return None, {"lf_status": "fail", "lf_failure_reason": "lf_invalid_params"}

    height, width, channels = image_array.shape
    blocks_h = height // block_size
    blocks_w = width // block_size
    total_blocks = blocks_h * blocks_w
    if total_blocks <= 0:
        return None, {"lf_status": "absent", "lf_absent_reason": "image_too_small_for_lf_blocks"}

    lf_selector = band_spec.get("lf_selector_summary") if isinstance(band_spec.get("lf_selector_summary"), dict) else {}
    region_ratio = float(lf_selector.get("region_ratio", 1.0))
    region_ratio = max(0.0, min(1.0, region_ratio))

    import random
    rng_seed = int(digests.canonical_sha256({"key_material": key_material, "tag": "lf_dct"})[:16], 16)
    rng = random.Random(rng_seed)
    all_block_ids = list(range(total_blocks))
    rng.shuffle(all_block_ids)
    selected_count = max(1, int(round(total_blocks * region_ratio)))
    selected_blocks = all_block_ids[:selected_count]
    bit_signs = {block_id: (-1.0 if rng.random() < 0.5 else 1.0) for block_id in selected_blocks}

    dct_matrix = _build_dct_matrix(block_size)
    score_acc = 0.0
    score_count = 0
    work = image_array.astype(np.float32)
    for block_id in selected_blocks:
        block_row = block_id // blocks_w
        block_col = block_id % blocks_w
        r0 = block_row * block_size
        c0 = block_col * block_size
        expected_sign = bit_signs[block_id]
        for channel_idx in range(channels):
            block = work[r0:r0 + block_size, c0:c0 + block_size, channel_idx]
            coeff = _dct2(block, dct_matrix)
            for ci, cj in coeff_indices:
                score_acc += expected_sign * float(coeff[ci, cj])
                score_count += 1

    if score_count <= 0:
        return None, {"lf_status": "absent", "lf_absent_reason": "no_lf_coefficients"}

    raw = score_acc / float(score_count)
    lf_score = float(0.5 + 0.5 * math.tanh(raw / 10.0))
    trace = {
        "lf_status": "ok",
        "lf_score_raw": float(raw),
        "lf_score_count": score_count,
        "selected_block_count": selected_count,
        "total_block_count": total_blocks,
        "band_spec_digest": digests.canonical_sha256(band_spec),
    }
    return lf_score, trace


def compute_lf_trace_digest(lf_trace_summary: Dict[str, Any]) -> str:
    """
    功能：计算 LF 追踪摘要 digest。

    Compute canonical LF trace digest from summary mapping.

    Args:
        lf_trace_summary: LF trace summary mapping.

    Returns:
        SHA256 digest string.
    """
    if not isinstance(lf_trace_summary, dict):
        raise TypeError("lf_trace_summary must be dict")
    return digests.canonical_sha256(lf_trace_summary)


def _build_dct_matrix(block_size: int) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    matrix = np.zeros((block_size, block_size), dtype=np.float32)
    scale0 = math.sqrt(1.0 / block_size)
    scale = math.sqrt(2.0 / block_size)
    for i in range(block_size):
        alpha = scale0 if i == 0 else scale
        for j in range(block_size):
            matrix[i, j] = alpha * math.cos(math.pi * (2 * j + 1) * i / (2.0 * block_size))
    return matrix


def _dct2(block: np.ndarray, dct_matrix: np.ndarray) -> np.ndarray:
    return dct_matrix @ block @ dct_matrix.T


def _idct2(coeff: np.ndarray, idct_matrix: np.ndarray) -> np.ndarray:
    return idct_matrix @ coeff @ idct_matrix.T


class LFCoderPRC:
    """
    功能：PRC paper-faithful LF coder with latent sign flipping and BP decoder.

    Implements latent-space sign flipping encoding and belief propagation decoding
    as specified in PRC-Watermark paper.

    Args:
        impl_id: Implementation identifier (must be lf_coder_prc_v1).
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If constructor inputs are invalid.
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

    def embed_apply(
        self,
        cfg: Dict[str, Any],
        latent_features: Any,
        plan_digest: str,
        cfg_digest: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        功能：PRC latent sign flipping embed.

        Apply latent-space sign flipping encoding with deterministic seed derivation.

        Args:
            cfg: Configuration dict with watermark.lf.* parameters.
            latent_features: Input latent features.
            plan_digest: Plan digest binding.
            cfg_digest: Optional cfg canonical digest.

        Returns:
            Dict with latent_features_embedded, lf_trace_summary, lf_trace_digest.

        Raises:
            TypeError: If inputs are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if not isinstance(plan_digest, str) or not plan_digest:
            raise TypeError("plan_digest must be non-empty str")

        lf_cfg = cfg.get("watermark", {}).get("lf", {})
        enabled = lf_cfg.get("enabled", False)
        if not enabled:
            return {
                "status": "absent",
                "latent_features_embedded": latent_features,
                "lf_trace_summary": {"absent": "lf_disabled"},
                "lf_trace_digest": digests.canonical_sha256({"absent": "lf_disabled"}),
            }

        coding_mode = lf_cfg.get("coding_mode", "latent_space_sign_flipping")
        if coding_mode != "latent_space_sign_flipping":
            return {
                "status": "failed",
                "latent_features_embedded": latent_features,
                "lf_failure_reason": "lf_coding_mode_mismatch",
                "lf_trace_digest": digests.canonical_sha256({"failure": "mode_mismatch"}),
            }

        variance = float(lf_cfg.get("variance", 1.5))
        message_length = int(lf_cfg.get("message_length", 64))
        ecc_sparsity = int(lf_cfg.get("ecc_sparsity", 3))

        # Generate message bits deterministically from plan_digest
        seed = int(digests.canonical_sha256({"plan_digest": plan_digest, "tag": "lf_message"})[:16], 16)
        import random
        rng = random.Random(seed)
        message_bits = [1 if rng.random() < 0.5 else -1 for _ in range(message_length)]
        ldpc_spec = build_ldpc_spec(
            message_length=message_length,
            ecc_sparsity=ecc_sparsity,
            seed_key=f"{plan_digest}:{message_length}:{ecc_sparsity}:embed",
        )
        code_bits = encode_message_bits(message_bits, ldpc_spec)
        block_length = int(ldpc_spec.get("n", len(code_bits)))

        # Latent sign flipping encoding
        flat_latents = _flatten_to_list(latent_features)
        if len(flat_latents) < block_length:
            return {
                "status": "failed",
                "latent_features_embedded": latent_features,
                "lf_failure_reason": "lf_insufficient_latent_dimension",
                "lf_trace_digest": digests.canonical_sha256({"failure": "insufficient_dim"}),
            }

        # Apply sign flipping to latents
        embedded_latents = list(flat_latents)
        for i in range(block_length):
            # Sign flip: latent *= code_bit * variance_scaling
            scale = variance * code_bits[i]
            embedded_latents[i] = embedded_latents[i] * scale if embedded_latents[i] != 0 else scale

        # Build trace summary
        parity_check_digest = ldpc_spec["parity_check_digest"]
        llr_summary_digest = digests.canonical_sha256({
            "variance": variance,
            "message_bits_digest": digests.canonical_sha256({"message_bits": message_bits}),
            "code_bits_digest": digests.canonical_sha256({"code_bits": code_bits}),
        })
        trace_summary = {
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "coding_mode": coding_mode,
            "variance": variance,
            "message_length": message_length,
            "block_length": block_length,
            "ecc_sparsity": ecc_sparsity,
            "parity_check_digest": parity_check_digest,
            "llr_summary_digest": llr_summary_digest,
            "ldpc_seed_digest": ldpc_spec["seed_digest"],
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "mode": "embed",
        }
        lf_trace_digest = digests.canonical_sha256(trace_summary)

        return {
            "status": "ok",
            "latent_features_embedded": embedded_latents,
            "lf_trace_summary": trace_summary,
            "lf_trace_digest": lf_trace_digest,
            "parity_check_digest": parity_check_digest,
            "llr_summary_digest": llr_summary_digest,
        }

    def detect_score(
        self,
        cfg: Dict[str, Any],
        latent_features: Any,
        plan_digest: str,
        cfg_digest: Optional[str] = None
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        功能：PRC belief propagation decoding.

        Detect watermark using belief propagation decoder.

        Args:
            cfg: Configuration dict.
            latent_features: Input latent features.
            plan_digest: Plan digest binding.
            cfg_digest: Optional cfg digest.

        Returns:
            Tuple of (lf_score, lf_detect_trace).

        Raises:
            TypeError: If inputs are invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        if not isinstance(plan_digest, str) or not plan_digest:
            raise TypeError("plan_digest must be non-empty str")

        lf_cfg = cfg.get("watermark", {}).get("lf", {})
        enabled = lf_cfg.get("enabled", False)
        if not enabled:
            return None, {
                "status": "absent",
                "lf_absent_reason": "lf_disabled",
                "lf_trace_digest": digests.canonical_sha256({"absent": "lf_disabled"}),
            }

        decoder = lf_cfg.get("decoder", "belief_propagation")
        if decoder != "belief_propagation":
            return None, {
                "status": "failed",
                "lf_failure_reason": "lf_decoder_mismatch",
                "lf_trace_digest": digests.canonical_sha256({"failure": "decoder_mismatch"}),
            }

        variance = float(lf_cfg.get("variance", 1.5))
        message_length = int(lf_cfg.get("message_length", 64))
        ecc_sparsity = int(lf_cfg.get("ecc_sparsity", 3))
        bp_iterations = int(lf_cfg.get("bp_iterations", 10))

        # Extract flattened latents
        flat_latents = _flatten_to_list(latent_features)
        ldpc_spec = build_ldpc_spec(
            message_length=message_length,
            ecc_sparsity=ecc_sparsity,
            seed_key=f"{plan_digest}:{message_length}:{ecc_sparsity}:embed",
        )
        block_length = int(ldpc_spec.get("n", message_length))

        if len(flat_latents) < block_length:
            return None, {
                "status": "failed",
                "lf_failure_reason": "lf_insufficient_latent_dimension",
                "available_latent_dim": int(len(flat_latents)),
                "required_block_length": int(block_length),
                "lf_trace_digest": digests.canonical_sha256({"failure": "insufficient_dim"}),
            }

        # Recover posteriors using erf-based recovery
        posteriors = recover_posteriors_erf(flat_latents[:block_length], variance)
        llr_values = [float((2.0 * p) / max(variance, 1e-8)) for p in posteriors]

        decode_result = decode_soft_llr(llr_values, ldpc_spec, bp_iterations)
        decoded_bits = decode_result["decoded_bits"]
        bp_converged = bool(decode_result["bp_converged"])
        bp_iteration_count = int(decode_result["bp_iteration_count"])

        seed = int(digests.canonical_sha256({"plan_digest": plan_digest, "tag": "lf_message"})[:16], 16)
        import random
        rng = random.Random(seed)
        expected_bits = [1 if rng.random() < 0.5 else -1 for _ in range(message_length)]

        agreement = 0
        for exp_bit, dec_bit in zip(expected_bits, decoded_bits[:message_length]):
            if exp_bit == dec_bit:
                agreement += 1

        # Score is defined as expected/decoded bit agreement ratio in [0, 1].
        lf_score = float(agreement) / float(message_length) if message_length > 0 else 0.0

        # Build trace
        llr_summary_digest = digests.canonical_sha256({
            "variance": variance,
            "posteriors_digest": digests.canonical_sha256({"posteriors": posteriors[:10]}),
            "decoded_bits_digest": digests.canonical_sha256({"decoded_bits": decoded_bits[:16]}),
        })
        trace = {
            "status": "ok",
            "lf_score": lf_score,
            "bp_iterations": bp_iterations,
            "bp_converged": bp_converged,
            "bp_iteration_count": bp_iteration_count,
            "block_length": block_length,
            "available_latent_dim": int(len(flat_latents)),
            "required_block_length": int(block_length),
            "syndrome_weight": int(decode_result["syndrome_weight"]),
            "llr_summary_digest": llr_summary_digest,
            "parity_check_digest": ldpc_spec["parity_check_digest"],
            "impl_id": self.impl_id,
            "impl_version": self.impl_version,
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
        }
        trace["lf_trace_digest"] = digests.canonical_sha256(trace)

        return lf_score, trace


def _flatten_to_list(value: Any) -> list:
    """
    功能：递归扁平化为浮点列表。

    Flatten nested structure to float list.

    Args:
        value: Input value (can be nested).

    Returns:
        Flattened list of floats.
    """
    result = []
    _flatten_recursive_to_list(value, result)
    return result


def _flatten_recursive_to_list(value: Any, sink: list) -> None:
    """
    功能：递归扁平化辅助函数。

    Recursive helper for flattening.

    Args:
        value: Current value.
        sink: Output list.

    Returns:
        None.
    """
    if value is None:
        return
    if isinstance(value, (int, float)):
        sink.append(float(value))
        return
    if isinstance(value, np.ndarray):
        for item in value.reshape(-1).tolist():
            _flatten_recursive_to_list(item, sink)
        return
    # 兼容 torch.Tensor 输入但不引入硬依赖。
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            tensor_np = value.detach().cpu().numpy()
            if isinstance(tensor_np, np.ndarray):
                for item in tensor_np.reshape(-1).tolist():
                    _flatten_recursive_to_list(item, sink)
                return
        except Exception:
            # tensor 转换失败时，回退到后续通用分支。
            pass
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            _flatten_recursive_to_list(value[key], sink)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_recursive_to_list(item, sink)
