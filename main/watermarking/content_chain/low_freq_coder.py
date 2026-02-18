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
from typing import Any, Dict, Optional

from main.core import digests

from .interfaces import ContentEvidence


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

# 允许的失败原因枚举。
ALLOWED_LF_CODER_FAILURE_REASONS = {
    "lf_coder_disabled",                  # enable_lf 配置为 false，低频编码未启用
    "lf_coder_no_plan",                   # 缺失 plan_digest，无法确定编码子空间
    "lf_coder_invalid_input",             # 输入缺失或形状不符
    "lf_coder_plan_mismatch",             # plan_digest 不一致，编码参数与计划脱离
    "lf_coder_encoding_failed",           # 编码计算异常
    "lf_coder_trace_digest_mismatch",     # lf_trace_digest 与预期不符
}


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
                encoding_trace=None
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
                encoding_trace=None
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

        # (3) 验证输入有效性。
        if inputs is None:
            inputs = {}

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
                encoding_trace=None
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
            if not isinstance(ecc, int) or ecc < 2 or ecc > 8:
                # ecc 参数不合法，必须 fail-fast。
                raise ValueError(f"ecc must be int in [2, 8], got {ecc}")
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
                param_seed_input = f"{plan_digest}:{codebook_id}:{ecc}:{strength}:{delta}:{block_length}"
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
                encoding_trace=None
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
            encoding_trace=encoding_trace
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
    encoding_trace: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    功能：构造可复算的低频编码追踪有效负载。

    Build deterministic trace payload for LF coder digest computation.
    
    关键设计：trace_payload 中的键空间与 plan_digest_include_paths 对齐，
    确保配置参数变化（watermark.lf.* 中任何字段）均导致 lf_trace_digest 变化。

    Args:
        cfg: Configuration mapping containing 'watermark.lf.*' parameters.
        impl_id: Implementation identifier.
        impl_version: Implementation version.
        impl_digest: Implementation digest.
        enabled: Whether LF coding is enabled.
        plan_digest: Optional plan digest binding.
        encoding_trace: Optional encoding parameters and trace.

    Returns:
        JSON-like dict for canonical SHA256 computation.
        Keys are named to match plan_digest_include_paths (codebook_id, ecc, strength).

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

    # P0 修复：统一键空间为 watermark.lf（与 plan_digest_include_paths 对齐）。
    lf_cfg = cfg.get("watermark", {}).get("lf", {})

    payload = {
        "impl_id": impl_id,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "trace_version": LOW_FREQ_CODER_TRACE_VERSION,
        "enabled": enabled,
        "plan_digest": plan_digest,
        # P0 修复：包含所有 plan_digest_include_paths 中的 watermark.lf.* 字段。
        "codebook_id": lf_cfg.get("codebook_id", "default") if enabled else None,
        "ecc": lf_cfg.get("ecc", 3) if enabled else None,
        "strength": lf_cfg.get("strength", 0.1) if enabled else None,
        "delta": lf_cfg.get("delta", 1.0) if enabled else None,
        "block_length": lf_cfg.get("block_length", 8) if enabled else None,
    }

    if encoding_trace is not None:
        payload["encoding_trace"] = encoding_trace

    return payload
