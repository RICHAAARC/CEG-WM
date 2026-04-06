"""
检测、评估与校准编排

功能说明：
- 执行检测编排流程，包括 plan_digest 一致性验证。

"""

from __future__ import annotations

import copy
import glob
import inspect
import math
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from main.core import digests
from main.core import records_io
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain import detector_scoring
from main.watermarking.content_chain import channel_hf
from main.watermarking.common.plan_digest_flow import PLAN_INPUT_SCHEMA_VERSION, verify_plan_digest
from main.watermarking.content_chain.high_freq_embedder import (
    HighFreqTruncationCodec,
    HIGH_FREQ_TRUNCATION_CODEC_ID,
    HIGH_FREQ_TRUNCATION_CODEC_VERSION,
    HF_FAILURE_RULE_VERSION,
)
from main.watermarking.content_chain.low_freq_coder import (
    LowFreqTemplateCodec,
    LOW_FREQ_TEMPLATE_CODEC_ID,
    LOW_FREQ_TEMPLATE_CODEC_VERSION,
)
from main.watermarking.content_chain import high_freq_embedder as high_freq_embedder_module
from main.watermarking.content_chain import low_freq_coder as low_freq_coder_module
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    build_runtime_jvp_operator_from_cache,
)
from main.watermarking.fusion import neyman_pearson
from main.watermarking.fusion.interfaces import FusionDecision
from main.watermarking.geometry_chain.sync.latent_sync_template import SyncRuntimeContext

GEO_AVAILABILITY_RULE_VERSION = "geo_availability_rule_v1"
from main.evaluation import protocol_loader as eval_protocol_loader
from main.evaluation import metrics as eval_metrics
from main.evaluation import report_builder as eval_report_builder
from main.evaluation import attack_coverage as eval_attack_coverage
from main.evaluation import workflow_inputs as eval_workflow_inputs
from main.policy.runtime_whitelist import PolicyPathSemantics, load_policy_path_semantics


def _as_dict_payload(value: Any) -> Dict[str, Any] | None:
    """
    功能：将对象规范化为 dict 负载。

    Convert a payload-like object to a dictionary.

    Args:
        value: Candidate payload object.

    Returns:
        Dictionary payload when available; otherwise None.
    """
    if isinstance(value, dict):
        return cast(Dict[str, Any], value)
    as_dict_method = getattr(value, "as_dict", None)
    if callable(as_dict_method):
        converted = as_dict_method()
        if isinstance(converted, dict):
            return cast(Dict[str, Any], converted)
    return None


def _coerce_optional_finite_score(value: Any) -> Optional[float]:
    """
    功能：将候选值解析为有限浮点分数。

    Coerce a candidate value into a finite float score.

    Args:
        value: Candidate score value.

    Returns:
        Finite float when available; otherwise None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if np.isfinite(numeric_value):
            return numeric_value
    return None


def _resolve_optional_finite_score_from_candidates(*values: Any) -> Optional[float]:
    """
    功能：按顺序解析第一个有限浮点分数，保留 0.0。 

    Resolve the first finite score from ordered candidates while preserving 0.0.

    Args:
        *values: Candidate score values.

    Returns:
        First finite float score when available; otherwise None.
    """
    for value in values:
        score = _coerce_optional_finite_score(value)
        if score is not None:
            return score
    return None


def _synchronize_content_score_aliases(content_evidence_payload: Dict[str, Any]) -> None:
    """
    功能：同步内容链正式分数字段与兼容别名。

    Synchronize canonical content and LF score fields with compatibility aliases.

    Args:
        content_evidence_payload: Mutable content evidence payload.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        raise TypeError("content_evidence_payload must be dict")

    score_parts_node = content_evidence_payload.get("score_parts")
    score_parts = cast(Dict[str, Any], score_parts_node) if isinstance(score_parts_node, dict) else None
    hf_summary_node = content_evidence_payload.get("hf_evidence_summary")
    hf_summary = cast(Dict[str, Any], hf_summary_node) if isinstance(hf_summary_node, dict) else None

    content_chain_score = (
        _coerce_optional_finite_score(content_evidence_payload.get(eval_metrics.CONTENT_CHAIN_SCORE_NAME))
        or _coerce_optional_finite_score(content_evidence_payload.get("score"))
        or _coerce_optional_finite_score(content_evidence_payload.get("content_score"))
    )
    if content_chain_score is not None:
        content_evidence_payload[eval_metrics.CONTENT_CHAIN_SCORE_NAME] = content_chain_score
        content_evidence_payload["score"] = content_chain_score
        content_evidence_payload["content_score"] = content_chain_score
        if score_parts is not None:
            score_parts[eval_metrics.CONTENT_CHAIN_SCORE_NAME] = content_chain_score
            score_parts["content_score"] = content_chain_score

    lf_channel_score = (
        _coerce_optional_finite_score(content_evidence_payload.get(eval_metrics.LF_CHANNEL_SCORE_NAME))
        or _coerce_optional_finite_score(content_evidence_payload.get("lf_score"))
    )
    if lf_channel_score is not None:
        content_evidence_payload[eval_metrics.LF_CHANNEL_SCORE_NAME] = lf_channel_score
        content_evidence_payload["lf_score"] = lf_channel_score
        if score_parts is not None:
            score_parts[eval_metrics.LF_CHANNEL_SCORE_NAME] = lf_channel_score
            score_parts["lf_score"] = lf_channel_score

    lf_correlation_score = (
        _coerce_optional_finite_score(content_evidence_payload.get(eval_metrics.LF_CORRELATION_SCORE_NAME))
        or _coerce_optional_finite_score(content_evidence_payload.get("detect_lf_score"))
    )
    if lf_correlation_score is not None:
        content_evidence_payload[eval_metrics.LF_CORRELATION_SCORE_NAME] = lf_correlation_score
        content_evidence_payload["detect_lf_score"] = lf_correlation_score
        if score_parts is not None:
            score_parts[eval_metrics.LF_CORRELATION_SCORE_NAME] = lf_correlation_score
            score_parts["detect_lf_score"] = lf_correlation_score

    hf_raw_energy = _resolve_optional_finite_score_from_candidates(
        content_evidence_payload.get("hf_raw_energy"),
        content_evidence_payload.get("hf_score"),
        hf_summary.get("hf_raw_energy") if isinstance(hf_summary, dict) else None,
        score_parts.get("hf_raw_energy") if isinstance(score_parts, dict) else None,
    )
    if hf_raw_energy is not None:
        content_evidence_payload["hf_raw_energy"] = hf_raw_energy
        if score_parts is not None:
            score_parts["hf_raw_energy"] = hf_raw_energy
        if hf_summary is not None:
            hf_summary["hf_raw_energy"] = hf_raw_energy

    hf_content_score = _resolve_optional_finite_score_from_candidates(
        content_evidence_payload.get("hf_content_score"),
        hf_summary.get("hf_content_score") if isinstance(hf_summary, dict) else None,
        score_parts.get("hf_content_score") if isinstance(score_parts, dict) else None,
    )
    if hf_content_score is not None:
        content_evidence_payload["hf_content_score"] = hf_content_score
        if score_parts is not None:
            score_parts["hf_content_score"] = hf_content_score
        if hf_summary is not None:
            hf_summary["hf_content_score"] = hf_content_score


def _call_content_extractor_extract(
    extractor: Any,
    cfg: Dict[str, Any],
    inputs: Optional[Dict[str, Any]],
    cfg_digest: Optional[str],
) -> Any:
    """
    功能：兼容不同 extract 签名调用 content_extractor。

    Call content_extractor.extract with backward-compatible signature handling.

    Args:
        extractor: Content extractor instance.
        cfg: Configuration mapping.
        inputs: Optional content input mapping.
        cfg_digest: Optional config digest.

    Returns:
        Extractor return payload.
    """
    extract_fn = getattr(extractor, "extract", None)
    if not callable(extract_fn):
        raise TypeError("content_extractor.extract must be callable")

    signature = inspect.signature(extract_fn)
    params = signature.parameters
    positional_args: list[Any] = []
    keyword_args: Dict[str, Any] = {}

    if "inputs" in params:
        keyword_args["inputs"] = inputs
    elif len(params) >= 2 and inputs is not None:
        positional_args.append(inputs)

    if "cfg_digest" in params:
        keyword_args["cfg_digest"] = cfg_digest
    elif len(params) >= 3 and cfg_digest is not None and len(positional_args) >= 1:
        positional_args.append(cfg_digest)

    return extract_fn(cfg, *positional_args, **keyword_args)


def _extract_input_attestation_payload(input_record: Optional[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    功能：从 embed 输入记录提取 attestation 载荷。

    Extract attestation payload from the embed-side input record.

    Args:
        input_record: Optional input record mapping.

    Returns:
        Attestation payload mapping or None.
    """
    if not isinstance(input_record, dict):
        return None
    attestation_node = input_record.get("attestation")
    attestation_payload = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    candidate_statement = attestation_payload.get("statement")
    if not isinstance(candidate_statement, dict):
        candidate_statement = input_record.get("attestation_statement")
    attestation_bundle = attestation_payload.get("signed_bundle")
    if not isinstance(attestation_bundle, dict):
        attestation_bundle = input_record.get("attestation_bundle")
    trace_commit = attestation_payload.get("trace_commit")
    if not isinstance(trace_commit, str) or not trace_commit:
        if isinstance(attestation_bundle, dict):
            bundle_trace_commit = attestation_bundle.get("trace_commit")
            if isinstance(bundle_trace_commit, str) and bundle_trace_commit:
                trace_commit = bundle_trace_commit
            else:
                trace_commit = None
        else:
            trace_commit = None
    if not isinstance(candidate_statement, dict):
        return None
    return {
        "candidate_statement": candidate_statement,
        "attestation_bundle": attestation_bundle if isinstance(attestation_bundle, dict) else None,
        "trace_commit": trace_commit,
    }


def _extract_negative_branch_attestation_provenance(
    input_record: Optional[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """
    功能：提取 negative branch 的 statement-only attestation provenance。 

    Extract statement-only attestation provenance for the negative branch.

    Args:
        input_record: Optional input record mapping.

    Returns:
        Provenance payload with candidate_statement when available; otherwise
        None.
    """
    if not isinstance(input_record, dict):
        return None

    provenance_node = input_record.get("negative_branch_source_attestation_provenance")
    provenance_payload = cast(Dict[str, Any], provenance_node) if isinstance(provenance_node, dict) else {}
    candidate_statement = provenance_payload.get("statement")
    if not isinstance(candidate_statement, dict):
        return None

    trace_commit = provenance_payload.get("trace_commit")
    if not isinstance(trace_commit, str) or not trace_commit:
        trace_commit = None

    return {
        "candidate_statement": candidate_statement,
        "attestation_bundle": None,
        "trace_commit": trace_commit,
        "provenance": provenance_payload,
    }


NEGATIVE_BRANCH_STATEMENT_ONLY_ATTESTATION_SOURCE = "negative_branch_statement_only_provenance"
STATEMENT_ONLY_PROVENANCE_NO_BUNDLE_STATUS = "statement_only_provenance_no_bundle"


def _resolve_detect_attestation_bundle_status(
    authenticity_status: Any,
    attestation_source: Any,
    bundle_verification: Optional[Dict[str, Any]],
    explicit_bundle_status: Any = None,
) -> Optional[str]:
    """
    功能：解析 detect 侧真实性层的 bundle_status 合同。 

    Resolve the canonical detect-side bundle_status contract.

    Args:
        authenticity_status: Authenticity status token.
        attestation_source: Detect attestation source token.
        bundle_verification: Optional bundle verification payload.
        explicit_bundle_status: Optional explicit bundle status override.

    Returns:
        Canonical bundle_status string when available; otherwise None.
    """
    if isinstance(explicit_bundle_status, str) and explicit_bundle_status:
        return explicit_bundle_status

    if isinstance(bundle_verification, dict):
        bundle_status = bundle_verification.get("status")
        if isinstance(bundle_status, str) and bundle_status:
            return bundle_status

    if (
        authenticity_status == "statement_only"
        and attestation_source == NEGATIVE_BRANCH_STATEMENT_ONLY_ATTESTATION_SOURCE
    ):
        return STATEMENT_ONLY_PROVENANCE_NO_BUNDLE_STATUS
    return None


def _build_detect_authenticity_result(
    authenticity_status: Any,
    statement_status: str,
    attestation_source: Any,
    bundle_verification: Optional[Dict[str, Any]] = None,
    explicit_bundle_status: Any = None,
    attestation_digest: Any = None,
    event_binding_digest: Any = None,
    trace_commit: Any = None,
) -> Dict[str, Any]:
    """
    功能：构造 detect 侧真实性层结果。 

    Build the canonical detect-side authenticity_result payload.

    Args:
        authenticity_status: Authenticity status token.
        statement_status: Statement parsing status token.
        attestation_source: Detect attestation source token.
        bundle_verification: Optional bundle verification payload.
        explicit_bundle_status: Optional explicit bundle status override.
        attestation_digest: Optional attestation digest.
        event_binding_digest: Optional event binding digest.
        trace_commit: Optional trace commit digest.

    Returns:
        Canonical authenticity_result mapping.
    """
    status_value = authenticity_status if isinstance(authenticity_status, str) and authenticity_status else "absent"
    authenticity_result = {
        "status": status_value,
        "bundle_status": _resolve_detect_attestation_bundle_status(
            authenticity_status=status_value,
            attestation_source=attestation_source,
            bundle_verification=bundle_verification,
            explicit_bundle_status=explicit_bundle_status,
        ),
        "statement_status": statement_status,
    }
    if isinstance(attestation_digest, str) and attestation_digest:
        authenticity_result["attestation_digest"] = attestation_digest
    if isinstance(event_binding_digest, str) and event_binding_digest:
        authenticity_result["event_binding_digest"] = event_binding_digest
    if isinstance(trace_commit, str) and trace_commit:
        authenticity_result["trace_commit"] = trace_commit
    return authenticity_result


def _attach_detect_attestation_source(
    attestation_result: Dict[str, Any],
    attestation_source: Any,
) -> Dict[str, Any]:
    """
    功能：向 detect attestation 结果附加来源标识。 

    Attach attestation_source into the persisted detect attestation payload.

    Args:
        attestation_result: Mutable detect attestation result mapping.
        attestation_source: Detect attestation source token.

    Returns:
        The same mapping with attestation_source when available.
    """
    if isinstance(attestation_source, str) and attestation_source:
        attestation_result["attestation_source"] = attestation_source
    return attestation_result


def _bind_detect_attestation_runtime_to_cfg(cfg: Dict[str, Any], attestation_context: Dict[str, Any]) -> None:
    """
    功能：将 detect 主链 attestation 运行时变量绑定到 cfg。

    Bind detect-side attestation runtime variables into cfg for LF/HF/GEO use.

    Args:
        cfg: Mutable configuration mapping.
        attestation_context: Detect attestation context mapping.

    Returns:
        None.
    """
    if not isinstance(attestation_context, dict):
        return
    runtime_bindings = attestation_context.get("runtime_bindings")
    if not isinstance(runtime_bindings, dict):
        return

    cfg["attestation_runtime"] = runtime_bindings
    cfg["attestation_digest"] = runtime_bindings.get("attestation_digest")
    cfg["attestation_event_digest"] = runtime_bindings.get("event_binding_digest")
    cfg["lf_attestation_event_digest"] = runtime_bindings.get("event_binding_digest")
    cfg["hf_attestation_event_digest"] = runtime_bindings.get("event_binding_digest")
    cfg["lf_attestation_key"] = runtime_bindings.get("k_lf")
    cfg["hf_attestation_key"] = runtime_bindings.get("k_hf")
    cfg["k_lf"] = runtime_bindings.get("k_lf")
    cfg["k_hf"] = runtime_bindings.get("k_hf")
    cfg["k_geo"] = runtime_bindings.get("k_geo")
    cfg["geo_anchor_seed"] = runtime_bindings.get("geo_anchor_seed")

    watermark_node = cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
    watermark_cfg = dict(watermark_cfg)
    watermark_cfg["attestation_digest"] = runtime_bindings.get("attestation_digest")
    watermark_cfg["attestation_event_digest"] = runtime_bindings.get("event_binding_digest")
    cfg["watermark"] = watermark_cfg


def _build_detect_hf_runtime_cfg(cfg: Dict[str, Any], plan_digest: Optional[str]) -> Dict[str, Any]:
    """
    功能：为 detect 侧 HF challenge 构造局部运行时 cfg。 

    Build a local runtime cfg for detect-side HF challenge derivation.

    Args:
        cfg: Base runtime configuration mapping.
        plan_digest: Canonical plan digest to bind into the HF path.

    Returns:
        Copied cfg with HF-local plan digest binding applied.

    Raises:
        TypeError: If cfg is not a dict.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    runtime_cfg = dict(cfg)
    watermark_node = runtime_cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
    watermark_cfg = dict(watermark_cfg)

    if isinstance(plan_digest, str) and plan_digest:
        runtime_cfg["plan_digest"] = plan_digest
        watermark_cfg["plan_digest"] = plan_digest

    runtime_cfg["watermark"] = watermark_cfg
    return runtime_cfg


def _prepare_detect_attestation_context(cfg: Dict[str, Any], input_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    功能：为 detect 主链准备 attestation 条件化上下文。

    Prepare attestation context for the detect main path. When the signed bundle
    authenticates successfully, the resulting event keys are injected into
    LF/HF/GEO execution. Statement-only mode is preserved as an explicit
    non-authentic layer.

    Args:
        cfg: Configuration mapping.
        input_record: Optional embed-side input record.

    Returns:
        Detect attestation context mapping.
    """
    attestation_node = cfg.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    if not bool(attestation_cfg.get("enabled", False)):
        return {
            "attestation_status": "absent",
            "attestation_absent_reason": "attestation_disabled",
        }

    payload = _extract_input_attestation_payload(input_record)
    attestation_source = "formal_input_payload"
    if not isinstance(payload, dict):
        negative_branch_provenance = _extract_negative_branch_attestation_provenance(input_record)
        if isinstance(negative_branch_provenance, dict):
            payload = negative_branch_provenance
            attestation_source = NEGATIVE_BRANCH_STATEMENT_ONLY_ATTESTATION_SOURCE
        else:
            return {
                "attestation_status": "absent",
                "attestation_absent_reason": "attestation_statement_absent",
            }

    if not isinstance(payload, dict):
        return {
            "attestation_status": "absent",
            "attestation_absent_reason": "attestation_statement_absent",
        }

    from main.watermarking.provenance.attestation_statement import (
        compute_attestation_digest,
        statement_from_dict,
        verify_signed_attestation_bundle,
    )
    from main.watermarking.provenance.key_derivation import derive_attestation_keys, derive_geo_anchor_seed

    candidate_statement = payload.get("candidate_statement")
    if not isinstance(candidate_statement, dict):
        return {
            "attestation_status": "absent",
            "attestation_absent_reason": "attestation_statement_absent",
        }

    try:
        statement = statement_from_dict(candidate_statement)
    except Exception as exc:
        return {
            "attestation_status": "mismatch",
            "attestation_failure_reason": f"attestation_statement_invalid:{type(exc).__name__}",
            "candidate_statement": candidate_statement,
        }

    attestation_digest = compute_attestation_digest(statement)
    trace_commit = payload.get("trace_commit")
    if not isinstance(trace_commit, str) or not trace_commit:
        trace_commit = None

    k_master = cfg.get("__attestation_verify_k_master__")
    if not isinstance(k_master, str) or not k_master:
        return {
            "attestation_status": "absent",
            "attestation_absent_reason": "attestation_master_key_missing",
            "candidate_statement": candidate_statement,
            "attestation_bundle": payload.get("attestation_bundle"),
            "attestation_digest": attestation_digest,
            "trace_commit": trace_commit,
        }

    bundle_verification = None
    attestation_bundle = payload.get("attestation_bundle")
    if isinstance(attestation_bundle, dict):
        bundle_verification = verify_signed_attestation_bundle(attestation_bundle, k_master)
        if bundle_verification.get("status") != "ok":
            return {
                "attestation_status": "mismatch",
                "candidate_statement": candidate_statement,
                "attestation_bundle": attestation_bundle,
                "bundle_verification": bundle_verification,
                "attestation_digest": attestation_digest,
                "trace_commit": trace_commit,
            }
        authenticity_status = "authentic"
    else:
        authenticity_status = "statement_only"

    event_binding_mode = "trajectory_bound" if bool(attestation_cfg.get("use_trajectory_mix", True)) else "statement_only"
    attest_keys = derive_attestation_keys(
        k_master,
        attestation_digest,
        trajectory_commit=trace_commit,
        event_binding_mode=event_binding_mode,
    )
    return {
        "attestation_status": "ok",
        "attestation_source": attestation_source,
        "authenticity_status": authenticity_status,
        "candidate_statement": candidate_statement,
        "attestation_bundle": attestation_bundle if isinstance(attestation_bundle, dict) else None,
        "bundle_verification": bundle_verification,
        "attestation_digest": attestation_digest,
        "event_binding_digest": attest_keys.event_binding_digest,
        "event_binding_mode": attest_keys.event_binding_mode,
        "trace_commit": trace_commit,
        "runtime_bindings": {
            "attestation_digest": attestation_digest,
            "event_binding_digest": attest_keys.event_binding_digest,
            "k_lf": attest_keys.k_lf,
            "k_hf": attest_keys.k_hf,
            "k_geo": attest_keys.k_geo,
            "geo_anchor_seed": derive_geo_anchor_seed(attest_keys.k_geo),
            "event_binding_mode": attest_keys.event_binding_mode,
        },
    }


def _build_detect_attestation_result(
    cfg: Dict[str, Any],
    attestation_context: Dict[str, Any],
    content_evidence_payload: Optional[Dict[str, Any]],
    geometry_evidence_payload: Optional[Dict[str, Any]],
    lf_attestation_features: Optional[Any] = None,
    lf_attestation_trace_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    功能：构造 detect 侧 attestation 分层结果。

    Build the detect-side layered attestation result with authenticity, image
    evidence, and final event-attested decision.

    Args:
        cfg: Configuration mapping.
        attestation_context: Prepared detect attestation context.
        content_evidence_payload: Optional content evidence mapping.
        geometry_evidence_payload: Optional geometry evidence mapping.
        lf_attestation_features: Optional LF attestation coefficient vector
            extracted from the detect main path.

    Returns:
        Attestation result mapping.
    """
    if not isinstance(attestation_context, dict):
        return {
            "status": "absent",
            "attestation_absent_reason": "attestation_context_invalid",
        }

    candidate_statement = attestation_context.get("candidate_statement")
    attestation_bundle = attestation_context.get("attestation_bundle")
    attestation_source = attestation_context.get("attestation_source")
    k_master = cfg.get("__attestation_verify_k_master__")
    if not isinstance(candidate_statement, dict):
        return _attach_detect_attestation_source({
            "status": attestation_context.get("attestation_status", "absent"),
            "attestation_absent_reason": attestation_context.get("attestation_absent_reason", "attestation_statement_absent"),
            "authenticity_result": _build_detect_authenticity_result(
                authenticity_status=attestation_context.get("authenticity_status", "absent"),
                statement_status="absent",
                attestation_source=attestation_source,
                bundle_verification=attestation_context.get("bundle_verification") if isinstance(attestation_context.get("bundle_verification"), dict) else None,
                explicit_bundle_status="absent",
            ),
            "image_evidence_result": {"status": "absent", "channel_scores": {"lf": None, "hf": None, "geo": None}},
            "final_event_attested_decision": {"status": "absent", "is_event_attested": False},
        }, attestation_source)
    if not isinstance(k_master, str) or not k_master:
        return _attach_detect_attestation_source({
            "status": "absent",
            "attestation_absent_reason": "attestation_master_key_missing",
            "authenticity_result": _build_detect_authenticity_result(
                authenticity_status=attestation_context.get("authenticity_status", "absent"),
                statement_status="parsed",
                attestation_source=attestation_source,
                bundle_verification=attestation_context.get("bundle_verification") if isinstance(attestation_context.get("bundle_verification"), dict) else None,
                explicit_bundle_status="absent",
            ),
            "image_evidence_result": {"status": "absent", "channel_scores": {"lf": None, "hf": None, "geo": None}},
            "final_event_attested_decision": {"status": "absent", "is_event_attested": False},
        }, attestation_source)

    geo_score = None
    if isinstance(geometry_evidence_payload, dict):
        raw_geo_score = geometry_evidence_payload.get("geo_score")
        if isinstance(raw_geo_score, (int, float)):
            geo_score = float(raw_geo_score)

    hf_values = None
    if isinstance(content_evidence_payload, dict):
        raw_hf_values = content_evidence_payload.get("hf_attestation_values")
        if raw_hf_values is None:
            score_parts_node = content_evidence_payload.get("score_parts")
            if isinstance(score_parts_node, dict):
                raw_hf_values = score_parts_node.get("hf_attestation_values")
        if raw_hf_values is not None:
            hf_values = raw_hf_values

    attestation_node = cfg.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}

    result = verify_attestation(
        k_master=k_master,
        candidate_statement=candidate_statement,
        attestation_bundle=attestation_bundle if isinstance(attestation_bundle, dict) else None,
        content_evidence=content_evidence_payload if isinstance(content_evidence_payload, dict) else None,
        cfg=cfg,
        hf_values=hf_values,
        lf_latent_features=lf_attestation_features,
        geo_score=geo_score,
        lf_weight=float(attestation_cfg.get("lf_weight", 0.5)),
        hf_weight=float(attestation_cfg.get("hf_weight", 0.3)),
        geo_weight=float(attestation_cfg.get("geo_weight", 0.2)),
        attested_threshold=float(attestation_cfg.get("threshold", 0.65)),
        attestation_decision_mode=str(attestation_cfg.get("decision_mode", "content_primary_geo_rescue")),
        geo_rescue_band_delta_low=float(attestation_cfg.get("rescue_band_delta_low", 0.05)),
        geo_rescue_min_score=float(attestation_cfg.get("geo_rescue_min_score", 0.3)),
        lf_params=lf_attestation_trace_context,
        detect_hf_plan_digest_used=(
            cfg.get("__detect_hf_plan_digest_used__")
            if isinstance(cfg.get("__detect_hf_plan_digest_used__"), str) and cfg.get("__detect_hf_plan_digest_used__")
            else None
        ),
        attestation_source=attestation_source if isinstance(attestation_source, str) else None,
    )
    lf_planner_risk_report = _build_lf_planner_risk_report_artifact(
        cast(Dict[str, Any], result.get("_lf_alignment_table_artifact")) if isinstance(result.get("_lf_alignment_table_artifact"), dict) else {},
        lf_attestation_trace_context,
    )
    if isinstance(lf_planner_risk_report, dict):
        result["_lf_planner_risk_report_artifact"] = lf_planner_risk_report
    geo_rescue_diagnostics = _build_geo_rescue_diagnostics_artifact(
        cfg,
        geometry_evidence_payload if isinstance(geometry_evidence_payload, dict) else None,
        result,
        attestation_digest=result.get("attestation_digest") if isinstance(result.get("attestation_digest"), str) else None,
        event_binding_digest=result.get("event_binding_digest") if isinstance(result.get("event_binding_digest"), str) else None,
        trace_commit=attestation_context.get("trace_commit") if isinstance(attestation_context.get("trace_commit"), str) else None,
    )
    if isinstance(geo_rescue_diagnostics, dict):
        result["_geo_rescue_diagnostics_artifact"] = geo_rescue_diagnostics
    result["status"] = result.get("verdict")
    return _attach_detect_attestation_source(result, attestation_source)


def _resolve_hf_detect_summary(content_evidence: Optional[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not isinstance(content_evidence, dict):
        return None
    hf_summary = content_evidence.get("hf_evidence_summary")
    if isinstance(hf_summary, dict):
        return cast(Dict[str, Any], hf_summary)
    score_parts = content_evidence.get("score_parts")
    if isinstance(score_parts, dict):
        hf_metrics = score_parts.get("hf_metrics")
        if isinstance(hf_metrics, dict):
            return cast(Dict[str, Any], hf_metrics)
    return None


def _resolve_trace_match_status(detect_value: Any, attestation_value: Any) -> str:
    if detect_value is None or attestation_value is None:
        return "absent"
    if detect_value == attestation_value:
        return "ok"
    return "mismatch"


def _merge_hf_attestation_trace(
    hf_trace: Optional[Dict[str, Any]],
    detect_summary: Optional[Dict[str, Any]],
    *,
    attestation_digest: str,
    event_binding_digest: str,
    trace_commit: Optional[str],
    hf_score: Optional[float],
    hf_decision_score: Optional[float],
    detect_hf_plan_digest_used: Optional[str] = None,
) -> Dict[str, Any] | None:
    if not isinstance(hf_trace, dict):
        return None

    detect_hf_challenge_digest = None
    detect_hf_challenge_source = None
    detect_hf_challenge_seed = None
    detect_hf_threshold_percentile_applied = None
    detect_hf_coeffs_retained_count = None
    detect_hf_retention_ratio = None
    detect_hf_trace_digest = None
    if isinstance(detect_summary, dict):
        detect_hf_challenge_digest = detect_summary.get("challenge_digest")
        detect_hf_challenge_source = detect_summary.get("challenge_source")
        detect_hf_challenge_seed = detect_summary.get("challenge_seed")
        detect_hf_threshold_percentile_applied = detect_summary.get("threshold_percentile_applied")
        detect_hf_coeffs_retained_count = detect_summary.get("coeffs_retained_count")
        detect_hf_retention_ratio = detect_summary.get("retention_ratio")
        detect_hf_trace_digest = detect_summary.get("hf_trace_digest")

    challenge_match_status = _resolve_trace_match_status(
        detect_hf_challenge_digest,
        hf_trace.get("hf_attestation_challenge_digest"),
    )
    threshold_match_status = _resolve_trace_match_status(
        detect_hf_threshold_percentile_applied,
        hf_trace.get("hf_attestation_threshold_percentile_applied"),
    )
    retained_count_match_status = _resolve_trace_match_status(
        detect_hf_coeffs_retained_count,
        hf_trace.get("hf_attestation_retained_count"),
    )

    comparison_statuses = [
        challenge_match_status,
        threshold_match_status,
        retained_count_match_status,
    ]
    if all(status == "absent" for status in comparison_statuses):
        trace_consistency = "absent"
    elif any(status == "mismatch" for status in comparison_statuses):
        trace_consistency = "mismatch"
    elif any(status == "absent" for status in comparison_statuses):
        trace_consistency = "partial"
    else:
        trace_consistency = "ok"

    return {
        "artifact_type": "hf_attestation_trace",
        "attestation_digest": attestation_digest,
        "event_binding_digest": event_binding_digest,
        "trace_commit": trace_commit,
        "hf_attestation_score": hf_score,
        "hf_attestation_decision_score": hf_decision_score,
        **hf_trace,
        "detect_hf_plan_digest_used": detect_hf_plan_digest_used,
        "detect_hf_challenge_digest": detect_hf_challenge_digest,
        "detect_hf_challenge_seed": detect_hf_challenge_seed,
        "detect_hf_challenge_source": detect_hf_challenge_source,
        "detect_hf_threshold_percentile_applied": detect_hf_threshold_percentile_applied,
        "detect_hf_coeffs_retained_count": detect_hf_coeffs_retained_count,
        "detect_hf_retention_ratio": detect_hf_retention_ratio,
        "detect_hf_trace_digest": detect_hf_trace_digest,
        "hf_attestation_challenge_match_status": challenge_match_status,
        "hf_attestation_threshold_match_status": threshold_match_status,
        "hf_attestation_retained_count_match_status": retained_count_match_status,
        "hf_attestation_trace_consistency": trace_consistency,
    }


def _build_detect_lf_observability_fields(detect_lf_status: Any) -> Dict[str, Any]:
    """
    功能：根据 exact LF helper 状态构造 detect LF 可观测字段。

    Build append-only detect LF observability fields from the existing helper
    status string without changing LF scoring semantics.

    Args:
        detect_lf_status: Raw status value returned by the exact LF helper.

    Returns:
        Mapping containing detect_lf_status and, when applicable, exactly one of
        detect_lf_failure_reason or detect_lf_absent_reason.
    """
    if not isinstance(detect_lf_status, str):
        return {}

    normalized_status = detect_lf_status.strip()
    if not normalized_status:
        return {}

    observability_fields: Dict[str, Any] = {
        "detect_lf_status": normalized_status,
    }
    lower_status = normalized_status.lower()

    if normalized_status == "ok" or normalized_status.startswith("ok_") or "drift_detected" in lower_status:
        return observability_fields

    failure_prefixes = (
        "cfg_invalid_type",
        "tfs_spec_missing_or_invalid",
        "projection_matrix_missing",
        "phi_dim_mismatch_",
        "lf_trajectory_score_failed:",
    )
    if any(lower_status.startswith(prefix) for prefix in failure_prefixes):
        observability_fields["detect_lf_failure_reason"] = normalized_status
        return observability_fields

    absent_statuses = {
        "no_trajectory_cache",
        "lf_basis_missing",
        "lf_plan_digest_missing_cannot_derive_codeword",
    }
    if (
        normalized_status in absent_statuses
        or normalized_status.startswith("trajectory_latent_absent:")
        or lower_status.startswith("absent_")
        or "absent" in lower_status
    ):
        observability_fields["detect_lf_absent_reason"] = normalized_status
        return observability_fields

    observability_fields["detect_lf_failure_reason"] = normalized_status
    return observability_fields


def _build_lf_attestation_trace_context(
    cfg: Dict[str, Any],
    plan_payload: Optional[Dict[str, Any]],
    lf_trace_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any] | None:
    plan_dict = _resolve_plan_dict(plan_payload)
    lf_basis_for_decode = plan_dict.get("lf_basis") if isinstance(plan_dict.get("lf_basis"), dict) else None
    if not isinstance(lf_basis_for_decode, dict):
        return None

    trajectory_feature_spec = lf_basis_for_decode.get("trajectory_feature_spec")
    if not isinstance(trajectory_feature_spec, dict):
        return None

    watermark_node = cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
    lf_cfg = cast(Dict[str, Any], watermark_cfg.get("lf")) if isinstance(watermark_cfg.get("lf"), dict) else {}

    basis_matrix_raw = lf_basis_for_decode.get("projection_matrix")
    basis_rank = lf_basis_for_decode.get("basis_rank")
    if basis_rank is None and basis_matrix_raw is not None:
        basis_matrix_np = np.asarray(basis_matrix_raw, dtype=np.float64)
        basis_rank = int(basis_matrix_np.shape[1])

    trace_bundle = lf_trace_bundle if isinstance(lf_trace_bundle, dict) else {}
    attestation_node = cfg.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}

    return {
        "variance": float(lf_cfg.get("variance", 1.5)),
        "message_length": int(lf_cfg.get("message_length", 64)),
        "ecc_sparsity": int(lf_cfg.get("ecc_sparsity", 3)),
        "basis_rank": int(basis_rank) if isinstance(basis_rank, (int, float)) else None,
        "edit_timestep": int(trajectory_feature_spec.get("edit_timestep", 0)),
        "trajectory_feature_spec": trajectory_feature_spec,
        "trajectory_feature_vector": trace_bundle.get("trajectory_feature_vector"),
        "trajectory_feature_digest": trace_bundle.get("trajectory_feature_digest"),
        "projected_lf_digest": trace_bundle.get("projected_lf_digest"),
        "plan_digest": trace_bundle.get("plan_digest"),
        "lf_basis_digest": trace_bundle.get("lf_basis_digest"),
        "basis_digest": trace_bundle.get("lf_basis_digest"),
        "projection_matrix_digest": trace_bundle.get("projection_matrix_digest"),
        "trajectory_feature_spec_digest": trace_bundle.get("trajectory_feature_spec_digest"),
        "projection_seed": trace_bundle.get("projection_seed"),
        "formal_exact_evidence_source": trace_bundle.get("formal_exact_evidence_source"),
        "event_binding_mode": "trajectory_bound" if bool(attestation_cfg.get("use_trajectory_mix", True)) else "statement_only",
    }


def _extract_embed_lf_closed_loop_context(input_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(input_record, dict):
        return {}
    embed_content_evidence = input_record.get("content_evidence")
    if not isinstance(embed_content_evidence, dict):
        embed_content_evidence = input_record.get("content_result")
    if not isinstance(embed_content_evidence, dict):
        return {}
    injection_metrics = embed_content_evidence.get("injection_metrics")
    if not isinstance(injection_metrics, dict):
        return {}
    lf_closed_loop_summary = injection_metrics.get("lf_closed_loop_summary")
    if not isinstance(lf_closed_loop_summary, dict):
        return {}
    edit_timestep_summary = injection_metrics.get("lf_edit_timestep_closed_loop_summary")
    if not isinstance(edit_timestep_summary, dict):
        edit_timestep_summary = {}
    terminal_step_summary = injection_metrics.get("lf_terminal_step_closed_loop_summary")
    if not isinstance(terminal_step_summary, dict):
        terminal_step_summary = {}
    return {
        "pre_injection_coeffs": lf_closed_loop_summary.get("pre_injection_coeffs"),
        "injected_template_coeffs": lf_closed_loop_summary.get("injected_template_coeffs"),
        "post_injection_coeffs": lf_closed_loop_summary.get("post_injection_coeffs"),
        "selected_step_post_coeffs": lf_closed_loop_summary.get("post_injection_coeffs"),
        "embed_edit_timestep_coeffs": edit_timestep_summary.get("post_injection_coeffs"),
        "embed_terminal_step_coeffs": terminal_step_summary.get("post_injection_coeffs"),
        "embed_expected_bit_signs": lf_closed_loop_summary.get("expected_bit_signs"),
        "embed_codeword_source": lf_closed_loop_summary.get("codeword_source"),
        "embed_attestation_event_digest": lf_closed_loop_summary.get("attestation_event_digest"),
        "embed_event_binding_mode": lf_closed_loop_summary.get("event_binding_mode"),
        "embed_basis_digest": lf_closed_loop_summary.get("basis_digest"),
        "embed_basis_binding_status": lf_closed_loop_summary.get("basis_binding_status"),
        "embed_closed_loop_digest": injection_metrics.get("lf_closed_loop_digest"),
        "embed_closed_loop_step_index": injection_metrics.get("lf_closed_loop_step_index"),
        "embed_closed_loop_selection_rule": injection_metrics.get("lf_closed_loop_selection_rule"),
        "embed_edit_timestep_step_index": injection_metrics.get("lf_edit_timestep_step_index"),
        "embed_terminal_step_index": injection_metrics.get("lf_terminal_step_index"),
    }


def _build_lf_alignment_table_artifact(
    lf_result: Dict[str, Any],
    *,
    attestation_digest: str,
    event_binding_digest: str,
    trace_commit: Optional[str],
) -> Dict[str, Any] | None:
    if not isinstance(lf_result, dict) or lf_result.get("status") != "ok":
        return None

    pre_coeffs = lf_result.get("pre_injection_coeffs")
    template_coeffs = lf_result.get("injected_template_coeffs")
    post_coeffs = lf_result.get("post_injection_coeffs")
    detect_coeffs = lf_result.get("detect_exact_timestep_coeffs")
    if not isinstance(detect_coeffs, list):
        detect_coeffs = lf_result.get("projected_lf_coeffs")
    expected_bit_signs = lf_result.get("expected_bit_signs")
    embed_expected_bit_signs = lf_result.get("embed_expected_bit_signs")
    if not all(isinstance(value, list) for value in [pre_coeffs, template_coeffs, post_coeffs, detect_coeffs, expected_bit_signs]):
        return None

    n_compare = int(lf_result.get("n_bits_compared") or 0)
    n_rows = min(n_compare, len(pre_coeffs), len(template_coeffs), len(post_coeffs), len(detect_coeffs), len(expected_bit_signs))
    if n_rows <= 0:
        return None

    expected = [int(expected_bit_signs[index]) for index in range(n_rows)]
    embed_expected = None
    embed_formal_expected_signs_match = False
    expected_signs_mismatch_reason = None
    if isinstance(embed_expected_bit_signs, list):
        embed_values = [int(embed_expected_bit_signs[index]) for index in range(min(n_rows, len(embed_expected_bit_signs)))]
        embed_expected = embed_values
        if len(embed_values) == n_rows:
            embed_formal_expected_signs_match = embed_values == expected
            if not embed_formal_expected_signs_match:
                expected_signs_mismatch_reason = "embed_formal_sign_values_differ"
        else:
            expected_signs_mismatch_reason = "embed_formal_sign_length_mismatch"
    else:
        expected_signs_mismatch_reason = "embed_expected_bit_signs_absent"
    pre_values = [float(pre_coeffs[index]) for index in range(n_rows)]
    template_values = [float(template_coeffs[index]) for index in range(n_rows)]
    post_values = [float(post_coeffs[index]) for index in range(n_rows)]
    detect_values = [float(detect_coeffs[index]) for index in range(n_rows)]

    signed_pre_alignment = [float(pre_values[index] * expected[index]) for index in range(n_rows)]
    signed_template_alignment = [float(template_values[index] * expected[index]) for index in range(n_rows)]
    signed_post_alignment = [float(post_values[index] * expected[index]) for index in range(n_rows)]
    signed_detect_alignment = [float(detect_values[index] * expected[index]) for index in range(n_rows)]

    template_margins = [abs(value) for value in signed_template_alignment if abs(value) > 0.0]
    alignment_margin_threshold = float(np.median(np.asarray(template_margins, dtype=np.float64)) * 0.5) if template_margins else 0.0
    pre_agreement_count = sum(1 for value in signed_pre_alignment if value > 0.0)
    post_agreement_count = sum(1 for value in signed_post_alignment if value > 0.0)
    detect_agreement_count = sum(1 for value in signed_detect_alignment if value > 0.0)
    strong_negative_pre_count = sum(1 for value in signed_pre_alignment if value < -alignment_margin_threshold)
    strong_negative_post_count = sum(1 for value in signed_post_alignment if value < -alignment_margin_threshold)
    strong_negative_detect_count = sum(1 for value in signed_detect_alignment if value < -alignment_margin_threshold)
    post_still_negative_count = sum(1 for value in signed_post_alignment if value < 0.0)
    post_crosses_target_halfspace_count = sum(
        1 for pre_value, post_value in zip(signed_pre_alignment, signed_post_alignment) if pre_value <= 0.0 and post_value > 0.0
    )
    detect_crosses_target_halfspace_count = sum(
        1 for post_value, detect_value in zip(signed_post_alignment, signed_detect_alignment) if post_value <= 0.0 and detect_value > 0.0
    )
    detect_reverted_after_post_positive_count = sum(
        1 for post_value, detect_value in zip(signed_post_alignment, signed_detect_alignment) if post_value > 0.0 and detect_value < 0.0
    )

    artifact = {
        "artifact_type": "lf_alignment_table",
        "attestation_digest": attestation_digest,
        "event_binding_digest": event_binding_digest,
        "trace_commit": trace_commit,
        "plan_digest": lf_result.get("plan_digest"),
        "lf_basis_digest": lf_result.get("lf_basis_digest"),
        "projection_matrix_digest": lf_result.get("projection_matrix_digest"),
        "embed_closed_loop_digest": lf_result.get("embed_closed_loop_digest"),
        "embed_closed_loop_step_index": lf_result.get("embed_closed_loop_step_index"),
        "embed_closed_loop_selection_rule": lf_result.get("embed_closed_loop_selection_rule"),
        "n_bits_compared": n_rows,
        "expected_bit_signs": expected,
        "embed_expected_bit_signs": embed_expected,
        "formal_expected_bit_signs": expected,
        "embed_formal_expected_signs_match": embed_formal_expected_signs_match,
        "expected_signs_mismatch_reason": expected_signs_mismatch_reason,
        "pre_injection_coeffs": pre_values,
        "injected_template_coeffs": template_values,
        "post_injection_coeffs": post_values,
        "detect_side_coeffs": detect_values,
        "signed_pre_alignment": signed_pre_alignment,
        "signed_template_alignment": signed_template_alignment,
        "signed_post_alignment": signed_post_alignment,
        "signed_detect_alignment": signed_detect_alignment,
        "alignment_margin_threshold": alignment_margin_threshold,
        "pre_agreement_count": pre_agreement_count,
        "post_agreement_count": post_agreement_count,
        "detect_agreement_count": detect_agreement_count,
        "strong_negative_pre_count": strong_negative_pre_count,
        "strong_negative_post_count": strong_negative_post_count,
        "strong_negative_detect_count": strong_negative_detect_count,
        "post_still_negative_count": post_still_negative_count,
        "post_crosses_target_halfspace_count": post_crosses_target_halfspace_count,
        "detect_crosses_target_halfspace_count": detect_crosses_target_halfspace_count,
        "detect_reverted_after_post_positive_count": detect_reverted_after_post_positive_count,
    }
    artifact["lf_alignment_table_digest"] = digests.canonical_sha256(artifact)
    return artifact


def _build_lf_retain_breakdown_artifact(
    lf_result: Dict[str, Any],
    *,
    attestation_digest: str,
    event_binding_digest: str,
    trace_commit: Optional[str],
) -> Dict[str, Any] | None:
    if not isinstance(lf_result, dict) or lf_result.get("status") != "ok":
        return None

    pre_coeffs = lf_result.get("pre_injection_coeffs")
    selected_step_post_coeffs = lf_result.get("selected_step_post_coeffs")
    if not isinstance(selected_step_post_coeffs, list):
        selected_step_post_coeffs = lf_result.get("post_injection_coeffs")
    edit_step_coeffs = lf_result.get("embed_edit_timestep_coeffs")
    terminal_step_coeffs = lf_result.get("embed_terminal_step_coeffs")
    detect_coeffs = lf_result.get("detect_exact_timestep_coeffs")
    if not isinstance(detect_coeffs, list):
        detect_coeffs = lf_result.get("projected_lf_coeffs")
    expected_bit_signs = lf_result.get("expected_bit_signs")
    injected_template_coeffs = lf_result.get("injected_template_coeffs")

    required_lists = [
        pre_coeffs,
        selected_step_post_coeffs,
        edit_step_coeffs,
        terminal_step_coeffs,
        detect_coeffs,
        expected_bit_signs,
    ]
    if not all(isinstance(value, list) for value in required_lists):
        return None

    n_compare = int(lf_result.get("n_bits_compared") or 0)
    n_rows = min(
        n_compare,
        len(cast(list[Any], pre_coeffs)),
        len(cast(list[Any], selected_step_post_coeffs)),
        len(cast(list[Any], edit_step_coeffs)),
        len(cast(list[Any], terminal_step_coeffs)),
        len(cast(list[Any], detect_coeffs)),
        len(cast(list[Any], expected_bit_signs)),
    )
    if n_rows <= 0:
        return None

    expected = [int(cast(list[Any], expected_bit_signs)[index]) for index in range(n_rows)]
    pre_values = [float(cast(list[Any], pre_coeffs)[index]) for index in range(n_rows)]
    selected_values = [float(cast(list[Any], selected_step_post_coeffs)[index]) for index in range(n_rows)]
    edit_values = [float(cast(list[Any], edit_step_coeffs)[index]) for index in range(n_rows)]
    terminal_values = [float(cast(list[Any], terminal_step_coeffs)[index]) for index in range(n_rows)]
    detect_values = [float(cast(list[Any], detect_coeffs)[index]) for index in range(n_rows)]

    stage_coeffs = {
        "pre_injection": pre_values,
        "selected_step_post": selected_values,
        "embed_edit_timestep": edit_values,
        "embed_terminal_step": terminal_values,
        "detect_exact_timestep": detect_values,
    }
    stage_signed_alignment = {
        stage_name: [float(values[index] * expected[index]) for index in range(n_rows)]
        for stage_name, values in stage_coeffs.items()
    }

    template_values = None
    if isinstance(injected_template_coeffs, list):
        template_values = [
            float(cast(list[Any], injected_template_coeffs)[index])
            for index in range(min(n_rows, len(cast(list[Any], injected_template_coeffs))))
        ]
    template_margins = [abs(value) for value in (template_values or []) if abs(value) > 0.0]
    alignment_margin_threshold = float(np.median(np.asarray(template_margins, dtype=np.float64)) * 0.5) if template_margins else 0.0

    def _build_stage_summary(stage_name: str) -> Dict[str, Any]:
        signed_values = stage_signed_alignment[stage_name]
        values_np = np.asarray(signed_values, dtype=np.float64)
        positive_count = sum(1 for value in signed_values if value > 0.0)
        strong_negative_count = sum(1 for value in signed_values if value < -alignment_margin_threshold)
        return {
            "stage_name": stage_name,
            "positive_count": positive_count,
            "positive_ratio": float(positive_count / float(n_rows)),
            "strong_negative_count": strong_negative_count,
            "mean_signed_alignment": float(np.mean(values_np)),
            "median_signed_alignment": float(np.median(values_np)),
        }

    def _build_segment(segment_name: str, from_stage: str, to_stage: str) -> Dict[str, Any]:
        from_values = stage_signed_alignment[from_stage]
        to_values = stage_signed_alignment[to_stage]
        deltas = [float(to_values[index] - from_values[index]) for index in range(n_rows)]
        from_positive_count = sum(1 for value in from_values if value > 0.0)
        to_positive_count = sum(1 for value in to_values if value > 0.0)
        retained_positive_count = sum(
            1 for from_value, to_value in zip(from_values, to_values) if from_value > 0.0 and to_value > 0.0
        )
        gained_positive_count = sum(
            1 for from_value, to_value in zip(from_values, to_values) if from_value <= 0.0 and to_value > 0.0
        )
        lost_positive_count = sum(
            1 for from_value, to_value in zip(from_values, to_values) if from_value > 0.0 and to_value <= 0.0
        )
        sign_flip_count = sum(
            1
            for from_value, to_value in zip(from_values, to_values)
            if (from_value > 0.0 and to_value <= 0.0) or (from_value <= 0.0 and to_value > 0.0)
        )
        retained_positive_ratio = None
        if from_positive_count > 0:
            retained_positive_ratio = float(retained_positive_count / float(from_positive_count))
        deltas_np = np.asarray(deltas, dtype=np.float64)
        return {
            "segment_name": segment_name,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "from_positive_count": from_positive_count,
            "to_positive_count": to_positive_count,
            "retained_positive_count": retained_positive_count,
            "retained_positive_ratio": retained_positive_ratio,
            "gained_positive_count": gained_positive_count,
            "lost_positive_count": lost_positive_count,
            "sign_flip_count": sign_flip_count,
            "mean_signed_delta": float(np.mean(deltas_np)),
            "median_signed_delta": float(np.median(deltas_np)),
        }

    stage_summaries = {
        stage_name: _build_stage_summary(stage_name)
        for stage_name in stage_coeffs.keys()
    }
    breakdown_segments = [
        _build_segment("pre_to_selected_step", "pre_injection", "selected_step_post"),
        _build_segment("selected_step_to_edit_timestep", "selected_step_post", "embed_edit_timestep"),
        _build_segment("edit_timestep_to_terminal", "embed_edit_timestep", "embed_terminal_step"),
        _build_segment("terminal_to_detect_exact_timestep", "embed_terminal_step", "detect_exact_timestep"),
        _build_segment("pre_to_detect_exact_timestep", "pre_injection", "detect_exact_timestep"),
    ]
    drift_segments = {
        segment["segment_name"]: segment
        for segment in breakdown_segments
        if segment["segment_name"] in {
            "selected_step_to_edit_timestep",
            "edit_timestep_to_terminal",
            "terminal_to_detect_exact_timestep",
        }
    }
    selected_step_mismatch_lost_positive_count = drift_segments.get("selected_step_to_edit_timestep", {}).get("lost_positive_count")
    embed_tail_drift_lost_positive_count = drift_segments.get("edit_timestep_to_terminal", {}).get("lost_positive_count")
    detect_reconstruction_drift_lost_positive_count = drift_segments.get("terminal_to_detect_exact_timestep", {}).get("lost_positive_count")
    pre_to_detect_retained_positive_ratio = breakdown_segments[-1].get("retained_positive_ratio")
    dominant_drift_segment = None
    if drift_segments:
        max_lost_positive_count = max(int(segment.get("lost_positive_count") or 0) for segment in drift_segments.values())
        if max_lost_positive_count <= 0:
            dominant_drift_segment = "no_positive_loss_detected"
        else:
            dominant_drift_segment = max(
                drift_segments.values(),
                key=lambda item: (
                    int(item.get("lost_positive_count") or 0),
                    int(item.get("sign_flip_count") or 0),
                ),
            ).get("segment_name")

    artifact = {
        "artifact_type": "lf_retain_breakdown",
        "attestation_digest": attestation_digest,
        "event_binding_digest": event_binding_digest,
        "trace_commit": trace_commit,
        "plan_digest": lf_result.get("plan_digest"),
        "lf_basis_digest": lf_result.get("lf_basis_digest"),
        "projection_matrix_digest": lf_result.get("projection_matrix_digest"),
        "embed_closed_loop_digest": lf_result.get("embed_closed_loop_digest"),
        "embed_closed_loop_step_index": lf_result.get("embed_closed_loop_step_index"),
        "embed_closed_loop_selection_rule": lf_result.get("embed_closed_loop_selection_rule"),
        "edit_timestep": lf_result.get("edit_timestep"),
        "embed_edit_timestep_step_index": lf_result.get("embed_edit_timestep_step_index"),
        "embed_terminal_step_index": lf_result.get("embed_terminal_step_index"),
        "n_bits_compared": n_rows,
        "expected_bit_signs": expected,
        "pre_injection_coeffs": pre_values,
        "selected_step_post_coeffs": selected_values,
        "embed_edit_timestep_coeffs": edit_values,
        "embed_terminal_step_coeffs": terminal_values,
        "detect_exact_timestep_coeffs": detect_values,
        "formal_exact_evidence_source": lf_result.get("formal_exact_evidence_source"),
        "formal_exact_object_binding_status": lf_result.get("formal_exact_object_binding_status"),
        "formal_exact_image_path_source": lf_result.get("formal_exact_image_path_source"),
        "lf_exact_repair_enabled": lf_result.get("lf_exact_repair_enabled"),
        "lf_exact_repair_mode": lf_result.get("lf_exact_repair_mode"),
        "lf_exact_repair_applied": lf_result.get("lf_exact_repair_applied"),
        "lf_exact_repair_summary": lf_result.get("lf_exact_repair_summary"),
        "stage_summaries": stage_summaries,
        "breakdown_segments": breakdown_segments,
        "breakdown_summary": {
            "selected_step_mismatch_lost_positive_count": selected_step_mismatch_lost_positive_count,
            "embed_tail_drift_lost_positive_count": embed_tail_drift_lost_positive_count,
            "detect_reconstruction_drift_lost_positive_count": detect_reconstruction_drift_lost_positive_count,
            "pre_to_detect_retained_positive_ratio": pre_to_detect_retained_positive_ratio,
            "dominant_drift_segment": dominant_drift_segment,
        },
    }
    protocol_control_section = _build_lf_protocol_control_section(
        lf_result,
        expected_bit_signs=expected,
        terminal_values=terminal_values,
        detect_values=detect_values,
    )
    if isinstance(protocol_control_section, dict):
        artifact.update(protocol_control_section)
        cross_seed_protocol_loss_count = int(artifact.get("cross_seed_protocol_loss_count") or 0)
        same_seed_residual_loss_count = int(artifact.get("same_seed_residual_loss_count") or 0)
        same_seed_residual_loss_ratio = artifact.get("same_seed_residual_loss_ratio")
        if (
            int(detect_reconstruction_drift_lost_positive_count or 0) <= 0
            and isinstance(pre_to_detect_retained_positive_ratio, (int, float))
            and float(pre_to_detect_retained_positive_ratio) >= 0.95
        ):
            if cross_seed_protocol_loss_count <= 0 and same_seed_residual_loss_count <= 0:
                artifact["protocol_root_cause_classification"] = "no_protocol_loss_detected"
            elif (
                cross_seed_protocol_loss_count <= 0
                and same_seed_residual_loss_count > 0
                and isinstance(same_seed_residual_loss_ratio, (int, float))
                and float(same_seed_residual_loss_ratio) <= 0.05
            ):
                artifact["protocol_root_cause_classification"] = "minor_same_seed_residual_not_primary"
            else:
                artifact["protocol_root_cause_classification"] = artifact.get("protocol_root_cause_classification")
            control_protocol_summary = artifact.get("control_protocol_summary")
            if isinstance(control_protocol_summary, dict):
                control_protocol_summary["protocol_root_cause_classification"] = artifact.get("protocol_root_cause_classification")
    artifact["lf_retain_breakdown_digest"] = digests.canonical_sha256(artifact)
    return artifact


def _build_lf_protocol_control_section(
    lf_result: Dict[str, Any],
    *,
    expected_bit_signs: list[int],
    terminal_values: list[float],
    detect_values: list[float],
) -> Dict[str, Any] | None:
    protocol_classification = lf_result.get("detect_protocol_classification")
    if not isinstance(protocol_classification, str):
        protocol_classification = "unknown"

    embed_seed = lf_result.get("embed_seed") if isinstance(lf_result.get("embed_seed"), int) else None
    detect_seed = lf_result.get("detect_seed") if isinstance(lf_result.get("detect_seed"), int) else None
    same_seed_available = bool(lf_result.get("same_seed_as_embed_available", False))
    same_seed_value = lf_result.get("same_seed_as_embed_value") if isinstance(lf_result.get("same_seed_as_embed_value"), int) else None
    same_seed_control_status = lf_result.get("same_seed_control_status") if isinstance(lf_result.get("same_seed_control_status"), str) else None
    same_seed_control_reason = lf_result.get("same_seed_control_reason") if isinstance(lf_result.get("same_seed_control_reason"), str) else None
    image_conditioned_reconstruction_available = bool(lf_result.get("image_conditioned_reconstruction_available", False))
    image_conditioned_reconstruction_status = (
        lf_result.get("image_conditioned_reconstruction_status")
        if isinstance(lf_result.get("image_conditioned_reconstruction_status"), str)
        else "not_implemented"
    )

    same_seed_control_coeffs = lf_result.get("detect_exact_timestep_coeffs_same_seed_control")
    if not isinstance(same_seed_control_coeffs, list) and bool(lf_result.get("same_seed_control_reused_formal_detect", False)):
        same_seed_control_coeffs = detect_values
    if isinstance(same_seed_control_coeffs, list):
        same_seed_control_coeffs = [float(value) for value in same_seed_control_coeffs]
    else:
        same_seed_control_coeffs = None

    def _build_transition(segment_name: str, from_signed_values: list[float], to_signed_values: list[float]) -> Dict[str, Any]:
        from_positive_count = sum(1 for value in from_signed_values if value > 0.0)
        to_positive_count = sum(1 for value in to_signed_values if value > 0.0)
        retained_positive_count = sum(
            1 for from_value, to_value in zip(from_signed_values, to_signed_values) if from_value > 0.0 and to_value > 0.0
        )
        lost_positive_count = sum(
            1 for from_value, to_value in zip(from_signed_values, to_signed_values) if from_value > 0.0 and to_value <= 0.0
        )
        gained_positive_count = sum(
            1 for from_value, to_value in zip(from_signed_values, to_signed_values) if from_value <= 0.0 and to_value > 0.0
        )
        sign_flip_count = sum(
            1
            for from_value, to_value in zip(from_signed_values, to_signed_values)
            if (from_value > 0.0 and to_value <= 0.0) or (from_value <= 0.0 and to_value > 0.0)
        )
        retained_positive_ratio = None
        if from_positive_count > 0:
            retained_positive_ratio = float(retained_positive_count / float(from_positive_count))
        delta_array = np.asarray(
            [float(to_value - from_value) for from_value, to_value in zip(from_signed_values, to_signed_values)],
            dtype=np.float64,
        )
        return {
            "segment_name": segment_name,
            "from_positive_count": from_positive_count,
            "to_positive_count": to_positive_count,
            "retained_positive_count": retained_positive_count,
            "retained_positive_ratio": retained_positive_ratio,
            "lost_positive_count": lost_positive_count,
            "gained_positive_count": gained_positive_count,
            "sign_flip_count": sign_flip_count,
            "mean_signed_delta": float(np.mean(delta_array)),
            "median_signed_delta": float(np.median(delta_array)),
        }

    result: Dict[str, Any] = {
        "embed_seed": embed_seed,
        "detect_seed": detect_seed,
        "same_seed_as_embed_available": same_seed_available,
        "same_seed_as_embed_value": same_seed_value,
        "detect_protocol_classification": protocol_classification,
        "image_conditioned_reconstruction_available": image_conditioned_reconstruction_available,
        "image_conditioned_reconstruction_status": image_conditioned_reconstruction_status,
        "same_seed_control_status": same_seed_control_status,
        "same_seed_control_reason": same_seed_control_reason,
        "same_seed_control_trace_digest": lf_result.get("same_seed_control_trace_digest"),
        "same_seed_control_trajectory_digest": lf_result.get("same_seed_control_trajectory_digest"),
        "detect_exact_timestep_coeffs_same_seed_control": same_seed_control_coeffs,
        eval_metrics.LF_CHANNEL_SCORE_NAME: lf_result.get(eval_metrics.LF_CHANNEL_SCORE_NAME),
        eval_metrics.LF_CORRELATION_SCORE_NAME: lf_result.get(eval_metrics.LF_CORRELATION_SCORE_NAME),
        "lf_attestation_score": lf_result.get("lf_attestation_score"),
        "cross_seed_protocol_loss_count": None,
        "same_seed_residual_loss_count": None,
        "cross_seed_protocol_loss_ratio": None,
        "same_seed_residual_loss_ratio": None,
        "protocol_root_cause_classification": "inconclusive",
        "control_protocol_segments": [],
        "control_protocol_summary": {
            "status": same_seed_control_status,
            "reason": same_seed_control_reason,
            "protocol_root_cause_classification": "inconclusive",
        },
    }

    if not same_seed_available or not isinstance(same_seed_control_coeffs, list):
        return result

    n_rows = min(len(expected_bit_signs), len(terminal_values), len(detect_values), len(same_seed_control_coeffs))
    if n_rows <= 0:
        return result

    terminal_signed = [float(terminal_values[index] * expected_bit_signs[index]) for index in range(n_rows)]
    detect_signed = [float(detect_values[index] * expected_bit_signs[index]) for index in range(n_rows)]
    same_seed_signed = [float(same_seed_control_coeffs[index] * expected_bit_signs[index]) for index in range(n_rows)]

    control_segments = [
        _build_transition("embed_terminal_to_detect_exact_detect_seed", terminal_signed, detect_signed),
        _build_transition("embed_terminal_to_detect_exact_embed_seed_control", terminal_signed, same_seed_signed),
        _build_transition("detect_exact_embed_seed_control_to_detect_exact_detect_seed", same_seed_signed, detect_signed),
    ]

    terminal_positive_count = sum(1 for value in terminal_signed if value > 0.0)
    same_seed_residual_loss_count = sum(
        1 for terminal_value, same_seed_value_item in zip(terminal_signed, same_seed_signed) if terminal_value > 0.0 and same_seed_value_item <= 0.0
    )
    cross_seed_protocol_loss_count = sum(
        1
        for terminal_value, same_seed_value_item, detect_value_item in zip(terminal_signed, same_seed_signed, detect_signed)
        if terminal_value > 0.0 and same_seed_value_item > 0.0 and detect_value_item <= 0.0
    )
    cross_seed_protocol_loss_ratio = None
    same_seed_residual_loss_ratio = None
    if terminal_positive_count > 0:
        cross_seed_protocol_loss_ratio = float(cross_seed_protocol_loss_count / float(terminal_positive_count))
        same_seed_residual_loss_ratio = float(same_seed_residual_loss_count / float(terminal_positive_count))

    protocol_root_cause_classification = "inconclusive"
    if cross_seed_protocol_loss_count <= 0 and same_seed_residual_loss_count <= 0:
        protocol_root_cause_classification = "inconclusive"
    elif cross_seed_protocol_loss_count > same_seed_residual_loss_count:
        protocol_root_cause_classification = "cross_seed_rerun_mismatch_dominant"
    elif same_seed_residual_loss_count > cross_seed_protocol_loss_count:
        protocol_root_cause_classification = "same_seed_residual_drift_dominant"
    elif cross_seed_protocol_loss_count > 0 and same_seed_residual_loss_count > 0:
        protocol_root_cause_classification = "mixed"

    result.update(
        {
            "cross_seed_protocol_loss_count": cross_seed_protocol_loss_count,
            "same_seed_residual_loss_count": same_seed_residual_loss_count,
            "cross_seed_protocol_loss_ratio": cross_seed_protocol_loss_ratio,
            "same_seed_residual_loss_ratio": same_seed_residual_loss_ratio,
            "protocol_root_cause_classification": protocol_root_cause_classification,
            "control_protocol_segments": control_segments,
            "control_protocol_summary": {
                "terminal_positive_count": terminal_positive_count,
                "cross_seed_protocol_loss_count": cross_seed_protocol_loss_count,
                "same_seed_residual_loss_count": same_seed_residual_loss_count,
                "cross_seed_protocol_loss_ratio": cross_seed_protocol_loss_ratio,
                "same_seed_residual_loss_ratio": same_seed_residual_loss_ratio,
                "protocol_root_cause_classification": protocol_root_cause_classification,
                "same_seed_control_status": same_seed_control_status,
                "same_seed_control_reason": same_seed_control_reason,
            },
        }
    )
    return result


def _extract_planner_posterior_context(input_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(input_record, dict):
        return {}

    plan_stats_node = input_record.get("plan_stats")
    plan_stats = cast(Dict[str, Any], plan_stats_node) if isinstance(plan_stats_node, dict) else {}
    route_basis_bridge_node = plan_stats.get("route_basis_bridge")
    route_basis_bridge = cast(Dict[str, Any], route_basis_bridge_node) if isinstance(route_basis_bridge_node, dict) else {}
    subspace_plan_node = input_record.get("subspace_plan")
    subspace_plan = cast(Dict[str, Any], subspace_plan_node) if isinstance(subspace_plan_node, dict) else {}

    planner_rank = subspace_plan.get("rank")
    if not isinstance(planner_rank, int):
        planner_rank = plan_stats.get("rank") if isinstance(plan_stats.get("rank"), int) else None

    return {
        "planner_rank": planner_rank,
        "basis_digest": input_record.get("basis_digest"),
        "route_basis_bridge": route_basis_bridge,
        "lf_feature_count": len(route_basis_bridge.get("lf_feature_cols", [])) if isinstance(route_basis_bridge.get("lf_feature_cols"), list) else None,
        "lf_decomposition_shape": plan_stats.get("lf_basis_shape"),
    }


def _resolve_lf_dimension_routing(route_basis_bridge: Dict[str, Any], dimension_index: int) -> Dict[str, Any]:
    lf_feature_cols = route_basis_bridge.get("lf_feature_cols")
    feature_col = None
    if isinstance(lf_feature_cols, list) and dimension_index < len(lf_feature_cols):
        raw_feature_col = lf_feature_cols[dimension_index]
        if isinstance(raw_feature_col, int):
            feature_col = raw_feature_col

    route_layer_node = route_basis_bridge.get("route_layer")
    route_layer = cast(Dict[str, Any], route_layer_node) if isinstance(route_layer_node, dict) else {}
    feature_bridge_node = route_basis_bridge.get("feature_bridge_layer")
    feature_bridge_layer = cast(Dict[str, Any], feature_bridge_node) if isinstance(feature_bridge_node, dict) else {}

    routing_tag = None
    if isinstance(feature_col, int):
        routing_tag = f"lf_feature_col:{feature_col}"

    decomposition_group = route_layer.get("feature_routing_mode")
    if not isinstance(decomposition_group, str) or not decomposition_group:
        decomposition_group = feature_bridge_layer.get("route_to_feature_bridge")
    if not isinstance(decomposition_group, str) or not decomposition_group:
        decomposition_group = None

    return {
        "lf_feature_col": feature_col,
        "routing_tag": routing_tag,
        "decomposition_group": decomposition_group,
    }


def _classify_lf_posterior_risk(
    *,
    n_bits_compared: int,
    strong_negative_pre_count: int,
    post_still_negative_count: int,
    post_crosses_target_halfspace_count: int,
    detect_reverted_after_post_positive_count: int,
) -> str:
    if n_bits_compared <= 0:
        return "mixed"

    dominant_count_threshold = max(2, int(math.ceil(float(n_bits_compared) * 0.25)))
    host_baseline_dominant = (
        strong_negative_pre_count >= dominant_count_threshold
        and post_still_negative_count >= dominant_count_threshold
        and detect_reverted_after_post_positive_count < post_still_negative_count
    )
    detect_trajectory_shift = (
        post_crosses_target_halfspace_count >= dominant_count_threshold
        and detect_reverted_after_post_positive_count >= max(2, int(math.ceil(float(post_crosses_target_halfspace_count) * 0.5)))
    )
    basis_sample_mismatch = (
        strong_negative_pre_count < dominant_count_threshold
        and post_crosses_target_halfspace_count < dominant_count_threshold
        and post_still_negative_count >= dominant_count_threshold
    )

    if host_baseline_dominant and not detect_trajectory_shift and not basis_sample_mismatch:
        return "host_baseline_dominant"
    if detect_trajectory_shift and not host_baseline_dominant and not basis_sample_mismatch:
        return "detect_trajectory_shift"
    if basis_sample_mismatch and not host_baseline_dominant and not detect_trajectory_shift:
        return "basis_sample_mismatch"
    return "mixed"


def _compute_lf_report_auxiliary_metrics(
    *,
    signed_pre_alignment: list[float],
    signed_post_alignment: list[float],
    signed_detect_alignment: list[float],
    detect_side_coeffs: list[float],
    planner_rank: int | None,
) -> Dict[str, Any]:
    """
    功能：为 LF planner posterior report 补齐治理要求的辅助统计字段。

    Compute governed auxiliary statistics required by the LF planner report
    contract. These values are descriptive only and do not control the primary
    posterior classification.

    Args:
        signed_pre_alignment: Signed pre-injection alignment values.
        signed_post_alignment: Signed post-injection alignment values.
        signed_detect_alignment: Signed detect-side alignment values.
        detect_side_coeffs: Detect-side LF coefficients.
        planner_rank: Optional planner rank.

    Returns:
        Mapping with governed auxiliary metric fields.
    """
    pre_array = np.asarray(signed_pre_alignment, dtype=np.float64)
    post_array = np.asarray(signed_post_alignment, dtype=np.float64)
    detect_array = np.asarray(signed_detect_alignment, dtype=np.float64)
    coeff_array = np.abs(np.asarray(detect_side_coeffs, dtype=np.float64))

    negative_pre = float(np.sum(pre_array < 0.0))
    non_negative_pre = float(np.sum(pre_array >= 0.0))
    host_baseline_ratio = float(negative_pre / max(non_negative_pre, 1.0))

    sign_chain = np.stack(
        [
            np.sign(pre_array),
            np.sign(post_array),
            np.sign(detect_array),
        ],
        axis=0,
    )
    stable_mask = np.all(sign_chain == sign_chain[0], axis=0)
    sign_stability = float(np.mean(stable_mask.astype(np.float64))) if stable_mask.size > 0 else 0.0

    template_gain = np.maximum(np.abs(post_array) - np.abs(pre_array), 0.0)
    residual_mass = np.maximum(-post_array, 0.0) + np.maximum(-detect_array, 0.0)
    reconstruction_residual_ratio = float(np.sum(residual_mass) / max(np.sum(np.abs(post_array)) + np.sum(template_gain), 1e-8))
    reconstruction_residual_ratio = max(0.0, min(reconstruction_residual_ratio, 1.0))

    total_energy = float(np.sum(coeff_array))
    if total_energy <= 0.0:
        top1_energy_ratio = 0.0
        topk_energy_ratio = 0.0
    else:
        sorted_energy = np.sort(coeff_array)[::-1]
        top1_energy_ratio = float(sorted_energy[0] / total_energy)
        topk = max(1, min(int(planner_rank) if isinstance(planner_rank, int) and planner_rank > 0 else 1, int(sorted_energy.shape[0])))
        topk_energy_ratio = float(np.sum(sorted_energy[:topk]) / total_energy)

    return {
        "host_baseline_ratio": host_baseline_ratio,
        "sign_stability": sign_stability,
        "reconstruction_residual_ratio": reconstruction_residual_ratio,
        "top1_energy_ratio": top1_energy_ratio,
        "topk_energy_ratio": topk_energy_ratio,
    }


def _build_lf_planner_risk_report_artifact(
    lf_alignment_table: Dict[str, Any],
    planner_context: Optional[Dict[str, Any]],
) -> Dict[str, Any] | None:
    if not isinstance(lf_alignment_table, dict):
        return None

    expected_bit_signs = lf_alignment_table.get("expected_bit_signs")
    signed_pre_alignment = lf_alignment_table.get("signed_pre_alignment")
    signed_template_alignment = lf_alignment_table.get("signed_template_alignment")
    signed_post_alignment = lf_alignment_table.get("signed_post_alignment")
    signed_detect_alignment = lf_alignment_table.get("signed_detect_alignment")
    detect_side_coeffs = lf_alignment_table.get("detect_side_coeffs")
    if not all(
        isinstance(value, list)
        for value in [
            expected_bit_signs,
            signed_pre_alignment,
            signed_template_alignment,
            signed_post_alignment,
            signed_detect_alignment,
            detect_side_coeffs,
        ]
    ):
        return None

    n_bits_compared = int(lf_alignment_table.get("n_bits_compared") or 0)
    n_rows = min(
        n_bits_compared,
        len(expected_bit_signs),
        len(signed_pre_alignment),
        len(signed_template_alignment),
        len(signed_post_alignment),
        len(signed_detect_alignment),
        len(detect_side_coeffs),
    )
    if n_rows <= 0:
        return None

    embed_formal_expected_signs_match = lf_alignment_table.get("embed_formal_expected_signs_match")
    expected_signs_mismatch_reason = lf_alignment_table.get("expected_signs_mismatch_reason")
    has_sign_source_mismatch = (
        embed_formal_expected_signs_match is False
        or (
            embed_formal_expected_signs_match is None
            and isinstance(expected_signs_mismatch_reason, str)
            and bool(expected_signs_mismatch_reason)
        )
    )

    planner_context_mapping = cast(Dict[str, Any], planner_context) if isinstance(planner_context, dict) else {}
    route_basis_bridge_node = planner_context_mapping.get("route_basis_bridge")
    route_basis_bridge = cast(Dict[str, Any], route_basis_bridge_node) if isinstance(route_basis_bridge_node, dict) else {}
    route_basis_bridge_digest = digests.canonical_sha256(route_basis_bridge) if route_basis_bridge else None
    planner_rank = planner_context_mapping.get("planner_rank") if isinstance(planner_context_mapping.get("planner_rank"), int) else None
    alignment_margin_threshold = float(lf_alignment_table.get("alignment_margin_threshold") or 0.0)
    confidence_threshold = max(abs(alignment_margin_threshold), 1e-8)

    per_dimension_summary = []
    high_confidence_mismatch_dimensions = []
    mismatch_feature_col_counts: Dict[str, int] = {}
    mismatch_dimension_indices = []
    mismatch_feature_cols = []

    for dimension_index in range(n_rows):
        signed_pre = float(signed_pre_alignment[dimension_index])
        signed_template = float(signed_template_alignment[dimension_index])
        signed_post = float(signed_post_alignment[dimension_index])
        signed_detect = float(signed_detect_alignment[dimension_index])
        detect_value = float(detect_side_coeffs[dimension_index])
        routing_info = _resolve_lf_dimension_routing(route_basis_bridge, dimension_index)
        post_positive = signed_post > 0.0
        detect_positive = signed_detect > 0.0
        pre_strong_negative = signed_pre < -confidence_threshold
        post_still_negative = signed_post < 0.0
        detect_reverted = signed_post > 0.0 and signed_detect < 0.0
        is_high_confidence_mismatch = signed_detect < -confidence_threshold

        per_dimension_summary.append(
            {
                "dimension_index": dimension_index,
                "expected_bit_sign": int(expected_bit_signs[dimension_index]),
                "signed_pre_alignment": signed_pre,
                "signed_template_alignment": signed_template,
                "signed_post_alignment": signed_post,
                "signed_detect_alignment": signed_detect,
                "post_positive": post_positive,
                "detect_positive": detect_positive,
                "is_high_confidence_mismatch": is_high_confidence_mismatch,
                "routing_tag": routing_info.get("routing_tag"),
                "decomposition_group": routing_info.get("decomposition_group"),
            }
        )

        if is_high_confidence_mismatch:
            high_confidence_mismatch_dimensions.append(
                {
                    "dimension_index": dimension_index,
                    "signed_pre_alignment": signed_pre,
                    "signed_post_alignment": signed_post,
                    "signed_detect_alignment": signed_detect,
                    "detect_side_coeff": detect_value,
                    "pre_strong_negative": pre_strong_negative,
                    "post_still_negative": post_still_negative,
                    "detect_reverted_after_post_positive": detect_reverted,
                    "routing_tag": routing_info.get("routing_tag"),
                    "decomposition_group": routing_info.get("decomposition_group"),
                }
            )
            mismatch_dimension_indices.append(dimension_index)
            feature_col = routing_info.get("lf_feature_col")
            if isinstance(feature_col, int):
                mismatch_feature_cols.append(feature_col)
                feature_col_key = str(feature_col)
                mismatch_feature_col_counts[feature_col_key] = mismatch_feature_col_counts.get(feature_col_key, 0) + 1

    strong_negative_pre_count = int(lf_alignment_table.get("strong_negative_pre_count") or 0)
    post_still_negative_count = int(lf_alignment_table.get("post_still_negative_count") or 0)
    post_crosses_target_halfspace_count = int(lf_alignment_table.get("post_crosses_target_halfspace_count") or 0)
    detect_reverted_after_post_positive_count = int(lf_alignment_table.get("detect_reverted_after_post_positive_count") or 0)
    detect_crosses_target_halfspace_count = int(lf_alignment_table.get("detect_crosses_target_halfspace_count") or 0)

    if has_sign_source_mismatch:
        risk_classification = "sign_source_mismatch"
    else:
        risk_classification = _classify_lf_posterior_risk(
            n_bits_compared=n_rows,
            strong_negative_pre_count=strong_negative_pre_count,
            post_still_negative_count=post_still_negative_count,
            post_crosses_target_halfspace_count=post_crosses_target_halfspace_count,
            detect_reverted_after_post_positive_count=detect_reverted_after_post_positive_count,
        )

    dominant_signal = "mixed"
    dominant_count = max(
        strong_negative_pre_count,
        post_still_negative_count,
        detect_reverted_after_post_positive_count,
        post_crosses_target_halfspace_count,
    )
    if risk_classification == "host_baseline_dominant":
        dominant_signal = "host_baseline_counts"
    elif risk_classification == "detect_trajectory_shift":
        dominant_signal = "detect_reversion_counts"
    elif risk_classification == "basis_sample_mismatch":
        dominant_signal = "post_still_negative_counts"
    elif risk_classification == "sign_source_mismatch":
        dominant_signal = "sign_source_mismatch"

    route_layer_node = route_basis_bridge.get("route_layer") if isinstance(route_basis_bridge, dict) else None
    route_layer = cast(Dict[str, Any], route_layer_node) if isinstance(route_layer_node, dict) else {}
    routing_pattern_summary = {
        "route_basis_bridge_digest": route_basis_bridge_digest,
        "route_source": route_layer.get("route_source"),
        "feature_routing_mode": route_layer.get("feature_routing_mode"),
        "lf_feature_cols_source": route_layer.get("lf_feature_cols_source"),
        "region_index_digest": route_basis_bridge.get("region_index_digest") if isinstance(route_basis_bridge, dict) else None,
        "mismatch_dimension_count": len(high_confidence_mismatch_dimensions),
        "mismatch_dimension_indices": mismatch_dimension_indices,
        "mismatch_feature_cols": mismatch_feature_cols,
        "mismatch_feature_col_counts": mismatch_feature_col_counts,
    }
    host_baseline_risk_summary = {
        "strong_negative_pre_count": strong_negative_pre_count,
        "post_still_negative_count": post_still_negative_count,
        "detect_reverted_after_post_positive_count": detect_reverted_after_post_positive_count,
        "post_crosses_target_halfspace_count": post_crosses_target_halfspace_count,
        "detect_crosses_target_halfspace_count": detect_crosses_target_halfspace_count,
        "dominant_signal": dominant_signal,
        "dominant_count": dominant_count,
    }
    primary_evidence = {
        "evidence_type": "lf_closed_loop_posterior_counts",
        "risk_classification_driver": risk_classification,
        "dominant_signal": dominant_signal,
        "supporting_counts": host_baseline_risk_summary,
        "high_confidence_mismatch_count": len(high_confidence_mismatch_dimensions),
        "confidence_threshold": confidence_threshold,
    }
    if has_sign_source_mismatch:
        primary_evidence["sign_source_mismatch_reason"] = expected_signs_mismatch_reason
    auxiliary_metrics = _compute_lf_report_auxiliary_metrics(
        signed_pre_alignment=[float(value) for value in signed_pre_alignment[:n_rows]],
        signed_post_alignment=[float(value) for value in signed_post_alignment[:n_rows]],
        signed_detect_alignment=[float(value) for value in signed_detect_alignment[:n_rows]],
        detect_side_coeffs=[float(value) for value in detect_side_coeffs[:n_rows]],
        planner_rank=planner_rank,
    )
    host_baseline_dominant_flag = risk_classification == "host_baseline_dominant"
    basis_sample_mismatch_flag = risk_classification == "basis_sample_mismatch"
    detect_trajectory_shift_flag = risk_classification == "detect_trajectory_shift"

    return {
        "artifact_type": "lf_planner_risk_report",
        "risk_report_version": "v1",
        "risk_classification": risk_classification,
        "lf_feature_count": planner_context_mapping.get("lf_feature_count"),
        "lf_decomposition_shape": planner_context_mapping.get("lf_decomposition_shape"),
        "planner_rank": planner_rank,
        "host_baseline_ratio": auxiliary_metrics["host_baseline_ratio"],
        "sign_stability": auxiliary_metrics["sign_stability"],
        "reconstruction_residual_ratio": auxiliary_metrics["reconstruction_residual_ratio"],
        "top1_energy_ratio": auxiliary_metrics["top1_energy_ratio"],
        "topk_energy_ratio": auxiliary_metrics["topk_energy_ratio"],
        "host_baseline_dominant_flag": host_baseline_dominant_flag,
        "basis_sample_mismatch_flag": basis_sample_mismatch_flag,
        "detect_trajectory_shift_flag": detect_trajectory_shift_flag,
        "route_basis_bridge_digest": route_basis_bridge_digest,
        "plan_digest": lf_alignment_table.get("plan_digest"),
        "basis_digest": planner_context_mapping.get("basis_digest"),
        "embed_formal_expected_signs_match": embed_formal_expected_signs_match,
        "expected_signs_mismatch_reason": expected_signs_mismatch_reason,
        "primary_evidence": primary_evidence,
        "per_dimension_summary": per_dimension_summary,
        "high_confidence_mismatch_dimensions": high_confidence_mismatch_dimensions,
        "routing_pattern_summary": routing_pattern_summary,
        "host_baseline_risk_summary": host_baseline_risk_summary,
    }


def _build_lf_attestation_trace_artifact(
    lf_result: Dict[str, Any],
    *,
    attestation_digest: str,
    event_binding_digest: str,
    trace_commit: Optional[str],
) -> Dict[str, Any] | None:
    if not isinstance(lf_result, dict) or lf_result.get("status") != "ok":
        return None

    return {
        "artifact_type": "lf_attestation_trace",
        "attestation_digest": attestation_digest,
        "event_binding_digest": event_binding_digest,
        "trace_commit": trace_commit,
        eval_metrics.CONTENT_CHAIN_SCORE_NAME: lf_result.get(eval_metrics.CONTENT_CHAIN_SCORE_NAME),
        eval_metrics.LF_CHANNEL_SCORE_NAME: lf_result.get(eval_metrics.LF_CHANNEL_SCORE_NAME),
        eval_metrics.LF_CORRELATION_SCORE_NAME: lf_result.get(eval_metrics.LF_CORRELATION_SCORE_NAME),
        "lf_attestation_score": lf_result.get("lf_attestation_score"),
        "agreement_count": lf_result.get("agreement_count"),
        "n_bits_compared": lf_result.get("n_bits_compared"),
        "basis_rank": lf_result.get("basis_rank"),
        "variance": lf_result.get("variance"),
        "edit_timestep": lf_result.get("edit_timestep"),
        "trajectory_feature_spec": lf_result.get("trajectory_feature_spec"),
        "trajectory_feature_vector": lf_result.get("trajectory_feature_vector"),
        "trajectory_feature_digest": lf_result.get("trajectory_feature_digest"),
        "projected_lf_coeffs": lf_result.get("projected_lf_coeffs"),
        "projected_lf_signs": lf_result.get("projected_lf_signs"),
        "projected_lf_digest": lf_result.get("projected_lf_digest"),
        "expected_bit_signs": lf_result.get("expected_bit_signs"),
        "posterior_values": lf_result.get("posterior_values"),
        "posterior_signs": lf_result.get("posterior_signs"),
        "posterior_margin_values": lf_result.get("posterior_margin_values"),
        "agreement_indices": lf_result.get("agreement_indices"),
        "mismatch_indices": lf_result.get("mismatch_indices"),
        "weakest_posterior_indices": lf_result.get("weakest_posterior_indices"),
        "weakest_posterior_margins": lf_result.get("weakest_posterior_margins"),
        "plan_digest": lf_result.get("plan_digest"),
        "lf_basis_digest": lf_result.get("lf_basis_digest"),
        "projection_matrix_digest": lf_result.get("projection_matrix_digest"),
        "trajectory_feature_spec_digest": lf_result.get("trajectory_feature_spec_digest"),
        "projection_seed": lf_result.get("projection_seed"),
        "lf_exact_repair_enabled": lf_result.get("lf_exact_repair_enabled"),
        "lf_exact_repair_mode": lf_result.get("lf_exact_repair_mode"),
        "lf_exact_repair_applied": lf_result.get("lf_exact_repair_applied"),
        "lf_exact_repair_summary": lf_result.get("lf_exact_repair_summary"),
        "lf_attestation_trace_digest": lf_result.get("lf_attestation_trace_digest"),
    }


def _build_geo_rescue_scale_classification(
    *,
    quality_score: Optional[float],
    template_match_score: Optional[float],
    geo_score_source: str,
    geo_rescue_min_score: float,
) -> str:
    if quality_score is None and template_match_score is None:
        return "metrics_absent"
    if quality_score is None or template_match_score is None:
        return "partial_metrics"
    if geo_score_source == "template_match_score":
        if quality_score >= geo_rescue_min_score and template_match_score < geo_rescue_min_score:
            return "quality_pass_template_fail_source_template"
        if template_match_score > 0.0 and quality_score / template_match_score >= 1.5:
            return "quality_exceeds_template_source_template"
        return "quality_template_consistent_source_template"
    if geo_score_source == "quality_score":
        if quality_score >= geo_rescue_min_score and template_match_score < geo_rescue_min_score:
            return "quality_pass_template_fail_source_quality"
        return "quality_template_consistent_source_quality"
    if geo_score_source == "template_confidence":
        if quality_score >= geo_rescue_min_score and template_match_score < geo_rescue_min_score:
            return "quality_pass_template_fail_source_template_confidence"
        return "quality_template_consistent_source_template_confidence"
    return "geo_source_unclassified"


def _summarize_geo_distribution(values: list[float]) -> Dict[str, Any] | None:
    if len(values) == 0:
        return None
    values_array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": int(values_array.size),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "mean": float(np.mean(values_array)),
        "median": float(np.median(values_array)),
    }


def _classify_template_score_scale_band(
    template_match_score: Optional[float],
    template_match_internal_threshold: Optional[float],
    geo_rescue_min_score: float,
) -> str:
    if template_match_score is None or template_match_internal_threshold is None:
        return "inconclusive"
    if template_match_score < template_match_internal_threshold:
        return "below_internal_threshold"
    if template_match_score < geo_rescue_min_score:
        return "between_internal_threshold_and_rescue_gate"
    return "at_or_above_rescue_gate"


def _classify_rescue_gate_scale(
    template_match_internal_threshold: Optional[float],
    geo_rescue_min_score: float,
    positive_template_to_gate_max_ratio: Optional[float],
) -> str:
    if template_match_internal_threshold is None or geo_rescue_min_score <= 0.0:
        return "inconclusive"
    threshold_ratio = float(template_match_internal_threshold / geo_rescue_min_score)
    if positive_template_to_gate_max_ratio is not None and positive_template_to_gate_max_ratio >= 1.0:
        return "template_scale_sufficient_for_rescue"
    if threshold_ratio <= 0.25:
        return "template_internal_threshold_far_below_rescue_gate"
    if threshold_ratio < 1.0:
        return "template_scale_and_rescue_gate_comparable"
    return "template_scale_sufficient_for_rescue"


def _classify_geo_repair_direction(
    *,
    quality_score: Optional[float],
    template_match_score: Optional[float],
    geo_score_source: str,
    geo_rescue_min_score: float,
    positive_template_to_gate_max_ratio: Optional[float],
    positive_template_to_internal_threshold_max_ratio: Optional[float],
    rescue_gate_scale_classification: str,
) -> str:
    if (
        positive_template_to_internal_threshold_max_ratio is not None
        and positive_template_to_internal_threshold_max_ratio < 1.0
    ):
        return "template_score_extraction_itself_too_weak"
    if (
        rescue_gate_scale_classification == "template_internal_threshold_far_below_rescue_gate"
        and positive_template_to_gate_max_ratio is not None
        and positive_template_to_gate_max_ratio < 0.5
    ):
        return "scale_misalignment_between_template_score_and_rescue_gate"
    if (
        quality_score is not None
        and template_match_score is not None
        and quality_score >= geo_rescue_min_score
        and template_match_score < geo_rescue_min_score
        and geo_score_source == "template_confidence"
    ):
        return "template_confidence_rebinding_active"
    if (
        quality_score is not None
        and template_match_score is not None
        and quality_score >= geo_rescue_min_score
        and template_match_score < geo_rescue_min_score
        and geo_score_source == "template_match_score"
    ):
        return "quality_good_template_bad_need_score_rebinding_or_recalibration"
    return "inconclusive"


def _build_geo_rescue_diagnostics_artifact(
    cfg: Dict[str, Any],
    geometry_evidence_payload: Optional[Dict[str, Any]],
    attestation_result: Dict[str, Any],
    *,
    attestation_digest: Optional[str],
    event_binding_digest: Optional[str],
    trace_commit: Optional[str],
) -> Dict[str, Any] | None:
    if not isinstance(cfg, dict) or not isinstance(attestation_result, dict):
        return None

    attestation_node = cfg.get("attestation")
    attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
    detect_node = cfg.get("detect")
    detect_cfg = cast(Dict[str, Any], detect_node) if isinstance(detect_node, dict) else {}
    geometry_node = detect_cfg.get("geometry")
    geometry_cfg = cast(Dict[str, Any], geometry_node) if isinstance(geometry_node, dict) else {}
    geo_repair_node = geometry_cfg.get("geo_score_repair")
    geo_repair_cfg = cast(Dict[str, Any], geo_repair_node) if isinstance(geo_repair_node, dict) else {}
    attested_threshold = float(attestation_cfg.get("threshold", 0.65))
    geo_rescue_band_delta_low = float(attestation_cfg.get("rescue_band_delta_low", 0.05))
    geo_rescue_min_score = float(attestation_cfg.get("geo_rescue_min_score", 0.3))

    geometry_payload = cast(Dict[str, Any], geometry_evidence_payload) if isinstance(geometry_evidence_payload, dict) else {}
    sync_metrics = geometry_payload.get("sync_metrics")
    if not isinstance(sync_metrics, dict):
        sync_result = geometry_payload.get("sync_result")
        if isinstance(sync_result, dict) and isinstance(sync_result.get("sync_quality_metrics"), dict):
            sync_metrics = cast(Dict[str, Any], sync_result.get("sync_quality_metrics"))
        else:
            sync_metrics = {}
    sync_result = geometry_payload.get("sync_result")
    sync_result_mapping = cast(Dict[str, Any], sync_result) if isinstance(sync_result, dict) else {}

    relation_binding = geometry_payload.get("relation_digest_binding")
    relation_binding_mapping = cast(Dict[str, Any], relation_binding) if isinstance(relation_binding, dict) else {}
    channel_scores = cast(Dict[str, Any], attestation_result.get("channel_scores")) if isinstance(attestation_result.get("channel_scores"), dict) else {}

    quality_score = float(sync_metrics.get("quality_score")) if isinstance(sync_metrics.get("quality_score"), (int, float)) else None
    template_match_score = None
    if isinstance(sync_metrics.get("template_match_score"), (int, float)):
        template_match_score = float(sync_metrics.get("template_match_score"))
    elif isinstance(sync_result_mapping.get("template_match_metrics"), dict):
        template_match_metrics = cast(Dict[str, Any], sync_result_mapping.get("template_match_metrics"))
        if isinstance(template_match_metrics.get("template_match_score"), (int, float)):
            template_match_score = float(template_match_metrics.get("template_match_score"))
    template_confidence = None
    if isinstance(sync_metrics.get("template_confidence"), (int, float)):
        template_confidence = float(sync_metrics.get("template_confidence"))
    elif isinstance(sync_result_mapping.get("template_match_metrics"), dict):
        template_match_metrics = cast(Dict[str, Any], sync_result_mapping.get("template_match_metrics"))
        if isinstance(template_match_metrics.get("template_confidence"), (int, float)):
            template_confidence = float(template_match_metrics.get("template_confidence"))
    uncertainty = float(sync_metrics.get("uncertainty")) if isinstance(sync_metrics.get("uncertainty"), (int, float)) else None
    geo_score = float(channel_scores.get("geo")) if isinstance(channel_scores.get("geo"), (int, float)) else None

    active_geo_score_source_candidate = sync_metrics.get("active_geo_score_source")
    geo_score_source_candidate = sync_metrics.get("geo_score_source")
    if isinstance(active_geo_score_source_candidate, str) and active_geo_score_source_candidate:
        geo_score_source = active_geo_score_source_candidate
    elif isinstance(geo_score_source_candidate, str) and geo_score_source_candidate:
        geo_score_source = geo_score_source_candidate
    elif geo_score is not None and template_match_score is not None and abs(geo_score - template_match_score) <= 1e-9:
        geo_score_source = "template_match_score"
    elif geo_score is not None and template_confidence is not None and abs(geo_score - template_confidence) <= 1e-9:
        geo_score_source = "template_confidence"
    elif geo_score is not None and quality_score is not None and abs(geo_score - quality_score) <= 1e-9:
        geo_score_source = "quality_score"
    else:
        geo_score_source = "other_or_absent"

    geo_score_repair_enabled = bool(sync_metrics.get("geo_score_repair_enabled", False))
    geo_score_repair_mode = sync_metrics.get("geo_score_repair_mode") if isinstance(sync_metrics.get("geo_score_repair_mode"), str) else None
    if not isinstance(geo_score_repair_mode, str) or not geo_score_repair_mode:
        geo_score_repair_mode = geo_repair_cfg.get("mode") if isinstance(geo_repair_cfg.get("mode"), str) else "template_confidence"
    geo_score_repair_active = bool(sync_metrics.get("geo_score_repair_active", False))
    geo_score_repair_summary = (
        cast(Dict[str, Any], sync_metrics.get("geo_score_repair_summary"))
        if isinstance(sync_metrics.get("geo_score_repair_summary"), dict)
        else None
    )
    geo_repair_enabled = bool(sync_metrics.get("geo_repair_enabled", geo_score_repair_enabled))
    geo_repair_mode = sync_metrics.get("geo_repair_mode") if isinstance(sync_metrics.get("geo_repair_mode"), str) else geo_score_repair_mode
    geo_repair_active = bool(sync_metrics.get("geo_repair_active", geo_score_repair_active))
    geo_repair_summary = (
        cast(Dict[str, Any], sync_metrics.get("geo_repair_summary"))
        if isinstance(sync_metrics.get("geo_repair_summary"), dict)
        else geo_score_repair_summary
    )

    content_attestation_score = attestation_result.get("content_attestation_score")
    if not isinstance(content_attestation_score, (int, float)):
        content_attestation_score = None
    content_gap_to_attested_threshold = None
    if isinstance(content_attestation_score, (int, float)):
        content_gap_to_attested_threshold = float(attested_threshold - float(content_attestation_score))

    quality_vs_template_ratio = None
    if quality_score is not None and template_match_score is not None and abs(template_match_score) > 1e-12:
        quality_vs_template_ratio = float(quality_score / template_match_score)

    geo_score_vs_rescue_min_ratio = None
    if geo_score is not None and geo_rescue_min_score > 0.0:
        geo_score_vs_rescue_min_ratio = float(geo_score / geo_rescue_min_score)

    template_match_internal_threshold = None
    if isinstance(sync_metrics.get("template_match_threshold"), (int, float)):
        template_match_internal_threshold = float(sync_metrics.get("template_match_threshold"))
    elif isinstance(sync_result_mapping.get("template_match_metrics"), dict):
        template_match_metrics = cast(Dict[str, Any], sync_result_mapping.get("template_match_metrics"))
        if isinstance(template_match_metrics.get("template_match_threshold"), (int, float)):
            template_match_internal_threshold = float(template_match_metrics.get("template_match_threshold"))

    template_match_threshold_to_rescue_min_ratio = None
    if template_match_internal_threshold is not None and geo_rescue_min_score > 0.0:
        template_match_threshold_to_rescue_min_ratio = float(template_match_internal_threshold / geo_rescue_min_score)

    geo_scale_control_context = cfg.get("__geo_rescue_scale_control_context__")
    geo_scale_control_mapping = cast(Dict[str, Any], geo_scale_control_context) if isinstance(geo_scale_control_context, dict) else {}
    positive_template_scores = [
        float(value)
        for value in cast(list[Any], geo_scale_control_mapping.get("positive_template_match_scores", []))
        if isinstance(value, (int, float))
    ]
    positive_quality_scores = [
        float(value)
        for value in cast(list[Any], geo_scale_control_mapping.get("positive_quality_scores", []))
        if isinstance(value, (int, float))
    ]
    negative_template_scores = [
        float(value)
        for value in cast(list[Any], geo_scale_control_mapping.get("negative_template_match_scores", []))
        if isinstance(value, (int, float))
    ]
    negative_quality_scores = [
        float(value)
        for value in cast(list[Any], geo_scale_control_mapping.get("negative_quality_scores", []))
        if isinstance(value, (int, float))
    ]

    if bool(geo_scale_control_mapping.get("current_sample_treated_as_positive", False)):
        if template_match_score is not None:
            positive_template_scores.append(template_match_score)
        if quality_score is not None:
            positive_quality_scores.append(quality_score)

    positive_template_match_score_summary = _summarize_geo_distribution(positive_template_scores)
    positive_quality_score_summary = _summarize_geo_distribution(positive_quality_scores)
    negative_template_match_score_summary = _summarize_geo_distribution(negative_template_scores)
    negative_quality_score_summary = _summarize_geo_distribution(negative_quality_scores)

    positive_template_to_gate_max_ratio = None
    if positive_template_match_score_summary is not None and geo_rescue_min_score > 0.0:
        positive_template_to_gate_max_ratio = float(
            positive_template_match_score_summary.get("max", 0.0) / geo_rescue_min_score
        )

    positive_template_to_internal_threshold_max_ratio = None
    if positive_template_match_score_summary is not None and template_match_internal_threshold not in {None, 0.0}:
        positive_template_to_internal_threshold_max_ratio = float(
            positive_template_match_score_summary.get("max", 0.0) / float(template_match_internal_threshold)
        )

    template_score_scale_band = _classify_template_score_scale_band(
        template_match_score,
        template_match_internal_threshold,
        geo_rescue_min_score,
    )
    rescue_gate_scale_classification = _classify_rescue_gate_scale(
        template_match_internal_threshold,
        geo_rescue_min_score,
        positive_template_to_gate_max_ratio,
    )
    geo_repair_direction_classification = _classify_geo_repair_direction(
        quality_score=quality_score,
        template_match_score=template_match_score,
        geo_score_source=geo_score_source,
        geo_rescue_min_score=geo_rescue_min_score,
        positive_template_to_gate_max_ratio=positive_template_to_gate_max_ratio,
        positive_template_to_internal_threshold_max_ratio=positive_template_to_internal_threshold_max_ratio,
        rescue_gate_scale_classification=rescue_gate_scale_classification,
    )

    artifact = {
        "artifact_type": "geo_rescue_diagnostics",
        "attestation_digest": attestation_digest,
        "event_binding_digest": event_binding_digest,
        "trace_commit": trace_commit,
        "decision_mode": attestation_result.get("attestation_decision_mode"),
        "content_attestation_score": content_attestation_score,
        "attested_threshold": attested_threshold,
        "geo_rescue_band_delta_low": geo_rescue_band_delta_low,
        "geo_rescue_band_lower_bound": attested_threshold - geo_rescue_band_delta_low,
        "geo_rescue_min_score": geo_rescue_min_score,
        "quality_score": quality_score,
        "template_match_score": template_match_score,
        "template_confidence": template_confidence,
        "geo_score": geo_score,
        "geo_score_source": geo_score_source,
        "active_geo_score_source": geo_score_source,
        "geo_score_repair_enabled": geo_score_repair_enabled,
        "geo_score_repair_mode": geo_score_repair_mode,
        "geo_score_repair_active": geo_score_repair_active,
        "geo_score_repair_summary": geo_score_repair_summary,
        "geo_repair_enabled": geo_repair_enabled,
        "geo_repair_mode": geo_repair_mode,
        "geo_repair_active": geo_repair_active,
        "geo_repair_summary": geo_repair_summary,
        "geo_rescue_eligible": attestation_result.get("geo_rescue_eligible"),
        "geo_rescue_applied": attestation_result.get("geo_rescue_applied"),
        "geo_not_used_reason": attestation_result.get("geo_not_used_reason"),
        "sync_status": geometry_payload.get("sync_status") or sync_result_mapping.get("sync_status") or sync_result_mapping.get("status"),
        "anchor_status": geometry_payload.get("anchor_status"),
        "relation_digest_binding_status": relation_binding_mapping.get("binding_status"),
        "uncertainty": uncertainty,
        "quality_vs_template_ratio": quality_vs_template_ratio,
        "geo_score_vs_rescue_min_ratio": geo_score_vs_rescue_min_ratio,
        "content_gap_to_attested_threshold": content_gap_to_attested_threshold,
        "geo_scale_classification": _build_geo_rescue_scale_classification(
            quality_score=quality_score,
            template_match_score=template_match_score,
            geo_score_source=geo_score_source,
            geo_rescue_min_score=geo_rescue_min_score,
        ),
        "template_match_internal_threshold": template_match_internal_threshold,
        "template_match_threshold_to_rescue_min_ratio": template_match_threshold_to_rescue_min_ratio,
        "template_score_scale_band": template_score_scale_band,
        "rescue_gate_scale_classification": rescue_gate_scale_classification,
        "positive_template_match_score_summary": positive_template_match_score_summary,
        "positive_quality_score_summary": positive_quality_score_summary,
        "negative_template_match_score_summary": negative_template_match_score_summary,
        "negative_quality_score_summary": negative_quality_score_summary,
        "positive_template_to_gate_max_ratio": positive_template_to_gate_max_ratio,
        "positive_template_to_internal_threshold_max_ratio": positive_template_to_internal_threshold_max_ratio,
        "geo_repair_direction_classification": geo_repair_direction_classification,
        "scale_control_scan_source": geo_scale_control_mapping.get("scan_source"),
        "scale_control_scan_glob": geo_scale_control_mapping.get("scan_glob"),
        "scale_control_scanned_record_count": geo_scale_control_mapping.get("scanned_record_count"),
        "scale_control_labelled_record_count": geo_scale_control_mapping.get("labelled_record_count"),
    }
    artifact["geo_rescue_diagnostics_digest"] = digests.canonical_sha256(artifact)
    return artifact


def _canonicalize_detect_runtime_mode(detect_runtime_mode: Any) -> Optional[str]:
    """
    功能：输出 detect_runtime_mode 的 canonical 语义名。

    Derive the canonical runtime mode name used by canonical-first consumers.

    Args:
        detect_runtime_mode: Detect runtime mode value.

    Returns:
        Canonical runtime mode string, or None when input is unavailable.
    """
    if not isinstance(detect_runtime_mode, str):
        return None

    normalized_mode = detect_runtime_mode.strip()
    if not normalized_mode:
        return None
    return normalized_mode


def resolve_detect_runtime_mode(record: Dict[str, Any]) -> Optional[str]:
    """
    功能：按 canonical-first 规则解析 detect runtime mode。

    Resolve detect runtime mode with canonical-first semantics.

    Args:
        record: Detect record mapping.

    Returns:
        Canonical runtime mode when available; otherwise the normalized legacy
        runtime mode. Returns None when neither field is available.

    Raises:
        TypeError: If record is not a dict.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be dict")

    canonical_mode = record.get("detect_runtime_mode_canonical")
    normalized_canonical = _canonicalize_detect_runtime_mode(canonical_mode)
    if normalized_canonical is not None:
        return normalized_canonical

    legacy_mode = record.get("detect_runtime_mode")
    return _canonicalize_detect_runtime_mode(legacy_mode)


def run_detect_orchestrator(
    cfg: Dict[str, Any],
    impl_set: BuiltImplSet,
    input_record: Optional[Dict[str, Any]] = None,
    cfg_digest: Optional[str] = None,
    trajectory_evidence: Optional[Dict[str, Any]] = None,
    injection_evidence: Optional[Dict[str, Any]] = None,
    content_result_override: Any | None = None,
    detect_plan_result_override: Any | None = None
) -> Dict[str, Any]:
    """
    功能：执行检测编排流程，包括 plan_digest 一致性验证。

    Execute detect workflow using injected implementations.
    Validates plan_digest consistency with embed-time plan_digest when available.
    Supports ablation flags: when ablation.normalized.enable_content=false,
    content_extractor returns status="absent" with no failure reason.

    Args:
        cfg: Config mapping (may differ from embed-time cfg).
        impl_set: Built implementation set.
        input_record: Optional input record mapping (contains embed-time plan_digest).
        cfg_digest: Optional cfg digest for detect-time cfg.
                   If None, plan_digest validation is skipped.
        trajectory_evidence: Optional trajectory tap evidence mapping.
        injection_evidence: Optional injection evidence mapping.
        content_result_override: Optional precomputed content result.
        detect_plan_result_override: Optional precomputed detect plan result.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
    """
    if content_result_override is not None and not isinstance(content_result_override, dict) and not hasattr(content_result_override, "as_dict"):
        # content_result_override 类型不符合预期，必须 fail-fast。
        raise TypeError("content_result_override must be dict, ContentEvidence, or None")
    if detect_plan_result_override is not None and not isinstance(detect_plan_result_override, dict) and not hasattr(detect_plan_result_override, "as_dict"):
        # detect_plan_result_override 类型不符合预期，必须 fail-fast。
        raise TypeError("detect_plan_result_override must be dict, SubspacePlan, or None")

    # 读取 ablation.normalized 开关；若缺失则默认全启用。
    ablation_normalized = _get_ablation_normalized(cfg)
    enable_content = ablation_normalized.get("enable_content", True)
    enable_geometry = ablation_normalized.get("enable_geometry", True)
    enable_sync = ablation_normalized.get("enable_sync", True)
    enable_anchor = ablation_normalized.get("enable_anchor", True)
    enable_image_sidecar = ablation_normalized.get("enable_image_sidecar", True)
    paper_cfg_raw = cfg.get("paper_faithfulness")
    paper_cfg: Dict[str, Any] = cast(Dict[str, Any], paper_cfg_raw) if isinstance(paper_cfg_raw, dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False))
    if paper_enabled:
        enable_image_sidecar = False  # 论文正式路径禁止 image-domain sidecar（v2.0 收口）。

    attestation_context = _prepare_detect_attestation_context(cfg, input_record)
    _bind_detect_attestation_runtime_to_cfg(cfg, attestation_context)

    detect_content_inputs = _build_content_inputs_for_detect(cfg, input_record)

    # Ablation: 禁用 content 模块时返回 absent 语义。
    content_result: Any
    if not enable_content:
        content_result = _build_ablation_absent_content_evidence("content_chain_disabled_by_ablation")
    elif content_result_override is not None:
        content_result = cast(Any, content_result_override)
    else:
        content_result = _call_content_extractor_extract(
            impl_set.content_extractor,
            cfg,
            detect_content_inputs,
            cfg_digest,
        )
    
    # Ablation: 禁用 geometry 模块时返回 absent 语义。
    if not enable_geometry:
        geometry_result = _build_ablation_absent_geometry_evidence("geometry_chain_disabled_by_ablation")
    else:
        geometry_result = _run_geometry_chain_with_sync(
            impl_set,
            cfg,
            enable_anchor=bool(enable_anchor),
            enable_sync=bool(enable_sync),
        )

    # (1) 统一转换 ContentEvidence / GeometryEvidence 数据类为 dict。
    # 优先使用 .as_dict() 方法；若不存在则直接使用数据类或字典。
    content_evidence_payload: Dict[str, Any] | None = _as_dict_payload(content_result)

    if trajectory_evidence is not None:
        if content_evidence_payload is None:
            content_evidence_payload = {}
        content_evidence_payload["trajectory_evidence"] = trajectory_evidence
        _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
    if injection_evidence is not None:
        if content_evidence_payload is None:
            content_evidence_payload = {}
        _merge_injection_evidence(content_evidence_payload, injection_evidence)
    if isinstance(content_evidence_payload, dict) and isinstance(detect_content_inputs, dict):
        detect_input_source = detect_content_inputs.get("input_source")
        if isinstance(detect_input_source, str) and detect_input_source:
            content_evidence_payload["input_source"] = detect_input_source
        detect_image_path_source = detect_content_inputs.get("image_path_source")
        if isinstance(detect_image_path_source, str) and detect_image_path_source:
            content_evidence_payload["image_path_source"] = detect_image_path_source

    geometry_evidence_payload: Dict[str, Any] | None = _as_dict_payload(geometry_result)

    # (2) 构造融合输入适配 dict，兼容不同 content/geometry evidence 数据结构。
    # 优先从 .as_dict() 结果中读取，但为向后兼容也检查数据类属性。
    content_evidence_adapted = _adapt_content_evidence_for_fusion(content_result)
    geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)

    planner_content_payload: Dict[str, Any] | None = content_evidence_payload
    planner_inputs = _build_planner_inputs_for_runtime(cfg, None, planner_content_payload)
    mask_digest = None
    if isinstance(planner_content_payload, dict):
        mask_digest = planner_content_payload.get("mask_digest")
    if not isinstance(mask_digest, str) or not mask_digest:
        if isinstance(input_record, dict):
            embed_content_evidence = input_record.get("content_evidence")
            if isinstance(embed_content_evidence, dict):
                planner_content_payload = cast(Dict[str, Any], embed_content_evidence)
                mask_digest = planner_content_payload.get("mask_digest")
                planner_inputs = _build_planner_inputs_for_runtime(cfg, None, planner_content_payload)
    if not isinstance(mask_digest, str) or not mask_digest:
        # detect-mode 前置阶段可能无法提供 mask_digest；此时回退到 embed-mode 提取，供 planner 使用。
        cfg_for_planner = dict(cfg)
        detect_cfg_node = cfg_for_planner.get("detect")
        if isinstance(detect_cfg_node, dict):
            detect_cfg_for_planner = dict(cast(Dict[str, Any], detect_cfg_node))
        else:
            detect_cfg_for_planner = {}
        detect_content_cfg_node = detect_cfg_for_planner.get("content")
        if isinstance(detect_content_cfg_node, dict):
            detect_content_cfg_for_planner = dict(cast(Dict[str, Any], detect_content_cfg_node))
        else:
            detect_content_cfg_for_planner = {}
        detect_content_cfg_for_planner["enabled"] = False
        detect_cfg_for_planner["content"] = detect_content_cfg_for_planner
        cfg_for_planner["detect"] = detect_cfg_for_planner
        planner_content_result = _call_content_extractor_extract(
            impl_set.content_extractor,
            cfg_for_planner,
            detect_content_inputs,
            cfg_digest,
        )
        planner_content_payload = _as_dict_payload(planner_content_result)
        if isinstance(planner_content_payload, dict):
            mask_digest = planner_content_payload.get("mask_digest")
        planner_inputs = _build_planner_inputs_for_runtime(cfg, None, planner_content_payload)

    planner_cfg_digest = cfg_digest
    planner_cfg = cfg
    if isinstance(input_record, dict):
        embed_cfg_digest = input_record.get("cfg_digest")
        if isinstance(embed_cfg_digest, str) and embed_cfg_digest:
            planner_cfg_digest = embed_cfg_digest
        embed_seed = input_record.get("seed")
        if isinstance(embed_seed, int):
            planner_cfg = dict(cfg)
            planner_cfg["seed"] = embed_seed

    detect_plan_result_obj: Any = cast(
        Any,
        detect_plan_result_override
        if detect_plan_result_override is not None
        else impl_set.subspace_planner.plan(
            planner_cfg,
            mask_digest=mask_digest,
            cfg_digest=planner_cfg_digest,
            inputs=planner_inputs,
        ),
    )

    expected_plan_digest = _resolve_expected_plan_digest(input_record)
    # formal path 语义闭包：不再从 cfg 回填 expected_plan_digest，测试模式与正式模式路径统一
    embed_time_plan_digest = expected_plan_digest
    embed_time_basis_digest = None
    embed_time_planner_impl_identity = None
    if isinstance(input_record, dict):
        embed_time_basis_digest = input_record.get("basis_digest")
        embed_time_planner_impl_identity = input_record.get("subspace_planner_impl_identity")

    detect_time_plan_digest = getattr(detect_plan_result_obj, "plan_digest", None)
    detect_time_basis_digest = getattr(detect_plan_result_obj, "basis_digest", None)
    detect_planner_input_digest = _extract_detect_planner_input_digest(detect_plan_result_obj)
    if detect_planner_input_digest is None:
        build_planner_input_digest = getattr(impl_set.subspace_planner, "_build_planner_input_digest", None)
        if callable(build_planner_input_digest):
            computed_digest = build_planner_input_digest(planner_inputs)
            if isinstance(computed_digest, str) and computed_digest:
                detect_planner_input_digest = computed_digest
    detect_time_planner_impl_identity = None
    detect_plan_node = getattr(detect_plan_result_obj, "plan", None)
    if isinstance(detect_plan_node, dict):
        detect_plan_node_payload = cast(Dict[str, Any], detect_plan_node)
        detect_time_planner_impl_identity = detect_plan_node_payload.get("planner_impl_identity")

    plan_payload = _as_dict_payload(detect_plan_result_obj)

    if isinstance(plan_payload, dict):
        if not isinstance(detect_time_plan_digest, str):
            payload_plan_digest = plan_payload.get("plan_digest")
            if isinstance(payload_plan_digest, str) and payload_plan_digest:
                detect_time_plan_digest = payload_plan_digest
        if not isinstance(detect_time_basis_digest, str):
            payload_basis_digest = plan_payload.get("basis_digest")
            if isinstance(payload_basis_digest, str) and payload_basis_digest:
                detect_time_basis_digest = payload_basis_digest
        if detect_time_planner_impl_identity is None:
            plan_node = plan_payload.get("plan")
            if isinstance(plan_node, dict):
                plan_node_payload = cast(Dict[str, Any], plan_node)
                detect_time_planner_impl_identity = plan_node_payload.get("planner_impl_identity")

    mismatch_reasons: list[str] = []
    if isinstance(expected_plan_digest, str) and expected_plan_digest:
        mismatch_reasons = _collect_plan_mismatch_reasons(
            embed_time_plan_digest=expected_plan_digest,
            detect_time_plan_digest=detect_time_plan_digest,
            embed_time_basis_digest=embed_time_basis_digest,
            detect_time_basis_digest=detect_time_basis_digest,
            embed_time_planner_impl_identity=embed_time_planner_impl_identity,
            detect_time_planner_impl_identity=detect_time_planner_impl_identity
        )
        plan_digest_status, plan_digest_reason = verify_plan_digest(
            expected_plan_digest,
            detect_time_plan_digest if isinstance(detect_time_plan_digest, str) else None,
        )
        if plan_digest_reason == "plan_digest_mismatch" and "plan_digest_mismatch" not in mismatch_reasons:
            mismatch_reasons.append("plan_digest_mismatch")
    else:
        plan_digest_status = "absent"
        plan_digest_reason = "plan_digest_absent"

    trajectory_status, trajectory_mismatch_reason = _evaluate_trajectory_consistency(
        input_record=input_record,
        trajectory_evidence=trajectory_evidence,
        detect_planner_input_digest=detect_planner_input_digest
    )
    if trajectory_status == "mismatch" and trajectory_mismatch_reason:
        mismatch_reasons.append(trajectory_mismatch_reason)

    injection_status, injection_mismatch_reason = _evaluate_injection_consistency(
        input_record=input_record,
        injection_evidence=injection_evidence
    )
    if injection_status == "mismatch" and injection_mismatch_reason:
        mismatch_reasons.append(injection_mismatch_reason)

    paper_impl_binding_status, paper_impl_binding_reason = _evaluate_paper_impl_binding_consistency(
        cfg=cfg,
        injection_evidence=injection_evidence,
        input_record=input_record,
    )
    if paper_impl_binding_status == "mismatch" and isinstance(paper_impl_binding_reason, str):
        mismatch_reasons.append(paper_impl_binding_reason)

    # (S-D) Paper Faithfulness: 验证 paper faithfulness 证据一致性（必达 
    # 注意：只 input_record 存在且包 paper_faithfulness 信息时才添加到全局 mismatch_reasons
    # 这样可以避免单元测试中使用不完整 input_record 时产生副作用
    paper_faithfulness_status, paper_absent_reasons, paper_mismatch_reasons, paper_fail_reasons = _evaluate_paper_faithfulness_consistency(
        input_record=input_record
    )
    
    # 仅当 paper_faithfulness 显式启用 input_record 包含对应字段时，才将缺失视为 mismatch 
    paper_cfg_raw = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_cfg_raw) if isinstance(paper_cfg_raw, dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False))
    if paper_enabled and input_record is not None and isinstance(input_record.get("paper_faithfulness"), dict):
        # 启用模式下，paper_faithfulness 缺失或不一致必须进 mismatch 门禁 
        if paper_mismatch_reasons:
            mismatch_reasons.extend(paper_mismatch_reasons)

    primary_mismatch_reason, primary_mismatch_field_path = _resolve_primary_mismatch(
        mismatch_reasons
    )

    trajectory_absent_forced = _is_embed_trajectory_explicit_absent(input_record)
    forced_mismatch = len(mismatch_reasons) > 0
    forced_absent = (
        not isinstance(expected_plan_digest, str) or
        not expected_plan_digest or
        (
            isinstance(injection_evidence, dict)
            and injection_evidence.get("injection_absent_reason") == "inference_failed"
            and not forced_mismatch
        ) or
        (
            isinstance(trajectory_evidence, dict)
            and trajectory_evidence.get("trajectory_absent_reason") == "inference_failed"
            and not forced_mismatch
        )
    )
    if trajectory_absent_forced:
        content_evidence_payload = {
            "status": "absent",
            "score": None,
            "plan_digest": detect_time_plan_digest,
            "basis_digest": detect_time_basis_digest,
            "content_failure_reason": "detector_no_plan_expected" if not isinstance(expected_plan_digest, str) or not expected_plan_digest else None,
            "score_parts": None,
            "lf_score": None,
            "hf_score": None,
            "audit": {
                "impl_identity": "detect_orchestrator",
                "impl_version": "v1",
                "impl_digest": digests.canonical_sha256({"impl_id": "detect_orchestrator", "impl_version": "v1"}),
                "trace_digest": digests.canonical_sha256({
                    "trajectory_status": trajectory_status,
                    "trajectory_mismatch_reason": trajectory_mismatch_reason,
                    "plan_digest_status": plan_digest_status
                })
            }
        }
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        _bind_scores_if_ok(content_evidence_payload)
        content_result = content_evidence_payload
        content_evidence_adapted = content_evidence_payload
        geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
        fusion_result = _build_absent_fusion_decision(cfg, content_evidence_adapted, geometry_evidence_adapted)
    elif forced_mismatch:
        content_evidence_payload = {
            "status": "mismatch",
            "score": None,
            "plan_digest": detect_time_plan_digest,
            "basis_digest": detect_time_basis_digest,
            "content_failure_reason": _resolve_mismatch_failure_reason(primary_mismatch_reason),
            "content_mismatch_reason": primary_mismatch_reason,
            "content_mismatch_field_path": primary_mismatch_field_path,
            "mismatch_reasons": list(mismatch_reasons),
            "score_parts": None,
            "lf_score": None,
            "hf_score": None,
            "audit": {
                "impl_identity": "detect_orchestrator",
                "impl_version": "v1",
                "impl_digest": digests.canonical_sha256({"impl_id": "detect_orchestrator", "impl_version": "v1"}),
                "trace_digest": digests.canonical_sha256({
                    "mismatch_reasons": mismatch_reasons,
                    "primary_mismatch_reason": primary_mismatch_reason,
                    "primary_mismatch_field_path": primary_mismatch_field_path
                })
            }
        }
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        _bind_scores_if_ok(content_evidence_payload)
        content_result = content_evidence_payload
        content_evidence_adapted = content_evidence_payload
        geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
        fusion_result = _build_mismatch_fusion_decision(cfg, content_evidence_adapted, geometry_evidence_adapted)
    elif forced_absent:
        content_evidence_payload = {
            "status": "absent",
            "score": None,
            "plan_digest": detect_time_plan_digest,
            "basis_digest": detect_time_basis_digest,
            "content_failure_reason": "detector_no_plan_expected" if not isinstance(expected_plan_digest, str) or not expected_plan_digest else None,
            "score_parts": None,
            "lf_score": None,
            "hf_score": None,
            "audit": {
                "impl_identity": "detect_orchestrator",
                "impl_version": "v1",
                "impl_digest": digests.canonical_sha256({"impl_id": "detect_orchestrator", "impl_version": "v1"}),
                "trace_digest": digests.canonical_sha256({
                    "trajectory_status": trajectory_status,
                    "trajectory_mismatch_reason": trajectory_mismatch_reason,
                    "plan_digest_status": plan_digest_status
                })
            }
        }
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        _bind_scores_if_ok(content_evidence_payload)
        content_result = content_evidence_payload
        content_evidence_adapted = content_evidence_payload
        geometry_evidence_adapted = _adapt_geometry_evidence_for_fusion(geometry_result)
        fusion_result = _build_absent_fusion_decision(cfg, content_evidence_adapted, geometry_evidence_adapted)
    else:
        hf_evidence = _build_hf_detect_evidence(
            impl_set=impl_set,
            cfg=cfg,
            cfg_digest=cfg_digest,
            plan_payload=plan_payload,
            plan_digest=detect_time_plan_digest,
            embed_time_plan_digest=embed_time_plan_digest,
            trajectory_evidence=trajectory_evidence,
        )
        if _is_image_domain_sidecar_enabled(cfg, ablation_override=bool(enable_image_sidecar)):
            lf_raw_score, hf_raw_score, raw_score_traces = _extract_content_raw_scores_from_image(
                cfg=cfg,
                input_record=input_record,
                plan_payload=plan_payload,
                plan_digest=detect_time_plan_digest,
                cfg_digest=cfg_digest,
            )
        else:
            sidecar_disabled_reason = _resolve_sidecar_disabled_reason(
                paper_enabled=paper_enabled,
                enable_image_sidecar=bool(enable_image_sidecar),
            )
            lf_raw_score, lf_raw_trace = _extract_lf_raw_score_from_trajectory(
                cfg=cfg,
                plan_payload=plan_payload,
                plan_digest=detect_time_plan_digest,
                cfg_digest=cfg_digest,
            )
            hf_raw_score = None
            raw_score_traces = {
                "lf": lf_raw_trace,
                "hf": {
                    "hf_status": "absent",
                    "hf_absent_reason": sidecar_disabled_reason,
                },
            }

        lf_detect_evidence = _build_lf_detect_evidence(
            lf_raw_score,
            raw_score_traces.get("lf"),
        )
        detector_inputs: Dict[str, Any] = {
            "expected_plan_digest": expected_plan_digest,
            "observed_plan_digest": detect_time_plan_digest,
            "plan_digest": detect_time_plan_digest,
            "lf_evidence": lf_detect_evidence,
            "hf_evidence": hf_evidence,
            "lf_detect_trace": raw_score_traces.get("lf"),
            "hf_detect_trace": raw_score_traces.get("hf"),
            "trajectory_evidence": trajectory_evidence,
            "injection_evidence": injection_evidence,
            "plan_payload": plan_payload,
        }
        content_result = impl_set.content_extractor.extract(
            cfg,
            inputs=detector_inputs,
            cfg_digest=cfg_digest,
        )
        content_evidence_payload = _adapt_content_evidence_for_fusion(content_result)
        content_result = content_evidence_payload
        if trajectory_evidence is not None:
            content_evidence_payload["trajectory_evidence"] = trajectory_evidence
            _inject_trajectory_audit_fields(content_evidence_payload, trajectory_evidence)
        if injection_evidence is not None:
            _merge_injection_evidence(content_evidence_payload, injection_evidence)
        score_parts_node = content_evidence_payload.get("score_parts")
        score_parts = cast(Dict[str, Any], score_parts_node) if isinstance(score_parts_node, dict) else {}
        if not isinstance(score_parts_node, dict):
            content_evidence_payload["score_parts"] = score_parts
        score_parts["lf_trajectory_detect_trace"] = raw_score_traces.get("lf")
        score_parts["hf_image_detect_trace"] = raw_score_traces.get("hf")
        lf_summary = lf_detect_evidence.get("lf_evidence_summary")
        if isinstance(lf_summary, dict):
            content_evidence_payload["lf_evidence_summary"] = lf_summary
            score_parts.setdefault("lf_metrics", lf_summary)
        hf_summary = hf_evidence.get("hf_evidence_summary")
        if isinstance(hf_summary, dict):
            content_evidence_payload["hf_evidence_summary"] = hf_summary
            score_parts.setdefault("hf_metrics", hf_summary)
            score_parts["hf_trajectory_detect_trace"] = hf_summary
        hf_attestation_values = hf_evidence.get("hf_attestation_values")
        if isinstance(hf_attestation_values, list):
            content_evidence_payload["hf_attestation_values"] = hf_attestation_values
            score_parts["hf_attestation_values"] = hf_attestation_values
        hf_attestation_status = hf_evidence.get("hf_attestation_status")
        if isinstance(hf_attestation_status, str) and hf_attestation_status:
            content_evidence_payload["hf_attestation_status"] = hf_attestation_status
            score_parts["hf_attestation_status"] = hf_attestation_status
        hf_attestation_failure_reason = hf_evidence.get("hf_attestation_failure_reason")
        if isinstance(hf_attestation_failure_reason, str) and hf_attestation_failure_reason:
            content_evidence_payload["hf_attestation_failure_reason"] = hf_attestation_failure_reason
            score_parts["hf_attestation_failure_reason"] = hf_attestation_failure_reason
        _bind_scores_if_ok(content_evidence_payload)
        content_evidence_adapted = _adapt_content_evidence_for_fusion(content_evidence_payload)
        
        #  input_record 中提 calibrate 生成 thresholds_artifact 
        # 并注入到 cfg 中供 fusion_rule.fuse() 使用（必须修正：threshold binding error） 
        if isinstance(input_record, dict) and "thresholds_artifact" in input_record:
            thresholds_artifact = input_record["thresholds_artifact"]
            if isinstance(thresholds_artifact, dict):
                cfg["__thresholds_artifact__"] = thresholds_artifact
        
        fusion_result = impl_set.fusion_rule.fuse(cfg, content_evidence_adapted, geometry_evidence_adapted)
    input_fields = len(input_record or {})

    # 实现 detect 侧同构分数与一致性校 
    detect_runtime_mode = "fallback_identity"  # 默认：未获得可用 detect 同构分数
    detect_traj_cache = cfg.get("__detect_trajectory_latent_cache__")

    if not forced_mismatch and isinstance(plan_payload, dict):
        # plan_payload  SubspacePlanEvidence  dict 化结构，
        # lf_basis/hf_basis  plan_payload["plan"] 内层，而非顶层 
        _plan_inner = plan_payload.get("plan")
        _plan_inner_dict = cast(Dict[str, Any], _plan_inner) if isinstance(_plan_inner, dict) else {}
        lf_basis = _plan_inner_dict.get("lf_basis")
        hf_basis = _plan_inner_dict.get("hf_basis")

        #  input_record 提取 embed 侧分数（兼容 content_evidence 承载） 
        embed_lf_score = None
        embed_hf_score = None
        if isinstance(input_record, dict):
            embed_lf_score = input_record.get("lf_score")
            embed_hf_score = input_record.get("hf_score")
            embed_content_evidence = input_record.get("content_evidence")
            if isinstance(embed_content_evidence, dict):
                embed_content_payload = cast(Dict[str, Any], embed_content_evidence)
                if embed_lf_score is None:
                    embed_lf_score = embed_content_payload.get("lf_score")
                if embed_hf_score is None:
                    embed_hf_score = embed_content_payload.get("hf_score")

        # --- 评分路径：trajectory cache 可用时，使用 TFSW z_{t_e} 精确评分；否则显式失效。---
        if detect_traj_cache is not None and not detect_traj_cache.is_empty():
            # 主路径：使用真实 z_{t_e} 执行 TFSW（exact-only）。
            # plan_digest 必须传入以派生 LDPC / Rademacher 模板（闭环验证要求）。
            _active_pd_audit = embed_time_plan_digest if isinstance(embed_time_plan_digest, str) and embed_time_plan_digest else detect_time_plan_digest
            detect_lf_score, detect_lf_status = detector_scoring.extract_lf_score_from_detect_trajectory(
                detect_traj_cache,
                lf_basis,
                embed_lf_score,
                cfg,
                plan_digest=_active_pd_audit,
            )
            detect_hf_score, detect_hf_status = detector_scoring.extract_hf_score_from_detect_trajectory(
                detect_traj_cache,
                hf_basis,
                embed_hf_score,
                cfg,
                plan_digest=_active_pd_audit,
            )
        else:
            detect_lf_score, detect_lf_status = None, "no_trajectory_cache"
            detect_hf_score, detect_hf_status = None, "no_trajectory_cache"

        # 校验 plan_digest 与 basis_digest 的一致性。
        embed_plan_digest = input_record.get("plan_digest") if input_record else None
        embed_basis_digest = input_record.get("basis_digest") if input_record else None

        plan_digest_consistent, plan_digest_reason = detector_scoring.validate_plan_digest_consistency(
            embed_plan_digest,
            detect_time_plan_digest
        )
        basis_digest_consistent, basis_digest_reason = detector_scoring.validate_basis_digest_consistency(
            embed_basis_digest,
            detect_time_basis_digest if isinstance(detect_time_basis_digest, str) else None
        )

        # 追加 detect 侧分数与一致性状态到 content_evidence
        content_evidence_payload["detect_lf_score"] = detect_lf_score
        content_evidence_payload[eval_metrics.LF_CORRELATION_SCORE_NAME] = detect_lf_score
        content_evidence_payload["detect_hf_score"] = detect_hf_score
        score_parts_node = content_evidence_payload.get("score_parts")
        if isinstance(score_parts_node, dict):
            score_parts = cast(Dict[str, Any], score_parts_node)
            score_parts[eval_metrics.LF_CORRELATION_SCORE_NAME] = detect_lf_score
            score_parts["detect_lf_score"] = detect_lf_score
        content_evidence_payload.update(_build_detect_lf_observability_fields(detect_lf_status))
        # hf_basis is None（detect plan 未提供 HF basis）时，显式写入 HF 缺失原因。
        if hf_basis is None:
            content_evidence_payload["detect_hf_score_absent_reason"] = "hf_basis_absent_in_detect_plan"

        lf_score_drift_status = None
        _lf_st = detect_lf_status or ""
        if _lf_st.startswith("ok_trajectory_ok_exact"):
            lf_score_drift_status = "ok"
        elif "drift_detected" in _lf_st:
            lf_score_drift_status = "drift_detected"

        hf_score_drift_status = None
        _hf_st = detect_hf_status or ""
        if _hf_st.startswith("ok_trajectory_ok_exact"):
            hf_score_drift_status = "ok"
        elif "drift_detected" in _hf_st:
            hf_score_drift_status = "drift_detected"

        if lf_score_drift_status is not None:
            content_evidence_payload["lf_score_drift_status"] = lf_score_drift_status
        if hf_score_drift_status is not None:
            content_evidence_payload["hf_score_drift_status"] = hf_score_drift_status

        if basis_digest_consistent:
            basis_digest_match = "consistent"
        elif "absent" in basis_digest_reason:
            basis_digest_match = "absent"
        else:
            basis_digest_match = "mismatch"

        if trajectory_status == "ok":
            trajectory_digest_match = "consistent"
        elif trajectory_status == "mismatch":
            trajectory_digest_match = "mismatch"
        else:
            trajectory_digest_match = "absent"

        content_evidence_payload["basis_digest_match"] = basis_digest_match
        content_evidence_payload["trajectory_digest_match"] = trajectory_digest_match

        if (not plan_digest_consistent and "absent" not in plan_digest_reason) or basis_digest_match == "mismatch" or trajectory_status == "mismatch":
            subspace_consistency_status = "inconsistent"
        elif "absent" in plan_digest_reason or basis_digest_match == "absent" or trajectory_status == "absent":
            subspace_consistency_status = "absent"
        else:
            subspace_consistency_status = "ok"

        content_evidence_payload["subspace_consistency_status"] = subspace_consistency_status

        subspace_semantics = _extract_subspace_evidence_semantics(plan_payload)
        evidence_level = subspace_semantics.get("evidence_level") if isinstance(subspace_semantics.get("evidence_level"), str) else "<absent>"
        subspace_primary_path = bool(evidence_level in {"primary", "hybrid"})
        pipeline_runtime_meta_raw = cfg.get("__pipeline_runtime_meta__")
        pipeline_runtime_meta = cast(Dict[str, Any], pipeline_runtime_meta_raw) if isinstance(pipeline_runtime_meta_raw, dict) else {}
        synthetic_pipeline_runtime = bool(pipeline_runtime_meta.get("synthetic_pipeline", False))
        content_evidence_payload["subspace_evidence_semantics"] = subspace_semantics
        content_evidence_payload["subspace_evidence_level"] = evidence_level
        content_evidence_payload["subspace_primary_path"] = subspace_primary_path
        content_evidence_payload["synthetic_pipeline_runtime"] = synthetic_pipeline_runtime

        runtime_built = bool(pipeline_runtime_meta.get("status") == "built")

        # 如果 detect 侧分数有效、未命中不一致且运行期为真实 synthetic pipeline，则标记为真实运行模式 
        _lf_ok = (detect_lf_status == "ok" or (detect_lf_status or "").startswith("ok_trajectory_"))
        _hf_ok = (detect_hf_status == "ok" or (detect_hf_status or "").startswith("ok_trajectory_"))
        content_failure_reason_value = content_evidence_payload.get("content_failure_reason")
        hf_only_sidecar_disabled = (
            isinstance(content_failure_reason_value, str)
            and content_failure_reason_value in {
                "image_domain_sidecar_disabled",
                "image_domain_sidecar_disabled_by_ablation",
                "formal_profile_sidecar_disabled",
                "ablation_sidecar_disabled",
            }
        )
        if (
            (_lf_ok or (hf_only_sidecar_disabled and _hf_ok))
            and subspace_consistency_status != "inconsistent"
            and runtime_built
            and (not synthetic_pipeline_runtime)
        ):
            detect_runtime_mode = "real"

    _bind_scores_if_ok(content_evidence_payload)
    _populate_detect_mask_digest_from_input_record(content_evidence_payload, input_record)
    if isinstance(content_evidence_payload, dict):
        if isinstance(attestation_context.get("event_binding_digest"), str):
            content_evidence_payload["attestation_event_digest"] = attestation_context.get("event_binding_digest")
        _synchronize_content_score_aliases(content_evidence_payload)
    if isinstance(geometry_evidence_payload, dict):
        if isinstance(attestation_context.get("event_binding_digest"), str):
            geometry_evidence_payload["attestation_event_digest"] = attestation_context.get("event_binding_digest")
        if isinstance(attestation_context.get("trace_commit"), str):
            geometry_evidence_payload["attestation_trace_commit"] = attestation_context.get("trace_commit")

    lf_attestation_features = None
    lf_attestation_trace_context = None
    if (
        isinstance(content_evidence_payload, dict)
        and content_evidence_payload.get("status") == "ok"
        and attestation_context.get("authenticity_status") == "authentic"
    ):
        lf_attestation_trace_bundle = (
            cast(Dict[str, Any], cfg.get("__lf_formal_exact_trace_bundle__"))
            if isinstance(cfg.get("__lf_formal_exact_trace_bundle__"), dict)
            else _extract_lf_attestation_trace_bundle_from_trajectory(
                cfg,
                plan_payload,
            )
        )
        if isinstance(lf_attestation_trace_bundle, dict):
            lf_attestation_features = lf_attestation_trace_bundle.get("lf_attestation_features")
        lf_attestation_trace_context = _build_lf_attestation_trace_context(
            cfg,
            plan_payload,
            lf_trace_bundle=lf_attestation_trace_bundle if isinstance(lf_attestation_trace_bundle, dict) else None,
        )
        if lf_attestation_trace_context is None:
            lf_attestation_trace_context = {}
        lf_attestation_trace_context.update(_extract_embed_lf_closed_loop_context(input_record))
        lf_attestation_trace_context.update(_extract_planner_posterior_context(input_record))
        lf_protocol_control_context = cfg.get("__lf_protocol_control_context__")
        if isinstance(lf_protocol_control_context, dict):
            lf_attestation_trace_context.update(cast(Dict[str, Any], lf_protocol_control_context))
        lf_formal_exact_context = cfg.get("__lf_formal_exact_context__")
        if isinstance(lf_formal_exact_context, dict):
            lf_attestation_trace_context.update(cast(Dict[str, Any], lf_formal_exact_context))

    attestation_result = _build_detect_attestation_result(
        cfg,
        attestation_context,
        content_evidence_payload if isinstance(content_evidence_payload, dict) else None,
        geometry_evidence_payload if isinstance(geometry_evidence_payload, dict) else None,
        lf_attestation_features=lf_attestation_features,
        lf_attestation_trace_context=lf_attestation_trace_context,
    )
    if isinstance(content_evidence_payload, dict) and isinstance(attestation_result, dict):
        lf_attestation_score = _coerce_optional_finite_score(attestation_result.get("lf_attestation_score"))
        if lf_attestation_score is not None:
            content_evidence_payload["lf_attestation_score"] = lf_attestation_score
        _synchronize_content_score_aliases(content_evidence_payload)

    # 删除临时 transient 字段，确保不写入 records
    cfg.pop("__detect_trajectory_latent_cache__", None)
    cfg.pop("__detect_pipeline_obj__", None)
    cfg.pop("__pipeline_runtime_meta__", None)
    cfg.pop("__detect_attention_maps__", None)
    cfg.pop("__detect_self_attention_maps__", None)
    cfg.pop("__runtime_self_attention_maps__", None)
    cfg.pop("__lf_formal_exact_trace_bundle__", None)
    cfg.pop("__lf_formal_exact_context__", None)
    cfg.pop("__lf_attacked_image_conditioned_latent_cache__", None)
    cfg.pop("__detect_hf_plan_digest_used__", None)

    plan_digest_mismatch_reason = plan_digest_reason if plan_digest_reason == "plan_digest_mismatch" else None

    fusion_result = _close_formal_required_chain_decision(
        cfg,
        content_evidence_payload if isinstance(content_evidence_payload, dict) else {},
        geometry_evidence_payload if isinstance(geometry_evidence_payload, dict) else {},
        fusion_result,
    )

    execution_report = _derive_execution_report_from_chain_states(
        content_evidence_payload=content_evidence_payload,
        geometry_evidence_payload=geometry_evidence_payload,
        fusion_result=fusion_result,
    )

    record: Dict[str, Any] = {
        "operation": "detect",
        "detect_runtime_mode": detect_runtime_mode,
        "detect_runtime_mode_canonical": _canonicalize_detect_runtime_mode(detect_runtime_mode),
        "detect_runtime_status": "active" if detect_runtime_mode == "real" else "fallback",
        "detect_runtime_is_fallback": (detect_runtime_mode != "real"),
        "image_path": "<absent>",
        "score": getattr(fusion_result, "evidence_summary", {}).get("content_score"),
        "execution_report": execution_report,
        "input_record_fields": input_fields,
        "plan_digest_expected": expected_plan_digest,
        "plan_digest_observed": detect_time_plan_digest,
        "plan_digest_status": plan_digest_status,
        "plan_digest_validation_status": plan_digest_status,
        "plan_digest_mismatch_reason": primary_mismatch_reason if forced_mismatch else plan_digest_mismatch_reason,
        "content_evidence_payload": content_evidence_payload,
        "geometry_evidence_payload": geometry_evidence_payload,
        "content_result": content_result,
        "geometry_result": geometry_result,
        "fusion_result": fusion_result,
        "attestation": attestation_result,
        # (S-D) Paper Faithfulness: 添加一致性验证结果（结构 failure semantics 
        "paper_faithfulness": {
            "status": paper_faithfulness_status,
            "absent_reasons": paper_absent_reasons,
            "mismatch_reasons": paper_mismatch_reasons,
            "fail_reasons": paper_fail_reasons
        }
    }

    _bind_actual_detect_planner_payload_to_record(
        record,
        input_record=input_record,
        detect_plan_result=detect_plan_result_obj,
        detect_plan_result_override=detect_plan_result_override,
        detect_time_plan_digest=detect_time_plan_digest,
        detect_time_basis_digest=detect_time_basis_digest,
        detect_time_planner_impl_identity=detect_time_planner_impl_identity,
        detect_planner_input_digest=detect_planner_input_digest,
    )

    # (append-only) 构建 final_decision 顶层稳定判决快照，供后续冻结与审查使用 
    # 所有字段从 fusion_result 只读投影，不替换原有 fusion_result 字段 
    try:
        _fd_audit = getattr(fusion_result, "audit", {})
        record["final_decision"] = {
            "decision_status": getattr(fusion_result, "decision_status", None),
            "is_watermarked": getattr(fusion_result, "is_watermarked", None),
            "routing_decisions": getattr(fusion_result, "routing_decisions", None),
            "threshold_source": _fd_audit.get("threshold_source") if isinstance(_fd_audit, dict) else None,
            "event_attested": bool(record.get("attestation", {}).get("final_event_attested_decision", {}).get("is_event_attested")) if isinstance(record.get("attestation"), dict) else False,
        }
    except Exception:
        record["final_decision"] = None

    return record


def _normalize_execution_chain_status(raw_status: Any) -> str:
    """
    功能：将链路状态归一化到 ok/absent/failed 三态 

    Normalize execution-chain status into canonical enum {ok, absent, failed}.

    Args:
        raw_status: Raw status token from runtime payload.

    Returns:
        Canonical status token.
    """
    if not isinstance(raw_status, str) or not raw_status:
        return "failed"
    normalized = raw_status.strip().lower()
    if normalized == "fail":
        return "failed"
    if normalized in {"failed", "error", "mismatch"}:
        return "failed"
    if normalized in {"absent", "none", "disabled", "not_applicable"}:
        return "absent"
    if normalized in {"ok", "synced", "accepted", "rejected", "abstain", "decided"}:
        return "ok"
    return "failed"


def _derive_execution_report_from_chain_states(
    content_evidence_payload: Any,
    geometry_evidence_payload: Any,
    fusion_result: Any,
) -> Dict[str, Any]:
    """
    功能：由 content/geometry/fusion 实际状态推 execution_report 

    Derive execution_report from actual chain payloads instead of hardcoded statuses.

    Args:
        content_evidence_payload: Content evidence payload mapping.
        geometry_evidence_payload: Geometry evidence payload mapping.
        fusion_result: Fusion decision object.

    Returns:
        Canonical execution_report mapping.
    """
    content_status_raw = None
    if isinstance(content_evidence_payload, dict):
        content_payload = cast(Dict[str, Any], content_evidence_payload)
        content_status_raw = content_payload.get("status")
    content_chain_status = _normalize_execution_chain_status(content_status_raw)

    geometry_status_raw = None
    if isinstance(geometry_evidence_payload, dict):
        geometry_payload = cast(Dict[str, Any], geometry_evidence_payload)
        geometry_status_raw = geometry_payload.get("status")
        if geometry_status_raw is None:
            geometry_status_raw = geometry_payload.get("sync_status")
    geometry_chain_status = _normalize_execution_chain_status(geometry_status_raw)

    fusion_status_raw = None
    if hasattr(fusion_result, "decision_status"):
        fusion_status_raw = getattr(fusion_result, "decision_status")
    fusion_chain_status = _normalize_execution_chain_status(fusion_status_raw)
    if fusion_chain_status == "failed" and content_chain_status == "ok" and geometry_chain_status == "ok":
        fusion_chain_status = "ok"

    return {
        "content_chain_status": content_chain_status,
        "geometry_chain_status": geometry_chain_status,
        "fusion_status": fusion_chain_status,
        "audit_obligations_satisfied": True,
    }


def _resolve_cfg_plan_digest(cfg: Dict[str, Any]) -> Optional[str]:
    """
    功能：从 cfg 读取 plan_digest。

    Resolve cfg-side plan_digest from cfg.

    Args:
        cfg: Configuration mapping.

    Returns:
        plan_digest string or None.
    """
    watermark_cfg = cfg.get("watermark")
    if not isinstance(watermark_cfg, dict):
        return None
    watermark_payload = cast(Dict[str, Any], watermark_cfg)
    candidate = watermark_payload.get("plan_digest")
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def _bind_scores_if_ok(content_evidence_payload: Dict[str, Any]) -> None:
    """
    功能：分数写入纪律收口， status=ok 允许数值分数 

    Enforce score write discipline: numeric score fields are allowed only when status="ok".

    Args:
        content_evidence_payload: Mutable content evidence mapping.

    Returns:
        None.
    """
    status_value = content_evidence_payload.get("status")
    if not isinstance(status_value, str):
        status_value = "failed"
        content_evidence_payload["status"] = status_value

    score_parts_node = content_evidence_payload.get("score_parts")
    score_parts: Optional[Dict[str, Any]] = None
    if isinstance(score_parts_node, dict):
        score_parts = cast(Dict[str, Any], score_parts_node)
    if status_value != "ok":
        content_evidence_payload["score"] = None
        content_evidence_payload["content_score"] = None
        content_evidence_payload[eval_metrics.CONTENT_CHAIN_SCORE_NAME] = None
        content_evidence_payload["lf_score"] = None
        content_evidence_payload[eval_metrics.LF_CHANNEL_SCORE_NAME] = None
        content_evidence_payload["hf_score"] = None
        content_evidence_payload["hf_raw_energy"] = None
        content_evidence_payload["hf_content_score"] = None
        content_evidence_payload["detect_lf_score"] = None
        content_evidence_payload[eval_metrics.LF_CORRELATION_SCORE_NAME] = None
        content_evidence_payload["detect_hf_score"] = None
        if score_parts is not None:
            for numeric_key in [
                "score",
                "lf_score",
                "hf_score",
                "hf_raw_energy",
                "hf_content_score",
                "content_score",
                eval_metrics.CONTENT_CHAIN_SCORE_NAME,
                eval_metrics.LF_CHANNEL_SCORE_NAME,
                eval_metrics.LF_CORRELATION_SCORE_NAME,
                "content_score",
                "detect_lf_score",
                "detect_hf_score",
            ]:
                if isinstance(score_parts.get(numeric_key), (int, float)):
                    score_parts[numeric_key] = None
        return

    for field_name in [
        "score",
        "content_score",
        eval_metrics.CONTENT_CHAIN_SCORE_NAME,
        "lf_score",
        eval_metrics.LF_CHANNEL_SCORE_NAME,
        "hf_score",
        "hf_raw_energy",
        "hf_content_score",
        "detect_lf_score",
        eval_metrics.LF_CORRELATION_SCORE_NAME,
    ]:
        score_value = content_evidence_payload.get(field_name)
        if score_value is None:
            continue
        if not isinstance(score_value, (int, float)) or not np.isfinite(float(score_value)):
            content_evidence_payload["status"] = "failed"
            content_evidence_payload["content_failure_reason"] = "detector_score_validation_failed"
            content_evidence_payload["score"] = None
            content_evidence_payload["lf_score"] = None
            content_evidence_payload["hf_score"] = None
            content_evidence_payload["hf_raw_energy"] = None
            content_evidence_payload["hf_content_score"] = None
            content_evidence_payload["score_parts"] = None
            return

    _synchronize_content_score_aliases(content_evidence_payload)


def _build_hf_detect_evidence(
    impl_set: BuiltImplSet,
    cfg: Dict[str, Any],
    cfg_digest: Optional[str],
    plan_payload: Optional[Dict[str, Any]],
    plan_digest: Optional[str],
    embed_time_plan_digest: Optional[str],
    trajectory_evidence: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    功能：构 detect  HF 证据 

    Build detect-side HF evidence under planner-defined plan.

    Args:
        cfg: Configuration mapping.
        cfg_digest: Optional cfg digest.
        plan_payload: Planner evidence mapping.
        plan_digest: Detect-time plan digest.
        embed_time_plan_digest: Embed-time plan digest.
        trajectory_evidence: Optional trajectory evidence mapping.

    Returns:
        HF evidence mapping.

    Raises:
        TypeError: If cfg is invalid.
    """
    embedder = impl_set.hf_embedder
    if embedder is None or not hasattr(embedder, "detect"):
        embedder = HighFreqTruncationCodec(
            impl_id=HIGH_FREQ_TRUNCATION_CODEC_ID,
            impl_version=HIGH_FREQ_TRUNCATION_CODEC_VERSION,
            impl_digest=digests.canonical_sha256(
                {
                    "impl_id": HIGH_FREQ_TRUNCATION_CODEC_ID,
                    "impl_version": HIGH_FREQ_TRUNCATION_CODEC_VERSION,
                }
            ),
        )

    plan_dict = _resolve_plan_dict(plan_payload)
    hf_basis = plan_dict.get("hf_basis") if isinstance(plan_dict.get("hf_basis"), dict) else None
    if not isinstance(hf_basis, dict):
        return {
            "status": "mismatch",
            "hf_score": None,
            "hf_trace_digest": None,
            "hf_evidence_summary": {
                "hf_status": "mismatch",
                "hf_failure_reason": "hf_subspace_missing",
            },
            "content_failure_reason": "hf_subspace_missing",
        }

    tfs = hf_basis.get("trajectory_feature_spec")
    if not isinstance(tfs, dict):
        return {
            "status": "failed",
            "hf_score": None,
            "hf_trace_digest": None,
            "hf_evidence_summary": {
                "hf_status": "failed",
                "hf_failure_reason": "hf_basis_missing_trajectory_feature_spec",
            },
            "content_failure_reason": "hf_basis_missing_trajectory_feature_spec",
        }

    detect_traj_cache = cfg.get("__detect_trajectory_latent_cache__")
    edit_timestep = int(tfs.get("edit_timestep", 0))
    detect_latent, resolution_status = detector_scoring.resolve_detect_trajectory_latent_for_timestep(
        detect_traj_cache,
        edit_timestep,
    )
    if detect_latent is None:
        return {
            "status": "absent",
            "hf_score": None,
            "hf_trace_digest": None,
            "hf_evidence_summary": {
                "hf_status": "absent",
                "hf_absent_reason": f"trajectory_latent_absent:{resolution_status}",
            },
            "content_failure_reason": f"trajectory_latent_absent:{resolution_status}",
        }

    expected_plan_digest = embed_time_plan_digest if isinstance(embed_time_plan_digest, str) and embed_time_plan_digest else plan_digest
    cfg.pop("__detect_hf_plan_digest_used__", None)
    hf_runtime_cfg = _build_detect_hf_runtime_cfg(cfg, expected_plan_digest)
    if isinstance(expected_plan_digest, str) and expected_plan_digest:
        cfg["__detect_hf_plan_digest_used__"] = expected_plan_digest
    detect_result = embedder.detect(
        latents_or_features=detect_latent,
        plan=plan_payload,
        cfg=hf_runtime_cfg,
        cfg_digest=cfg_digest,
        expected_plan_digest=expected_plan_digest,
    )
    evidence: Dict[str, Any]
    if isinstance(detect_result, tuple):
        detect_tuple = cast(tuple[Any, ...], detect_result)
        if len(detect_tuple) == 2:
            hf_score = detect_tuple[0]
            evidence_node = detect_tuple[1]
            evidence = cast(Dict[str, Any], evidence_node) if isinstance(evidence_node, dict) else {
                "status": "failed",
                "hf_score": None,
                "hf_trace_digest": None,
                "hf_evidence_summary": {
                    "hf_status": "failed",
                    "hf_failure_reason": "hf_detection_invalid_output",
                },
                "content_failure_reason": "hf_detection_invalid_output",
            }
        else:
            hf_score = None
            evidence = {
                "status": "failed",
                "hf_score": None,
                "hf_trace_digest": None,
                "hf_evidence_summary": {
                    "hf_status": "failed",
                    "hf_failure_reason": "hf_detection_invalid_output",
                },
                "content_failure_reason": "hf_detection_invalid_output",
            }
    else:
        hf_score = None
        evidence = {
            "status": "failed",
            "hf_score": None,
            "hf_trace_digest": None,
            "hf_evidence_summary": {
                "hf_status": "failed",
                "hf_failure_reason": "hf_detection_invalid_output",
            },
            "content_failure_reason": "hf_detection_invalid_output",
        }
    if "hf_score" not in evidence:
        evidence["hf_score"] = hf_score
    if "hf_evidence_summary" not in evidence:
        evidence["hf_evidence_summary"] = {
            "hf_status": evidence.get("status"),
            "hf_absent_reason": evidence.get("hf_absent_reason"),
            "hf_failure_reason": evidence.get("hf_failure_reason"),
        }
    if evidence.get("status") == "ok":
        summary_node = evidence.get("hf_evidence_summary")
        summary = cast(Dict[str, Any], summary_node) if isinstance(summary_node, dict) else {}
        if not isinstance(summary_node, dict):
            evidence["hf_evidence_summary"] = summary
        try:
            from main.watermarking.content_chain.high_freq_embedder import _prepare_hf_feature_vector

            feature_vector = _prepare_hf_feature_vector(detect_latent, hf_basis)
            coeffs = channel_hf.compute_hf_basis_projection(feature_vector, hf_basis)
            evidence["hf_attestation_values"] = np.asarray(coeffs, dtype=np.float32).reshape(-1).tolist()
            evidence["hf_attestation_status"] = "ok"
            evidence.pop("hf_attestation_failure_reason", None)
            summary["hf_attestation_status"] = "ok"
            summary.pop("hf_attestation_failure_reason", None)
        except Exception as exc:
            # HF attestation 投影失败必须可审计，禁止静默吞掉异常。
            failure_reason = f"hf_attestation_projection_failed:{type(exc).__name__}"
            evidence["hf_attestation_status"] = "failed"
            evidence["hf_attestation_failure_reason"] = failure_reason
            summary["hf_attestation_status"] = "failed"
            summary["hf_attestation_failure_reason"] = failure_reason
    return evidence


def _extract_lf_raw_score_from_trajectory(
    cfg: Dict[str, Any],
    plan_payload: Optional[Dict[str, Any]],
    plan_digest: Optional[str],
    cfg_digest: Optional[str],
    latent_cache: Any = None,
    detect_path: str = "low_freq_template_trajectory",
) -> tuple[Optional[float], Dict[str, Any]]:
    """
    功能：通过 detect 侧 trajectory 路径提取 LF 原始分数与 trace。

    Extract LF score and trace from the detect-side trajectory path.

    Args:
        cfg: Configuration mapping.
        plan_payload: Planner payload mapping.
        plan_digest: Detect-side plan digest.
        cfg_digest: Detect-side config digest.

    Returns:
        Tuple of LF score and LF trace mapping.
    """
    plan_dict = _resolve_plan_dict(plan_payload)
    lf_basis_for_decode = plan_dict.get("lf_basis") if isinstance(plan_dict.get("lf_basis"), dict) else None
    detect_traj_cache = latent_cache if latent_cache is not None else cfg.get("__detect_trajectory_latent_cache__")
    if lf_basis_for_decode is None:
        return None, {
            "lf_status": "absent",
            "lf_absent_reason": "lf_basis_missing",
            "lf_detect_path": detect_path,
        }
    if not isinstance(plan_digest, str) or not plan_digest:
        return None, {
            "lf_status": "absent",
            "lf_absent_reason": "lf_plan_digest_missing",
            "lf_detect_path": detect_path,
        }
    if detect_traj_cache is None or detect_traj_cache.is_empty():
        return None, {
            "lf_status": "absent",
            "lf_absent_reason": "lf_timestep_unresolved",
            "lf_detect_path": detect_path,
        }

    tfs = lf_basis_for_decode.get("trajectory_feature_spec")
    if not isinstance(tfs, dict) or tfs.get("feature_operator") != "masked_normalized_random_projection":
        return None, {
            "lf_status": "failed",
            "lf_failure_reason": "lf_trajectory_feature_spec_invalid",
            "lf_detect_path": detect_path,
        }

    edit_timestep = int(tfs.get("edit_timestep", 0))
    z_t, resolution_status = detector_scoring.resolve_detect_trajectory_latent_for_timestep(
        detect_traj_cache, edit_timestep
    )
    if z_t is None:
        return None, {
            "lf_status": "absent",
            "lf_absent_reason": "lf_timestep_unresolved",
            "lf_resolution_status": resolution_status,
            "lf_detect_path": detect_path,
        }

    try:
        from main.watermarking.content_chain.subspace.trajectory_feature_space import (
            extract_trajectory_feature_np,
        )

        phi = extract_trajectory_feature_np(np.asarray(z_t, dtype=np.float64), tfs)
        lf_impl_digest = digests.canonical_sha256(
            {
                "impl_id": LOW_FREQ_TEMPLATE_CODEC_ID,
                "impl_version": LOW_FREQ_TEMPLATE_CODEC_VERSION,
            }
        )
        lf_coder = LowFreqTemplateCodec(LOW_FREQ_TEMPLATE_CODEC_ID, LOW_FREQ_TEMPLATE_CODEC_VERSION, lf_impl_digest)
        lf_score, lf_detect_trace = lf_coder.detect_score(
            cfg=cfg,
            latent_features=phi,
            plan_digest=plan_digest,
            cfg_digest=cfg_digest,
            lf_basis=lf_basis_for_decode,
        )
        lf_trace = {
            "lf_status": lf_detect_trace.get("status", "failed"),
            "lf_score": lf_score,
            "lf_trace_digest": lf_detect_trace.get("lf_trace_digest"),
            "bp_converged": lf_detect_trace.get("bp_converged"),
            "bp_iteration_count": lf_detect_trace.get("bp_iteration_count"),
            "parity_check_digest": lf_detect_trace.get("parity_check_digest"),
            "lf_failure_reason": lf_detect_trace.get("lf_failure_reason"),
            "bp_converge_status": lf_detect_trace.get("bp_converge_status"),
            "lf_detect_path": detect_path,
        }
        failure_reason = lf_trace.get("lf_failure_reason")
        if isinstance(failure_reason, str) and failure_reason:
            if failure_reason == "lf_basis_required_but_absent":
                lf_trace["lf_status"] = "absent"
                lf_trace["lf_absent_reason"] = "lf_basis_missing"
            elif failure_reason == "projection_matrix_missing":
                lf_trace["lf_status"] = "absent"
                lf_trace["lf_absent_reason"] = "lf_projection_matrix_missing"
            elif failure_reason.startswith("phi_dim_mismatch"):
                lf_trace["lf_status"] = "failed"
                lf_trace["lf_failure_reason"] = "lf_dimension_mismatch"
            elif failure_reason == "plan_digest_missing":
                lf_trace["lf_status"] = "absent"
                lf_trace["lf_absent_reason"] = "lf_plan_digest_missing"
        return lf_score, lf_trace
    except Exception as exc:
        return None, {
            "lf_status": "failed",
            "lf_failure_reason": f"lf_trajectory_score_failed:{type(exc).__name__}",
            "lf_detect_path": detect_path,
        }


def _is_attacked_positive_input_record(input_record: Optional[Dict[str, Any]]) -> bool:
    """
    功能：判定当前 detect 输入是否属于 PW03 attacked_positive。 

    Detect whether the current record is a PW03 attacked_positive input.

    Args:
        input_record: Detect input record mapping.

    Returns:
        True when the input record is attacked_positive; otherwise False.
    """
    if not isinstance(input_record, dict):
        return False
    return input_record.get("sample_role") == "attacked_positive"


def _update_lf_trace_with_formal_exact_context(
    lf_trace: Dict[str, Any],
    lf_formal_exact_context: Dict[str, Any],
    image_path_source: Optional[str],
) -> None:
    """
    功能：将 formal exact 上下文追加到 LF trace，便于审计 attacked_positive 路径。 

    Append formal exact context fields into LF trace for attacked-positive auditing.

    Args:
        lf_trace: Mutable LF trace mapping.
        lf_formal_exact_context: Formal exact context mapping.
        image_path_source: Resolved detect image path source.

    Returns:
        None.
    """
    formal_exact_object_binding_status = lf_formal_exact_context.get("formal_exact_object_binding_status")
    if isinstance(formal_exact_object_binding_status, str) and formal_exact_object_binding_status:
        lf_trace["formal_exact_object_binding_status"] = formal_exact_object_binding_status

    formal_exact_evidence_source = lf_formal_exact_context.get("formal_exact_evidence_source")
    if isinstance(formal_exact_evidence_source, str) and formal_exact_evidence_source:
        lf_trace["formal_exact_evidence_source"] = formal_exact_evidence_source

    formal_exact_image_path_source = lf_formal_exact_context.get("formal_exact_image_path_source")
    if isinstance(formal_exact_image_path_source, str) and formal_exact_image_path_source:
        lf_trace["formal_exact_image_path_source"] = formal_exact_image_path_source

    image_conditioned_reconstruction_status = lf_formal_exact_context.get("image_conditioned_reconstruction_status")
    if isinstance(image_conditioned_reconstruction_status, str) and image_conditioned_reconstruction_status:
        lf_trace["image_conditioned_reconstruction_status"] = image_conditioned_reconstruction_status

    image_conditioned_reconstruction_available = lf_formal_exact_context.get("image_conditioned_reconstruction_available")
    if isinstance(image_conditioned_reconstruction_available, bool):
        lf_trace["image_conditioned_reconstruction_available"] = image_conditioned_reconstruction_available

    if isinstance(image_path_source, str) and image_path_source:
        lf_trace.setdefault("image_path_source", image_path_source)


def _build_attacked_positive_lf_unavailable_trace(
    lf_formal_exact_context: Dict[str, Any],
    image_path_source: Optional[str],
) -> Dict[str, Any]:
    """
    功能：为 attacked_positive 构造 fail-closed 的 LF unavailable trace。 

    Build a fail-closed LF trace when attacked-positive image-conditioned evidence
    is unavailable.

    Args:
        lf_formal_exact_context: Formal exact context mapping.
        image_path_source: Resolved detect image path source.

    Returns:
        LF trace mapping with explicit unavailable semantics.
    """
    image_conditioned_reconstruction_status = lf_formal_exact_context.get("image_conditioned_reconstruction_status")
    lf_trace: Dict[str, Any] = {
        "lf_status": "absent",
        "lf_absent_reason": "attack_image_conditioned_evidence_unavailable",
        "lf_detect_path": "low_freq_template_image_conditioned_attack",
    }
    if image_conditioned_reconstruction_status == "ok":
        lf_trace["lf_status"] = "failed"
        lf_trace.pop("lf_absent_reason", None)
        lf_trace["lf_failure_reason"] = "attack_image_conditioned_latent_cache_missing"
    _update_lf_trace_with_formal_exact_context(lf_trace, lf_formal_exact_context, image_path_source)
    return lf_trace


def _extract_lf_attestation_trace_bundle_from_trajectory(
    cfg: Dict[str, Any],
    plan_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """
    功能：从 detect 主路径提取 LF attestation 细粒度轨迹证据。

    Extract fine-grained LF attestation evidence from the detect-side
    trajectory cache and LF basis.

    Args:
        cfg: Configuration mapping.
        plan_payload: Planner payload mapping.

    Returns:
        LF trace bundle when available; otherwise None.
    """
    plan_dict = _resolve_plan_dict(plan_payload)
    lf_basis_for_decode = plan_dict.get("lf_basis") if isinstance(plan_dict.get("lf_basis"), dict) else None
    if not isinstance(lf_basis_for_decode, dict):
        return None

    detect_traj_cache = cfg.get("__detect_trajectory_latent_cache__")
    if detect_traj_cache is None or detect_traj_cache.is_empty():
        return None

    tfs = lf_basis_for_decode.get("trajectory_feature_spec")
    if not isinstance(tfs, dict) or tfs.get("feature_operator") != "masked_normalized_random_projection":
        return None

    basis_matrix_raw = lf_basis_for_decode.get("projection_matrix")
    if basis_matrix_raw is None:
        return None

    edit_timestep = int(tfs.get("edit_timestep", 0))
    detect_latent, _ = detector_scoring.resolve_detect_trajectory_latent_for_timestep(
        detect_traj_cache,
        edit_timestep,
    )
    if detect_latent is None:
        return None

    try:
        from main.watermarking.content_chain.subspace.trajectory_feature_space import (
            extract_trajectory_feature_np,
        )

        phi = extract_trajectory_feature_np(np.asarray(detect_latent, dtype=np.float64), tfs)
        basis_matrix_np = np.asarray(basis_matrix_raw, dtype=np.float64)
        basis_rank = int(lf_basis_for_decode.get("basis_rank", basis_matrix_np.shape[1]))

        if hasattr(phi, "detach"):
            latents_flat = phi.detach().cpu().numpy().astype(np.float64).reshape(-1)
        else:
            latents_flat = np.asarray(phi, dtype=np.float64).reshape(-1)

        if latents_flat.shape[0] != basis_matrix_np.shape[0]:
            latent_proj_spec = lf_basis_for_decode.get("latent_projection_spec")
            if not isinstance(latent_proj_spec, dict):
                return None
            feature_dim = int(latent_proj_spec.get("feature_dim", basis_matrix_np.shape[0]))
            proj_seed = int(latent_proj_spec.get("seed", 0))
            timestep_index = int(latent_proj_spec.get("edit_timestep", 0))
            sample_index = int(latent_proj_spec.get("sample_idx", 0))
            projection_seed = proj_seed + 7919 + timestep_index * 131 + sample_index
            index_rng = np.random.default_rng(projection_seed)
            projection_indices = index_rng.integers(0, max(1, latents_flat.shape[0]), size=feature_dim)
            latents_flat = latents_flat[projection_indices]

        if latents_flat.shape[0] != basis_matrix_np.shape[0]:
            return None

        coeffs_arr = np.dot(latents_flat, basis_matrix_np)
        projected_lf_coeffs = [float(value) for value in coeffs_arr[:basis_rank].tolist()]
        trajectory_feature_vector = [float(value) for value in latents_flat.tolist()]
        plan_digest = plan_payload.get("plan_digest") if isinstance(plan_payload, dict) else None
        basis_digest = plan_payload.get("basis_digest") if isinstance(plan_payload, dict) else None
        projection_seed = tfs.get("projection_seed")
        return {
            "lf_attestation_features": projected_lf_coeffs,
            "trajectory_feature_vector": trajectory_feature_vector,
            "trajectory_feature_digest": digests.canonical_sha256(trajectory_feature_vector),
            "projected_lf_digest": digests.canonical_sha256(projected_lf_coeffs),
            "plan_digest": plan_digest if isinstance(plan_digest, str) and plan_digest else None,
            "lf_basis_digest": (
                basis_digest if isinstance(basis_digest, str) and basis_digest else digests.canonical_sha256(lf_basis_for_decode)
            ),
            "projection_matrix_digest": digests.canonical_sha256(basis_matrix_raw),
            "trajectory_feature_spec_digest": digests.canonical_sha256(tfs),
            "projection_seed": int(projection_seed) if isinstance(projection_seed, (int, float)) else None,
        }
    except Exception:
        return None


def _extract_lf_attestation_features_from_trajectory(
    cfg: Dict[str, Any],
    plan_payload: Optional[Dict[str, Any]],
) -> Optional[List[float]]:
    """
    功能：从 detect 主路径提取 LF attestation 所需的系数向量。

    Extract the LF coefficient vector required by the attestation main path
    from the detect-side trajectory cache and LF basis.

    Args:
        cfg: Configuration mapping.
        plan_payload: Planner payload mapping.

    Returns:
        LF coefficient vector when available; otherwise None.
    """
    trace_bundle = _extract_lf_attestation_trace_bundle_from_trajectory(cfg, plan_payload)
    if not isinstance(trace_bundle, dict):
        return None
    lf_attestation_features = trace_bundle.get("lf_attestation_features")
    return cast(Optional[List[float]], lf_attestation_features if isinstance(lf_attestation_features, list) else None)


def _extract_content_raw_scores_from_image(
    cfg: Dict[str, Any],
    input_record: Optional[Dict[str, Any]],
    plan_payload: Optional[Dict[str, Any]],
    plan_digest: Optional[str],
    cfg_digest: Optional[str],
) -> tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    功能：从图像提取 LF/HF 原始分数 

            lf_trace_bundle=lf_attestation_trace_bundle if isinstance(lf_attestation_trace_bundle, dict) else None,
    Extract LF/HF raw scores from image artifact for calibration-ready evidence.

    Args:
        cfg: Configuration mapping.
        input_record: Embed-side record mapping.
        plan_payload: Planner payload mapping.
        plan_digest: Detect-side plan digest.
        cfg_digest: Detect-side config digest.

    Returns:
        Tuple of (lf_score, hf_score, traces).
    """
    plan_dict = _resolve_plan_dict(plan_payload)
    band_spec_node = plan_dict.get("band_spec")
    band_spec = cast(Dict[str, Any], band_spec_node) if isinstance(band_spec_node, dict) else {}
    routing_summary_node = band_spec.get("hf_selector_summary")
    routing_summary = cast(Dict[str, Any], routing_summary_node) if isinstance(routing_summary_node, dict) else {}

    watermark_cfg_node = cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_cfg_node) if isinstance(watermark_cfg_node, dict) else {}

    key_material = digests.canonical_sha256(
        {
            "plan_digest": plan_digest,
            "cfg_digest": cfg_digest,
            "key_id": watermark_cfg.get("key_id"),
            "pattern_id": watermark_cfg.get("pattern_id"),
        }
    )

    lf_cfg_node = watermark_cfg.get("lf")
    lf_cfg = cast(Dict[str, Any], lf_cfg_node) if isinstance(lf_cfg_node, dict) else {}
    ecc_value = lf_cfg.get("ecc", 3)
    lf_score = None
    lf_trace: Dict[str, Any] = {"lf_status": "absent", "lf_absent_reason": "lf_unavailable"}
    lf_formal_exact_context_node = cfg.get("__lf_formal_exact_context__")
    lf_formal_exact_context = (
        cast(Dict[str, Any], lf_formal_exact_context_node)
        if isinstance(lf_formal_exact_context_node, dict)
        else {}
    )
    attacked_positive_input_record = _is_attacked_positive_input_record(input_record)

    image_path, image_path_source = _resolve_detect_image_path_with_source(cfg, input_record)
    image_array: Optional[Any] = None
    if image_path is not None:
        try:
            image_array = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        except Exception:
            image_array = None

    if isinstance(ecc_value, str) and ecc_value == "sparse_ldpc":
        if attacked_positive_input_record:
            attack_image_conditioned_cache = cfg.get("__lf_attacked_image_conditioned_latent_cache__")
            if (
                attack_image_conditioned_cache is None
                or not hasattr(attack_image_conditioned_cache, "is_empty")
                or attack_image_conditioned_cache.is_empty()
            ):
                lf_trace = _build_attacked_positive_lf_unavailable_trace(
                    lf_formal_exact_context,
                    image_path_source if isinstance(image_path_source, str) else None,
                )
            else:
                lf_score, lf_trace = _extract_lf_raw_score_from_trajectory(
                    cfg=cfg,
                    plan_payload=plan_payload,
                    plan_digest=plan_digest,
                    cfg_digest=cfg_digest,
                    latent_cache=attack_image_conditioned_cache,
                    detect_path="low_freq_template_image_conditioned_attack",
                )
                _update_lf_trace_with_formal_exact_context(
                    lf_trace,
                    lf_formal_exact_context,
                    image_path_source if isinstance(image_path_source, str) else None,
                )
        else:
            lf_score, lf_trace = _extract_lf_raw_score_from_trajectory(
                cfg=cfg,
                plan_payload=plan_payload,
                plan_digest=plan_digest,
                cfg_digest=cfg_digest,
            )
    else:
        raise RuntimeError(
            "image_dct_fallback path reached in _extract_content_raw_scores_from_image: "
            "this path was removed in v2.0 single-path closure. "
            "Only ecc='sparse_ldpc' trajectory path is permitted. "
            "Ensure image_domain_sidecar_enabled=false in paper mode config."
        )

    hf_cfg_node = watermark_cfg.get("hf")
    hf_cfg = cast(Dict[str, Any], hf_cfg_node) if isinstance(hf_cfg_node, dict) else {}
    hf_enabled = bool(hf_cfg.get("enabled", False))
    if hf_enabled:
        if image_array is None:
            hf_score = None
            hf_trace = {"hf_status": "absent", "hf_absent_reason": "detect_image_absent"}
        else:
            raise RuntimeError(
                "hf_image_texture_score path reached in _extract_content_raw_scores_from_image: "
                "this path was removed in v2.0 single-path closure. "
                "Only HighFreqTruncationCodec class entry is permitted for HF detection. "
                "Ensure image_domain_sidecar_enabled=false in paper mode config."
            )
    else:
        hf_score = None
        hf_trace = {"hf_status": "absent", "hf_absent_reason": "hf_disabled_by_config"}

    if isinstance(image_path_source, str) and image_path_source:
        lf_trace["image_path_source"] = image_path_source
        hf_trace["image_path_source"] = image_path_source

    return lf_score, hf_score, {"lf": lf_trace, "hf": hf_trace}


def _bind_raw_scores_to_content_payload(
    content_evidence_payload: Dict[str, Any],
    lf_score: Optional[float],
    hf_score: Optional[float],
    traces: Dict[str, Any],
) -> None:
    """
    功能：将 LF/HF 原始分数与 trace 写入 content evidence。

    Bind LF/HF raw scores and traces into content evidence score_parts.
    """
    score_parts_node = content_evidence_payload.get("score_parts")
    score_parts: Dict[str, Any]
    if isinstance(score_parts_node, dict):
        score_parts = cast(Dict[str, Any], score_parts_node)
    else:
        score_parts = {}
        content_evidence_payload["score_parts"] = score_parts

    lf_node = traces.get("lf")
    lf_trace = cast(Dict[str, Any], lf_node) if isinstance(lf_node, dict) else {}
    hf_node = traces.get("hf")
    hf_trace = cast(Dict[str, Any], hf_node) if isinstance(hf_node, dict) else {}

    content_evidence_payload["lf_score"] = lf_score
    content_evidence_payload[eval_metrics.LF_CHANNEL_SCORE_NAME] = lf_score
    score_parts["lf_detect_trace"] = lf_trace
    score_parts[eval_metrics.LF_CHANNEL_SCORE_NAME] = lf_score
    score_parts["lf_score"] = lf_score
    lf_template_status = lf_trace.get("lf_status")
    if isinstance(lf_template_status, str) and lf_template_status:
        score_parts["lf_template_status"] = lf_template_status
    # 补齐 lf_status 顶层口径（if-not-in 守卫，不覆写统一提取器已写入值）。
    if "lf_status" not in score_parts and isinstance(lf_template_status, str) and lf_template_status:
        score_parts["lf_status"] = lf_template_status

    # （P1 修复）BP 收敛状态降级守卫：
    # 当 bp_converge_status="degraded" 且 lf_status 仍为 "ok" 时，将顶层 lf_status 覆写为 "degraded"。
    # 不改 low_freq_coder trace["status"]，不影响统一提取器决策链，仅修正诊断字段语义。
    _bp_converge_status = lf_trace.get("bp_converge_status")
    if _bp_converge_status == "degraded" and score_parts.get("lf_status") == "ok":
        score_parts["lf_status"] = "degraded"
        score_parts["lf_status_degraded_reason"] = "bp_not_converged"

    hf_status = hf_trace.get("hf_status")
    if hf_status == "absent" and hf_trace.get("hf_absent_reason") == "hf_disabled_by_config":
        content_evidence_payload.pop("hf_score", None)
        content_evidence_payload.pop("hf_trace_digest", None)
        score_parts.pop("hf_status", None)
        score_parts.pop("hf_metrics", None)
        score_parts.pop("hf_absent_reason", None)
        score_parts.pop("hf_failure_reason", None)
        score_parts.pop("hf_detect_trace", None)
    else:
        content_evidence_payload["hf_score"] = hf_score
        score_parts["hf_detect_trace"] = hf_trace
        score_parts["hf_status"] = hf_status if isinstance(hf_status, str) else "failed"
        if "hf_absent_reason" in hf_trace:
            score_parts["hf_absent_reason"] = hf_trace.get("hf_absent_reason")
        if "hf_failure_reason" in hf_trace:
            score_parts["hf_failure_reason"] = hf_trace.get("hf_failure_reason")

    _synchronize_content_score_aliases(content_evidence_payload)


def _populate_detect_mask_digest_from_input_record(
    content_evidence_payload: Dict[str, Any],
    input_record: Optional[Dict[str, Any]],
) -> None:
    """
    功能：当 detect content 成功但 mask_digest 缺失时，透传 input_record 中的值。

    Populate detect-side mask_digest from input_record when status is ok but digest is absent.

    Args:
        content_evidence_payload: Mutable detect content payload.
        input_record: Optional upstream record payload.

    Returns:
        None.
    """
    if not isinstance(content_evidence_payload, dict):
        return
    if content_evidence_payload.get("status") != "ok":
        return

    current_mask_digest = content_evidence_payload.get("mask_digest")
    if isinstance(current_mask_digest, str) and current_mask_digest:
        return

    if not isinstance(input_record, dict):
        return
    input_content_node = input_record.get("content_evidence")
    if not isinstance(input_content_node, dict):
        return
    input_content_payload = cast(Dict[str, Any], input_content_node)
    input_mask_digest = input_content_payload.get("mask_digest")
    if isinstance(input_mask_digest, str) and input_mask_digest:
        content_evidence_payload["mask_digest"] = input_mask_digest


def _resolve_detect_image_path_with_source(
    cfg: Dict[str, Any],
    input_record: Optional[Dict[str, Any]],
) -> tuple[Optional[Path], Optional[str]]:
    candidates: list[tuple[str, Any]] = [
        ("cfg.__detect_input_image_path__", cfg.get("__detect_input_image_path__")),
        ("cfg.input_image_path", cfg.get("input_image_path")),
    ]

    if isinstance(input_record, dict):
        candidates.extend(
            [
                ("input_record.watermarked_path", input_record.get("watermarked_path")),
                ("input_record.image_path", input_record.get("image_path")),
            ]
        )
        record_inputs = input_record.get("inputs")
        if isinstance(record_inputs, dict):
            candidates.append(("input_record.inputs.input_image_path", record_inputs.get("input_image_path")))

    for source_name, value in candidates:
        if isinstance(value, str) and value:
            path = Path(value).resolve()
            if path.exists() and path.is_file():
                return path, source_name
    return None, None


def _resolve_detect_image_path(cfg: Dict[str, Any], input_record: Optional[Dict[str, Any]]) -> Optional[Path]:
    resolved_path, _ = _resolve_detect_image_path_with_source(cfg, input_record)
    return resolved_path


def _build_content_inputs_for_detect(
    cfg: Dict[str, Any],
    input_record: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    功能：构 detect 阶段 content extractor 主输入 

    Build content extractor inputs for detect stage with explicit input priority.
    Falls back to reading input_image_path from input_record when available.

    Args:
        cfg: Configuration mapping.
        input_record: Optional embed/detect input record.

    Returns:
        Input mapping with explicit source marker when available, otherwise None.
    """
    explicit_image = cfg.get("__detect_input_image__")
    explicit_latent = None

    if isinstance(input_record, dict):
        if explicit_image is None:
            explicit_image = input_record.get("image")
        if explicit_latent is None:
            explicit_latent = input_record.get("latent")

        record_inputs_node = input_record.get("inputs")
        record_inputs = cast(Dict[str, Any], record_inputs_node) if isinstance(record_inputs_node, dict) else {}
        if explicit_image is None:
            explicit_image = record_inputs.get("image")
        if explicit_latent is None:
            explicit_latent = record_inputs.get("latent")

    inputs: Dict[str, Any] = {}
    if explicit_image is not None:
        inputs["image"] = explicit_image
        inputs["input_source"] = "image"
        if explicit_latent is not None:
            inputs["latent"] = explicit_latent
        return inputs

    if explicit_latent is not None:
        inputs["latent"] = explicit_latent
        inputs["input_source"] = "latent"
        return inputs

    image_path, image_path_source = _resolve_detect_image_path_with_source(cfg, input_record)
    if image_path is not None:
        inputs["image_path"] = str(image_path)
        inputs["input_source"] = "image_path"
        if isinstance(image_path_source, str) and image_path_source:
            inputs["image_path_source"] = image_path_source

    input_content_evidence: Dict[str, Any] = {}
    if isinstance(input_record, dict):
        content_node = input_record.get("content_evidence")
        if isinstance(content_node, dict):
            input_content_evidence = cast(Dict[str, Any], content_node)

    expected_plan_digest = _resolve_expected_plan_digest(input_record)
    if isinstance(expected_plan_digest, str) and expected_plan_digest:
        inputs["expected_plan_digest"] = expected_plan_digest

    observed_plan_digest = input_content_evidence.get("plan_digest")
    if isinstance(observed_plan_digest, str) and observed_plan_digest:
        inputs["observed_plan_digest"] = observed_plan_digest
        inputs["plan_digest"] = observed_plan_digest

    for evidence_key in ["lf_evidence", "hf_evidence", "statistics", "injection_evidence"]:
        evidence_value = input_content_evidence.get(evidence_key)
        if evidence_value is not None:
            inputs[evidence_key] = evidence_value

    if inputs:
        return inputs
    return None


def _build_lf_detect_evidence(
    lf_score: Optional[float],
    lf_trace: Any,
) -> Dict[str, Any]:
    """
    功能：将 detect 侧 LF 闭环结果封装为正式 evidence。

    Build LF detect evidence so UnifiedContentExtractor consumes structured evidence rather than
    proxy raw-score side channels.

    Args:
        lf_score: Detect-side LF score.
        lf_trace: Detect-side LF trace mapping.

    Returns:
        LF evidence mapping.
    """
    trace_payload = cast(Dict[str, Any], lf_trace) if isinstance(lf_trace, dict) else {}
    lf_status = trace_payload.get("lf_status") if isinstance(trace_payload.get("lf_status"), str) else "absent"
    if lf_status == "ok" and isinstance(lf_score, (int, float)):
        return {
            "status": "ok",
            "lf_score": float(lf_score),
            "lf_trace_digest": trace_payload.get("lf_trace_digest"),
            "bp_converged": trace_payload.get("bp_converged"),
            "bp_iteration_count": trace_payload.get("bp_iteration_count"),
            "parity_check_digest": trace_payload.get("parity_check_digest"),
            "lf_evidence_summary": {
                "lf_status": "ok",
                "lf_detect_variant": trace_payload.get("lf_detect_path"),
                "bp_converged": trace_payload.get("bp_converged"),
                "bp_iteration_count": trace_payload.get("bp_iteration_count"),
                "parity_check_digest": trace_payload.get("parity_check_digest"),
            },
        }
    if lf_status == "mismatch":
        return {
            "status": "mismatch",
            "lf_score": None,
            "content_failure_reason": trace_payload.get("lf_failure_reason") or "lf_plan_mismatch",
            "lf_trace_digest": trace_payload.get("lf_trace_digest"),
            "lf_evidence_summary": {
                "lf_status": "mismatch",
                "lf_failure_reason": trace_payload.get("lf_failure_reason") or "lf_plan_mismatch",
            },
        }
    if lf_status == "failed":
        return {
            "status": "failed",
            "lf_score": None,
            "content_failure_reason": trace_payload.get("lf_failure_reason") or "lf_detection_failed",
            "lf_trace_digest": trace_payload.get("lf_trace_digest"),
            "lf_evidence_summary": {
                "lf_status": "failed",
                "lf_failure_reason": trace_payload.get("lf_failure_reason") or "lf_detection_failed",
            },
        }
    return {
        "status": "absent",
        "lf_score": None,
        "content_failure_reason": trace_payload.get("lf_absent_reason"),
        "lf_trace_digest": trace_payload.get("lf_trace_digest"),
        "lf_evidence_summary": {
            "lf_status": "absent",
            "lf_absent_reason": trace_payload.get("lf_absent_reason"),
        },
    }


def _resolve_sidecar_disabled_reason(
    paper_enabled: bool,
    enable_image_sidecar: bool,
) -> str:
    """
    功能：区分 formal profile 与 ablation 的 sidecar 关闭语义。

    Resolve detect-side sidecar-disabled semantics.

    Args:
        paper_enabled: Whether paper faithfulness is enabled.
        enable_image_sidecar: Effective sidecar switch after ablation normalization.

    Returns:
        Canonical sidecar-disabled reason string.
    """
    if paper_enabled:
        return "formal_profile_sidecar_disabled"
    if not enable_image_sidecar:
        return "ablation_sidecar_disabled"
    return "image_domain_sidecar_disabled"


def _resolve_plan_dict(plan_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(plan_payload, dict):
        plan_node = plan_payload.get("plan")
        if isinstance(plan_node, dict):
            return cast(Dict[str, Any], plan_node)
        return plan_payload
    return {}


def _build_lf_image_embed_params_for_detect(cfg: Dict[str, Any]) -> Dict[str, Any]:
    watermark_node = cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
    lf_node = watermark_cfg.get("lf")
    lf_cfg = cast(Dict[str, Any], lf_node) if isinstance(lf_node, dict) else {}
    ecc_value = lf_cfg.get("ecc", 1)
    redundancy = ecc_value if isinstance(ecc_value, int) else 1
    return {
        "dct_block_size": int(lf_cfg.get("dct_block_size", 8)),
        "lf_coeff_indices": lf_cfg.get("lf_coeff_indices", [[1, 1], [1, 2], [2, 1]]),
        "alpha": float(lf_cfg.get("strength", 1.5)),
        "redundancy": int(redundancy),
        "variance": float(lf_cfg.get("variance", 1.5)),
    }


def _build_hf_image_embed_params_for_detect(cfg: Dict[str, Any]) -> Dict[str, Any]:
    watermark_node = cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
    hf_node = watermark_cfg.get("hf")
    hf_cfg = cast(Dict[str, Any], hf_node) if isinstance(hf_node, dict) else {}
    return {
        "beta": float(hf_cfg.get("tau", 2.0)),
        "tail_truncation_ratio": float(hf_cfg.get("tail_truncation_ratio", 0.1)),
        "tail_truncation_mode": hf_cfg.get("tail_truncation_mode", "projection_tail_truncation"),
        "sampling_stride": int(hf_cfg.get("sampling_stride", 1)),
    }


def _merge_injection_evidence(content_evidence_payload: Dict[str, Any], injection_evidence: Dict[str, Any]) -> None:
    """
    功能：合并注入证据到 content_evidence 
    
    Merge injection evidence into content evidence payload using registered fields.

    Args:
        content_evidence_payload: Mutable content evidence mapping.
        injection_evidence: Injection evidence mapping.

    Returns:
        None.
    """
    content_evidence_payload["injection_status"] = injection_evidence.get("status")
    content_evidence_payload["injection_absent_reason"] = injection_evidence.get("injection_absent_reason")
    content_evidence_payload["injection_failure_reason"] = injection_evidence.get("injection_failure_reason")
    content_evidence_payload["injection_trace_digest"] = injection_evidence.get("injection_trace_digest")
    content_evidence_payload["injection_params_digest"] = injection_evidence.get("injection_params_digest")
    content_evidence_payload["injection_metrics"] = injection_evidence.get("injection_metrics")
    content_evidence_payload["subspace_binding_digest"] = injection_evidence.get("subspace_binding_digest")
    content_evidence_payload["lf_impl_binding"] = injection_evidence.get("lf_impl_binding")
    content_evidence_payload["hf_impl_binding"] = injection_evidence.get("hf_impl_binding")


def _evaluate_injection_consistency(
    input_record: Optional[Dict[str, Any]],
    injection_evidence: Optional[Dict[str, Any]]
) -> tuple[str, Optional[str]]:
    """
    功能：校验 embed/detect 两端注入证据一致性。
    
    Evaluate injection evidence consistency between embed-time record and detect-time runtime.

    Args:
        input_record: Embed-time input record mapping or None.
        injection_evidence: Detect-time injection evidence mapping or None.

    Returns:
        Tuple of (status, mismatch_reason_or_none).
        status is one of: "ok", "absent", "mismatch".
    """
    embed_injection: Optional[Dict[str, Any]] = None
    if isinstance(input_record, dict):
        for key in ["content_evidence_payload", "content_evidence", "content_result"]:
            candidate = input_record.get(key)
            if isinstance(candidate, dict) and "injection_status" in candidate:
                embed_injection = cast(Dict[str, Any], candidate)
                break

    if embed_injection is None:
        # 向后兼容：embed 未提供注入证据时不触发缺失分支。
        return "ok", None
    if injection_evidence is None:
        return "absent", "injection_evidence_missing"

    embed_status = embed_injection.get("injection_status")
    detect_status = injection_evidence.get("status")
    if embed_status == "mismatch" or detect_status == "mismatch":
        return "mismatch", "injection_status_mismatch"
    if embed_status != "ok" or detect_status != "ok":
        return "absent", "injection_status_not_ok"

    embed_trace_digest = embed_injection.get("injection_trace_digest")
    detect_trace_digest = injection_evidence.get("injection_trace_digest")
    embed_params_digest = embed_injection.get("injection_params_digest")
    detect_params_digest = injection_evidence.get("injection_params_digest")
    embed_binding_digest = embed_injection.get("subspace_binding_digest")
    detect_binding_digest = injection_evidence.get("subspace_binding_digest")

    # injection_trace_digest、injection_params_digest、subspace_binding_digest 均为运行时摘要。
    # 这些摘要与 latent 值相关：embed 与 detect 使用不同推理种子时，会生成不同的 latent 轨迹。
    # 注入到不同 latent 上的修改向量和幅度天然不同，因此三者跨 run 不要求相等。
    # 此处仅保留格式有效性检查，不再作相等性门禁，避免因预期差异触发 false mismatch。
    # 真实的计划一致性由 plan_digest 与 plan_override_for_orchestrator 机制保证。
    if not isinstance(embed_trace_digest, str) or not isinstance(detect_trace_digest, str):
        # trace digest 格式无效：降级为 absent 而非 mismatch，不阻断主链。
        return "absent", "injection_trace_digest_invalid"
    if not isinstance(embed_params_digest, str) or not isinstance(detect_params_digest, str):
        # params digest 格式无效：降级为 absent 而非 mismatch，不阻断主链。
        return "absent", "injection_params_digest_invalid"
    # run 间等值校验已移除：trace/params/binding digest 均为 latent-dependent，不可跨 run 比对。
    return "ok", None


def _evaluate_paper_impl_binding_consistency(
    cfg: Dict[str, Any],
    injection_evidence: Optional[Dict[str, Any]],
    input_record: Optional[Dict[str, Any]] = None,
) -> tuple[str, Optional[str]]:
    """
    功能：在 paper 模式下校 HF/LF impl 绑定一致性 

    Validate impl binding consistency for paper mode and reject fallback-only claims.

    Args:
        cfg: Runtime configuration mapping.
        injection_evidence: Injection evidence mapping.

    Returns:
        Tuple of (status, reason) where status in {ok, absent, mismatch}.
    """
    paper_cfg_node = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_cfg_node) if isinstance(paper_cfg_node, dict) else {}
    if not bool(paper_cfg.get("enabled", False)):
        return "ok", None

    watermark_node = cfg.get("watermark")
    watermark_cfg = cast(Dict[str, Any], watermark_node) if isinstance(watermark_node, dict) else {}
    lf_node = watermark_cfg.get("lf")
    lf_cfg = cast(Dict[str, Any], lf_node) if isinstance(lf_node, dict) else {}
    ecc_value = lf_cfg.get("ecc", "sparse_ldpc")
    if isinstance(ecc_value, int):
        return "mismatch", "lf_ecc_int_not_allowed_under_paper_mode"

    if isinstance(injection_evidence, dict):
        detect_status = injection_evidence.get("status")
        if isinstance(detect_status, str) and detect_status != "ok":
            return "absent", "paper_impl_binding_injection_status_not_ok"

    resolved_binding_source: Optional[Dict[str, Any]] = injection_evidence if isinstance(injection_evidence, dict) else None
    if resolved_binding_source is None:
        resolved_binding_source = _extract_embed_impl_binding_source(input_record)
    if resolved_binding_source is None:
        return "absent", "paper_impl_binding_evidence_absent"

    for channel_name in ["lf_impl_binding", "hf_impl_binding"]:
        binding_node = resolved_binding_source.get(channel_name)
        if not isinstance(binding_node, dict):
            return "mismatch", f"{channel_name}_missing_under_paper_mode"
        binding_payload = cast(Dict[str, Any], binding_node)
        impl_selected = binding_payload.get("impl_selected")
        if not isinstance(impl_selected, str) or not impl_selected:
            return "mismatch", f"{channel_name}_impl_selected_absent"
        evidence_level = binding_payload.get("evidence_level")
        # 正式路径只允许 primary / ablation_disabled；其余为非正式路径绑定。
        if evidence_level not in {"primary", "ablation_disabled", None}:
            return "mismatch", f"{channel_name}_non_primary_binding_under_paper_mode"
    return "ok", None


def _extract_embed_impl_binding_source(input_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    功能：从 embed 输入记录提取 impl 绑定证据来源 

    Extract LF/HF impl binding source from embed-time record fields.

    Args:
        input_record: Optional embed-time input record mapping.

    Returns:
        Mapping containing lf_impl_binding/hf_impl_binding if available.
    """
    if input_record is None:
        return None

    for field_name in ["content_evidence_payload", "content_evidence", "content_result"]:
        candidate = input_record.get(field_name)
        if isinstance(candidate, dict):
            candidate_payload = cast(Dict[str, Any], candidate)
            has_lf = isinstance(candidate_payload.get("lf_impl_binding"), dict)
            has_hf = isinstance(candidate_payload.get("hf_impl_binding"), dict)
            if has_lf or has_hf:
                return candidate_payload
    return None


def _evaluate_paper_faithfulness_consistency(
    input_record: Optional[Dict[str, Any]]
) -> tuple[str, list[str], list[str], list[str]]:
    """
    功能：校 paper faithfulness 证据一致性（S-D 必达） 

    Evaluate paper faithfulness evidence consistency between embed-time record.
    Validates: pipeline_fingerprint_digest, injection_site_digest, paper_spec_digest.
    Returns structured failure semantics: absent_reasons / mismatch_reasons / fail_reasons.

    Args:
        input_record: Embed-time input record mapping or None.

    Returns:
        Tuple of (status, absent_reasons, mismatch_reasons, fail_reasons).
        status is one of: "ok", "absent", "mismatch", "failed".
        absent_reasons: list of tokens for missing required evidence (non-empty if status="absent").
        mismatch_reasons: list of tokens for inconsistent evidence (non-empty if status="mismatch").
        fail_reasons: list of tokens for failed validation (non-empty if status="failed").

    Raises:
        TypeError: If input_record type is invalid.
    """
    absent_reasons: list[str] = []
    mismatch_reasons: list[str] = []
    fail_reasons: list[str] = []

    if input_record is None:
        absent_reasons.append("input_record_is_none")
        return "absent", absent_reasons, mismatch_reasons, fail_reasons

    # 提取 embed-time paper faithfulness 证据 
    embed_content_evidence: Optional[Dict[str, Any]] = None
    for key in ["content_evidence_payload", "content_evidence", "content_result"]:
        candidate = input_record.get(key)
        if isinstance(candidate, dict):
            embed_content_evidence = cast(Dict[str, Any], candidate)
            break

    # (1) 验证 content_evidence 存在性（整体 absent 前置检查） 
    if not isinstance(embed_content_evidence, dict):
        absent_reasons.append("content_evidence_absent")
        return "absent", absent_reasons, mismatch_reasons, fail_reasons

    # content_evidence 存在说明 embed 侧运行了，后续缺失归类为 mismatch 
    paper_node = input_record.get("paper_faithfulness")
    embed_paper_faithfulness = cast(Dict[str, Any], paper_node) if isinstance(paper_node, dict) else None

    # (2) 验证 paper_spec_digest 存在性（mismatch vs fail） 
    if embed_paper_faithfulness is not None:
        spec_digest = embed_paper_faithfulness.get("spec_digest")
        if spec_digest == "<absent>":
            mismatch_reasons.append("paper_spec_digest_absent_or_invalid")
        elif spec_digest == "<failed>":
            fail_reasons.append("paper_spec_digest_marked_failed")
        elif not isinstance(spec_digest, str) or not spec_digest:
            mismatch_reasons.append("paper_spec_digest_missing")
    else:
        mismatch_reasons.append("paper_faithfulness_section_absent")

    # (3) 验证 pipeline_fingerprint_digest 存在性（mismatch vs fail） 
    pipeline_fingerprint_digest = embed_content_evidence.get("pipeline_fingerprint_digest")
    if pipeline_fingerprint_digest == "<absent>":
        mismatch_reasons.append("pipeline_fingerprint_digest_marked_absent")
    elif pipeline_fingerprint_digest == "<failed>":
        fail_reasons.append("pipeline_fingerprint_digest_marked_failed")
    elif not isinstance(pipeline_fingerprint_digest, str) or not pipeline_fingerprint_digest:
        mismatch_reasons.append("pipeline_fingerprint_digest_missing")

    # (4) 验证 injection_site_digest 存在性（mismatch vs fail） 
    injection_site_digest = embed_content_evidence.get("injection_site_digest")
    if injection_site_digest == "<absent>":
        mismatch_reasons.append("injection_site_digest_marked_absent")
    elif injection_site_digest == "<failed>":
        fail_reasons.append("injection_site_digest_marked_failed")
    elif not isinstance(injection_site_digest, str) or not injection_site_digest:
        mismatch_reasons.append("injection_site_digest_missing")

    # (5) 验证 alignment_digest 存在性（mismatch vs fail） 
    alignment_digest = embed_content_evidence.get("alignment_digest")
    if alignment_digest == "<absent>":
        mismatch_reasons.append("alignment_digest_marked_absent")
    elif alignment_digest == "<failed>":
        fail_reasons.append("alignment_digest_marked_failed")
    elif not isinstance(alignment_digest, str) or not alignment_digest:
        mismatch_reasons.append("alignment_digest_missing")

    # (6) 决定最 status（优先级：failed > mismatch > absent > ok） 
    if len(fail_reasons) > 0:
        return "failed", absent_reasons, mismatch_reasons, fail_reasons
    if len(mismatch_reasons) > 0:
        return "mismatch", absent_reasons, mismatch_reasons, fail_reasons
    if len(absent_reasons) > 0:
        return "absent", absent_reasons, mismatch_reasons, fail_reasons

    return "ok", absent_reasons, mismatch_reasons, fail_reasons


def _extract_lf_evidence_from_input_record(input_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    功能：从 embed record 中提 LF 证据 

    Extract LF evidence payload from embed-time input record.

    Args:
        input_record: Optional input record mapping.

    Returns:
        LF evidence mapping or None.
    """
    if input_record is None:
        return None
    for key in ["content_evidence_payload", "content_evidence", "content_result"]:
        candidate = input_record.get(key)
        if not isinstance(candidate, dict):
            continue
        candidate_payload = cast(Dict[str, Any], candidate)
        if "lf_score" in candidate_payload or "lf_trace_digest" in candidate_payload or candidate_payload.get("status") in {"ok", "failed", "mismatch", "absent"}:
            return candidate_payload
    return None


def _resolve_expected_plan_digest(input_record: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    功能：从输入记录解析 expected plan_digest 

    Resolve expected plan digest strictly from bound input record payload.

    Args:
        input_record: Embed-side bound input record.

    Returns:
        Expected plan digest string or None.
    """
    if input_record is None:
        return None

    direct = input_record.get("plan_digest")
    if isinstance(direct, str) and direct:
        return direct

    for key in ["content_evidence_payload", "content_evidence", "content_result"]:
        payload = input_record.get(key)
        if not isinstance(payload, dict):
            continue
        payload_mapping = cast(Dict[str, Any], payload)
        candidate = payload_mapping.get("plan_digest")
        if isinstance(candidate, str) and candidate:
            return candidate

        injection_site_spec = payload_mapping.get("injection_site_spec")
        if isinstance(injection_site_spec, dict):
            injection_site_spec_payload = cast(Dict[str, Any], injection_site_spec)
            injection_rule_summary = injection_site_spec_payload.get("injection_rule_summary")
            if isinstance(injection_rule_summary, dict):
                injection_rule_summary_payload = cast(Dict[str, Any], injection_rule_summary)
                summary_plan_digest = injection_rule_summary_payload.get("plan_digest")
                if isinstance(summary_plan_digest, str) and summary_plan_digest:
                    return summary_plan_digest

    embed_trace = input_record.get("embed_trace")
    if isinstance(embed_trace, dict):
        embed_trace_payload = cast(Dict[str, Any], embed_trace)
        trace_plan_digest = embed_trace_payload.get("plan_digest")
        if isinstance(trace_plan_digest, str) and trace_plan_digest:
            return trace_plan_digest

        trace_injection = embed_trace_payload.get("injection_evidence")
        if isinstance(trace_injection, dict):
            trace_injection_payload = cast(Dict[str, Any], trace_injection)
            trace_injection_plan_digest = trace_injection_payload.get("plan_digest")
            if isinstance(trace_injection_plan_digest, str) and trace_injection_plan_digest:
                return trace_injection_plan_digest

    top_level_injection = input_record.get("injection_evidence")
    if isinstance(top_level_injection, dict):
        top_level_injection_payload = cast(Dict[str, Any], top_level_injection)
        injection_plan_digest = top_level_injection_payload.get("plan_digest")
        if isinstance(injection_plan_digest, str) and injection_plan_digest:
            return injection_plan_digest

    subspace_plan = input_record.get("subspace_plan")
    if isinstance(subspace_plan, dict):
        subspace_plan_payload = cast(Dict[str, Any], subspace_plan)
        if subspace_plan_payload:
            return digests.canonical_sha256(subspace_plan_payload)
    return None


def _collect_plan_mismatch_reasons(
    embed_time_plan_digest: Any,
    detect_time_plan_digest: Any,
    embed_time_basis_digest: Any,
    detect_time_basis_digest: Any,
    embed_time_planner_impl_identity: Any,
    detect_time_planner_impl_identity: Any
) -> list[str]:
    """
    功能：收集计划锚点不一致原因 

    Collect mismatch reasons for plan/basis/impl identity anchors.

    Args:
        embed_time_plan_digest: Embed-time plan digest.
        detect_time_plan_digest: Detect-time recomputed plan digest.
        embed_time_basis_digest: Embed-time basis digest.
        detect_time_basis_digest: Detect-time recomputed basis digest.
        embed_time_planner_impl_identity: Embed-time planner impl identity payload.
        detect_time_planner_impl_identity: Detect-time planner impl identity payload.

    Returns:
        List of mismatch reason tokens.
    """
    reasons: list[str] = []
    if isinstance(embed_time_plan_digest, str) and isinstance(detect_time_plan_digest, str):
        if embed_time_plan_digest != detect_time_plan_digest:
            reasons.append("plan_digest_mismatch")
    if isinstance(embed_time_basis_digest, str) and isinstance(detect_time_basis_digest, str):
        if embed_time_basis_digest != detect_time_basis_digest:
            reasons.append("basis_digest_mismatch")
    if isinstance(embed_time_planner_impl_identity, dict) and isinstance(detect_time_planner_impl_identity, dict):
        if embed_time_planner_impl_identity != detect_time_planner_impl_identity:
            reasons.append("planner_impl_identity_mismatch")
    return reasons


def _evaluate_trajectory_consistency(
    input_record: Optional[Dict[str, Any]],
    trajectory_evidence: Optional[Dict[str, Any]],
    detect_planner_input_digest: Optional[str]
) -> tuple[str, Optional[str]]:
    """
    功能：校验 embed/detect 两端 trajectory 证据一致性。

    Evaluate trajectory evidence consistency between embed-time record and detect-time runtime.

    Args:
        input_record: Embed-time input record mapping or None.
        trajectory_evidence: Detect-time trajectory evidence mapping or None.
        detect_planner_input_digest: Detect-time planner input digest.

    Returns:
        Tuple of (status, mismatch_reason_or_none).
        status is one of: "ok", "absent", "mismatch".

    Raises:
        TypeError: If inputs are invalid.
    """
    embed_trajectory_evidence: Optional[Dict[str, Any]] = None
    if input_record is not None:
        candidate = None
        for key in ["content_evidence_payload", "content_evidence", "content_result"]:
            payload = input_record.get(key)
            if isinstance(payload, dict) and "trajectory_evidence" in payload:
                payload_mapping = cast(Dict[str, Any], payload)
                candidate = payload_mapping.get("trajectory_evidence")
                break
        if candidate is None and "trajectory_evidence" in input_record:
            candidate = input_record.get("trajectory_evidence")
        if candidate is not None and not isinstance(candidate, dict):
            # embed 记录中的 trajectory_evidence 类型不合法，必须 fail-fast 
            raise TypeError("embed trajectory_evidence must be dict or None")
        if isinstance(candidate, dict):
            embed_trajectory_evidence = cast(Dict[str, Any], candidate)

    embed_planner_input_digest = _extract_embed_planner_input_digest(input_record)

    if embed_trajectory_evidence is None and trajectory_evidence is None:
        return "absent", None
    if embed_trajectory_evidence is None or trajectory_evidence is None:
        return "absent", None

    embed_status = _resolve_trajectory_tap_status(embed_trajectory_evidence)
    detect_status = _resolve_trajectory_tap_status(trajectory_evidence)
    if embed_status != "ok" or detect_status != "ok":
        return "absent", None

    if not isinstance(embed_planner_input_digest, str) or not embed_planner_input_digest:
        return "absent", None
    if not isinstance(detect_planner_input_digest, str) or not detect_planner_input_digest:
        return "absent", None

    embed_spec_digest = embed_trajectory_evidence.get("trajectory_spec_digest")
    detect_spec_digest = trajectory_evidence.get("trajectory_spec_digest")
    embed_trajectory_digest = embed_trajectory_evidence.get("trajectory_digest")
    detect_trajectory_digest = trajectory_evidence.get("trajectory_digest")

    if not isinstance(embed_spec_digest, str) or not isinstance(detect_spec_digest, str):
        return "mismatch", "trajectory_evidence_invalid"
    if not isinstance(embed_trajectory_digest, str) or not isinstance(detect_trajectory_digest, str):
        return "mismatch", "trajectory_evidence_invalid"

    if embed_spec_digest != detect_spec_digest:
        return "mismatch", "trajectory_spec_digest_mismatch"
    if embed_trajectory_digest != detect_trajectory_digest:
        return "mismatch", "trajectory_digest_mismatch"
    if embed_planner_input_digest != detect_planner_input_digest:
        return "mismatch", "plan_digest_mismatch"
    return "ok", None


def _extract_embed_planner_input_digest(input_record: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    功能：从 embed 记录提取 planner_input_digest 

    Extract embed-time planner_input_digest from record payload.

    Args:
        input_record: Embed-time record mapping.

    Returns:
        Planner input digest string or None.
    """
    if input_record is None:
        return None

    subspace_plan = input_record.get("subspace_plan")
    if isinstance(subspace_plan, dict):
        subspace_plan_payload = cast(Dict[str, Any], subspace_plan)
        direct_digest = subspace_plan_payload.get("planner_input_digest")
        if isinstance(direct_digest, str) and direct_digest:
            return direct_digest
        verifiable_spec = subspace_plan_payload.get("verifiable_input_domain_spec")
        if isinstance(verifiable_spec, dict):
            verifiable_spec_payload = cast(Dict[str, Any], verifiable_spec)
            digest_value = verifiable_spec_payload.get("planner_input_digest")
            if isinstance(digest_value, str) and digest_value:
                return digest_value

    content_payload = input_record.get("content_evidence_payload")
    if isinstance(content_payload, dict):
        content_payload_mapping = cast(Dict[str, Any], content_payload)
        nested = content_payload_mapping.get("subspace_plan")
        if isinstance(nested, dict):
            nested_payload = cast(Dict[str, Any], nested)
            direct_digest = nested_payload.get("planner_input_digest")
            if isinstance(direct_digest, str) and direct_digest:
                return direct_digest
            verifiable_spec = nested_payload.get("verifiable_input_domain_spec")
            if isinstance(verifiable_spec, dict):
                verifiable_spec_payload = cast(Dict[str, Any], verifiable_spec)
                digest_value = verifiable_spec_payload.get("planner_input_digest")
                if isinstance(digest_value, str) and digest_value:
                    return digest_value
    return None


def _is_embed_trajectory_explicit_absent(input_record: Optional[Dict[str, Any]]) -> bool:
    """
    功能：判 embed  trajectory 证据是否显式 absent 

    Determine whether embed-side trajectory evidence is explicitly absent.

    Args:
        input_record: Embed-time input record mapping.

    Returns:
        True if embed trajectory evidence exists and status is absent.
    """
    if input_record is None:
        return False

    embed_trajectory_evidence: Optional[Dict[str, Any]] = None
    for key in ["content_evidence_payload", "content_evidence", "content_result"]:
        payload = input_record.get(key)
        if isinstance(payload, dict) and "trajectory_evidence" in payload:
            payload_mapping = cast(Dict[str, Any], payload)
            candidate = payload_mapping.get("trajectory_evidence")
            if isinstance(candidate, dict):
                embed_trajectory_evidence = cast(Dict[str, Any], candidate)
            break
    if embed_trajectory_evidence is None and "trajectory_evidence" in input_record:
        top_candidate = input_record.get("trajectory_evidence")
        if isinstance(top_candidate, dict):
            embed_trajectory_evidence = cast(Dict[str, Any], top_candidate)
    if embed_trajectory_evidence is None:
        return False

    embed_status = _resolve_trajectory_tap_status(embed_trajectory_evidence)
    return embed_status == "absent"


def _inject_trajectory_audit_fields(
    content_evidence_payload: Dict[str, Any],
    trajectory_evidence: Dict[str, Any]
) -> None:
    """
    功能：将轨迹 tap 子状态写 content_evidence.audit（兼容新旧字段） 

    Inject trajectory tap status fields into content_evidence.audit.

    Args:
        content_evidence_payload: Content evidence payload mapping.
        trajectory_evidence: Trajectory evidence mapping.

    Returns:
        None.
    """
    audit = content_evidence_payload.get("audit")
    if not isinstance(audit, dict):
        audit = {}
        content_evidence_payload["audit"] = audit

    tap_status = _resolve_trajectory_tap_status(trajectory_evidence)
    tap_absent_reason = _resolve_trajectory_absent_reason(trajectory_evidence)

    if isinstance(tap_status, str) and tap_status:
        audit["trajectory_tap_status"] = tap_status
    if isinstance(tap_absent_reason, str) and tap_absent_reason:
        audit["trajectory_absent_reason"] = tap_absent_reason


def _resolve_trajectory_tap_status(trajectory_evidence: Dict[str, Any]) -> Optional[str]:
    """
    功能：优先读 trajectory audit 子状态，兼容 status 字段 

    Resolve trajectory tap status with new-field-first compatibility.

    Args:
        trajectory_evidence: Trajectory evidence mapping.

    Returns:
        trajectory tap status string or None.
    """
    audit_node = trajectory_evidence.get("audit")
    audit = cast(Dict[str, Any], audit_node) if isinstance(audit_node, dict) else None
    if audit is not None:
        value = audit.get("trajectory_tap_status")
        if isinstance(value, str) and value:
            return value

    compat_value = trajectory_evidence.get("status")
    if isinstance(compat_value, str) and compat_value:
        return compat_value
    return None


def _resolve_trajectory_absent_reason(trajectory_evidence: Dict[str, Any]) -> Optional[str]:
    """
    功能：优先读 trajectory audit 缺失原因，兼容旧字段 

    Resolve trajectory absent reason with new-field-first compatibility.

    Args:
        trajectory_evidence: Trajectory evidence mapping.

    Returns:
        Trajectory absent reason string or None.
    """
    audit_node = trajectory_evidence.get("audit")
    audit = cast(Dict[str, Any], audit_node) if isinstance(audit_node, dict) else None
    if audit is not None:
        value = audit.get("trajectory_absent_reason")
        if isinstance(value, str) and value:
            return value

    compat_value = trajectory_evidence.get("trajectory_absent_reason")
    if isinstance(compat_value, str) and compat_value:
        return compat_value
    return None


def _extract_detect_planner_input_digest(detect_plan_result: Any) -> Optional[str]:
    """
    功能：从 detect 侧规划结果提 planner_input_digest 

    Extract detect-time planner_input_digest from planner output.

    Args:
        detect_plan_result: Planner result object.

    Returns:
        Planner input digest string or None.
    """
    if detect_plan_result is None:
        return None

    plan_node = getattr(detect_plan_result, "plan", None)
    if isinstance(plan_node, dict):
        detect_plan_mapping = cast(Dict[str, Any], plan_node)
        direct_digest = detect_plan_mapping.get("planner_input_digest")
        if isinstance(direct_digest, str) and direct_digest:
            return direct_digest
        verifiable_spec = detect_plan_mapping.get("verifiable_input_domain_spec")
        if isinstance(verifiable_spec, dict):
            verifiable_spec_payload = cast(Dict[str, Any], verifiable_spec)
            digest_value = verifiable_spec_payload.get("planner_input_digest")
            if isinstance(digest_value, str) and digest_value:
                return digest_value

    if isinstance(detect_plan_result, dict):
        detect_plan_result_payload = cast(Dict[str, Any], detect_plan_result)
        direct_digest = detect_plan_result_payload.get("planner_input_digest")
        if isinstance(direct_digest, str) and direct_digest:
            return direct_digest
        verifiable_spec = detect_plan_result_payload.get("verifiable_input_domain_spec")
        if isinstance(verifiable_spec, dict):
            verifiable_spec_payload = cast(Dict[str, Any], verifiable_spec)
            digest_value = verifiable_spec_payload.get("planner_input_digest")
            if isinstance(digest_value, str) and digest_value:
                return digest_value
    return None


def _bind_actual_detect_planner_payload_to_record(
    record: Dict[str, Any],
    *,
    input_record: Optional[Dict[str, Any]],
    detect_plan_result: Any,
    detect_plan_result_override: Any,
    detect_time_plan_digest: Any,
    detect_time_basis_digest: Any,
    detect_time_planner_impl_identity: Any,
    detect_planner_input_digest: Optional[str],
) -> None:
    """
    功能：将 detect 实际消费的 planner payload 追加写回记录顶层。

    Bind the planner payload actually consumed by detect into top-level record
    fields in an append-only manner.

    Args:
        record: Mutable detect record mapping.
        input_record: Detect input record when available.
        detect_plan_result: Actual detect-time planner result object or mapping.
        detect_plan_result_override: Optional override payload supplied by CLI.
        detect_time_plan_digest: Actual detect-time plan digest.
        detect_time_basis_digest: Actual detect-time basis digest.
        detect_time_planner_impl_identity: Actual detect-time planner impl identity.
        detect_planner_input_digest: Actual detect-time planner input digest.

    Returns:
        None.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be dict")
    if input_record is not None and not isinstance(input_record, dict):
        raise TypeError("input_record must be dict or None")

    def _clone_mapping(mapping_value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(mapping_value, dict):
            return None
        payload = cast(Dict[str, Any], mapping_value)
        if not payload:
            return None
        return copy.deepcopy(payload)

    actual_subspace_plan = None
    actual_planner_impl_identity = None
    actual_plan_input_digest = None
    actual_plan_input_schema_version = None

    if detect_plan_result_override is not None and isinstance(input_record, dict):
        actual_subspace_plan = _clone_mapping(input_record.get("subspace_plan"))
        actual_planner_impl_identity = _clone_mapping(input_record.get("subspace_planner_impl_identity"))
        input_plan_input_digest = input_record.get("plan_input_digest")
        if isinstance(input_plan_input_digest, str) and input_plan_input_digest:
            actual_plan_input_digest = input_plan_input_digest
        input_plan_input_schema_version = input_record.get("plan_input_schema_version")
        if isinstance(input_plan_input_schema_version, str) and input_plan_input_schema_version:
            actual_plan_input_schema_version = input_plan_input_schema_version

    if actual_subspace_plan is None:
        actual_subspace_plan = _clone_mapping(getattr(detect_plan_result, "plan", None))
    if actual_subspace_plan is None and isinstance(detect_plan_result, dict):
        detect_plan_result_payload = cast(Dict[str, Any], detect_plan_result)
        actual_subspace_plan = _clone_mapping(detect_plan_result_payload.get("plan"))

    if actual_planner_impl_identity is None:
        actual_planner_impl_identity = _clone_mapping(detect_time_planner_impl_identity)
    if actual_planner_impl_identity is None and isinstance(actual_subspace_plan, dict):
        actual_planner_impl_identity = _clone_mapping(actual_subspace_plan.get("planner_impl_identity"))

    if not isinstance(actual_plan_input_digest, str) or not actual_plan_input_digest:
        if isinstance(detect_planner_input_digest, str) and detect_planner_input_digest:
            actual_plan_input_digest = detect_planner_input_digest
    if not isinstance(actual_plan_input_schema_version, str) or not actual_plan_input_schema_version:
        actual_plan_input_schema_version = PLAN_INPUT_SCHEMA_VERSION

    if isinstance(actual_subspace_plan, dict) and actual_subspace_plan:
        record["subspace_plan"] = actual_subspace_plan
    if isinstance(actual_planner_impl_identity, dict) and actual_planner_impl_identity:
        record["subspace_planner_impl_identity"] = actual_planner_impl_identity
    if isinstance(actual_plan_input_digest, str) and actual_plan_input_digest:
        record["plan_input_digest"] = actual_plan_input_digest
    if isinstance(actual_plan_input_schema_version, str) and actual_plan_input_schema_version:
        record["plan_input_schema_version"] = actual_plan_input_schema_version
    if isinstance(detect_time_plan_digest, str) and detect_time_plan_digest:
        record["plan_digest"] = detect_time_plan_digest
    if isinstance(detect_time_basis_digest, str) and detect_time_basis_digest:
        record["basis_digest"] = detect_time_basis_digest


def _resolve_mismatch_failure_reason(primary_mismatch_reason: str) -> str:
    """
    功能：将 mismatch 原因映射 content_failure_reason 

    Map mismatch reason token to content_failure_reason enum.

    Args:
        primary_mismatch_reason: Primary mismatch reason token.

    Returns:
        content_failure_reason enum string.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not primary_mismatch_reason:
        # primary_mismatch_reason 类型不合法，必须 fail-fast 
        raise TypeError("primary_mismatch_reason must be non-empty str")

    reason_map = {
        "plan_digest_mismatch": "detector_plan_mismatch",
        "basis_digest_mismatch": "detector_plan_mismatch",
        "planner_impl_identity_mismatch": "detector_plan_mismatch",
        "trajectory_spec_digest_mismatch": "detector_plan_mismatch",
        "trajectory_digest_mismatch": "detector_plan_mismatch",
        "trajectory_evidence_invalid": "detector_plan_mismatch",
        "injection_trace_digest_mismatch": "detector_plan_mismatch",
        "injection_params_digest_mismatch": "detector_plan_mismatch",
        "injection_trace_digest_invalid": "detector_plan_mismatch",
        "injection_params_digest_invalid": "detector_plan_mismatch",
        "injection_status_mismatch": "detector_plan_mismatch",
        "injection_subspace_binding_digest_mismatch": "detector_plan_mismatch",
        # (S-D) Paper Faithfulness mismatch reasons
        "paper_spec_digest_absent_or_invalid": "detector_plan_mismatch",
        "pipeline_fingerprint_digest_absent_or_invalid": "detector_plan_mismatch",
        "injection_site_digest_absent_or_invalid": "detector_plan_mismatch",
        "alignment_digest_absent_or_invalid": "detector_plan_mismatch",
        "paper_faithfulness_section_absent": "detector_plan_mismatch",
        "content_evidence_absent": "detector_plan_mismatch",
    }
    return reason_map.get(primary_mismatch_reason, "detector_plan_mismatch")


def _build_absent_fusion_decision(
    cfg: Dict[str, Any],
    content_evidence_adapted: Dict[str, Any],
    geometry_evidence_adapted: Dict[str, Any]
) -> FusionDecision:
    """
    功能：构 absent 的融合判决 

    Build a FusionDecision for absent trajectory evidence.

    Args:
        cfg: Configuration mapping.
        content_evidence_adapted: Adapted content evidence mapping.
        geometry_evidence_adapted: Adapted geometry evidence mapping.

    Returns:
        FusionDecision with decision_status="abstain" and score-free evidence summary.

    Raises:
        TypeError: If inputs are invalid.
    """
    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)

    evidence_summary: Dict[str, Any] = {
        "content_score": None,
        "geometry_score": geometry_evidence_adapted.get("geo_score"),
        "content_status": "absent",
        "geometry_status": geometry_evidence_adapted.get("status", "absent"),
        "fusion_rule_id": "detect_absent_guard"
    }
    audit: Dict[str, Any] = {
        "guard": "trajectory_absent_guard",
        "reason": content_evidence_adapted.get("content_failure_reason", "detector_no_evidence")
    }
    return FusionDecision(
        is_watermarked=None,
        decision_status="abstain",
        thresholds_digest=thresholds_digest,
        evidence_summary=evidence_summary,
        audit=audit
    )


def _resolve_primary_mismatch(mismatch_reasons: list[str]) -> tuple[str, str]:
    """
    功能：选择单一 mismatch 原因并返回对应字段路径 

    Resolve a single primary mismatch reason and its field path.

    Args:
        mismatch_reasons: Collected mismatch reason tokens.

    Returns:
        Tuple of (primary_reason, field_path).
    """
    reason_to_field_path = {
        "plan_digest_mismatch": "content_evidence.plan_digest",
        "basis_digest_mismatch": "content_evidence.basis_digest",
        "planner_impl_identity_mismatch": "content_evidence.planner_impl_identity",
        "trajectory_spec_digest_mismatch": "content_evidence.trajectory_evidence.trajectory_spec_digest",
        "trajectory_digest_mismatch": "content_evidence.trajectory_evidence.trajectory_digest",
        "trajectory_evidence_invalid": "content_evidence.trajectory_evidence",
        "injection_trace_digest_mismatch": "content_evidence.injection_trace_digest",
        "injection_params_digest_mismatch": "content_evidence.injection_params_digest",
        "injection_trace_digest_invalid": "content_evidence.injection_trace_digest",
        "injection_params_digest_invalid": "content_evidence.injection_params_digest",
        "injection_status_mismatch": "content_evidence.injection_status",
        "injection_subspace_binding_digest_mismatch": "content_evidence.subspace_binding_digest",
        "lf_impl_binding_missing_under_paper_mode": "content_evidence.lf_impl_binding",
        "hf_impl_binding_missing_under_paper_mode": "content_evidence.hf_impl_binding",
        "lf_impl_binding_impl_selected_absent": "content_evidence.lf_impl_binding.impl_selected",
        "hf_impl_binding_impl_selected_absent": "content_evidence.hf_impl_binding.impl_selected",
        "lf_impl_binding_non_primary_binding_under_paper_mode": "content_evidence.lf_impl_binding.evidence_level",
        "hf_impl_binding_non_primary_binding_under_paper_mode": "content_evidence.hf_impl_binding.evidence_level",
        "lf_ecc_int_not_allowed_under_paper_mode": "watermark.lf.ecc",
        # (S-D) Paper Faithfulness mismatch field paths
        "paper_spec_digest_absent_or_invalid": "paper_faithfulness.spec_digest",
        "pipeline_fingerprint_digest_absent_or_invalid": "content_evidence.pipeline_fingerprint_digest",
        "injection_site_digest_absent_or_invalid": "content_evidence.injection_site_digest",
        "alignment_digest_absent_or_invalid": "content_evidence.alignment_digest",
        "paper_faithfulness_section_absent": "paper_faithfulness",
        "content_evidence_absent": "content_evidence",
    }
    for token in [
        "trajectory_digest_mismatch",
        "plan_digest_mismatch",
        "basis_digest_mismatch",
        "planner_impl_identity_mismatch",
        "trajectory_spec_digest_mismatch",
        "trajectory_evidence_invalid",
        "injection_trace_digest_mismatch",
        "injection_params_digest_mismatch",
        "injection_trace_digest_invalid",
        "injection_params_digest_invalid",
        "injection_status_mismatch",
        "injection_subspace_binding_digest_mismatch",
        "lf_impl_binding_missing_under_paper_mode",
        "hf_impl_binding_missing_under_paper_mode",
        "lf_impl_binding_impl_selected_absent",
        "hf_impl_binding_impl_selected_absent",
        "lf_impl_binding_non_primary_binding_under_paper_mode",
        "hf_impl_binding_non_primary_binding_under_paper_mode",
        "lf_ecc_int_not_allowed_under_paper_mode",
        # (S-D) Paper Faithfulness mismatch priority
        "paper_spec_digest_absent_or_invalid",
        "pipeline_fingerprint_digest_absent_or_invalid",
        "injection_site_digest_absent_or_invalid",
        "alignment_digest_absent_or_invalid",
        "paper_faithfulness_section_absent",
        "content_evidence_absent",
    ]:
        if token in mismatch_reasons:
            return token, reason_to_field_path[token]
    return "unknown_mismatch", "content_evidence"


def _build_mismatch_fusion_decision(
    cfg: Dict[str, Any],
    content_evidence_adapted: Dict[str, Any],
    geometry_evidence_adapted: Dict[str, Any]
) -> FusionDecision:
    """
    功能：构 mismatch 的融合失败判决 

    Build a single-path FusionDecision for mismatch failures.

    Args:
        cfg: Configuration mapping.
        content_evidence_adapted: Adapted content evidence mapping.
        geometry_evidence_adapted: Adapted geometry evidence mapping.

    Returns:
        FusionDecision with decision_status="error" and score-free evidence summary.
    """
    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)

    evidence_summary: Dict[str, Any] = {
        "content_score": None,
        "geometry_score": geometry_evidence_adapted.get("geo_score"),
        "content_status": "mismatch",
        "geometry_status": geometry_evidence_adapted.get("status", "absent"),
        "fusion_rule_id": "detect_mismatch_guard"
    }
    audit: Dict[str, Any] = {
        "guard": "plan_anchor_consistency",
        "reason": content_evidence_adapted.get("content_failure_reason", "detector_plan_mismatch")
    }
    return FusionDecision(
        is_watermarked=None,
        decision_status="error",
        thresholds_digest=thresholds_digest,
        evidence_summary=evidence_summary,
        audit=audit
    )


def _build_planner_inputs_for_runtime(
    cfg: Dict[str, Any],
    trajectory_evidence: Optional[Dict[str, Any]] = None,
    content_evidence_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    功能：构造规划器输入签名 

    Build deterministic planner input signature from runtime cfg.

    Args:
        cfg: Configuration mapping.
        trajectory_evidence: Optional trajectory tap evidence.

    Returns:
        Planner input mapping.
    """
    generation_node = cfg.get("generation")
    generation_cfg = cast(Dict[str, Any], generation_node) if isinstance(generation_node, dict) else {}
    model_node = cfg.get("model")
    model_cfg = cast(Dict[str, Any], model_node) if isinstance(model_node, dict) else {}

    trace_signature: Dict[str, Any] = {
        "num_inference_steps": cfg.get("inference_num_steps", generation_cfg.get("num_inference_steps", 16)),
        "guidance_scale": cfg.get("inference_guidance_scale", generation_cfg.get("guidance_scale", 7.0)),
        "height": cfg.get("inference_height", model_cfg.get("height", 512)),
        "width": cfg.get("inference_width", model_cfg.get("width", 512)),
    }
    inputs: Dict[str, Any] = {"trace_signature": trace_signature}
    runtime_pipeline = cfg.get("__detect_pipeline_obj__")
    if runtime_pipeline is not None:
        inputs["pipeline"] = runtime_pipeline
    #  detect  per-step latent 缓存传递给 planner（内存传递，不写 records） 
    runtime_traj_cache = cfg.get("__detect_trajectory_latent_cache__")
    if runtime_traj_cache is not None:
        inputs["trajectory_latent_cache"] = runtime_traj_cache
        runtime_jvp_operator = build_runtime_jvp_operator_from_cache(cfg, runtime_traj_cache)
        if callable(runtime_jvp_operator):
            inputs["jvp_operator"] = runtime_jvp_operator
    if trajectory_evidence is not None:
        inputs["trajectory_evidence"] = trajectory_evidence
    if content_evidence_payload is not None:
        mask_stats_node = content_evidence_payload.get("mask_stats")
        if isinstance(mask_stats_node, dict):
            mask_stats = cast(Dict[str, Any], mask_stats_node)
            inputs["mask_summary"] = dict(mask_stats)
            routing_digest = mask_stats.get("routing_digest")
            if isinstance(routing_digest, str) and routing_digest:
                inputs["routing_digest"] = routing_digest
    return inputs


def run_calibrate_orchestrator(cfg: Dict[str, Any], impl_set: BuiltImplSet) -> Dict[str, Any]:
    """
    功能：执行校准流程并产出 NP 阈值工件 

    Execute calibration workflow and build NP thresholds artifacts.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
    """
    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    target_fpr = thresholds_spec.get("target_fpr")
    if not isinstance(target_fpr, (int, float)):
        raise TypeError("thresholds_spec.target_fpr must be number")
    score_name = _resolve_score_name_for_stats(cfg, mode="calibration")

    detect_records = _load_records_for_calibration(cfg)
    scores, strata_info = load_scores_for_calibration(detect_records, cfg, score_name=score_name)
    threshold_value, order_stat_info = compute_np_threshold(scores, float(target_fpr))
    sampling_policy_node = strata_info.get("sampling_policy")
    sampling_policy = cast(Dict[str, Any], sampling_policy_node) if isinstance(sampling_policy_node, dict) else {}
    null_source = sampling_policy.get("null_source") if isinstance(sampling_policy.get("null_source"), str) else "<absent>"
    n_selected_null = sampling_policy.get("n_selected_null") if isinstance(sampling_policy.get("n_selected_null"), int) else len(scores)

    threshold_key_used = neyman_pearson.format_fpr_key_canonical(float(target_fpr))
    threshold_id = f"{score_name}_np_{threshold_key_used}"
    thresholds_artifact: Dict[str, Any] = {
        "calibration_version": "np_v1",
        "rule_id": neyman_pearson.RULE_ID,
        "rule_version": neyman_pearson.RULE_VERSION,
        "threshold_id": threshold_id,
        "score_name": score_name,
        "target_fpr": float(target_fpr),
        "threshold_value": float(threshold_value),
        "threshold_key_used": threshold_key_used,
        "quantile_rule": "higher",
        "ties_policy": "strict_upper_bound",
        "threshold_value_semantics": "strict_upper_bound",
        "decision_operator": "score_greater_equal_threshold_value",
        "selected_order_stat_score": float(order_stat_info.get("selected_order_stat_score")),
        "effective_relation_to_selected_order_stat": "score_strictly_greater_than_selected_order_stat_score",
    }
    threshold_metadata_artifact: Dict[str, Any] = {
        "calibration_version": "np_v1",
        "rule_id": neyman_pearson.RULE_ID,
        "rule_version": neyman_pearson.RULE_VERSION,
        "method": "neyman_pearson_v1",
        "score_name": score_name,
        "target_fpr": float(target_fpr),
        "null_source": null_source,
        "n_null": n_selected_null,
        "n_samples": len(scores),
        "calibration_date": "1970-01-01",
        "quantile_method": "higher",
        "ties_policy": "strict_upper_bound",
        "threshold_value_semantics": "strict_upper_bound",
        "decision_operator": "score_greater_equal_threshold_value",
        "selected_order_stat_score": float(order_stat_info.get("selected_order_stat_score")),
        "target_fprs": [float(target_fpr)],
        "order_statistics": order_stat_info,
        "stratification": strata_info,
        "sample_digest": digests.canonical_sha256({"scores": [round(float(v), 12) for v in scores]}),
    }
    # 过滤出仅 null 样本（与 load_scores_for_calibration 保持一致：排除 label=True 的正样本） 
    # null_strata / conditional_fpr 语义要求统计对象仅为 null（负）样本，不能混入正样本 
    has_explicit_labels = strata_info.get("sampling_policy", {}).get("records_with_explicit_label", False)
    if has_explicit_labels:
        null_records_for_stats = [r for r in detect_records if _resolve_calibration_label(r) is not True]
    else:
        null_records_for_stats = detect_records

    threshold_metadata_artifact["null_strata"] = _compute_null_strata_for_calibration(
        null_records_for_stats,
        float(threshold_value),
        cfg,
        score_name=score_name,
    )
    threshold_metadata_artifact["conditional_fpr"] = _compute_conditional_fpr_for_calibration(
        null_records_for_stats,
        float(threshold_value),
        score_name=score_name,
    )
    threshold_metadata_artifact["conditional_fpr_records"] = _compute_conditional_fpr_records_for_calibration(
        null_records_for_stats,
        float(threshold_value),
        cfg,
        score_name=score_name,
    )

    record: Dict[str, Any] = {
        "operation": "calibrate",
        "calibration_is_fallback": False,
        "calibration_mode": "real",
        "protocol": "neyman_pearson",
        "threshold_key_used": threshold_key_used,
        "threshold_id": threshold_id,
        "calibration_samples": len(scores),
        "calibration_summary": {
            "score_name": score_name,
            "target_fpr": float(target_fpr),
            "threshold_value": float(threshold_value),
            "selected_order_stat_score": float(order_stat_info.get("selected_order_stat_score")),
            "order_statistics": order_stat_info,
            "stratification": strata_info,
        },
        "thresholds_artifact": thresholds_artifact,
        "threshold_metadata_artifact": threshold_metadata_artifact,
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
    }
    return record


def _pick_first_non_empty_string(values: list[Any]) -> Optional[str]:
    """
    功能：从候选值列表中提取首个非空字符串 

    Select the first non-empty string from candidate values.

    Args:
        values: Candidate values.

    Returns:
        The first non-empty string, or None when not found.
    """
    for value in values:
        if isinstance(value, str) and value and value != "<absent>":
            return value
    return None


def _resolve_cfg_digest_for_evaluate(cfg: Dict[str, Any], detect_records: list[Dict[str, Any]]) -> str:
    """
    功能：解 evaluate 报告 cfg_digest 锚点 

    Resolve cfg_digest anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        detect_records: Loaded detect records.

    Returns:
        Resolved cfg_digest anchor string.
    """
    from_cfg = _pick_first_non_empty_string([
        cfg.get("__evaluate_cfg_digest__"),
        cfg.get("cfg_digest"),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    for record in detect_records:
        from_record = _pick_first_non_empty_string([
            record.get("cfg_digest"),
        ])
        if isinstance(from_record, str):
            return from_record
    return "<absent>"


def _resolve_plan_digest_for_evaluate(cfg: Dict[str, Any], detect_records: list[Dict[str, Any]]) -> str:
    """
    功能：解 evaluate 报告 plan_digest 锚点 

    Resolve plan_digest anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        detect_records: Loaded detect records.

    Returns:
        Resolved plan_digest anchor string.
    """
    from_cfg = _pick_first_non_empty_string([
        cfg.get("__evaluate_plan_digest__"),
        _resolve_cfg_plan_digest(cfg),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    plan_digests: list[str] = []
    for record in detect_records:
        resolved = _pick_first_non_empty_string([
            record.get("plan_digest"),
            record.get("expected_plan_digest"),
        ])
        if isinstance(resolved, str):
            plan_digests.append(resolved)

    unique_plan_digests = sorted(set(plan_digests))
    if len(unique_plan_digests) == 1:
        return unique_plan_digests[0]
    if len(unique_plan_digests) > 1:
        return digests.canonical_sha256({"evaluate_plan_digest_candidates": unique_plan_digests})

    fallback_signatures: list[Dict[str, str]] = []
    for record in detect_records:
        attack_node = record.get("attack")
        attack = cast(Dict[str, Any], attack_node) if isinstance(attack_node, dict) else {}
        fallback_signatures.append({
            "family": str(attack.get("family", "unknown")),
            "params_version": str(attack.get("params_version", "unknown")),
        })
    if fallback_signatures:
        return digests.canonical_sha256({
            "rule": "evaluate_plan_digest_fallback",
            "attack_signatures": fallback_signatures,
        })
    return "<absent>"


def _resolve_threshold_metadata_digest_for_evaluate(
    cfg: Dict[str, Any],
    thresholds_path: Path,
    detect_records: list[Dict[str, Any]],
) -> str:
    """
    功能：解 evaluate 报告 threshold_metadata_digest 锚点 

    Resolve threshold metadata digest anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        thresholds_path: Threshold artifact path.
        detect_records: Loaded detect records.

    Returns:
        Resolved threshold metadata digest anchor string.
    """
    from_cfg = _pick_first_non_empty_string([
        cfg.get("__evaluate_threshold_metadata_digest__"),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    evaluate_cfg_node = cfg.get("evaluate")
    evaluate_cfg = cast(Dict[str, Any], evaluate_cfg_node) if isinstance(evaluate_cfg_node, dict) else {}
    candidate_path_nodes: list[Any] = [
        cfg.get("__evaluate_threshold_metadata_artifact_path__"),
        evaluate_cfg.get("threshold_metadata_artifact_path"),
        str(thresholds_path.parent / "threshold_metadata_artifact.json"),
        str(thresholds_path.parent / "threshold_metadata.json"),
    ]
    candidate_paths = [path for path in candidate_path_nodes if isinstance(path, str) and path]
    for path_str in candidate_paths:
        path_obj = Path(path_str).resolve()
        if not path_obj.exists() or not path_obj.is_file():
            continue
        try:
            payload = records_io.read_json(str(path_obj))
        except Exception:
            # metadata 工件不可读时跳过当前候选，继续尝试其他来源 
            continue
        if isinstance(payload, dict):
            return digests.canonical_sha256(payload)

    for record in detect_records:
        resolved = _pick_first_non_empty_string([
            record.get("threshold_metadata_digest"),
        ])
        if isinstance(resolved, str):
            return resolved
    return "<absent>"


def _resolve_impl_digest_for_evaluate(cfg: Dict[str, Any], detect_records: list[Dict[str, Any]]) -> str:
    """
    功能：解 evaluate 报告 impl_digest 锚点 

    Resolve implementation digest anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        detect_records: Loaded detect records.

    Returns:
        Resolved impl_digest anchor string.
    """
    from_cfg = _pick_first_non_empty_string([
        cfg.get("__impl_digest__"),
        cfg.get("impl_set_capabilities_extended_digest"),
        cfg.get("impl_set_capabilities_digest"),
        cfg.get("impl_identity_digest"),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    for record in detect_records:
        resolved = _pick_first_non_empty_string([
            record.get("impl_set_capabilities_extended_digest"),
            record.get("impl_set_capabilities_digest"),
            record.get("impl_identity_digest"),
            record.get("impl_digest"),
        ])
        if isinstance(resolved, str):
            return resolved
    return "<absent>"


def _resolve_policy_path_for_evaluate(cfg: Dict[str, Any], detect_records: list[Dict[str, Any]]) -> str:
    """
    功能：解 evaluate 报告 policy_path 锚点 

    Resolve policy_path anchor for evaluation report.

    Args:
        cfg: Configuration mapping.
        detect_records: Loaded detect records.

    Returns:
        Resolved policy_path anchor string.
    """
    from_cfg = _pick_first_non_empty_string([
        cfg.get("__policy_path__"),
        cfg.get("policy_path"),
    ])
    if isinstance(from_cfg, str):
        return from_cfg

    for record in detect_records:
        resolved = _pick_first_non_empty_string([
            record.get("policy_path"),
        ])
        if isinstance(resolved, str):
            return resolved
    return "<absent>"


def run_evaluate_orchestrator(cfg: Dict[str, Any], impl_set: BuiltImplSet) -> Dict[str, Any]:
    """
    功能：执行只读阈值评估流程 

    Execute evaluation workflow in readonly-threshold mode.

    Args:
        cfg: Config mapping.
        impl_set: Built implementation set.

    Returns:
        Business fields mapping for record.

    Raises:
        TypeError: If inputs are invalid.
    """
    thresholds_path = _resolve_thresholds_path_for_evaluate(cfg)
    thresholds_obj = load_thresholds_artifact_controlled(str(thresholds_path))
    if _is_formal_event_attestation_mainline(cfg):
        threshold_score_name = thresholds_obj.get("score_name", "content_score")
        if not isinstance(threshold_score_name, str) or not threshold_score_name:
            raise TypeError("thresholds artifact score_name must be non-empty str")
        eval_metrics.raise_if_legacy_event_attestation_alias_requested(
            threshold_score_name,
            consumer="run_evaluate_orchestrator.thresholds_artifact",
        )
    detect_records = _load_records_for_evaluate(cfg)
    
    # 记录 evaluate 开始前的 thresholds digest。
    thresholds_digest_before = digests.canonical_sha256(thresholds_obj)
    
    # 使用 evaluation 模块代替内联逻辑。
    attack_protocol_spec = eval_protocol_loader.load_attack_protocol_spec(cfg)
    
    # 计算 overall grouped metrics。
    aggregated_metrics = eval_metrics.aggregate_metrics(
        detect_records,
        thresholds_obj,
        attack_protocol_spec,
    )
    metrics_obj = aggregated_metrics.get("metrics_overall", {})
    breakdown = aggregated_metrics.get("breakdown", {})
    
    # 重新加载 thresholds 工件并核对 digest。
    thresholds_obj_after = load_thresholds_artifact_controlled(str(thresholds_path))
    thresholds_digest_after = digests.canonical_sha256(thresholds_obj_after)
    
    if thresholds_digest_before != thresholds_digest_after:
        # thresholds 工件在 evaluate 过程中被修改，违反 NP 规则。
        raise RuntimeError(
            f"thresholds 工件只读性验证失败\n"
            f"  - 路径: {thresholds_path}\n"
            f"  - digest_before: {thresholds_digest_before}\n"
            f"  - digest_after: {thresholds_digest_after}\n"
            f"  - 原因: evaluate 侧修改或污染了 thresholds 工件"
        )
    attack_group_metrics = aggregated_metrics.get("metrics_by_attack_condition", [])
    ablation_digest = _compute_ablation_digest_for_report(cfg)
    ablation_digest_extended = _compute_ablation_digest_extended_for_report(cfg)
    attack_trace_digest = _collect_attack_trace_digest(detect_records)
    coverage_manifest = eval_attack_coverage.compute_attack_coverage_manifest()
    attack_coverage_digest = coverage_manifest.get("attack_coverage_digest", "<absent>")
    
    # 构造条件指标容器（向后兼容） 
    conditional_metrics = eval_report_builder.build_conditional_metrics_container(
        attack_protocol_spec.get("version", "<absent>"),
        attack_group_metrics,
    )

    _ = impl_set

    fusion_result = FusionDecision(
        is_watermarked=None,
        decision_status="abstain",
        thresholds_digest=digests.canonical_sha256(thresholds_obj),
        evidence_summary={
            "content_score": None,
            "geometry_score": None,
            "content_status": "aggregate",
            "geometry_status": "aggregate",
            "fusion_rule_id": "evaluate_readonly_thresholds",
        },
        audit={
            "impl_identity": "evaluate_orchestrator",
            "mode": "readonly_thresholds",
            "threshold_key_used": thresholds_obj.get("threshold_key_used"),
            "n_total": metrics_obj.get("n_total"),
        },
        fusion_rule_version="v1",
        used_threshold_id=thresholds_obj.get("threshold_id") if isinstance(thresholds_obj.get("threshold_id"), str) else None,
    )

    # 使用 report_builder 组装完整报告 
    thresholds_digest = digests.canonical_sha256(thresholds_obj)
    threshold_metadata_digest = _resolve_threshold_metadata_digest_for_evaluate(
        cfg,
        thresholds_path,
        detect_records,
    )
    plan_digest = _resolve_plan_digest_for_evaluate(cfg, detect_records)
    impl_digest = _resolve_impl_digest_for_evaluate(cfg, detect_records)
    fusion_rule_version = thresholds_obj.get("rule_version", "<absent>")
    policy_path = _resolve_policy_path_for_evaluate(cfg, detect_records)
    cfg_digest = _resolve_cfg_digest_for_evaluate(cfg, detect_records)
    
    report_obj = eval_report_builder.build_eval_report(
        cfg_digest=cfg_digest,
        plan_digest=plan_digest,
        thresholds_digest=thresholds_digest,
        threshold_metadata_digest=threshold_metadata_digest,
        impl_digest=impl_digest,
        fusion_rule_version=fusion_rule_version,
        attack_protocol_version=attack_protocol_spec.get("version", "<absent>"),
        attack_protocol_digest=attack_protocol_spec.get("attack_protocol_digest", "<absent>"),
        policy_path=policy_path,
        metrics_overall=metrics_obj,
        metrics_by_attack_condition=attack_group_metrics,
        thresholds_artifact=thresholds_obj,
        attack_protocol_spec=attack_protocol_spec,  # (向后兼容)
        ablation_digest=ablation_digest,
        attack_trace_digest=attack_trace_digest,
        attack_coverage_digest=attack_coverage_digest,
    )
    anchors = report_obj.get("anchors") if isinstance(report_obj.get("anchors"), dict) else None
    if isinstance(anchors, dict):
        anchors["ablation_digest_extended"] = ablation_digest_extended
    
    # append-only 加入 readonly guard 记录
    report_obj["thresholds_readonly_guard"] = {
        "digest_before": thresholds_digest_before,
        "digest_after": thresholds_digest_after,
        "unchanged": (thresholds_digest_before == thresholds_digest_after),
        "guard_version": "v1",
    }

    record: Dict[str, Any] = {
        "operation": "evaluate",
        "evaluation_is_fallback": False,
        "evaluation_mode": "real",
        "metrics": metrics_obj,
        "evaluation_breakdown": breakdown,
        "conditional_metrics": conditional_metrics,
        "evaluation_report": report_obj,
        "thresholds_artifact": thresholds_obj,
        "threshold_key_used": thresholds_obj.get("threshold_key_used"),
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "ok",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True
        },
        "test_samples": int(metrics_obj.get("n_total", 0)),
        "fusion_result": fusion_result
    }
    return record


def load_scores_for_calibration(
    records: list[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    score_name: str = "content_score",
) -> tuple[list[float], Dict[str, Any]]:
    """
    功能：从 detect records 加载校准分数 

    Load calibration scores from detect records using strict status filtering.

    Args:
        records: Detect records list.
        cfg: Optional runtime config used for strict calibration filtering.

    Returns:
        Tuple of (scores, strata_info).
    """
    if cfg is not None and not isinstance(cfg, dict):
        raise TypeError("cfg must be dict or None")
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")
    if _is_formal_event_attestation_mainline(cfg):
        eval_metrics.raise_if_legacy_event_attestation_alias_requested(
            score_name,
            consumer="load_scores_for_calibration",
        )

    scores: list[float] = []
    total = len(records)
    valid = 0
    rejected = 0
    rejected_label_missing = 0
    rejected_label_positive = 0
    rejected_synthetic_fallback = 0
    rejected_synthetic_negative_closure = 0
    rejected_formal_sidecar_marker = 0

    calibration_cfg: Dict[str, Any] = {}
    if isinstance(cfg, dict):
        calibration_node = cfg.get("calibration")
        if isinstance(calibration_node, dict):
            calibration_cfg = cast(Dict[str, Any], calibration_node)
    exclude_formal_sidecar_marker = bool(calibration_cfg.get("exclude_formal_sidecar_disabled_marker", False))
    exclude_synthetic_negative_closure = bool(calibration_cfg.get("exclude_synthetic_negative_closure_marker", False))

    has_explicit_labels = False
    for item in records:
        if _resolve_calibration_label(item) is not None:
            has_explicit_labels = True
            break

    null_source = "status_ok_unlabeled_detect_records"
    if has_explicit_labels:
        null_source = "label_false_from_detect_records"

    for item in records:
        content_payload = item.get("content_evidence_payload")
        status_value = None
        score_value = None
        if isinstance(content_payload, dict):
            content_payload_mapping = cast(Dict[str, Any], content_payload)
            status_value = content_payload_mapping.get("status")
            if _is_synthetic_fallback_calibration_sample(content_payload_mapping):
                rejected += 1
                rejected_synthetic_fallback += 1
                continue
            if exclude_formal_sidecar_marker:
                usage_value = content_payload_mapping.get("calibration_sample_usage")
                if usage_value == "formal_with_sidecar_disabled_marker":
                    rejected += 1
                    rejected_formal_sidecar_marker += 1
                    continue
            if exclude_synthetic_negative_closure:
                usage_value = content_payload_mapping.get("calibration_sample_usage")
                if usage_value == "synthetic_negative_for_ground_truth_closure":
                    rejected += 1
                    rejected_synthetic_negative_closure += 1
                    continue
        score_value = _extract_score_for_stats(item, score_name)
        if score_value is None:
            rejected += 1
            continue

        if has_explicit_labels:
            resolved_label = _resolve_calibration_label(item)
            if resolved_label is None:
                rejected += 1
                rejected_label_missing += 1
                continue
            if resolved_label is True:
                rejected += 1
                rejected_label_positive += 1
                continue

        scores.append(float(score_value))
        valid += 1

    if len(scores) == 0:
        raise ValueError(f"calibration requires at least one valid {score_name} sample")

    strata_info: Dict[str, Any] = {
        "global": {
            "n_total": total,
            "n_valid": valid,
            "n_rejected": rejected,
        },
        "sampling_policy": {
            "score_name": score_name,
            "null_source": null_source,
            "label_field_candidates": ["label", "ground_truth", "is_watermarked"],
            "records_with_explicit_label": has_explicit_labels,
            "n_rejected_label_missing": rejected_label_missing,
            "n_rejected_label_positive": rejected_label_positive,
            "n_rejected_synthetic_fallback": rejected_synthetic_fallback,
            "n_rejected_synthetic_negative_closure": rejected_synthetic_negative_closure,
            "n_rejected_formal_sidecar_marker": rejected_formal_sidecar_marker,
            "exclude_formal_sidecar_disabled_marker": exclude_formal_sidecar_marker,
            "exclude_synthetic_negative_closure_marker": exclude_synthetic_negative_closure,
            "n_selected_null": valid,
        },
    }
    return scores, strata_info


def _resolve_score_name_for_stats(cfg: Dict[str, Any], mode: str) -> str:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if mode not in {"calibration", "evaluate"}:
        raise ValueError("mode must be one of {'calibration', 'evaluate'}")

    override_key = "__calibration_score_name__" if mode == "calibration" else "__evaluate_score_name__"
    override_value = cfg.get(override_key)
    if isinstance(override_value, str) and override_value:
        return override_value

    section_key = "calibration" if mode == "calibration" else "evaluate"
    section_node = cfg.get(section_key)
    section_cfg = cast(Dict[str, Any], section_node) if isinstance(section_node, dict) else {}
    section_value = section_cfg.get("score_name")
    if isinstance(section_value, str) and section_value:
        return section_value
    return "content_score"


def _is_formal_event_attestation_mainline(cfg: Optional[Dict[str, Any]]) -> bool:
    """
    功能：判定当前 cfg 是否属于 formal event-attestation 主线。

    Determine whether the current config belongs to the formal mainline that
    must reject legacy event-attestation alias artifacts.

    Args:
        cfg: Optional configuration mapping.

    Returns:
        True when paper_faithfulness is enabled; otherwise False.
    """
    if not isinstance(cfg, dict):
        return False

    paper_node = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_node) if isinstance(paper_node, dict) else {}
    return bool(paper_cfg.get("enabled", False))


def _is_synthetic_fallback_calibration_sample(content_payload: Dict[str, Any]) -> bool:
    """
    功能：判定样本是否属 synthetic fallback 校准样本 

    Determine whether calibration sample is synthetic fallback and must be excluded.

    Args:
        content_payload: Content evidence payload mapping.

    Returns:
        True when sample is synthetic fallback, otherwise False.
    """
    if not isinstance(content_payload, dict):
        raise TypeError("content_payload must be dict")

    synthetic_flag = content_payload.get("calibration_sample_is_synthetic_fallback")
    if synthetic_flag is True:
        return True

    origin_value = content_payload.get("calibration_sample_origin")
    usage_value = content_payload.get("calibration_sample_usage")
    if isinstance(origin_value, str) and isinstance(usage_value, str):
        if origin_value in {"synthetic_fallback", "sidecar_disabled_fallback"} and "synthetic" in usage_value:
            return True

    return False


def _resolve_calibration_label(record: Dict[str, Any]) -> Optional[bool]:
    """
    功能：从 detect record 解析校准标签 

    Resolve calibration label from detect record candidates.

    Args:
        record: Detect record mapping.

    Returns:
        Boolean label or None when missing.
    """
    for key_name in ["label", "ground_truth", "is_watermarked"]:
        value = record.get(key_name)
        if isinstance(value, bool):
            return value
    return None


def compute_np_threshold(scores: list[float], target_fpr: float) -> tuple[float, Dict[str, Any]]:
    """
    功能：按 order-statistics 计算 NP 阈值 

    Compute Neyman-Pearson threshold using higher quantile order statistics.

    Args:
        scores: Null distribution scores.
        target_fpr: Target false positive rate.

    Returns:
        Tuple of (threshold_value, order_stat_info).
    """
    if len(scores) == 0:
        raise ValueError("scores must be non-empty list")

    threshold_value, order_stat_info = neyman_pearson.compute_np_threshold_from_scores(
        scores,
        float(target_fpr),
        quantile_rule="higher",
    )
    return float(threshold_value), order_stat_info


def load_thresholds_artifact_controlled(path: str) -> Dict[str, Any]:
    """
    功能：只读加载阈值工件 

    Load thresholds artifact in read-only mode with schema checks.

    Args:
        path: Thresholds artifact path.

    Returns:
        Thresholds artifact mapping.
    """
    if not path:
        raise TypeError("path must be non-empty str")
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_file():
        raise ValueError(f"thresholds artifact not found: {path}")
    payload = records_io.read_json(path)
    if not isinstance(payload, dict):
        raise TypeError("thresholds artifact must be dict")
    payload_dict = cast(Dict[str, Any], payload)
    required = ["threshold_id", "score_name", "target_fpr", "threshold_value", "threshold_key_used"]
    for field_name in required:
        if field_name not in payload_dict:
            raise ValueError(f"thresholds artifact missing field: {field_name}")
    threshold_value = payload_dict.get("threshold_value")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")
    return payload_dict


def evaluate_records_against_threshold(
    records: list[Dict[str, Any]],
    thresholds_obj: Dict[str, Any],
    attack_protocol_spec: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    功能：使用只读阈值评 detect 记录 

    Evaluate detect records using precomputed thresholds artifact only.
    Now delegates to evaluation module for metric computation.

    Args:
        records: Detect records list.
        thresholds_obj: Thresholds artifact mapping.
        attack_protocol_spec: Optional attack protocol spec.

    Returns:
        Tuple of (metrics, breakdown, conditional_metrics).
    """
    threshold_value = thresholds_obj.get("threshold_value")
    if not isinstance(threshold_value, (int, float)):
        raise TypeError("threshold_value must be number")
    threshold_float = float(threshold_value)
    score_name = thresholds_obj.get("score_name", "content_score")
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("thresholds artifact score_name must be non-empty str")

    # 使用 evaluation 模块计算指标 
    if attack_protocol_spec is None:
        attack_protocol_spec = {
            "version": "<absent>",
            "family_field_candidates": ["attack_family", "attack.family", "attack.type"],
            "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
        }

    # Overall metrics 
    metrics, breakdown = eval_metrics.compute_overall_metrics(records, threshold_float, score_name=score_name)
    
    # 补充 thresholds 工件元数据 
    metrics["metric_version"] = "tpr_at_fpr_v1"
    metrics["score_name"] = score_name
    metrics["target_fpr"] = thresholds_obj.get("target_fpr")
    metrics["threshold_value"] = threshold_float
    metrics["threshold_key_used"] = thresholds_obj.get("threshold_key_used")

    # Grouped metrics 
    attack_group_metrics = eval_metrics.compute_attack_group_metrics(
        records,
        threshold_float,
        attack_protocol_spec,
        score_name=score_name,
    )

    # 计算条件指标中的 "items"（旧字段，用于向后兼容） 
    conditional_metrics_old = _compute_conditional_metrics_for_evaluate(records, threshold_float, score_name=score_name)
    additional_items = conditional_metrics_old.get("items", [])

    # 构造条件指标容器（向后兼容） 
    conditional_metrics = eval_report_builder.build_conditional_metrics_container(
        attack_protocol_spec.get("version", "<absent>"),
        attack_group_metrics,
        additional_items=additional_items,
    )

    return metrics, breakdown, conditional_metrics


def _load_records_for_calibration(cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
    records_glob = cfg.get("__calibration_detect_records_glob__")
    if not isinstance(records_glob, str) or not records_glob:
        calibration_cfg_node = cfg.get("calibration")
        calibration_cfg = cast(Dict[str, Any], calibration_cfg_node) if isinstance(calibration_cfg_node, dict) else {}
        records_glob_candidate = calibration_cfg.get("detect_records_glob")
        records_glob = records_glob_candidate if isinstance(records_glob_candidate, str) else None
    if not isinstance(records_glob, str) or not records_glob:
        raise ValueError("calibration.detect_records_glob is required")
    return _load_records_by_glob(records_glob)


def _load_records_for_evaluate(cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
    records_glob = cfg.get("__evaluate_detect_records_glob__")
    if not isinstance(records_glob, str) or not records_glob:
        evaluate_cfg_node = cfg.get("evaluate")
        evaluate_cfg = cast(Dict[str, Any], evaluate_cfg_node) if isinstance(evaluate_cfg_node, dict) else {}
        records_glob_candidate = evaluate_cfg.get("detect_records_glob")
        records_glob = records_glob_candidate if isinstance(records_glob_candidate, str) else None
    if not isinstance(records_glob, str) or not records_glob:
        raise ValueError("evaluate.detect_records_glob is required")
    records = _load_records_by_glob(records_glob)

    evaluate_cfg_node = cfg.get("evaluate")
    evaluate_cfg = cast(Dict[str, Any], evaluate_cfg_node) if isinstance(evaluate_cfg_node, dict) else {}
    exclude_synthetic_negative_closure = bool(evaluate_cfg.get("exclude_synthetic_negative_closure_marker", False))
    if not exclude_synthetic_negative_closure:
        return records

    filtered_records: list[Dict[str, Any]] = []
    for item in records:
        content_node = item.get("content_evidence_payload")
        if not isinstance(content_node, dict):
            filtered_records.append(item)
            continue
        content_payload = cast(Dict[str, Any], content_node)
        if _is_synthetic_negative_closure_sample(content_payload):
            continue
        filtered_records.append(item)
    return filtered_records


def _resolve_thresholds_path_for_evaluate(cfg: Dict[str, Any]) -> Path:
    thresholds_path = cfg.get("__evaluate_thresholds_path__")
    if not isinstance(thresholds_path, str) or not thresholds_path:
        evaluate_cfg_node = cfg.get("evaluate")
        evaluate_cfg = cast(Dict[str, Any], evaluate_cfg_node) if isinstance(evaluate_cfg_node, dict) else {}
        thresholds_path_candidate = evaluate_cfg.get("thresholds_path")
        thresholds_path = thresholds_path_candidate if isinstance(thresholds_path_candidate, str) else None
    if not isinstance(thresholds_path, str) or not thresholds_path:
        raise ValueError("evaluate.thresholds_path is required")
    path_obj = Path(thresholds_path).resolve()
    if not path_obj.exists() or not path_obj.is_file():
        raise ValueError(f"evaluate thresholds_path not found: {path_obj}")
    return path_obj


def _load_records_by_glob(records_glob: str) -> list[Dict[str, Any]]:
    if not records_glob:
        raise TypeError("records_glob must be non-empty str")
    matched_paths = sorted(glob.glob(records_glob, recursive=True))
    if len(matched_paths) == 0:
        raise ValueError(f"no detect records matched: {records_glob}")
    records: list[Dict[str, Any]] = []
    for path_str in matched_paths:
        path_obj = Path(path_str)
        if not path_obj.is_file():
            continue
        payload = records_io.read_json(str(path_obj))
        if isinstance(payload, dict):
            records.append(cast(Dict[str, Any], payload))
    if len(records) == 0:
        raise ValueError(f"no valid detect records loaded from: {records_glob}")
    return records


def _extract_ground_truth_label(record: Dict[str, Any]) -> Optional[bool]:
    for key_name in ["ground_truth_is_watermarked", "is_watermarked_gt", "label"]:
        value = record.get(key_name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
    return None


def _extract_geometry_score(record: Dict[str, Any]) -> Optional[float]:
    geometry_node = record.get("geometry_evidence_payload")
    if not isinstance(geometry_node, dict):
        return None
    geometry_payload = cast(Dict[str, Any], geometry_node)
    status_value = geometry_payload.get("status")
    if status_value != "ok":
        return None
    for key_name in ["score", "geo_score"]:
        value = geometry_payload.get(key_name)
        if isinstance(value, (int, float)):
            value_float = float(value)
            if np.isfinite(value_float):
                return value_float
    return None


def _extract_score_for_stats(record: Dict[str, Any], score_name: str) -> Optional[float]:
    if not isinstance(record, dict):
        return None
    if not isinstance(score_name, str) or not score_name:
        raise TypeError("score_name must be non-empty str")

    if eval_metrics.is_content_chain_score_name(score_name):
        score_value, _ = eval_workflow_inputs._resolve_content_score_source(record)
    elif eval_metrics.is_lf_channel_score_name(score_name):
        content_node = record.get("content_evidence_payload")
        if not isinstance(content_node, dict):
            return None
        content_payload = cast(Dict[str, Any], content_node)
        if content_payload.get("status") != "ok":
            return None
        score_value = content_payload.get(eval_metrics.LF_CHANNEL_SCORE_NAME)
        if not isinstance(score_value, (int, float)):
            score_value = content_payload.get("lf_score")
    elif score_name == "content_attestation_score":
        attestation_node = record.get("attestation")
        if not isinstance(attestation_node, dict):
            return None
        attestation_payload = cast(Dict[str, Any], attestation_node)
        image_evidence_result_node = attestation_payload.get("image_evidence_result")
        if not isinstance(image_evidence_result_node, dict):
            return None
        image_evidence_result = cast(Dict[str, Any], image_evidence_result_node)
        if image_evidence_result.get("status") != "ok":
            return None
        formal_score_name = image_evidence_result.get("content_attestation_score_name")
        if isinstance(formal_score_name, str) and formal_score_name and formal_score_name != "content_attestation_score":
            return None
        score_value = image_evidence_result.get("content_attestation_score")
    elif score_name == "event_attestation_score":
        attestation_node = record.get("attestation")
        if not isinstance(attestation_node, dict):
            return None
        attestation_payload = cast(Dict[str, Any], attestation_node)
        final_decision_node = attestation_payload.get("final_event_attested_decision")
        if not isinstance(final_decision_node, dict):
            return None
        final_decision = cast(Dict[str, Any], final_decision_node)
        formal_score_name = final_decision.get("event_attestation_score_name")
        if isinstance(formal_score_name, str) and formal_score_name and formal_score_name != "event_attestation_score":
            return None
        score_value = final_decision.get("event_attestation_score")
    elif score_name == "event_attestation_statistics_score":
        eval_metrics.raise_if_legacy_event_attestation_alias_requested(
            score_name,
            consumer="_extract_score_for_stats",
        )
        return None
    else:
        raise ValueError(f"unsupported score_name: {score_name}")

    if not isinstance(score_value, (int, float)):
        return None
    score_float = float(score_value)
    if not np.isfinite(score_float):
        return None
    return score_float


def _extract_content_score_for_stats(record: Dict[str, Any]) -> Optional[float]:
    return _extract_score_for_stats(record, "content_score")


def _build_rescue_band_spec_for_detect(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：在 detect 侧构 rescue band 参数 

    Build rescue-band parameters for detect-side statistics.

    Args:
        cfg: Configuration mapping.

    Returns:
        Rescue band parameter mapping.
    """
    return {
        "base_threshold": float(cfg.get("rescue_band_base_threshold", 0.5)),
        "delta_low": float(cfg.get("rescue_band_delta_low", 0.05)),
        "delta_high": float(cfg.get("rescue_band_delta_high", 0.05)),
        "geo_gate_lower": float(cfg.get("geo_gate_lower", 0.3)),
        "geo_gate_upper": float(cfg.get("geo_gate_upper", 0.7)),
    }


def _compute_null_strata_for_calibration(
    records: list[Dict[str, Any]],
    threshold_value: float,
    cfg: Dict[str, Any],
    score_name: str = "content_score",
) -> Dict[str, Any]:
    valid_content = 0
    geometry_available = 0
    geometry_unavailable = 0
    rescue_candidate = 0

    rescue_spec = _build_rescue_band_spec_for_detect(cfg)
    delta_low = float(rescue_spec.get("delta_low", 0.05))
    lower_bound = float(threshold_value) - delta_low
    for item in records:
        score_float = _extract_score_for_stats(item, score_name)
        if score_float is None:
            continue
        valid_content += 1

        geo_score = _extract_geometry_score(item)
        if geo_score is None:
            geometry_unavailable += 1
        else:
            geometry_available += 1

        if lower_bound <= score_float < float(threshold_value):
            rescue_candidate += 1

    return {
        "global_valid": {
            "n": int(valid_content),
        },
        "geometry_available": {
            "n": int(geometry_available),
        },
        "geometry_unavailable": {
            "n": int(geometry_unavailable),
        },
        "rescue_candidate": {
            "n": int(rescue_candidate),
            "window": f"[threshold-{delta_low}, threshold)",
        },
    }


def _compute_conditional_fpr_records_for_calibration(
    records: list[Dict[str, Any]],
    threshold_value: float,
    cfg: Dict[str, Any],
    score_name: str = "content_score",
) -> list[Dict[str, Any]]:
    threshold_float = float(threshold_value)
    rescue_spec = _build_rescue_band_spec_for_detect(cfg)
    delta_low = float(rescue_spec.get("delta_low", 0.05))
    delta_high = float(rescue_spec.get("delta_high", 0.05))
    geo_gate_lower = float(rescue_spec.get("geo_gate_lower", 0.3))
    geo_gate_upper = float(rescue_spec.get("geo_gate_upper", 0.7))
    align_quality_threshold = _resolve_align_quality_threshold(cfg)
    config_anchors: Dict[str, Any] = {
        "threshold_value": round(threshold_float, 12),
        "rescue_band_delta_low": round(delta_low, 12),
        "rescue_band_delta_high": round(delta_high, 12),
        "geo_gate_lower": round(geo_gate_lower, 12),
        "geo_gate_upper": round(geo_gate_upper, 12),
        "align_quality_threshold": round(align_quality_threshold, 12),
        "hf_failure_rule_version": HF_FAILURE_RULE_VERSION,
        "geo_availability_rule_version": GEO_AVAILABILITY_RULE_VERSION,
    }

    def _make_record(condition_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        sample_count = int(len(payload["sample_ids"]))
        positive_count = int(payload["positive"])
        empirical_fpr = float(positive_count / sample_count) if sample_count > 0 else None
        unavailable_count = int(payload.get("unavailable", 0))
        score_summary = _summarize_scores(payload["scores"])
        sample_set_digest = digests.canonical_sha256(sorted(payload["sample_ids"]))
        digest_payload: Dict[str, Any] = {
            "condition_id": condition_id,
            "definition": payload["definition"],
            "config_anchors": config_anchors,
            "sample_count": sample_count,
            "positive_count": positive_count,
            "unavailable_count": unavailable_count,
            "score_summary": score_summary,
            "sample_set_digest": sample_set_digest,
        }
        record: Dict[str, Any] = {
            "condition_id": condition_id,
            "definition": payload["definition"],
            "sample_count": sample_count,
            "empirical_fpr": empirical_fpr,
            "inputs_digest": digests.canonical_sha256(digest_payload),
        }
        if payload.get("include_unavailable_count", False):
            record["unavailable_count"] = unavailable_count
        return record

    condition_buffers: Dict[str, Dict[str, Any]] = {
        "global": {
            "definition": "all valid null samples",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "geometry_available": {
            "definition": "geometry score available",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "geometry_unavailable": {
            "definition": "geometry score unavailable",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "rescue_band_candidate": {
            "definition": f"content score in rescue band [threshold-{delta_low}, threshold) with delta_high={delta_high}",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "geo_gate_applied": {
            "definition": f"geo gate applied with bounds [{geo_gate_lower}, {geo_gate_upper}]",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "align_quality_ge_threshold": {
            "definition": f"alignment quality >= {align_quality_threshold}",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
            "unavailable": 0,
            "include_unavailable_count": True,
        },
        "hf_failure_rule": {
            "definition": f"HF failure decision rule version {HF_FAILURE_RULE_VERSION}",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
        "geo_availability_rule": {
            "definition": f"geo availability decision rule version {GEO_AVAILABILITY_RULE_VERSION}",
            "scores": [],
            "positive": 0,
            "sample_ids": set(),
        },
    }

    rescue_lower = threshold_float - delta_low
    for index, item in enumerate(records):
        score_float = _extract_score_for_stats(item, score_name)
        if score_float is None:
            continue
        pred_positive = bool(score_float >= threshold_float)
        sample_id = _build_calibration_sample_id(item, index)

        _update_condition_buffer(condition_buffers, "global", score_float, pred_positive, sample_id)

        geo_score = _extract_geometry_score(item)
        if geo_score is None:
            _update_condition_buffer(condition_buffers, "geometry_unavailable", score_float, pred_positive, sample_id)
        else:
            _update_condition_buffer(condition_buffers, "geometry_available", score_float, pred_positive, sample_id)

        if rescue_lower <= score_float < threshold_float:
            _update_condition_buffer(condition_buffers, "rescue_band_candidate", score_float, pred_positive, sample_id)

        if _extract_geo_gate_applied(item) is True:
            _update_condition_buffer(condition_buffers, "geo_gate_applied", score_float, pred_positive, sample_id)

        align_quality_value = _extract_align_quality_value(item)
        if align_quality_value is None:
            condition_buffers["align_quality_ge_threshold"]["unavailable"] += 1
        elif align_quality_value >= align_quality_threshold:
            _update_condition_buffer(condition_buffers, "align_quality_ge_threshold", score_float, pred_positive, sample_id)

        # (新增) 跟踪 HF 失败规则事件
        hf_failure_decision = _extract_hf_failure_decision(item)
        if hf_failure_decision is True:
            _update_condition_buffer(condition_buffers, "hf_failure_rule", score_float, pred_positive, sample_id)

        # (新增) 跟踪 Geo 可用性规则事 
        geo_available = _extract_geo_available(item)
        if geo_available is True:
            _update_condition_buffer(condition_buffers, "geo_availability_rule", score_float, pred_positive, sample_id)

    ordered_conditions = [
        "global",
        "geometry_available",
        "geometry_unavailable",
        "rescue_band_candidate",
        "geo_gate_applied",
        "align_quality_ge_threshold",
        "hf_failure_rule",
        "geo_availability_rule",
    ]
    return [
        _make_record(condition_id=condition_id, payload=condition_buffers[condition_id])
        for condition_id in ordered_conditions
    ]


def _compute_conditional_fpr_for_calibration(
    records: list[Dict[str, Any]],
    threshold_value: float,
    score_name: str = "content_score",
) -> Dict[str, Any]:
    global_total = 0
    global_fp = 0
    geo_available_total = 0
    geo_available_fp = 0
    geo_unavailable_total = 0
    geo_unavailable_fp = 0

    for item in records:
        score_float = _extract_score_for_stats(item, score_name)
        if score_float is None:
            continue
        pred_positive = bool(score_float >= float(threshold_value))

        global_total += 1
        if pred_positive:
            global_fp += 1

        geo_score = _extract_geometry_score(item)
        if geo_score is None:
            geo_unavailable_total += 1
            if pred_positive:
                geo_unavailable_fp += 1
        else:
            geo_available_total += 1
            if pred_positive:
                geo_available_fp += 1

    def _pack(condition_id: str, total_count: int, fp_count: int) -> Dict[str, Any]:
        fpr_value = float(fp_count / total_count) if total_count > 0 else None
        return {
            "condition_id": condition_id,
            "n": int(total_count),
            "fp": int(fp_count),
            "fpr_empirical": fpr_value,
        }

    return {
        "definition": "null-only empirical FPR conditioned on geometry availability",
        "items": [
            _pack("global", global_total, global_fp),
            _pack("geometry_available", geo_available_total, geo_available_fp),
            _pack("geometry_unavailable", geo_unavailable_total, geo_unavailable_fp),
        ],
    }


def _update_condition_buffer(
    condition_buffers: Dict[str, Dict[str, Any]],
    condition_id: str,
    score_value: float,
    pred_positive: bool,
    sample_id: str,
) -> None:
    if condition_id not in condition_buffers:
        # 条件不存在属于调用方逻辑错误，必 fail-fast 
        raise ValueError(f"unknown condition_id: {condition_id}")
    payload = condition_buffers[condition_id]
    payload["scores"].append(float(score_value))
    payload["sample_ids"].add(sample_id)
    if pred_positive:
        payload["positive"] += 1


def _summarize_scores(scores: list[float]) -> Dict[str, Any]:
    valid_scores = [float(value) for value in scores if np.isfinite(float(value))]
    if len(valid_scores) == 0:
        return {
            "count": 0,
            "p50": None,
            "p90": None,
        }
    arr = np.asarray(valid_scores, dtype=float)
    return {
        "count": int(arr.size),
        "p50": float(np.quantile(arr, 0.50, method="higher")),
        "p90": float(np.quantile(arr, 0.90, method="higher")),
    }


def _build_calibration_sample_id(record: Dict[str, Any], index: int) -> str:
    identity_payload: Dict[str, Any] = {
        "index": int(index),
        "cfg_digest": record.get("cfg_digest"),
        "plan_digest": record.get("plan_digest"),
        "image_path": record.get("image_path"),
        "label": record.get("label"),
    }
    return digests.canonical_sha256(identity_payload)


def _resolve_align_quality_threshold(cfg: Dict[str, Any]) -> float:
    for candidate_path in [
        "align_quality_threshold",
        "geometry_align_quality_threshold",
        "geometry.align_quality_threshold",
        "evaluate.align_quality_threshold",
    ]:
        value = _extract_nested_value(cfg, candidate_path)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return 0.5


def _extract_align_quality_value(record: Dict[str, Any]) -> Optional[float]:
    geometry_node = record.get("geometry_evidence_payload")
    if not isinstance(geometry_node, dict):
        return None
    geometry_payload = cast(Dict[str, Any], geometry_node)
    for dotted_path in [
        "sync_metrics.align_quality",
        "sync_metrics.alignment_quality",
        "stability_metrics.align_quality",
        "stability_metrics.alignment_quality",
    ]:
        value = _extract_nested_value(geometry_payload, dotted_path)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return None


def _extract_hf_failure_decision(record: Dict[str, Any]) -> Optional[bool]:
    """
    从记录中提取 HF 失败决策字段 

    Extract HF failure decision from content evidence.

    Args:
        record: Detection record mapping.

    Returns:
        HF failure decision (bool) or None if not available.
    """
    content_node = record.get("content_evidence_payload")
    if not isinstance(content_node, dict):
        return None
    content_payload = cast(Dict[str, Any], content_node)
    hf_failure_decision = content_payload.get("hf_failure_decision")
    if isinstance(hf_failure_decision, bool):
        return hf_failure_decision
    return None


def _extract_geo_available(record: Dict[str, Any]) -> Optional[bool]:
    """
    从记录中提取几何可用性字段 

    Extract geometry availability decision from geometry evidence.

    Args:
        record: Detection record mapping.

    Returns:
        Geometry available decision (bool) or None if not available.
    """
    geometry_node = record.get("geometry_evidence_payload")
    if not isinstance(geometry_node, dict):
        return None
    geometry_payload = cast(Dict[str, Any], geometry_node)
    geo_available = geometry_payload.get("geo_available")
    if isinstance(geo_available, bool):
        return geo_available
    return None


def _extract_geo_gate_applied(record: Dict[str, Any]) -> Optional[bool]:
    """
    功能：从 decision 审计区提 geo gate 是否生效 

    Extract geo-gate-applied flag from decision payload.

    Args:
        record: Detection record mapping.

    Returns:
        Geo gate flag or None when unavailable.
    """
    decision_payload = record.get("decision")
    if not isinstance(decision_payload, dict):
        return None
    decision_mapping = cast(Dict[str, Any], decision_payload)
    routing_decisions = decision_mapping.get("routing_decisions")
    if isinstance(routing_decisions, dict):
        routing_mapping = cast(Dict[str, Any], routing_decisions)
        value = routing_mapping.get("geo_gate_applied")
        if isinstance(value, bool):
            return value
    audit_payload = decision_mapping.get("audit")
    if isinstance(audit_payload, dict):
        audit_mapping = cast(Dict[str, Any], audit_payload)
        value = audit_mapping.get("geo_gate_applied")
        if isinstance(value, bool):
            return value
    return None


def _compute_conditional_metrics_for_evaluate(
    records: list[Dict[str, Any]],
    threshold_value: float,
    score_name: str = "content_score",
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, int]] = {
        "global": {"tp": 0, "fp": 0, "pos": 0, "neg": 0, "accepted": 0},
        "geometry_available": {"tp": 0, "fp": 0, "pos": 0, "neg": 0, "accepted": 0},
        "geometry_unavailable": {"tp": 0, "fp": 0, "pos": 0, "neg": 0, "accepted": 0},
    }

    for item in records:
        score_float = _extract_score_for_stats(item, score_name)
        if score_float is None:
            continue
        gt_value = _extract_ground_truth_label(item)
        if gt_value is None:
            continue

        pred_positive = bool(score_float >= float(threshold_value))
        group_name = "geometry_available" if _extract_geometry_score(item) is not None else "geometry_unavailable"

        for key_name in ["global", group_name]:
            group = groups[key_name]
            group["accepted"] += 1
            if gt_value:
                group["pos"] += 1
                if pred_positive:
                    group["tp"] += 1
            else:
                group["neg"] += 1
                if pred_positive:
                    group["fp"] += 1

    items: list[Dict[str, Any]] = []
    for condition_id, group in groups.items():
        tpr_value = float(group["tp"] / group["pos"]) if group["pos"] > 0 else None
        fpr_value = float(group["fp"] / group["neg"]) if group["neg"] > 0 else None
        items.append(
            {
                "condition_id": condition_id,
                "n_accepted": int(group["accepted"]),
                "n_pos": int(group["pos"]),
                "n_neg": int(group["neg"]),
                "tpr_at_fpr": tpr_value,
                "fpr_empirical": fpr_value,
            }
        )

    return {
        "version": "conditional_eval_v1",
        "items": items,
    }


def _extract_nested_value(payload: Dict[str, Any], dotted_path: str) -> Any:
    if not dotted_path:
        return None
    cursor: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor_mapping = cast(Dict[str, Any], cursor)
        cursor = cursor_mapping.get(part)
    return cursor


def _adapt_content_evidence_for_fusion(content_evidence: Any) -> Dict[str, Any]:
    """
    功能：将 ContentEvidence 数据类适配为融合规则期望的字典格式 

    Adapt ContentEvidence (dataclass or dict) to fusion rule expected format.
    Prioritizes .as_dict() method; falls back to direct dict or attribute extraction.

    Args:
        content_evidence: ContentEvidence dataclass, dict, or compatible object.

    Returns:
        Dict with fields expected by fusion rule (status, score, etc.).

    Raises:
        TypeError: If content_evidence type is unrecognized.
    """
    if isinstance(content_evidence, dict):
        # 已是字典，直接返回；但需确保 content_score 字段存在（fusion rule 读取该键 
        # 而部分来源只 score 键） 
        result_dict = cast(Dict[str, Any], content_evidence)
        if "content_score" not in result_dict and "score" in result_dict:
            result_dict["content_score"] = result_dict["score"]
        _synchronize_content_score_aliases(result_dict)
        return result_dict
    
    # 尝试 .as_dict() 方法 
    if hasattr(content_evidence, "as_dict") and callable(content_evidence.as_dict):
        try:
            converted = content_evidence.as_dict()
            if isinstance(converted, dict):
                converted_dict = cast(Dict[str, Any], converted)
                if "content_score" not in converted_dict and "score" in converted_dict:
                    converted_dict["content_score"] = converted_dict["score"]
                _synchronize_content_score_aliases(converted_dict)
                return converted_dict
        except Exception:
            # 如果 .as_dict() 失败，继续尝试属性提取 
            pass
    
    # 从数据类属性直接构造 
    adapted: Dict[str, Any] = {}
    
    # 提取关键字段（来 ContentEvidence 冻结结构） 
    for field_name in ["status", "score", "audit", "mask_digest", "mask_stats",
                       "plan_digest", "basis_digest", "lf_trace_digest", "hf_trace_digest",
                       "lf_score", "hf_score", "score_parts", "trajectory_evidence",
                       "content_failure_reason"]:
        if hasattr(content_evidence, field_name):
            adapted[field_name] = getattr(content_evidence, field_name)
    
    # 确保 content_score 字段存在：fusion rule 读取 content_score 
    #  ContentEvidence 数据类只 score 字段（两者语义等价） 
    if "content_score" not in adapted and "score" in adapted:
        adapted["content_score"] = adapted["score"]
    _synchronize_content_score_aliases(adapted)
    
    return adapted if adapted else {"status": "unknown"}


def _adapt_geometry_evidence_for_fusion(geometry_evidence: Any) -> Dict[str, Any]:
    """
    功能：将 GeometryEvidence 数据类适配为融合规则期望的字典格式 

    Adapt GeometryEvidence (dataclass or dict) to fusion rule expected format.
    Prioritizes .as_dict() method; falls back to direct dict or attribute extraction.

    Args:
        geometry_evidence: GeometryEvidence dataclass, dict, or compatible object.

    Returns:
        Dict with fields expected by fusion rule (status, geo_score, etc.).

    Raises:
        TypeError: If geometry_evidence type is unrecognized.
    """
    if isinstance(geometry_evidence, dict):
        # 已是字典，直接返回 
        return cast(Dict[str, Any], geometry_evidence)
    
    # 尝试 .as_dict() 方法 
    if hasattr(geometry_evidence, "as_dict") and callable(geometry_evidence.as_dict):
        try:
            converted = geometry_evidence.as_dict()
            if isinstance(converted, dict):
                return cast(Dict[str, Any], converted)
        except Exception:
            # 如果 .as_dict() 失败，继续尝试属性提取 
            pass
    
    # 从数据类属性直接构造 
    adapted: Dict[str, Any] = {}
    
    # 提取关键字段（来 GeometryEvidence 冻结结构） 
    for field_name in [
        "status",
        "geo_score",
        "audit",
        "anchor_digest",
        "anchor_config_digest",
        "anchor_metrics",
        "stability_metrics",
        "sync_digest",
        "sync_metrics",
        "sync_config_digest",
        "sync_quality_metrics",
        "resolution_binding",
        "geo_score_direction",
        "geo_failure_reason",
        "geometry_failure_reason",
    ]:
        if hasattr(geometry_evidence, field_name):
            adapted[field_name] = getattr(geometry_evidence, field_name)
    
    return adapted if adapted else {"status": "unknown"}


def _is_image_domain_sidecar_enabled(cfg: Dict[str, Any], ablation_override: bool | None = None) -> bool:
    """
    功能：解析图像域 sidecar 开关 

    Resolve whether image-domain detector sidecar is enabled.

    Args:
        cfg: Configuration mapping.

    Returns:
        True if sidecar is enabled.
    """
    if ablation_override is not None and not ablation_override:
        return False
    detect_runtime_node = cfg.get("detect_runtime")
    detect_runtime_cfg = cast(Dict[str, Any], detect_runtime_node) if isinstance(detect_runtime_node, dict) else {}
    explicit = detect_runtime_cfg.get("image_domain_sidecar_enabled")
    if isinstance(explicit, bool):
        return explicit
    paper_node = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_node) if isinstance(paper_node, dict) else {}
    if bool(paper_cfg.get("enabled", False)):
        return False
    # v2.0 正式路径收口后，sidecar 默认禁用；必须显式配置才能启用 
    return False


def _is_synthetic_negative_closure_sample(content_payload: Dict[str, Any]) -> bool:
    """
    功能：判定样本是否为 synthetic negative closure 标记样本 

    Determine whether sample is marked as synthetic negative closure.

    Args:
        content_payload: Content evidence payload mapping.

    Returns:
        True when sample should be excluded as synthetic negative closure.
    """
    if not isinstance(content_payload, dict):
        raise TypeError("content_payload must be dict")
    usage_value = content_payload.get("calibration_sample_usage")
    return usage_value == "synthetic_negative_for_ground_truth_closure"


def _resolve_runtime_self_attention_maps(cfg: Dict[str, Any]) -> Any:
    """
    功能：解 detect 侧真 self-attention maps 载荷 

    Resolve runtime self-attention maps from detect transient fields.

    Args:
        cfg: Configuration mapping.

    Returns:
        Attention maps payload or None.
    """
    for key_name in [
        "__detect_attention_maps__",
        "__detect_self_attention_maps__",
        "__runtime_self_attention_maps__",
    ]:
        candidate = cfg.get(key_name)
        if candidate is not None:
            return candidate
    return None


def _extract_subspace_evidence_semantics(plan_payload: Any) -> Dict[str, Any]:
    """
    功能：从计划载荷中提取子空间证据语义 

    Extract subspace evidence semantics from planner payload.

    Args:
        plan_payload: Plan payload mapping.

    Returns:
        Semantics mapping or empty dict.
    """
    if not isinstance(plan_payload, dict):
        return {}
    plan_payload_mapping = cast(Dict[str, Any], plan_payload)
    direct_value = plan_payload_mapping.get("subspace_evidence_semantics")
    if isinstance(direct_value, dict):
        return cast(Dict[str, Any], direct_value)
    plan_node = plan_payload_mapping.get("plan")
    if isinstance(plan_node, dict):
        plan_node_mapping = cast(Dict[str, Any], plan_node)
        nested_value = plan_node_mapping.get("subspace_evidence_semantics")
        if isinstance(nested_value, dict):
            return cast(Dict[str, Any], nested_value)
    plan_stats = plan_payload_mapping.get("plan_stats")
    if isinstance(plan_stats, dict):
        plan_stats_mapping = cast(Dict[str, Any], plan_stats)
        stats_value = plan_stats_mapping.get("subspace_evidence_semantics")
        if isinstance(stats_value, dict):
            return cast(Dict[str, Any], stats_value)
    return {}


def _resolve_policy_path_semantics_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：解析当前运行绑定的 policy_path semantics 载荷。

    Resolve active policy_path semantics payload from bound fact source.

    Args:
        cfg: Configuration mapping.

    Returns:
        Policy semantics mapping, or empty dict when unavailable.
    """
    semantics_value = cfg.get("__policy_path_semantics__")
    if isinstance(semantics_value, PolicyPathSemantics):
        semantics_data = semantics_value.data
        return cast(Dict[str, Any], semantics_data) if isinstance(semantics_data, dict) else {}
    if isinstance(semantics_value, dict):
        return cast(Dict[str, Any], semantics_value)
    try:
        semantics_obj = load_policy_path_semantics()
    except Exception:
        return {}
    return cast(Dict[str, Any], semantics_obj.data) if isinstance(semantics_obj.data, dict) else {}


def _resolve_required_chain_policy(
    cfg: Dict[str, Any],
    chain_name: str,
) -> Dict[str, Any] | None:
    """
    功能：从 facts source 解析当前 policy_path 的必需链失败语义。

    Resolve required-chain failure semantics from active policy facts.

    Args:
        cfg: Configuration mapping.
        chain_name: Required chain name.

    Returns:
        Chain policy mapping, or None when chain is not required.
    """
    if not isinstance(chain_name, str) or not chain_name:
        return None
    policy_path = cfg.get("policy_path")
    if not isinstance(policy_path, str) or not policy_path:
        return None
    semantics_payload = _resolve_policy_path_semantics_payload(cfg)
    policy_paths = semantics_payload.get("policy_paths")
    if not isinstance(policy_paths, dict):
        return None
    policy_spec = policy_paths.get(policy_path)
    if not isinstance(policy_spec, dict):
        return None
    required_chains = policy_spec.get("required_chains")
    on_chain_failure = policy_spec.get("on_chain_failure")
    if not isinstance(required_chains, dict) or not isinstance(on_chain_failure, dict):
        return None
    if required_chains.get(chain_name) is not True:
        return None
    chain_policy = on_chain_failure.get(chain_name)
    if not isinstance(chain_policy, dict):
        return None
    resolved_policy = dict(chain_policy)
    resolved_policy["policy_path"] = policy_path
    resolved_policy["chain_name"] = chain_name
    return resolved_policy


def _resolve_detect_sync_latents(cfg: Dict[str, Any]) -> Any:
    """
    功能：解析 detect 侧 sync 主路径使用的运行时 latent。 

    Resolve canonical detect-time latent input for sync-primary geometry.

    Args:
        cfg: Configuration mapping.

    Returns:
        Latest cached detect latent when available; otherwise None.
    """
    detect_traj_cache = cfg.get("__detect_trajectory_latent_cache__")
    if detect_traj_cache is None:
        return None
    available_steps = getattr(detect_traj_cache, "available_steps", None)
    get_step = getattr(detect_traj_cache, "get", None)
    if not callable(available_steps) or not callable(get_step):
        return None
    try:
        cached_steps = available_steps()
    except Exception:
        return None
    if not isinstance(cached_steps, list) or len(cached_steps) == 0:
        return None
    latest_step = cached_steps[-1]
    if not isinstance(latest_step, int):
        return None
    try:
        return get_step(latest_step)
    except Exception:
        return None


def _build_pre_sync_relation_binding(
    geometry_extractor: Any,
    cfg: Dict[str, Any],
    runtime_inputs: Dict[str, Any],
) -> Dict[str, Any] | None:
    """
    功能：在 sync-primary 前解析可复用的 relation 绑定。 

    Resolve authoritative relation binding from authentic runtime attention
    before sync execution, without changing sync-primary semantics.

    Args:
        geometry_extractor: Geometry extractor instance.
        cfg: Configuration mapping.
        runtime_inputs: Runtime input mapping.

    Returns:
        Mapping with relation binding fields, or None when unavailable.
    """
    if geometry_extractor is None:
        return None
    if not isinstance(runtime_inputs, dict):
        return None
    relation_binding: Dict[str, Any] = {
        "binding_source": "authentic_runtime_self_attention",
    }
    try:
        precomputed_result = _run_geometry_extractor_with_runtime_inputs(
            geometry_extractor,
            cfg,
            runtime_inputs,
        )
    except Exception as exc:
        relation_binding["binding_status"] = "failed"
        relation_binding["geometry_failure_reason"] = f"pre_sync_relation_binding_failed: {type(exc).__name__}"
        return relation_binding
    if not isinstance(precomputed_result, dict):
        relation_binding["binding_status"] = "invalid"
        relation_binding["geometry_failure_reason"] = "pre_sync_relation_binding_non_mapping"
        return relation_binding
    precomputed_mapping = cast(Dict[str, Any], precomputed_result)
    binding_status = _normalize_geometry_chain_status(precomputed_mapping.get("status"))
    relation_binding["binding_status"] = binding_status
    geometry_failure_reason = precomputed_mapping.get("geometry_failure_reason")
    if isinstance(geometry_failure_reason, str) and geometry_failure_reason:
        relation_binding["geometry_failure_reason"] = geometry_failure_reason
    geometry_absent_reason = precomputed_mapping.get("geometry_absent_reason")
    if isinstance(geometry_absent_reason, str) and geometry_absent_reason:
        relation_binding["geometry_absent_reason"] = geometry_absent_reason
    if binding_status != "ok":
        return relation_binding
    relation_digest = precomputed_mapping.get("relation_digest")
    if not isinstance(relation_digest, str) or not relation_digest:
        relation_binding["binding_status"] = "invalid"
        relation_binding["geometry_failure_reason"] = "pre_sync_relation_digest_missing"
        return relation_binding
    relation_binding["relation_digest"] = relation_digest
    relation_binding["relation_digest_source"] = "authentic_runtime_self_attention"
    anchor_digest = precomputed_mapping.get("anchor_digest")
    if isinstance(anchor_digest, str) and anchor_digest:
        relation_binding["anchor_digest"] = anchor_digest
    return relation_binding


def _normalize_detect_sync_failure_reason(sync_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：修正 detect 侧 sync 失败原因的审计语义。 

    Normalize sync failure semantics for detect-path diagnostics.

    Args:
        sync_result: Sync result mapping.

    Returns:
        Normalized sync result mapping.
    """
    if not isinstance(sync_result, dict):
        return sync_result
    normalized_result = dict(sync_result)
    absent_reason = normalized_result.get("geometry_absent_reason")
    if absent_reason == "relation_digest_absent_embed_mode":
        normalized_result["geometry_absent_reason_raw"] = absent_reason
        normalized_result["geometry_absent_reason"] = "detect_sync_relation_binding_missing"
    elif absent_reason == "latents_missing":
        normalized_result["geometry_absent_reason_raw"] = absent_reason
        normalized_result["geometry_absent_reason"] = "detect_sync_latents_missing"
    return normalized_result


def _close_formal_required_chain_decision(
    cfg: Dict[str, Any],
    content_evidence_payload: Dict[str, Any],
    geometry_evidence_payload: Dict[str, Any],
    fusion_result: FusionDecision,
) -> FusionDecision:
    """
    功能：在 formal path 下按 facts source 前置收口必需链判决语义。

    Close formal required-chain semantics before record write.

    Args:
        cfg: Configuration mapping.
        content_evidence_payload: Content evidence payload mapping.
        geometry_evidence_payload: Geometry evidence payload mapping.
        fusion_result: Original fusion decision.

    Returns:
        Possibly normalized FusionDecision for formal record emission.
    """
    chain_payloads = {
        "content": content_evidence_payload if isinstance(content_evidence_payload, dict) else {},
        "geometry": geometry_evidence_payload if isinstance(geometry_evidence_payload, dict) else {},
    }
    for chain_name, chain_payload in chain_payloads.items():
        chain_policy = _resolve_required_chain_policy(cfg, chain_name)
        if not isinstance(chain_policy, dict):
            continue
        chain_status = chain_payload.get("status")
        if chain_status == "ok":
            continue
        set_decision_to = chain_policy.get("set_decision_to")
        if set_decision_to is not None:
            continue

        normalized_audit = dict(fusion_result.audit) if isinstance(fusion_result.audit, dict) else {}
        record_fail_reason = chain_policy.get("record_fail_reason")
        if isinstance(record_fail_reason, str) and record_fail_reason:
            normalized_audit["failure_reason"] = record_fail_reason
        normalized_audit["formal_policy_path"] = chain_policy.get("policy_path")
        normalized_audit["formal_required_chain"] = chain_name

        chain_failure_reason = chain_payload.get(f"{chain_name}_failure_reason")
        if isinstance(chain_failure_reason, str) and chain_failure_reason:
            normalized_audit[f"{chain_name}_failure_reason"] = chain_failure_reason
        chain_absent_reason = chain_payload.get(f"{chain_name}_absent_reason")
        if isinstance(chain_absent_reason, str) and chain_absent_reason:
            normalized_audit[f"{chain_name}_absent_reason"] = chain_absent_reason
        if chain_name == "geometry":
            geometry_absent_reason_raw = chain_payload.get("geometry_absent_reason_raw")
            if isinstance(geometry_absent_reason_raw, str) and geometry_absent_reason_raw:
                normalized_audit["geometry_absent_reason_raw"] = geometry_absent_reason_raw

        normalized_summary = dict(fusion_result.evidence_summary)
        normalized_summary[f"{chain_name}_status"] = chain_status
        action = chain_policy.get("action")
        decision_status = "abstain" if action == "abstain" else "error"
        return FusionDecision(
            is_watermarked=None,
            decision_status=decision_status,
            thresholds_digest=fusion_result.thresholds_digest,
            evidence_summary=normalized_summary,
            audit=normalized_audit,
            fusion_rule_version=fusion_result.fusion_rule_version,
            used_threshold_id=fusion_result.used_threshold_id,
            routing_decisions=fusion_result.routing_decisions,
            routing_digest=fusion_result.routing_digest,
            conditional_fpr_notes=fusion_result.conditional_fpr_notes,
        )
    return fusion_result


def _build_geometry_runtime_inputs(
    cfg: Dict[str, Any],
    sync_result: Dict[str, Any] | None = None,
    anchor_result: Dict[str, Any] | None = None,
    relation_binding: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    功能：构造几何链运行时输入域 

    Build geometry runtime input payload for sync and extractor.

    Args:
        cfg: Configuration mapping.
        sync_result: Optional sync result mapping.
        anchor_result: Optional anchor result mapping.

    Returns:
        Runtime inputs mapping.
    """
    runtime_inputs: Dict[str, Any] = {
        "pipeline": cfg.get("__detect_pipeline_obj__"),
        "rng": cfg.get("rng"),
    }
    latents = _resolve_detect_sync_latents(cfg)
    if latents is not None:
        runtime_inputs["latents"] = latents
    paper_node = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_node) if isinstance(paper_node, dict) else {}
    paper_enabled = bool(paper_cfg.get("enabled", False))
    if isinstance(relation_binding, dict):
        relation_digest = relation_binding.get("relation_digest")
        if isinstance(relation_digest, str) and relation_digest:
            runtime_inputs["relation_digest"] = relation_digest
        anchor_digest = relation_binding.get("anchor_digest")
        if isinstance(anchor_digest, str) and anchor_digest:
            runtime_inputs["anchor_digest"] = anchor_digest
        runtime_inputs["relation_binding"] = relation_binding
    prebuilt_attention_maps = _resolve_runtime_self_attention_maps(cfg)
    if prebuilt_attention_maps is not None:
        runtime_inputs["attention_maps"] = prebuilt_attention_maps
        runtime_inputs["attention_maps_source"] = "runtime_self_attention"
        runtime_inputs["attention_maps_evidence_level"] = "primary"
        capture_source = cfg.get("__runtime_self_attention_source__")
        if isinstance(capture_source, str) and capture_source:
            runtime_inputs["attention_capture_source"] = capture_source
        else:
            runtime_inputs["attention_capture_source"] = "hook_capture"
    elif paper_enabled:
        # paper 正式路径要求 runtime self-attention；若缺失则返回 absent。
        runtime_inputs["attention_maps_source"] = "absent"
        runtime_inputs["attention_maps_missing_reason"] = "runtime_self_attention_missing_under_paper_mode"
        return runtime_inputs
    if isinstance(anchor_result, dict):
        relation_digest = anchor_result.get("relation_digest")
        if isinstance(relation_digest, str) and relation_digest:
            runtime_inputs["relation_digest"] = relation_digest
        anchor_digest = anchor_result.get("anchor_digest")
        if isinstance(anchor_digest, str) and anchor_digest:
            runtime_inputs["anchor_digest"] = anchor_digest
        runtime_inputs["anchor_result"] = anchor_result
    if isinstance(sync_result, dict):
        relation_digest = sync_result.get("relation_digest_bound")
        if isinstance(relation_digest, str) and relation_digest:
            runtime_inputs["relation_digest"] = relation_digest
        sync_digest = sync_result.get("sync_digest")
        if isinstance(sync_digest, str) and sync_digest:
            runtime_inputs["sync_digest"] = sync_digest
        runtime_inputs["sync_result"] = sync_result
    # embed  latent 空间统计（由 run_detect.py  input_record 注入），
    #  sync 模块 cross-comparison 替代单侧统计 
    embed_latent_stats = cfg.get("__embed_latent_spatial_stats__")
    if isinstance(embed_latent_stats, dict):
        runtime_inputs["embed_latent_stats"] = embed_latent_stats
    return runtime_inputs


def _run_sync_module_for_detect(sync_module: Any, cfg: Dict[str, Any], runtime_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：在 detect 侧执行同步模块 

    Execute sync module for detect runtime.

    Args:
        sync_module: Runtime sync module instance.
        cfg: Configuration mapping.
        runtime_inputs: Runtime input mapping.

    Returns:
        Sync result mapping.
    """
    if sync_module is None:
        return {"status": "absent", "geometry_absent_reason": "sync_module_absent"}
    pipeline_obj = runtime_inputs.get("pipeline")
    latents = runtime_inputs.get("latents")
    sync_with_context = getattr(sync_module, "sync_with_context", None)
    has_sync = hasattr(sync_module, "sync") and callable(getattr(sync_module, "sync", None))
    if callable(sync_with_context):
        try:
            sync_ctx = SyncRuntimeContext(
                pipeline=pipeline_obj,
                latents=latents,
                rng=runtime_inputs.get("rng"),
                trajectory_evidence=None
            )
            try:
                sync_result = sync_with_context(cfg, sync_ctx, runtime_inputs=runtime_inputs)
            except TypeError:
                sync_result = sync_with_context(cfg, sync_ctx)
            if isinstance(sync_result, dict):
                normalized: Dict[str, Any] = dict(cast(Dict[str, Any], sync_result))
                raw_status = normalized.get("sync_status")
                if not isinstance(raw_status, str) or not raw_status:
                    raw_status = normalized.get("status")
                if isinstance(raw_status, str) and raw_status:
                    lowered = raw_status.lower()
                    if lowered == "fail":
                        lowered = "failed"
                    if lowered in {"ok", "absent", "mismatch", "failed"}:
                        normalized["sync_status"] = lowered
                        normalized["status"] = lowered
                relation_binding = runtime_inputs.get("relation_binding")
                if isinstance(relation_binding, dict):
                    normalized["relation_binding_diagnostics"] = dict(relation_binding)
                return _normalize_detect_sync_failure_reason(normalized)
        except Exception as exc:
            return {
                "status": "failed",
                "geometry_failure_reason": f"sync_with_context_failed: {type(exc).__name__}",
            }
    if not has_sync:
        return {"status": "absent", "geometry_absent_reason": "sync_module_missing_sync_method"}
    try:
        sync_result = sync_module.sync(cfg)
        if isinstance(sync_result, dict):
            normalized_sync_result = dict(cast(Dict[str, Any], sync_result))
            relation_binding = runtime_inputs.get("relation_binding")
            if isinstance(relation_binding, dict):
                normalized_sync_result["relation_binding_diagnostics"] = dict(relation_binding)
            return _normalize_detect_sync_failure_reason(normalized_sync_result)
    except Exception as exc:
        return {
            "status": "failed",
            "geometry_failure_reason": f"sync_failed: {type(exc).__name__}",
        }
    return {"status": "absent", "geometry_absent_reason": "sync_module_returned_non_mapping"}


def _run_geometry_chain_with_sync(
    impl_set: BuiltImplSet,
    cfg: Dict[str, Any],
    *,
    enable_anchor: bool = True,
    enable_sync: bool = True,
) -> Any:
    """
    功能：detect 几何链按主辅层级执行：sync 优先，anchor 仅在 sync 成功后启用 

    Run detect geometry chain with sync-primary/anchor-secondary hard gate.
    When sync_primary_anchor_secondary is enabled, anchor extraction is
    gated on sync success to prevent "stable but untrustworthy" pseudo
    geometry evidence.

    Args:
        impl_set: Built runtime implementation set.
        cfg: Configuration mapping.

    Returns:
        Geometry evidence mapping.
    """
    sync_primary_mode = _is_sync_primary_anchor_secondary_enabled(cfg)

    if sync_primary_mode:
        # sync_primary 模式：先执行 sync（主几何证据），再按 sync 结果门控 anchor（辅锚点） 
        # 研究目标：Self-Attention 辅锚点仅在主同步成功后启用 
        #
        # sync 仍为主证据，但 relation binding 需先由真实 runtime self-attention 解析，
        # 否则 sync 无法在 formal detect 路径闭合自身输入契约。
        relation_binding = None
        if enable_sync:
            sync_base_inputs = _build_geometry_runtime_inputs(cfg)
            relation_binding = _build_pre_sync_relation_binding(
                impl_set.geometry_extractor,
                cfg,
                sync_base_inputs,
            )
            if isinstance(relation_binding, dict):
                sync_base_inputs = _build_geometry_runtime_inputs(cfg, relation_binding=relation_binding)
            sync_module = getattr(impl_set, "sync_module", None)
            sync_result: Dict[str, Any] = _run_sync_module_for_detect(sync_module, cfg, sync_base_inputs)
        else:
            sync_result = {
                "status": "absent",
                "sync_status": "absent",
                "geometry_absent_reason": "sync_disabled_by_ablation",
            }

        sync_status = _normalize_geometry_chain_status(
            sync_result.get("sync_status") or sync_result.get("status")
        )

        # 辅锚点硬门控：仅 sync 成功时才执行 anchor 
        # 避免产出"稳定但不可信"的伪几何证据 
        anchor_gated_out = False
        anchor_result: Dict[str, Any]
        if enable_anchor:
            if sync_status != "ok":
                anchor_result = {
                    "status": "absent",
                    "geo_score": None,
                    "geometry_absent_reason": "anchor_gated_by_sync_failure",
                    "anchor_gate_detail": {
                        "gate_policy": "sync_primary_anchor_secondary_hard_gate",
                        "sync_status": sync_status,
                        "reason": "attention anchor suppressed because primary sync did not succeed",
                    },
                    "relation_digest": None,
                    "anchor_digest": None,
                }
                anchor_gated_out = True
            else:
                base_inputs = _build_geometry_runtime_inputs(
                    cfg,
                    sync_result=sync_result,
                    relation_binding=relation_binding,
                )
                anchor_result_raw = _run_geometry_extractor_with_runtime_inputs(
                    impl_set.geometry_extractor, cfg, base_inputs
                )
                if not isinstance(anchor_result_raw, dict):
                    anchor_result = {
                        "status": "failed",
                        "geo_score": None,
                        "geometry_failure_reason": "geometry_anchor_result_non_mapping",
                    }
                else:
                    anchor_result = cast(Dict[str, Any], anchor_result_raw)
        else:
            anchor_result = {
                "status": "absent",
                "geo_score": None,
                "geometry_absent_reason": "anchor_disabled_by_ablation",
                "relation_digest": None,
                "anchor_digest": None,
            }
            anchor_gated_out = False
    else:
        # 兼容模式：先执行 anchor，再执行 sync（保持旧有排序） 
        anchor_gated_out = False
        base_inputs = _build_geometry_runtime_inputs(cfg)
        base_inputs["sync_result"] = {
            "status": "absent",
            "sync_status": "pending_anchor_first",
            "geometry_absent_reason": "sync_not_executed_yet",
        }

        if enable_anchor:
            anchor_result_raw = _run_geometry_extractor_with_runtime_inputs(
                impl_set.geometry_extractor,
                cfg,
                base_inputs,
            )
            if not isinstance(anchor_result_raw, dict):
                anchor_result = {
                    "status": "failed",
                    "geo_score": None,
                    "geometry_failure_reason": "geometry_anchor_result_non_mapping",
                }
            else:
                anchor_result = cast(Dict[str, Any], anchor_result_raw)
        else:
            anchor_result = {
                "status": "absent",
                "geo_score": None,
                "geometry_absent_reason": "anchor_disabled_by_ablation",
                "relation_digest": None,
                "anchor_digest": None,
            }

        if enable_sync:
            sync_inputs = _build_geometry_runtime_inputs(
                cfg,
                anchor_result=anchor_result,
            )
            sync_module = getattr(impl_set, "sync_module", None)
            sync_result = _run_sync_module_for_detect(sync_module, cfg, sync_inputs)
        else:
            sync_result = {
                "status": "absent",
                "sync_status": "absent",
                "geometry_absent_reason": "sync_disabled_by_ablation",
            }

    # (3) 合并几何证据
    geometry_result: Dict[str, Any] = dict(anchor_result)
    geometry_result.setdefault("sync_result", sync_result)
    geometry_result.setdefault("anchor_result", anchor_result)
    geometry_result.setdefault("anchor_status", anchor_result.get("status"))
    geometry_result["anchor_gated_by_sync"] = anchor_gated_out
    anchor_relation_digest = anchor_result.get("relation_digest")
    if isinstance(anchor_relation_digest, str) and anchor_relation_digest:
        geometry_result.setdefault("relation_digest", anchor_relation_digest)
    sync_status_val = sync_result.get("sync_status") or sync_result.get("status")
    if isinstance(sync_status_val, str) and sync_status_val and "sync_status" not in geometry_result:
        geometry_result["sync_status"] = sync_status_val
    if "sync_metrics" not in geometry_result:
        geometry_result["sync_metrics"] = sync_result.get("sync_quality_metrics")
    sync_quality_semantics = sync_result.get("sync_quality_semantics")
    if isinstance(sync_quality_semantics, dict):
        geometry_result["sync_quality_semantics"] = sync_quality_semantics
    relation_digest_bound = sync_result.get("relation_digest_bound")
    if isinstance(relation_digest_bound, str) and relation_digest_bound:
        geometry_result["relation_digest_bound"] = relation_digest_bound
    # sync_digest 提升：将 sync_result.sync_digest 暴露于顶层， assert_paper_mechanisms 读取 
    if not isinstance(geometry_result.get("sync_digest"), str) or not geometry_result.get("sync_digest"):
        sync_digest_val = sync_result.get("sync_digest")
        if isinstance(sync_digest_val, str) and sync_digest_val:
            geometry_result["sync_digest"] = sync_digest_val
    geometry_result["relation_digest_binding"] = {
        "anchor_relation_digest": anchor_relation_digest if isinstance(anchor_relation_digest, str) else None,
        "sync_relation_digest_bound": relation_digest_bound if isinstance(relation_digest_bound, str) else None,
        "binding_status": "matched" if isinstance(anchor_relation_digest, str) and isinstance(relation_digest_bound, str) and anchor_relation_digest == relation_digest_bound else "mismatch_or_absent",
    }
    relation_binding_diagnostics = sync_result.get("relation_binding_diagnostics")
    if isinstance(relation_binding_diagnostics, dict):
        geometry_result["relation_binding_diagnostics"] = dict(relation_binding_diagnostics)
    geometry_absent_reason_raw = sync_result.get("geometry_absent_reason_raw")
    if isinstance(geometry_absent_reason_raw, str) and geometry_absent_reason_raw:
        geometry_result["geometry_absent_reason_raw"] = geometry_absent_reason_raw
    geometry_result = _enforce_sync_primary_anchor_secondary(
        cfg=cfg,
        geometry_result=geometry_result,
        anchor_result=anchor_result,
        sync_result=sync_result,
    )
    # sync_primary 模式 sync 成功时， sync geo_score（quality_score）写 geometry_result 
    # 确保 _extract_geometry_score 可读取到有效浮点分数 
    if sync_primary_mode:
        sync_status_for_geo = _normalize_geometry_chain_status(
            sync_result.get("sync_status") or sync_result.get("status")
        )
        if sync_status_for_geo == "ok":
            sync_geo = sync_result.get("geo_score")
            if isinstance(sync_geo, (int, float)) and np.isfinite(float(sync_geo)):
                geometry_result["geo_score"] = float(sync_geo)
    return geometry_result


def _is_sync_primary_anchor_secondary_enabled(cfg: Dict[str, Any]) -> bool:
    """
    功能：解 detect 几何链主辅证据切换开关 

    Resolve controlled switch for sync-primary and anchor-secondary semantics.

    Args:
        cfg: Runtime configuration mapping.

    Returns:
        True when sync-primary mode is enabled.
    """
    detect_node = cfg.get("detect")
    detect_cfg = cast(Dict[str, Any], detect_node) if isinstance(detect_node, dict) else {}
    geometry_node = detect_cfg.get("geometry")
    geometry_cfg = cast(Dict[str, Any], geometry_node) if isinstance(geometry_node, dict) else {}
    explicit_switch = geometry_cfg.get("sync_primary_anchor_secondary")
    if isinstance(explicit_switch, bool):
        return explicit_switch

    paper_node = cfg.get("paper_faithfulness")
    paper_cfg = cast(Dict[str, Any], paper_node) if isinstance(paper_node, dict) else {}
    return bool(paper_cfg.get("enabled", False))


def _normalize_geometry_chain_status(raw_status: Any) -> str:
    """
    功能：归一化几何链状态到 ok/absent/mismatch/failed 

    Normalize geometry chain status into canonical enum.

    Args:
        raw_status: Raw status token.

    Returns:
        Canonical status token.
    """
    if not isinstance(raw_status, str) or not raw_status:
        return "failed"
    normalized = raw_status.strip().lower()
    if normalized == "fail":
        return "failed"
    if normalized in {"ok", "absent", "mismatch", "failed"}:
        return normalized
    if normalized in {"error"}:
        return "failed"
    return "failed"


def _enforce_sync_primary_anchor_secondary(
    *,
    cfg: Dict[str, Any],
    geometry_result: Dict[str, Any],
    anchor_result: Dict[str, Any],
    sync_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    功能：在受控开关下执行 sync 主证据、anchor 辅证据语义 

    Enforce sync-primary and anchor-secondary semantics with rollback-safe switch.

    Args:
        cfg: Runtime configuration mapping.
        geometry_result: Geometry payload to mutate.
        anchor_result: Anchor extraction payload.
        sync_result: Sync module payload.

    Returns:
        Updated geometry payload.
    """
    enabled = _is_sync_primary_anchor_secondary_enabled(cfg)
    anchor_status = _normalize_geometry_chain_status(anchor_result.get("status"))
    sync_status = _normalize_geometry_chain_status(sync_result.get("sync_status") or sync_result.get("status"))

    geometry_result["geometry_evidence_hierarchy"] = {
        "policy_version": "sync_primary_anchor_secondary",
        "switch_enabled": enabled,
        "primary_source": "sync" if enabled else "anchor",
        "secondary_source": "anchor" if enabled else "sync",
        "anchor_status": anchor_status,
        "sync_status": sync_status,
    }

    if not enabled:
        return geometry_result

    geometry_result["status"] = sync_status
    geometry_result["sync_status"] = sync_status
    geometry_result["anchor_status"] = anchor_status
    geometry_result["relation_digest_primary_source"] = "anchor_compat"

    if sync_status != "ok":
        geometry_result["geo_score"] = None
        failure_reason = sync_result.get("geometry_failure_reason")
        absent_reason = sync_result.get("geometry_absent_reason")
        if isinstance(failure_reason, str) and failure_reason:
            geometry_result["geometry_failure_reason"] = failure_reason
        elif isinstance(absent_reason, str) and absent_reason:
            geometry_result["geometry_absent_reason"] = absent_reason

    return geometry_result


def _run_geometry_extractor_with_runtime_inputs(
    geometry_extractor: Any,
    cfg: Dict[str, Any],
    runtime_inputs: Dict[str, Any] | None = None
) -> Any:
    """
    功能：以兼容方式调用 geometry extractor 

    Invoke geometry extractor with runtime inputs when supported.

    Args:
        geometry_extractor: Geometry extractor instance.
        cfg: Configuration mapping.

    Returns:
        Geometry extraction output.
    """
    if runtime_inputs is None:
        runtime_inputs = _build_geometry_runtime_inputs(cfg)
    extract_method = getattr(geometry_extractor, "extract", None)
    if not callable(extract_method):
        # geometry_extractor 协议不合法，必须 fail-fast 
        raise TypeError("geometry_extractor.extract must be callable")
    try:
        extracted = extract_method(cfg, inputs=runtime_inputs)
        if isinstance(extracted, dict):
            extracted_mapping = cast(Dict[str, Any], extracted)
            capture_source = runtime_inputs.get("attention_capture_source")
            if isinstance(capture_source, str) and capture_source:
                extracted_mapping["attention_capture_source"] = capture_source
            attention_source = runtime_inputs.get("attention_maps_source")
            if isinstance(attention_source, str) and attention_source and "attention_capture_source" not in extracted_mapping:
                extracted_mapping["attention_capture_source"] = attention_source
        return cast(Any, extracted)
    except TypeError:
        # 兼容旧实现：仅接收 cfg 参数。
        return cast(Any, extract_method(cfg))


def _get_ablation_normalized(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：读取 ablation.normalized 开关段。

    Read ablation.normalized switch settings from cfg.

    Args:
        cfg: Configuration mapping.

    Returns:
        ablation.normalized dict (empty if missing).

    Raises:
        TypeError: If cfg is invalid.
    """
    ablation = cfg.get("ablation")
    if not isinstance(ablation, dict):
        return {}
    ablation_mapping = cast(Dict[str, Any], ablation)
    normalized = ablation_mapping.get("normalized")
    if not isinstance(normalized, dict):
        return {}
    return cast(Dict[str, Any], normalized)


def _compute_ablation_digest_for_report(cfg: Dict[str, Any]) -> str:
    """
    功能：计算评测报告使用的 ablation_digest。

    Compute canonical ablation digest from normalized ablation config.

    Args:
        cfg: Runtime config mapping.

    Returns:
        Canonical digest string.
    """
    ablation_normalized = _get_ablation_normalized(cfg)
    return digests.canonical_sha256(ablation_normalized)


def _compute_ablation_digest_extended_for_report(cfg: Dict[str, Any]) -> str:
    """
    功能：计算扩展口径 ablation_digest_extended。

    Compute expanded ablation digest that binds high-impact runtime switches.

    Args:
        cfg: Runtime config mapping.

    Returns:
        Canonical digest string.
    """
    ablation_normalized = _get_ablation_normalized(cfg)
    payload: Dict[str, Any] = {
        "ablation_normalized": ablation_normalized,
        "detect_runtime_image_domain_sidecar_enabled": _is_image_domain_sidecar_enabled(cfg),
        "detect_runtime_explicit": (
            cfg.get("detect_runtime", {}).get("image_domain_sidecar_enabled")
            if isinstance(cfg.get("detect_runtime"), dict)
            else None
        ),
    }
    return digests.canonical_sha256(payload)


def _collect_attack_trace_digest(records: list[Dict[str, Any]]) -> str:
    """
    功能：聚合 detect records 中的攻击追踪摘要。

    Collect deterministic aggregate digest from per-record attack traces.

    Args:
        records: Detect records list.

    Returns:
        Aggregate digest string or "<absent>".
    """
    trace_digests: list[str] = []
    for item in records:
        direct_digest = item.get("attack_trace_digest")
        if isinstance(direct_digest, str) and direct_digest:
            trace_digests.append(direct_digest)
            continue

        attack_trace = item.get("attack_trace")
        if isinstance(attack_trace, dict):
            attack_trace_mapping = cast(Dict[str, Any], attack_trace)
            nested_digest = attack_trace_mapping.get("attack_trace_digest")
            if isinstance(nested_digest, str) and nested_digest:
                trace_digests.append(nested_digest)

    if len(trace_digests) == 0:
        return "<absent>"
    return digests.canonical_sha256(sorted(trace_digests))


def _build_ablation_absent_content_evidence(absent_reason: str) -> Dict[str, Any]:
    """
    功能：构建 ablation 禁用时的 content_evidence absent 语义。

    Build content_evidence with status="absent" for ablation-disabled modules.

    Args:
        absent_reason: Absence reason string (e.g., "content_chain_disabled_by_ablation").

    Returns:
        ContentEvidence-compatible dict with status="absent", score=None.

    Raises:
        TypeError: If absent_reason is invalid.
    """
    if not absent_reason:
        raise TypeError("absent_reason must be non-empty str")
    return {
        "status": "absent",
        "score": None,
        "audit": {
            "impl_identity": "ablation_switchboard",
            "impl_version": "v1",
            "impl_digest": digests.canonical_sha256({"impl_id": "ablation_switchboard", "impl_version": "v1"}),
            "trace_digest": digests.canonical_sha256({"absent_reason": absent_reason})
        },
        "mask_digest": None,
        "mask_stats": None,
        "plan_digest": None,
        "basis_digest": None,
        "lf_trace_digest": None,
        "hf_trace_digest": None,
        "lf_score": None,
        "hf_score": None,
        "score_parts": {
            "routing_digest": "<absent>",
            "routing_absent_reason": absent_reason,
        },
        "content_failure_reason": None  # absent 状态下无失败原因。
    }


def _build_ablation_absent_geometry_evidence(absent_reason: str) -> Dict[str, Any]:
    """
    功能：构建 ablation 禁用时的 geometry_evidence absent 语义。

    Build geometry_evidence with status="absent" for ablation-disabled modules.

    Args:
        absent_reason: Absence reason string (e.g., "geometry_chain_disabled_by_ablation").

    Returns:
        GeometryEvidence-compatible dict with status="absent", score=None.

    Raises:
        TypeError: If absent_reason is invalid.
    """
    if not absent_reason:
        raise TypeError("absent_reason must be non-empty str")
    return {
        "status": "absent",
        "geo_score": None,
        "audit": {
            "impl_identity": "ablation_switchboard",
            "impl_version": "v1",
            "impl_digest": digests.canonical_sha256({"impl_id": "ablation_switchboard", "impl_version": "v1"}),
            "trace_digest": digests.canonical_sha256({"absent_reason": absent_reason})
        },
        "sync": {
            "status": "absent",
            "reason": absent_reason
        },
        "anchor_digest": None,
        "anchor_metrics": None,
        "sync_digest": None,
        "sync_metrics": None,
        "geo_failure_reason": None  # absent 状态下无失败原因。
    }


# ---------------------------------------------------------------------------
# Cryptographic generation attestation 验证（附加函数，不修改既有 run_detect_orchestrator）
# ---------------------------------------------------------------------------

def verify_attestation(
    k_master: str,
    candidate_statement: Dict[str, Any],
    attestation_bundle: Optional[Dict[str, Any]] = None,
    content_evidence: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    hf_values: Optional[Any] = None,
    lf_latent_features: Optional[Any] = None,
    geo_score: Optional[float] = None,
    *,
    lf_weight: float = 0.5,
    hf_weight: float = 0.3,
    geo_weight: float = 0.2,
    attested_threshold: float = 0.65,
    attestation_decision_mode: str = "content_primary_geo_rescue",
    geo_rescue_band_delta_low: float = 0.05,
    geo_rescue_min_score: float = 0.3,
    lf_params: Optional[Dict[str, Any]] = None,
    detect_hf_plan_digest_used: Optional[str] = None,
    attestation_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    功能：验证图像是否来自一次真实生成事件（cryptographic generation attestation）。

    Verify whether an image originated from a specific generation event by
    reconstructing the attestation keys from a candidate statement and measuring
    multi-channel score fusion.

    Verification answers the proposition:
        "Does this image originate from this specific generation event?"
    rather than merely:
        "Does this image contain a watermark?"

    Detection flow:
        1. Reconstruct statement from candidate dict.
        2. Compute d_A = SHA256(CanonicalJSON(statement)).
        3. Derive keys: k_LF, k_HF, k_GEO, k_TR via HKDF(K_master, d_A).
        4. Compute S_LF = LF attestation score (latent posterior vs payload).
        5. Compute S_HF = HF truncation attestation score from supplied HF feature values.
        6. Compute S_GEO from provided geo_score.
        7. Compute a content-primary attestation score from LF/HF channels.
        8. Apply geometry as one-way rescue only when configured and eligible.
        9. Output attested | mismatch | absent.

    Output semantics:
        - "attested": score >= attested_threshold; image originates from this event.
        - "mismatch": statement fields mismatched or score < attested_threshold.
        - "absent": required inputs (LF latents or HF values) are not available.

    Args:
        k_master: Master key (hex str) used for key derivation.
        candidate_statement: Dict representing the candidate attestation statement.
        content_evidence: Optional content detection evidence dict (for lf_score fallback).
        cfg: Optional runtime configuration mapping used to derive canonical HF
            truncation semantics for attestation.
        hf_values: Optional HF channel feature values for HF truncation attestation scoring.
        lf_latent_features: Optional LF latent features for attestation bit correlation.
        geo_score: Optional geometry chain score (0-1), passed through.
        lf_weight: Score weight for LF channel (default 0.5).
        hf_weight: Score weight for HF channel (default 0.3).
        geo_weight: Score weight for GEO channel (default 0.2).
        attested_threshold: Fusion score threshold for "attested" verdict (default 0.65).
        attestation_decision_mode: Attestation decision mode.
        geo_rescue_band_delta_low: Lower rescue-band width below attested_threshold.
        geo_rescue_min_score: Minimum geometry score required for rescue.
        lf_params: Optional LF parameter dict for attestation score computation.
        detect_hf_plan_digest_used: Detect-side canonical plan digest used by
            the HF challenge path.

    Returns:
        Dict with:
        - "verdict": "attested" | "mismatch" | "absent".
        - "fusion_score": float or None.
        - "content_attestation_score": float or None.
                - "final_event_attested_decision": dict with the event-level verdict,
                    primary event_attestation_score, and a legacy alias mirror for
                    old statistics readers.
        - "channel_scores": dict with lf, hf, geo sub-scores.
        - "attestation_digest": d_A used for key derivation.
        - "statement": echoed candidate statement dict.
        - "attestation_trace_digest": reproducible audit digest.
        - "mismatch_reasons": list of strings explaining any mismatch.

    Raises:
        TypeError: If k_master or candidate_statement has wrong type.
        ValueError: If k_master is empty.
    """
    from main.watermarking.provenance.attestation_statement import (
        statement_from_dict,
        compute_attestation_digest,
        verify_statement_fields,
        verify_signed_attestation_bundle,
    )
    from main.watermarking.provenance.key_derivation import (
        derive_attestation_keys,
    )
    from main.watermarking.content_chain.low_freq_coder import (
        compute_lf_attestation_score,
    )
    from main.watermarking.content_chain.high_freq_embedder import (
        compute_hf_attestation_score,
    )

    if not k_master:
        raise ValueError("k_master must be non-empty str")

    lf_params = dict(lf_params) if isinstance(lf_params, dict) else {}
    if isinstance(content_evidence, dict):
        content_evidence_payload = cast(Dict[str, Any], content_evidence)
        for field_name, field_value in [
            (eval_metrics.CONTENT_CHAIN_SCORE_NAME, content_evidence_payload.get(eval_metrics.CONTENT_CHAIN_SCORE_NAME)),
            (eval_metrics.LF_CHANNEL_SCORE_NAME, content_evidence_payload.get(eval_metrics.LF_CHANNEL_SCORE_NAME)),
            (eval_metrics.LF_CORRELATION_SCORE_NAME, content_evidence_payload.get(eval_metrics.LF_CORRELATION_SCORE_NAME)),
        ]:
            numeric_value = _coerce_optional_finite_score(field_value)
            if numeric_value is not None:
                lf_params[field_name] = numeric_value

    mismatch_reasons: list[str] = []
    bundle_verification: Optional[Dict[str, Any]] = None
    authenticity_status = "statement_only"
    trace_commit: Optional[str] = None

    if attestation_bundle is not None:
        try:
            bundle_verification = verify_signed_attestation_bundle(attestation_bundle, k_master)
        except Exception as exc:
            mismatch_reasons.append(f"bundle_verification_failed: {exc}")
            return {
                "verdict": "mismatch",
                "fusion_score": None,
                "channel_scores": {"lf": None, "hf": None, "geo": None},
                "attestation_digest": None,
                "statement": candidate_statement,
                "attestation_trace_digest": None,
                "mismatch_reasons": mismatch_reasons,
                "bundle_verification": {"status": "mismatch", "mismatch_reasons": mismatch_reasons},
                "authenticity_result": _build_detect_authenticity_result(
                    authenticity_status="mismatch",
                    statement_status="unknown",
                    attestation_source=attestation_source,
                    bundle_verification={"status": "mismatch", "mismatch_reasons": mismatch_reasons},
                    explicit_bundle_status="mismatch",
                ),
            }
        if bundle_verification.get("status") != "ok":
            mismatch_reasons.extend(list(bundle_verification.get("mismatch_reasons") or []))
            return {
                "verdict": "mismatch",
                "fusion_score": None,
                "channel_scores": {"lf": None, "hf": None, "geo": None},
                "attestation_digest": bundle_verification.get("attestation_digest"),
                "statement": candidate_statement,
                "attestation_trace_digest": None,
                "mismatch_reasons": mismatch_reasons,
                "bundle_verification": bundle_verification,
                "authenticity_result": _build_detect_authenticity_result(
                    authenticity_status="mismatch",
                    statement_status="parsed",
                    attestation_source=attestation_source,
                    bundle_verification=bundle_verification,
                    explicit_bundle_status=bundle_verification.get("status"),
                ),
            }
        authenticity_status = "authentic"
        bundle_trace_commit = attestation_bundle.get("trace_commit")
        if isinstance(bundle_trace_commit, str) and bundle_trace_commit:
            trace_commit = bundle_trace_commit

    # (1) 验证并重建 statement。
    if not verify_statement_fields(candidate_statement):
        mismatch_reasons.append("statement_fields_invalid")
        return {
            "verdict": "mismatch",
            "fusion_score": None,
            "channel_scores": {"lf": None, "hf": None, "geo": None},
            "attestation_digest": None,
            "statement": candidate_statement,
            "attestation_trace_digest": None,
            "mismatch_reasons": mismatch_reasons,
        }

    try:
        statement = statement_from_dict(candidate_statement)
    except (TypeError, ValueError) as exc:
        mismatch_reasons.append(f"statement_parse_failed: {exc}")
        return {
            "verdict": "mismatch",
            "fusion_score": None,
            "channel_scores": {"lf": None, "hf": None, "geo": None},
            "attestation_digest": None,
            "statement": candidate_statement,
            "attestation_trace_digest": None,
            "mismatch_reasons": mismatch_reasons,
            "authenticity_result": _build_detect_authenticity_result(
                authenticity_status="mismatch",
                statement_status="parse_failed",
                attestation_source=attestation_source,
                bundle_verification=bundle_verification,
            ),
        }

    # (2) 计算 attestation digest d_A。
    d_a = compute_attestation_digest(statement)

    # (3) 派生四类子密钥。
    try:
        attestation_node = cfg.get("attestation") if isinstance(cfg, dict) else None
        attestation_cfg = cast(Dict[str, Any], attestation_node) if isinstance(attestation_node, dict) else {}
        event_binding_mode = "trajectory_bound" if bool(attestation_cfg.get("use_trajectory_mix", True)) else "statement_only"
        attest_keys = derive_attestation_keys(
            k_master,
            d_a,
            trajectory_commit=trace_commit,
            event_binding_mode=event_binding_mode,
        )
    except (TypeError, ValueError) as exc:
        mismatch_reasons.append(f"key_derivation_failed: {exc}")
        return {
            "verdict": "mismatch",
            "fusion_score": None,
            "channel_scores": {"lf": None, "hf": None, "geo": None},
            "attestation_digest": d_a,
            "statement": candidate_statement,
            "attestation_trace_digest": None,
            "mismatch_reasons": mismatch_reasons,
            "authenticity_result": _build_detect_authenticity_result(
                authenticity_status="mismatch",
                statement_status="parsed",
                attestation_source=attestation_source,
                bundle_verification=bundle_verification,
                attestation_digest=d_a,
                trace_commit=trace_commit,
            ),
        }

    # (4) 计算各通道 attestation 得分。
    s_lf: Optional[float] = None
    s_hf: Optional[float] = None
    s_hf_raw: Optional[float] = None
    s_geo: Optional[float] = None
    hf_attestation_trace: Optional[Dict[str, Any]] = None
    lf_attestation_trace: Optional[Dict[str, Any]] = None
    lf_alignment_table: Optional[Dict[str, Any]] = None
    lf_retain_breakdown: Optional[Dict[str, Any]] = None

    # LF 通道：基于 latent 后验与 attestation payload 的符号一致率。
    if lf_latent_features is not None:
        try:
            lf_result = compute_lf_attestation_score(
                latent_features=lf_latent_features,
                k_lf=attest_keys.k_lf,
                attestation_digest=attest_keys.event_binding_digest,
                lf_params=lf_params,
            )
            if lf_result.get("status") == "ok":
                if isinstance(lf_params, dict):
                    for field_name in [
                        "selected_step_post_coeffs",
                        "embed_edit_timestep_coeffs",
                        "embed_terminal_step_coeffs",
                        "detect_exact_timestep_coeffs",
                        "formal_exact_evidence_source",
                        "formal_exact_object_binding_status",
                        "formal_exact_image_path_source",
                        "embed_edit_timestep_step_index",
                        "embed_terminal_step_index",
                        "embed_seed",
                        "detect_seed",
                        "same_seed_as_embed_available",
                        "same_seed_as_embed_value",
                        "detect_protocol_classification",
                        "image_conditioned_reconstruction_available",
                        "image_conditioned_reconstruction_status",
                        "same_seed_control_status",
                        "same_seed_control_reason",
                        "same_seed_control_reused_formal_detect",
                        "same_seed_control_trace_digest",
                        "same_seed_control_trajectory_digest",
                        "detect_exact_timestep_coeffs_same_seed_control",
                        "lf_exact_repair_enabled",
                        "lf_exact_repair_mode",
                        "lf_exact_repair_applied",
                        "lf_exact_repair_summary",
                        eval_metrics.CONTENT_CHAIN_SCORE_NAME,
                        eval_metrics.LF_CHANNEL_SCORE_NAME,
                        eval_metrics.LF_CORRELATION_SCORE_NAME,
                    ]:
                        if field_name in lf_params and field_name not in lf_result:
                            lf_result[field_name] = lf_params.get(field_name)
                s_lf = lf_result.get("lf_attestation_score")
                lf_attestation_trace = _build_lf_attestation_trace_artifact(
                    lf_result,
                    attestation_digest=d_a,
                    event_binding_digest=attest_keys.event_binding_digest,
                    trace_commit=trace_commit,
                )
                lf_alignment_table = _build_lf_alignment_table_artifact(
                    lf_result,
                    attestation_digest=d_a,
                    event_binding_digest=attest_keys.event_binding_digest,
                    trace_commit=trace_commit,
                )
                lf_retain_breakdown = _build_lf_retain_breakdown_artifact(
                    lf_result,
                    attestation_digest=d_a,
                    event_binding_digest=attest_keys.event_binding_digest,
                    trace_commit=trace_commit,
                )
        except Exception:
            mismatch_reasons.append("lf_attestation_score_failed")
    elif content_evidence is not None:
        # 回退：使用现有内容检测分数作为 LF 代理。
        raw_lf = content_evidence.get("lf_score")
        if isinstance(raw_lf, (int, float)):
            s_lf = float(raw_lf)

    # HF 通道：基于 truncation-constrained HF 能量证据。
    if hf_values is not None:
        try:
            hf_result = compute_hf_attestation_score(
                hf_values=hf_values,
                k_hf=attest_keys.k_hf,
                attestation_event_digest=attest_keys.event_binding_digest,
                plan_digest=statement.plan_digest,
                cfg=cfg,
            )
            if isinstance(hf_result.get("hf_attestation_trace"), dict):
                hf_attestation_trace = cast(Dict[str, Any], hf_result.get("hf_attestation_trace"))
            if hf_result.get("status") == "ok":
                raw_hf_score = hf_result.get("hf_attestation_score")
                decision_hf_score = hf_result.get("hf_attestation_decision_score")
                if isinstance(raw_hf_score, (int, float)):
                    s_hf_raw = float(raw_hf_score)
                if isinstance(decision_hf_score, (int, float)):
                    s_hf = float(decision_hf_score)
                else:
                    s_hf = s_hf_raw
        except Exception:
            mismatch_reasons.append("hf_attestation_score_failed")

    # GEO 通道：直接使用调用方提供的几何链得分。
    if geo_score is not None:
        s_geo = float(max(0.0, min(1.0, geo_score)))

    if attestation_decision_mode not in {"weighted_sum", "content_primary_geo_rescue"}:
        raise ValueError(
            "attestation_decision_mode must be one of {'weighted_sum', 'content_primary_geo_rescue'}"
        )

    # (5) 检查是否缺少必要输入。
    if s_lf is None and s_hf is None and s_geo is None:
        return {
            "verdict": "absent",
            "fusion_score": None,
            "content_attestation_score": None,
            "channel_scores": {"lf": None, "hf": None, "geo": None},
            "attestation_digest": d_a,
            "event_binding_digest": attest_keys.event_binding_digest,
            "statement": candidate_statement,
            "attestation_trace_digest": None,
            "mismatch_reasons": ["all_channel_scores_absent"],
            "attestation_decision_mode": attestation_decision_mode,
            "geo_rescue_eligible": False,
            "geo_rescue_applied": False,
            "geo_not_used_reason": "all_channel_scores_absent",
            "authenticity_result": _build_detect_authenticity_result(
                authenticity_status=authenticity_status,
                statement_status="parsed",
                attestation_source=attestation_source,
                bundle_verification=bundle_verification,
                attestation_digest=d_a,
                event_binding_digest=attest_keys.event_binding_digest,
                trace_commit=trace_commit,
            ),
            "image_evidence_result": {
                "status": "absent",
                "channel_scores": {"lf": None, "hf": None, "geo": None},
                "content_attestation_score": None,
                "decision_mode": attestation_decision_mode,
                "geo_rescue_eligible": False,
                "geo_rescue_applied": False,
                "geo_not_used_reason": "all_channel_scores_absent",
            },
            "final_event_attested_decision": {
                "status": "absent",
                "is_event_attested": False,
            },
        }

    # (6) 先计算内容侧 attestation 主分（LF/HF 归一化）。
    content_weights: Dict[str, float] = {}
    if s_lf is not None:
        content_weights["lf"] = lf_weight
    if s_hf is not None:
        content_weights["hf"] = hf_weight

    content_total_weight = sum(content_weights.values())
    content_attestation_score: Optional[float]
    if content_total_weight < 1e-9:
        content_attestation_score = None
    else:
        content_channel_vals: Dict[str, float] = {"lf": s_lf or 0.0, "hf": s_hf or 0.0}
        content_attestation_score = sum(
            content_channel_vals[ch] * w / content_total_weight
            for ch, w in content_weights.items()
        )
        content_attestation_score = float(max(0.0, min(1.0, content_attestation_score)))

    geo_rescue_eligible = False
    geo_rescue_applied = False
    geo_not_used_reason: Optional[str] = None

    # (7) 按配置模式判定图像侧 attestation 结果。
    if attestation_decision_mode == "weighted_sum":
        effective_weights: Dict[str, float] = {}
        if s_lf is not None:
            effective_weights["lf"] = lf_weight
        if s_hf is not None:
            effective_weights["hf"] = hf_weight
        if s_geo is not None:
            effective_weights["geo"] = geo_weight

        total_weight = sum(effective_weights.values())
        if total_weight < 1e-9:
            fusion_score = None
            image_evidence_status = "absent"
        else:
            channel_vals: Dict[str, float] = {"lf": s_lf or 0.0, "hf": s_hf or 0.0, "geo": s_geo or 0.0}
            fusion_score = sum(
                channel_vals[ch] * w / total_weight
                for ch, w in effective_weights.items()
            )
            fusion_score = float(max(0.0, min(1.0, fusion_score)))
            image_evidence_status = "ok"

        if s_geo is None:
            geo_not_used_reason = "geometry_absent"
        else:
            geo_not_used_reason = "decision_mode_weighted_sum"

        if fusion_score is None:
            verdict = "absent"
        elif fusion_score >= attested_threshold:
            if authenticity_status == "authentic":
                verdict = "attested"
            else:
                verdict = "absent"
                mismatch_reasons.append("bundle_authenticity_absent")
        else:
            verdict = "mismatch"
            mismatch_reasons.append(f"fusion_score_below_threshold: {fusion_score:.4f} < {attested_threshold}")
    else:
        fusion_score = content_attestation_score
        if content_attestation_score is None:
            verdict = "absent"
            image_evidence_status = "absent"
            geo_not_used_reason = "content_attestation_evidence_absent"
            mismatch_reasons.append("content_attestation_evidence_absent")
        else:
            image_evidence_status = "ok"
            if authenticity_status != "authentic":
                verdict = "absent"
                geo_not_used_reason = "bundle_authenticity_absent"
                mismatch_reasons.append("bundle_authenticity_absent")
            elif content_attestation_score >= attested_threshold:
                verdict = "attested"
                geo_not_used_reason = "content_attestation_threshold_met"
            else:
                rescue_lower_bound = attested_threshold - geo_rescue_band_delta_low
                geo_rescue_eligible = bool(
                    s_geo is not None and rescue_lower_bound <= content_attestation_score < attested_threshold
                )
                if geo_rescue_eligible and s_geo is not None and s_geo >= geo_rescue_min_score:
                    verdict = "attested"
                    geo_rescue_applied = True
                else:
                    verdict = "mismatch"
                    mismatch_reasons.append(
                        f"content_attestation_score_below_threshold: {content_attestation_score:.4f} < {attested_threshold}"
                    )
                    if s_geo is None:
                        geo_not_used_reason = "geometry_absent"
                    elif content_attestation_score < rescue_lower_bound:
                        geo_not_used_reason = "content_score_outside_rescue_band"
                    elif s_geo < geo_rescue_min_score:
                        geo_not_used_reason = "geometry_score_below_rescue_min"
                    else:
                        geo_not_used_reason = "geometry_rescue_not_applied"

    bundle_status = _resolve_detect_attestation_bundle_status(
        authenticity_status=authenticity_status,
        attestation_source=attestation_source,
        bundle_verification=bundle_verification,
    )
    authenticity_result = _build_detect_authenticity_result(
        authenticity_status=authenticity_status,
        statement_status="parsed",
        attestation_source=attestation_source,
        bundle_verification=bundle_verification,
        explicit_bundle_status=bundle_status,
        attestation_digest=d_a,
        event_binding_digest=attest_keys.event_binding_digest,
        trace_commit=trace_commit,
    )
    image_evidence_result = {
        "status": image_evidence_status,
        "channel_scores": {"lf": s_lf, "hf": s_hf, "geo": s_geo},
        "channel_scores_raw": {"lf": s_lf, "hf": s_hf_raw, "geo": s_geo},
        "lf_attestation_score": s_lf,
        eval_metrics.LF_CHANNEL_SCORE_NAME: lf_params.get(eval_metrics.LF_CHANNEL_SCORE_NAME),
        eval_metrics.LF_CORRELATION_SCORE_NAME: lf_params.get(eval_metrics.LF_CORRELATION_SCORE_NAME),
        "fusion_score": fusion_score,
        "content_attestation_score": content_attestation_score,
        "content_attestation_score_name": "content_attestation_score",
        "content_attestation_score_semantics": "lf_and_hf_decision_fusion_before_geo_rescue",
        "hf_attestation_score": s_hf_raw,
        "hf_attestation_decision_score": s_hf,
        "threshold": attested_threshold,
        "decision_mode": attestation_decision_mode,
        "geo_rescue_eligible": geo_rescue_eligible,
        "geo_rescue_applied": geo_rescue_applied,
        "geo_not_used_reason": geo_not_used_reason,
    }
    hf_trace_artifact = _merge_hf_attestation_trace(
        hf_attestation_trace,
        _resolve_hf_detect_summary(content_evidence),
        attestation_digest=d_a,
        event_binding_digest=attest_keys.event_binding_digest,
        trace_commit=trace_commit,
        hf_score=s_hf_raw,
        hf_decision_score=s_hf,
        detect_hf_plan_digest_used=detect_hf_plan_digest_used,
    )
    event_attestation_score = (
        content_attestation_score
        if verdict == "attested"
        else (0.0 if content_attestation_score is not None else None)
    )
    final_event_attested_decision = {
        "status": verdict,
        "is_event_attested": bool(verdict == "attested"),
        "authenticity_status": authenticity_status,
        "image_evidence_status": image_evidence_status,
        "event_attestation_score": event_attestation_score,
        "event_attestation_score_name": "event_attestation_score",
        "event_attestation_score_semantics": "content_attestation_score_if_event_attested_else_zero_when_content_score_present",
        eval_metrics.LF_CHANNEL_SCORE_NAME: lf_params.get(eval_metrics.LF_CHANNEL_SCORE_NAME),
        eval_metrics.LF_CORRELATION_SCORE_NAME: lf_params.get(eval_metrics.LF_CORRELATION_SCORE_NAME),
        "lf_attestation_score": s_lf,
    }

    # (7) 构造审计摘要（可复算）。
    trace_payload: Dict[str, Any] = {
        "attestation_digest": d_a,
        "event_binding_digest": attest_keys.event_binding_digest,
        "lf_weight": lf_weight,
        "hf_weight": hf_weight,
        "geo_weight": geo_weight,
        "attested_threshold": attested_threshold,
        "attestation_decision_mode": attestation_decision_mode,
        "geo_rescue_band_delta_low": geo_rescue_band_delta_low,
        "geo_rescue_min_score": geo_rescue_min_score,
        "s_lf": round(s_lf, 6) if s_lf is not None else None,
        "s_hf": round(s_hf, 6) if s_hf is not None else None,
        "s_geo": round(s_geo, 6) if s_geo is not None else None,
        "content_attestation_score": round(content_attestation_score, 6) if content_attestation_score is not None else None,
        "fusion_score": round(fusion_score, 6) if fusion_score is not None else None,
        "verdict": verdict,
        "geo_rescue_eligible": geo_rescue_eligible,
        "geo_rescue_applied": geo_rescue_applied,
        "geo_not_used_reason": geo_not_used_reason,
        "bundle_status": bundle_status,
        "authenticity_status": authenticity_status,
        "image_evidence_status": image_evidence_status,
    }
    from main.core import digests as _digests
    attestation_trace_digest = _digests.canonical_sha256(trace_payload)

    return {
        "verdict": verdict,
        "fusion_score": fusion_score,
        "content_attestation_score": content_attestation_score,
        eval_metrics.CONTENT_CHAIN_SCORE_NAME: lf_params.get(eval_metrics.CONTENT_CHAIN_SCORE_NAME),
        eval_metrics.LF_CHANNEL_SCORE_NAME: lf_params.get(eval_metrics.LF_CHANNEL_SCORE_NAME),
        eval_metrics.LF_CORRELATION_SCORE_NAME: lf_params.get(eval_metrics.LF_CORRELATION_SCORE_NAME),
        "lf_attestation_score": s_lf,
        "channel_scores": {"lf": s_lf, "hf": s_hf, "geo": s_geo},
        "channel_scores_raw": {"lf": s_lf, "hf": s_hf_raw, "geo": s_geo},
        "hf_attestation_score": s_hf_raw,
        "hf_attestation_decision_score": s_hf,
        "attestation_digest": d_a,
        "event_binding_digest": attest_keys.event_binding_digest,
        "statement": candidate_statement,
        "attestation_trace_digest": attestation_trace_digest,
        "mismatch_reasons": mismatch_reasons,
        "attestation_decision_mode": attestation_decision_mode,
        "geo_rescue_eligible": geo_rescue_eligible,
        "geo_rescue_applied": geo_rescue_applied,
        "geo_not_used_reason": geo_not_used_reason,
        "bundle_verification": bundle_verification,
        "authenticity_result": authenticity_result,
        "image_evidence_result": image_evidence_result,
        "final_event_attested_decision": final_event_attested_decision,
        "_lf_attestation_trace_artifact": lf_attestation_trace,
        "_lf_alignment_table_artifact": lf_alignment_table,
        "_lf_retain_breakdown_artifact": lf_retain_breakdown,
        "_hf_attestation_trace_artifact": hf_trace_artifact,
    }
