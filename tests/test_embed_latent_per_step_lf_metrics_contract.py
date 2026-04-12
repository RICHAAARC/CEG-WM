"""
文件目的：验证 latent per-step embed 主路径会生成并保留 LF metrics 摘要。
Module type: General module
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, cast

from main.cli.run_common import bind_impl_identity_fields
from main.core import config_loader, records_io, schema
from main.core.contracts import bind_contract_to_record
from main.diffusion.sd3.provenance import (
    build_pipeline_provenance,
    compute_pipeline_provenance_canon_sha256,
)
from main.policy.runtime_whitelist import bind_semantics_to_record, bind_whitelist_to_record
from main.registries import pipeline_registry
from main.registries.runtime_resolver import BuiltImplSet, ImplIdentity
from main.watermarking.embed.orchestrator import run_embed_orchestrator
from paper_workflow.scripts.pw_common import build_payload_reference_sidecar_payload


class StubComponent:
    """
    功能：提供 impl 绑定测试所需的最小版本与 digest 字段。

    Minimal runtime component used for impl identity binding tests.

    Args:
        impl_id: Implementation identifier.

    Returns:
        None.
    """

    def __init__(self, impl_id: str) -> None:
        if not isinstance(impl_id, str) or not impl_id:
            raise TypeError("impl_id must be non-empty str")
        self.impl_id = impl_id
        self.impl_version = "v1"
        self.impl_digest = _hex_anchor(f"{impl_id}:digest")


class StubSyncModule(StubComponent):
    """
    功能：提供 orchestrator 所需的最小 sync 行为。

    Minimal sync module used by embed orchestrator contract tests.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(self) -> None:
        super().__init__("geometry_latent_sync_sd3")

    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：返回最小 sync 结果。

        Return the minimal sync result required by the orchestrator.

        Args:
            cfg: Runtime configuration mapping.

        Returns:
            Minimal sync payload.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        return {
            "status": "absent",
            "sync_absent_reason": "sync_not_required_for_contract_test",
        }


def _hex_anchor(seed_text: str) -> str:
    """
    功能：生成稳定的 64 位十六进制摘要字符串。

    Build a stable SHA256 hex anchor for test payloads.

    Args:
        seed_text: Seed text used for hashing.

    Returns:
        Lowercase SHA256 hex digest.
    """
    if not isinstance(seed_text, str) or not seed_text:
        raise TypeError("seed_text must be non-empty str")
    return hashlib.sha256(seed_text.encode("utf-8")).hexdigest()


def _build_cfg() -> Dict[str, Any]:
    """
    功能：构造 latent per-step LF summary 合同测试配置。

    Build the minimal runtime config required by the latent per-step LF summary
    contract tests.

    Args:
        None.

    Returns:
        Runtime configuration mapping.
    """
    return {
        "policy_path": "content_np_geo_rescue",
        "target_fpr": 1e-6,
        "paper_faithfulness": {"enabled": True},
        "attestation": {"enabled": False},
        "watermark": {
            "key_id": "test-key",
            "pattern_id": "test-pattern",
            "hf": {"enabled": False},
            "lf": {
                "enabled": True,
                "message_length": 32,
                "ecc_sparsity": 3,
            },
        },
    }


def _build_impl_set() -> BuiltImplSet:
    """
    功能：构造 orchestrator 合同测试所需的最小 impl_set。

    Build the minimal implementation set required by the orchestrator contract
    tests.

    Args:
        None.

    Returns:
        Built implementation set.
    """
    return BuiltImplSet(
        content_extractor=StubComponent("unified_content_extractor"),
        geometry_extractor=StubComponent("attention_anchor_extractor"),
        fusion_rule=StubComponent("fusion_neyman_pearson"),
        subspace_planner=StubComponent("subspace_planner"),
        sync_module=StubSyncModule(),
        hf_embedder=StubComponent("high_freq_truncation_codec"),
        lf_coder=StubComponent("low_freq_template_codec"),
    )


def _build_content_result() -> Dict[str, Any]:
    """
    功能：构造最小 content_evidence 负载。

    Build the minimal content evidence payload used by the tests.

    Args:
        None.

    Returns:
        Content evidence mapping.
    """
    return {
        "status": "ok",
        "mask_digest": _hex_anchor("mask"),
        "score_parts": {},
        "audit": {},
    }


def _build_subspace_result() -> Dict[str, Any]:
    """
    功能：构造最小 subspace planner 结果。

    Build the minimal planner result required by plan digest binding.

    Args:
        None.

    Returns:
        Subspace planner result mapping.
    """
    return {
        "plan": {
            "rank": 4,
            "planner_impl_identity": {"impl_id": "subspace_planner", "impl_version": "v1"},
            "planner_params": {"rank": 4, "target": "latent_step_contract"},
            "verifiable_input_domain_spec": {"planner_input_digest": _hex_anchor("planner_input")},
        },
        "basis_digest": _hex_anchor("basis"),
        "plan_stats": {"rank": 4},
    }


def _build_injection_evidence(basis_digest: str) -> Dict[str, Any]:
    """
    功能：构造带 LF closed-loop 摘要的最小 injection_evidence。

    Build the minimal injection evidence payload that carries authoritative LF
    closed-loop evidence.

    Args:
        basis_digest: Canonical basis digest bound to the runtime path.

    Returns:
        Injection evidence mapping.
    """
    if not isinstance(basis_digest, str) or not basis_digest:
        raise TypeError("basis_digest must be non-empty str")
    return {
        "status": "ok",
        "injection_trace_digest": _hex_anchor("injection_trace"),
        "injection_params_digest": _hex_anchor("injection_params"),
        "lf_impl_binding": {
            "impl_selected": "low_freq_template_codec",
            "evidence_level": "primary",
        },
        "hf_impl_binding": {
            "impl_selected": "high_freq_truncation_codec",
            "evidence_level": "channel_absent",
        },
        "injection_metrics": {
            "lf_closed_loop_summary": {
                "coeffs_count": 96,
                "codeword_source": "plan_digest_fallback",
                "basis_digest": basis_digest,
                "attestation_event_digest": None,
            },
            "lf_closed_loop_digest": _hex_anchor("lf_closed_loop"),
            "lf_closed_loop_step_index": 6,
            "lf_closed_loop_candidate_count": 1,
            "lf_codeword_source": "plan_digest_fallback",
            "lf_basis_digest": basis_digest,
            "event_binding_mode": "trajectory_bound",
        },
    }


def _build_record_under_test() -> tuple[Dict[str, Any], Dict[str, Any], BuiltImplSet]:
    """
    功能：运行 orchestrator 并返回待测记录。

    Run the embed orchestrator and return the contract-test record.

    Args:
        None.

    Returns:
        Tuple of (record, cfg, impl_set).
    """
    cfg = _build_cfg()
    impl_set = _build_impl_set()
    subspace_result = _build_subspace_result()
    basis_digest = cast(str, subspace_result["basis_digest"])
    record = run_embed_orchestrator(
        cfg,
        impl_set,
        _hex_anchor("cfg_digest"),
        injection_evidence=_build_injection_evidence(basis_digest),
        content_result_override=_build_content_result(),
        subspace_result_override=subspace_result,
    )
    return record, cfg, impl_set


def _bind_record_for_write(
    record: Dict[str, Any],
    cfg: Dict[str, Any],
    impl_set: BuiltImplSet,
) -> tuple[Any, Any, Any, Any, Any]:
    """
    功能：为 records 写盘补齐合同与事实源字段。

    Bind contract, whitelist, semantics, and impl identity fields before schema
    validation and controlled record writing.

    Args:
        record: Mutable record mapping.
        cfg: Runtime configuration mapping.
        impl_set: Built implementation set.

    Returns:
        Tuple of loaded fact-source objects and interpretation.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be dict")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    contracts, interpretation = config_loader.load_frozen_contracts_interpretation()
    whitelist = config_loader.load_runtime_whitelist()
    semantics = config_loader.load_policy_path_semantics()
    injection_scope_manifest = config_loader.load_injection_scope_manifest()

    record["cfg_digest"] = _hex_anchor("cfg_digest")
    record["policy_path"] = str(cfg["policy_path"])
    pipeline_provenance = build_pipeline_provenance(
        {
            "model_id": "sd3-test",
            "resolved_model_id": "sd3-test",
            "model_source": "hf",
            "resolved_model_source": "hf",
            "hf_revision": "main",
            "resolved_revision": "main",
            "model_source_resolution": "fallback_to_requested_model_source",
            "local_snapshot_status": "absent",
            "local_files_only": False,
        },
        pipeline_registry.SD3_DIFFUSERS_SHELL_ID,
        {},
    )
    record["pipeline_impl_id"] = pipeline_registry.SD3_DIFFUSERS_SHELL_ID
    record["pipeline_provenance"] = pipeline_provenance
    record["pipeline_provenance_canon_sha256"] = compute_pipeline_provenance_canon_sha256(pipeline_provenance)

    identity = ImplIdentity(
        content_extractor_id="unified_content_extractor",
        geometry_extractor_id="attention_anchor_extractor",
        fusion_rule_id="fusion_neyman_pearson",
        subspace_planner_id="subspace_planner",
        sync_module_id="geometry_latent_sync_sd3",
    )
    bind_impl_identity_fields(record, identity, impl_set, contracts)
    bind_contract_to_record(record, contracts)
    bind_whitelist_to_record(record, whitelist)
    bind_semantics_to_record(record, semantics)
    schema.ensure_required_fields(record, cfg, interpretation)
    schema.validate_record(record, interpretation=interpretation)
    return contracts, interpretation, whitelist, semantics, injection_scope_manifest


def test_run_embed_orchestrator_binds_lf_metrics_for_latent_per_step_path() -> None:
    """
    功能：latent per-step 主路径必须把 LF 摘要绑定到 embed_trace 与 score_parts。

    Verify that the latent per-step embed path binds LF summary fields into
    both embed_trace and content_evidence.score_parts.

    Args:
        None.

    Returns:
        None.
    """
    record, _cfg, _impl_set = _build_record_under_test()

    embed_trace = cast(Dict[str, Any], record["embed_trace"])
    lf_trace_summary = cast(Dict[str, Any], embed_trace["lf_trace_summary"])
    content_evidence = cast(Dict[str, Any], record["content_evidence"])
    score_parts = cast(Dict[str, Any], content_evidence["score_parts"])
    basis_digest = cast(str, record["basis_digest"])

    assert record["embed_mode"] == "latent_step_injection_v1"
    assert lf_trace_summary["lf_status"] == "ok"
    assert lf_trace_summary["lf_mode"] == "latent_step_injection_v1"
    assert lf_trace_summary["message_length"] == 32
    assert lf_trace_summary["ecc_sparsity"] == 3
    assert lf_trace_summary["message_source"] == "plan_digest"
    assert lf_trace_summary["plan_digest"] == record["plan_digest"]
    assert lf_trace_summary["basis_digest"] == basis_digest
    assert isinstance(lf_trace_summary["parity_check_digest"], str)
    assert len(lf_trace_summary["parity_check_digest"]) == 64
    assert score_parts["lf_metrics"] == lf_trace_summary

    sidecar_payload = build_payload_reference_sidecar_payload(
        family_id="family_contract",
        stage_name="pw01",
        event_id="evt_contract",
        event_index=0,
        sample_role="positive_source",
        prompt_sha256=_hex_anchor("prompt"),
        seed=7,
        embed_record=record,
    )
    assert sidecar_payload["message_length"] == 32
    assert sidecar_payload["plan_digest"] == record["plan_digest"]
    assert sidecar_payload["basis_digest"] == basis_digest
    assert sidecar_payload["message_source"] == "plan_digest"


def test_latent_per_step_lf_metrics_survive_records_write(tmp_path: Path) -> None:
    """
    功能：latent per-step LF 摘要在 schema 校验与 records 写盘后仍必须保留。

    Verify that latent per-step LF summary fields survive schema validation and
    the final controlled records write.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    record, cfg, impl_set = _build_record_under_test()
    contracts, _interpretation, whitelist, semantics, injection_scope_manifest = _bind_record_for_write(record, cfg, impl_set)

    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    record_path = records_dir / "embed_record.json"

    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        records_io.write_json(str(record_path), record)

    persisted_record = json.loads(record_path.read_text(encoding="utf-8"))
    embed_trace = cast(Dict[str, Any], persisted_record["embed_trace"])
    lf_trace_summary = cast(Dict[str, Any], embed_trace["lf_trace_summary"])
    content_evidence = cast(Dict[str, Any], persisted_record["content_evidence"])
    score_parts = cast(Dict[str, Any], content_evidence["score_parts"])

    assert persisted_record["schema_version"] == schema.RECORD_SCHEMA_VERSION
    assert lf_trace_summary["message_length"] == 32
    assert lf_trace_summary["ecc_sparsity"] == 3
    assert score_parts["lf_metrics"] == lf_trace_summary

    sidecar_payload = build_payload_reference_sidecar_payload(
        family_id="family_contract",
        stage_name="pw01",
        event_id="evt_contract",
        event_index=0,
        sample_role="positive_source",
        prompt_sha256=_hex_anchor("prompt"),
        seed=7,
        embed_record=persisted_record,
    )
    assert sidecar_payload["message_length"] == 32
    assert sidecar_payload["plan_digest"] == persisted_record["plan_digest"]
    assert sidecar_payload["basis_digest"] == persisted_record["basis_digest"]