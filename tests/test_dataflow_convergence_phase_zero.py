"""
File purpose: Dataflow Convergence 闃舵鍥炲綊娴嬭瘯銆?
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from main.core import config_loader, digests, schema
from main.core.contracts import get_contract_interpretation, load_frozen_contracts
from main.core.injection_scope import load_injection_scope_manifest
from main.core.records_io import bound_fact_sources
from main.policy.runtime_whitelist import load_policy_path_semantics, load_runtime_whitelist
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.subspace.planner_interface import SubspacePlanEvidence
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.watermarking.embed.orchestrator import run_embed_orchestrator
from main.watermarking.fusion.interfaces import FusionDecision


def _hex_anchor(tag: str) -> str:
    """
    鍔熻兘锛氱敓鎴愮ǔ瀹氬崄鍏繘鍒舵憳瑕併€?

    Build deterministic SHA256 hex digest anchor.

    Args:
        tag: Input tag string.

    Returns:
        SHA256 hex digest string.
    """
    return digests.canonical_sha256({"tag": tag})


def _write_png(path: Path) -> None:
    """
    鍔熻兘锛氬啓鍏ユ渶灏忔湁鏁?PNG 鏂囦欢銆?

    Write a minimal valid PNG bytes payload.

    Args:
        path: Destination path.

    Returns:
        None.
    """
    png_bytes = bytes(
        [
            137, 80, 78, 71, 13, 10, 26, 10,
            0, 0, 0, 13, 73, 72, 68, 82,
            0, 0, 0, 1, 0, 0, 0, 1,
            8, 2, 0, 0, 0, 144, 119, 83,
            222, 0, 0, 0, 12, 73, 68, 65,
            84, 8, 153, 99, 248, 15, 4, 0,
            9, 251, 3, 253, 160, 92, 203, 75,
            0, 0, 0, 0, 73, 69, 78, 68,
            174, 66, 96, 130,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


class _ContentExtractorStub:
    def extract(self, cfg: Dict[str, Any], cfg_digest: Optional[str] = None, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _ = cfg_digest
        _ = inputs
        return {
            "status": "ok",
            "mask_digest": _hex_anchor("mask"),
            "score": 0.5,
            "lf_score": 0.5,
        }


class _GeometryExtractorStub:
    def extract(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        _ = cfg
        return {"status": "absent", "geo_score": None}


class _FusionRuleStub:
    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> FusionDecision:
        _ = cfg
        _ = content_evidence
        _ = geometry_evidence
        return FusionDecision(
            is_watermarked=False,
            decision_status="decided",
            thresholds_digest=_hex_anchor("thresholds"),
            evidence_summary={
                "content_score": content_evidence.get("score"),
                "geometry_score": geometry_evidence.get("geo_score"),
                "content_status": content_evidence.get("status", "absent"),
                "geometry_status": geometry_evidence.get("status", "absent"),
                "fusion_rule_id": "fusion_stub",
            },
            audit={"impl": "fusion_stub"},
        )


class _SyncStub:
    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        _ = cfg
        return {"status": "ok"}


class _SubspacePlannerStub:
    def __init__(self, plan_digest: str, basis_digest: str) -> None:
        self.plan_digest = plan_digest
        self.basis_digest = basis_digest
        self.impl_identity = {
            "impl_id": "subspace_planner",
            "impl_version": "v2",
            "impl_digest": _hex_anchor("subspace_planner_impl"),
        }

    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: Optional[str] = None,
        cfg_digest: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> SubspacePlanEvidence:
        _ = cfg
        _ = mask_digest
        _ = cfg_digest
        _ = inputs
        plan_obj: Dict[str, Any] = {
            "rank": 8,
            "energy_ratio": 0.9,
            "planner_impl_identity": self.impl_identity,
            "verifiable_input_domain_spec": {
                "planner_input_digest": _hex_anchor("planner_input")
            },
        }
        return SubspacePlanEvidence(
            status="ok",
            plan=plan_obj,
            basis_digest=self.basis_digest,
            plan_digest=self.plan_digest,
            audit={
                "impl_identity": "subspace_planner",
                "impl_version": "v2",
                "impl_digest": _hex_anchor("planner_audit_impl"),
                "trace_digest": _hex_anchor("planner_trace"),
            },
            plan_stats={"rank": 8, "energy_ratio": 0.9},
            plan_failure_reason=None,
        )


def _build_impl_set(plan_digest: str, basis_digest: str) -> BuiltImplSet:
    return BuiltImplSet(
        content_extractor=_ContentExtractorStub(),
        geometry_extractor=_GeometryExtractorStub(),
        fusion_rule=_FusionRuleStub(),
        subspace_planner=_SubspacePlannerStub(plan_digest=plan_digest, basis_digest=basis_digest),
        sync_module=_SyncStub(),
    )


def _build_base_cfg() -> Dict[str, Any]:
    return {
        "policy_path": "content_only",
        "evaluate": {
            "target_fpr": 1e-6
        },
        "embed": {
            "output_artifact_rel_path": "watermarked/watermarked_identity_v0.png"
        },
        "detect": {
            "content": {"enabled": True},
            "geometry": {"enabled": False},
        },
        "watermark": {
            "plan_digest": _hex_anchor("cfg_injected_plan_digest"),
            "subspace": {"enabled": True},
            "hf": {"enabled": False},
        },
    }


def _set_value_by_field_path(mapping: Dict[str, Any], field_path: str, value: Any) -> None:
    """
    鍔熻兘锛氭寜鐐硅矾寰勫啓鍏ユ槧灏勫瓧娈点€?

    Set a nested mapping value by dotted field path.

    Args:
        mapping: Target mapping.
        field_path: Dotted path.
        value: Value to assign.

    Returns:
        None.
    """
    current = mapping
    segments = field_path.split(".")
    for segment in segments[:-1]:
        if segment not in current or not isinstance(current[segment], dict):
            current[segment] = {}
        current = current[segment]
    current[segments[-1]] = value


def _build_minimal_valid_record(interpretation: Any) -> Dict[str, Any]:
    """
    鍔熻兘锛氭瀯寤哄彲閫氳繃 validate_record 鐨勬渶灏忚褰曘€?

    Build a minimal record satisfying required fields.

    Args:
        interpretation: Contract interpretation instance.

    Returns:
        Minimal record mapping.
    """
    record: Dict[str, Any] = {}
    for field_path in interpretation.required_record_fields:
        if field_path == "schema_version":
            _set_value_by_field_path(record, field_path, schema.RECORD_SCHEMA_VERSION)
        else:
            _set_value_by_field_path(record, field_path, "stub")
    _set_value_by_field_path(record, interpretation.records_schema.decision_field_path, None)
    return record


def test_detect_must_not_read_plan_digest_from_cfg() -> None:
    plan_digest = _hex_anchor("expected_plan")
    basis_digest = _hex_anchor("basis")
    cfg = _build_base_cfg()
    impl_set = _build_impl_set(plan_digest=plan_digest, basis_digest=basis_digest)

    input_record: Dict[str, Any] = {
        "plan_digest": plan_digest,
        "basis_digest": basis_digest,
        "subspace_planner_impl_identity": impl_set.subspace_planner.impl_identity,
    }

    record = run_detect_orchestrator(cfg, impl_set, input_record=input_record, cfg_digest=_hex_anchor("cfg"))

    assert record["plan_digest_status"] == "ok"
    assert record["plan_digest_expected"] == plan_digest
    assert record["plan_digest_observed"] == plan_digest
    assert record["detect_runtime_mode"] in {"real", "fallback_identity"}
    assert record["detect_runtime_mode_canonical"] in {"real", "fallback_identity"}
    assert record["detect_runtime_is_fallback"] == (record["detect_runtime_mode"] != "real")


def test_embed_produces_real_artifact_and_records_anchor(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_image = tmp_path / "inputs" / "input.png"
    _write_png(input_image)

    contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
    whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
    semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
    injection_scope_manifest = load_injection_scope_manifest(config_loader.INJECTION_SCOPE_MANIFEST_PATH)

    plan_digest = _hex_anchor("embed_plan")
    basis_digest = _hex_anchor("embed_basis")
    impl_set = _build_impl_set(plan_digest=plan_digest, basis_digest=basis_digest)

    cfg = _build_base_cfg()
    cfg["__run_root_dir__"] = str(run_root)
    cfg["__artifacts_dir__"] = str(artifacts_dir)
    cfg["__embed_input_image_path__"] = str(input_image)

    with bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        record = run_embed_orchestrator(cfg, impl_set, cfg_digest=_hex_anchor("cfg_embed"))

    output_path = Path(record["watermarked_path"])
    assert output_path.exists()
    assert record["artifact_sha256"] == digests.file_sha256(output_path)
    assert record["input_sha256"] == digests.file_sha256(input_image)
    assert record["plan_digest"] == plan_digest
    assert record["basis_digest"] == basis_digest
    assert record["content_evidence"]["basis_digest"] == basis_digest


def test_plan_digest_roundtrip_embed_to_detect_ok(tmp_path: Path) -> None:
    run_root = tmp_path / "roundtrip"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_image = tmp_path / "inputs" / "roundtrip.png"
    _write_png(input_image)

    contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
    whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
    semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
    injection_scope_manifest = load_injection_scope_manifest(config_loader.INJECTION_SCOPE_MANIFEST_PATH)

    plan_digest = _hex_anchor("roundtrip_plan")
    basis_digest = _hex_anchor("roundtrip_basis")
    impl_set = _build_impl_set(plan_digest=plan_digest, basis_digest=basis_digest)

    cfg = _build_base_cfg()
    cfg["__run_root_dir__"] = str(run_root)
    cfg["__artifacts_dir__"] = str(artifacts_dir)
    cfg["__embed_input_image_path__"] = str(input_image)

    with bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        embed_record = run_embed_orchestrator(cfg, impl_set, cfg_digest=_hex_anchor("cfg_roundtrip_embed"))

    detect_record = run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=embed_record,
        cfg_digest=_hex_anchor("cfg_roundtrip_detect"),
    )

    assert detect_record["plan_digest_status"] == "ok"
    assert detect_record["plan_digest_expected"] == plan_digest
    assert detect_record["plan_digest_observed"] == plan_digest


def test_plan_digest_mismatch_triggers_mismatch_semantics() -> None:
    cfg = _build_base_cfg()
    observed_plan_digest = _hex_anchor("observed_plan")
    basis_digest = _hex_anchor("basis")
    impl_set = _build_impl_set(plan_digest=observed_plan_digest, basis_digest=basis_digest)

    input_record: Dict[str, Any] = {
        "plan_digest": _hex_anchor("expected_plan_other"),
        "basis_digest": basis_digest,
        "subspace_planner_impl_identity": impl_set.subspace_planner.impl_identity,
    }

    record = run_detect_orchestrator(cfg, impl_set, input_record=input_record, cfg_digest=_hex_anchor("cfg_detect"))

    assert record["plan_digest_status"] == "mismatch"
    assert record["plan_digest_mismatch_reason"] == "plan_digest_mismatch"
    assert record["content_result"]["status"] == "mismatch"


def test_append_only_fields_validate_with_interpretation() -> None:
    contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
    interpretation = get_contract_interpretation(contracts)

    record = _build_minimal_valid_record(interpretation)
    record["plan_input_digest"] = _hex_anchor("plan_input")
    record["plan_input_schema_version"] = "v2"
    record["input_sha256"] = _hex_anchor("input")
    record["artifact_sha256"] = _hex_anchor("artifact")
    record["plan_digest_expected"] = _hex_anchor("plan_expected")
    record["plan_digest_observed"] = _hex_anchor("plan_observed")
    record["plan_digest_status"] = "ok"

    schema.validate_record(record, interpretation=interpretation)

    old_record = _build_minimal_valid_record(interpretation)
    schema.validate_record(old_record, interpretation=interpretation)


def test_embed_identity_trace_marks_baseline_fields(tmp_path: Path) -> None:
    run_root = tmp_path / "embed_trace"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_image = tmp_path / "inputs" / "trace_input.png"
    _write_png(input_image)

    contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
    whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
    semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
    injection_scope_manifest = load_injection_scope_manifest(config_loader.INJECTION_SCOPE_MANIFEST_PATH)

    plan_digest = _hex_anchor("embed_trace_plan")
    basis_digest = _hex_anchor("embed_trace_basis")
    impl_set = _build_impl_set(plan_digest=plan_digest, basis_digest=basis_digest)

    cfg = _build_base_cfg()
    cfg["__run_root_dir__"] = str(run_root)
    cfg["__artifacts_dir__"] = str(artifacts_dir)
    cfg["__embed_input_image_path__"] = str(input_image)

    with bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        record = run_embed_orchestrator(cfg, impl_set, cfg_digest=_hex_anchor("cfg_embed_trace"))

    embed_trace = record.get("embed_trace")
    assert isinstance(embed_trace, dict)
    assert embed_trace.get("embed_mode") == "content_real_v1"
    assert embed_trace.get("note") == "content_embedding_real_v1"
    assert isinstance(embed_trace.get("lf_trace_digest"), str)
    assert isinstance(embed_trace.get("lf_trace_summary"), dict)

