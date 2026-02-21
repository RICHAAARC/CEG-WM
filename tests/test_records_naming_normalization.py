"""
File purpose: Records 命名规范化回归测试。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from main.core import config_loader, digests
from main.core.contracts import load_frozen_contracts
from main.core.injection_scope import load_injection_scope_manifest
from main.core.records_io import bound_fact_sources
from main.policy.runtime_whitelist import load_policy_path_semantics, load_runtime_whitelist
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.subspace.planner_interface import SubspacePlanEvidence
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.watermarking.embed.orchestrator import run_embed_orchestrator


FORBIDDEN_TOKENS = ["demo", "synthetic", "debug", "test_only"]


def _hex_anchor(tag: str) -> str:
    """
    功能：生成稳定十六进制摘要。

    Build deterministic SHA256 hex digest anchor.

    Args:
        tag: Input tag string.

    Returns:
        SHA256 hex digest string.
    """
    return digests.canonical_sha256({"tag": tag})


def _write_png(path: Path) -> None:
    """
    功能：写入最小有效 PNG 文件。

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
        _ = cfg
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
    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> Dict[str, Any]:
        _ = cfg
        _ = content_evidence
        _ = geometry_evidence
        return {
            "decision_status": "abstain",
            "is_watermarked": None,
        }


class _SyncStub:
    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        _ = cfg
        return {"status": "ok"}


class _SubspacePlannerStub:
    def __init__(self, plan_digest: str, basis_digest: str) -> None:
        self.plan_digest = plan_digest
        self.basis_digest = basis_digest
        self.impl_identity = {
            "impl_id": "subspace_planner_v1",
            "impl_version": "v1",
            "impl_digest": _hex_anchor("planner_impl"),
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
            "band_spec": {
                "lf_selector_summary": {
                    "selector": "mask_false_or_low_texture",
                    "region_ratio": 0.65,
                },
                "hf_selector_summary": {
                    "selector": "mask_true_or_high_texture",
                    "region_ratio": 0.35,
                },
            },
        }
        return SubspacePlanEvidence(
            status="ok",
            plan=plan_obj,
            basis_digest=self.basis_digest,
            plan_digest=self.plan_digest,
            audit={
                "impl_identity": "subspace_planner_v1",
                "impl_version": "v1",
                "impl_digest": _hex_anchor("planner_audit"),
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
            "target_fpr": 1e-6,
        },
        "embed": {
            "output_artifact_rel_path": "watermarked/watermarked_identity_v0.png",
        },
        "detect": {
            "content": {"enabled": True},
            "geometry": {"enabled": False},
        },
        "watermark": {
            "plan_digest": _hex_anchor("cfg_plan"),
            "subspace": {"enabled": True},
            "hf": {"enabled": False},
        },
    }


def _assert_no_forbidden_tokens(value: Any, path: str) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        for token in FORBIDDEN_TOKENS:
            if token in lowered:
                raise AssertionError(f"forbidden token '{token}' found at {path}: {value}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_forbidden_tokens(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _assert_no_forbidden_tokens(item, f"{path}[{idx}]")
        return


def test_records_identity_value_is_normalized_or_absent(tmp_path: Path) -> None:
    """
    功能：验证 identity 模式的命名规范化。

    Validate identity-mode naming normalization for embed records.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
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
    cfg["embed"]["test_mode_identity"] = True
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

    embed_trace = record.get("embed_trace")
    assert isinstance(embed_trace, dict)
    assert embed_trace.get("embed_mode") == "baseline_identity_v0"
    assert embed_trace.get("identity_mode") is True
    assert embed_trace.get("identity_reason") == "identity_pipeline"


def test_records_forbidden_tokens_not_present_in_values(tmp_path: Path) -> None:
    """
    功能：验证关键记录字段不包含禁止词。

    Validate forbidden tokens are absent from key record values.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
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
        embed_record = run_embed_orchestrator(cfg, impl_set, cfg_digest=_hex_anchor("cfg_embed"))

    detect_record = run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=embed_record,
        cfg_digest=_hex_anchor("cfg_detect"),
    )

    _assert_no_forbidden_tokens(embed_record, "embed_record")
    _assert_no_forbidden_tokens(detect_record, "detect_record")
