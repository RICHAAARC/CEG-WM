"""
File purpose: C3 鐪熷疄宓屽叆涓庢渶灏忔娴嬮棴鐜洖褰掓祴璇曘€?
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from main.core import config_loader, digests
from main.core.injection_scope import load_injection_scope_manifest
from main.core.records_io import bound_fact_sources
from main.policy.runtime_whitelist import load_policy_path_semantics, load_runtime_whitelist
from main.core.contracts import load_frozen_contracts
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.subspace.planner_interface import SubspacePlanEvidence
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    _build_high_freq_cfg_binding,
    _build_low_freq_cfg_binding,
)
from main.watermarking.embed.orchestrator import run_embed_orchestrator


def _hex_anchor(tag: str) -> str:
    return digests.canonical_sha256({"tag": tag})


def _write_input_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for row in range(64):
        for col in range(64):
            arr[row, col, 0] = (row * 3 + col * 5) % 256
            arr[row, col, 1] = (row * 7 + col * 11) % 256
            arr[row, col, 2] = (row * 13 + col * 17) % 256
    Image.fromarray(arr).save(path, format="PNG")


class _ContentExtractorStub:
    def extract(self, cfg: Dict[str, Any], cfg_digest: Optional[str] = None, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _ = cfg
        _ = cfg_digest
        _ = inputs
        return {
            "status": "ok",
            "mask_digest": _hex_anchor("mask"),
            "mask_stats": {
                "routing_digest": _hex_anchor("routing"),
                "routing_summary": {
                    "region_ratio": 0.35,
                },
            },
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
        "embed": {
            "output_artifact_rel_path": "watermarked/watermarked.png"
        },
        "watermark": {
            "key_id": "k1",
            "pattern_id": "p1",
            "lf": {
                "enabled": True,
                "strength": 1.5,
                "ecc": 2,
                "variance": 1.5,
                "dct_block_size": 8,
                "lf_coeff_indices": [[1, 1], [1, 2], [2, 1]],
            },
            "hf": {
                "enabled": True,
                "tail_truncation_ratio": 0.1,
                "tail_truncation_mode": "top_k_per_latent",
                "tau": 2.0,
                "sampling_stride": 1,
            },
        },
    }


def _run_embed_with_context(tmp_path: Path, cfg: Dict[str, Any], input_image: Path) -> Dict[str, Any]:
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(cfg)
    cfg["__run_root_dir__"] = str(run_root)
    cfg["__artifacts_dir__"] = str(artifacts_dir)
    cfg["__embed_input_image_path__"] = str(input_image)

    contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
    whitelist = load_runtime_whitelist(config_loader.RUNTIME_WHITELIST_PATH)
    semantics = load_policy_path_semantics(config_loader.POLICY_PATH_SEMANTICS_PATH)
    injection_scope_manifest = load_injection_scope_manifest(config_loader.INJECTION_SCOPE_MANIFEST_PATH)

    impl_set = _build_impl_set(plan_digest=_hex_anchor("plan"), basis_digest=_hex_anchor("basis"))
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
        return run_embed_orchestrator(cfg, impl_set, cfg_digest=_hex_anchor("cfg"))


def test_embed_produces_non_identical_watermarked_image(tmp_path: Path) -> None:
    input_image = tmp_path / "inputs" / "input.png"
    _write_input_image(input_image)
    cfg = _build_base_cfg()

    record = _run_embed_with_context(tmp_path, cfg, input_image)

    src = np.asarray(Image.open(input_image).convert("RGB"), dtype=np.int16)
    dst = np.asarray(Image.open(record["watermarked_path"]).convert("RGB"), dtype=np.int16)
    diff = np.abs(src - dst).sum()
    assert diff > 0


def test_artifact_sha256_is_anchored_and_recomputable(tmp_path: Path) -> None:
    input_image = tmp_path / "inputs" / "input.png"
    _write_input_image(input_image)
    cfg = _build_base_cfg()

    record = _run_embed_with_context(tmp_path, cfg, input_image)

    output_path = Path(record["watermarked_path"])
    assert output_path.exists()
    assert record["artifact_sha256"] == digests.file_sha256(output_path)
    assert record["watermarked_artifact_sha256"] == digests.file_sha256(output_path)


def test_enable_high_freq_false_hf_fields_absent_and_lf_unchanged(tmp_path: Path) -> None:
    input_image = tmp_path / "inputs" / "input.png"
    _write_input_image(input_image)

    cfg_a = _build_base_cfg()
    cfg_a["watermark"]["hf"]["enabled"] = False
    cfg_a["watermark"]["hf"]["tau"] = 2.0

    cfg_b = _build_base_cfg()
    cfg_b["watermark"]["hf"]["enabled"] = False
    cfg_b["watermark"]["hf"]["tau"] = 9.0

    record_a = _run_embed_with_context(tmp_path / "a", cfg_a, input_image)
    record_b = _run_embed_with_context(tmp_path / "b", cfg_b, input_image)

    evidence_a = record_a["content_evidence"]
    evidence_b = record_b["content_evidence"]
    assert "hf_trace_digest" not in evidence_a
    assert "hf_score" not in evidence_a
    assert evidence_a.get("lf_trace_digest") == evidence_b.get("lf_trace_digest")


def test_all_embedding_params_are_in_digest_scope() -> None:
    cfg_lf_a = _build_base_cfg()
    cfg_lf_b = _build_base_cfg()
    cfg_lf_b["watermark"]["lf"]["dct_block_size"] = 16

    lf_a = _build_low_freq_cfg_binding(cfg_lf_a)
    lf_b = _build_low_freq_cfg_binding(cfg_lf_b)
    assert lf_a["lf_cfg_digest"] != lf_b["lf_cfg_digest"]

    cfg_hf_a = _build_base_cfg()
    cfg_hf_b = _build_base_cfg()
    cfg_hf_b["watermark"]["hf"]["tau"] = 3.5

    hf_a = _build_high_freq_cfg_binding(cfg_hf_a)
    hf_b = _build_high_freq_cfg_binding(cfg_hf_b)
    assert hf_a["hf_cfg_digest"] != hf_b["hf_cfg_digest"]


def test_no_write_bypass_for_watermarked_artifact(tmp_path: Path, monkeypatch: Any) -> None:
    input_image = tmp_path / "inputs" / "input.png"
    _write_input_image(input_image)
    cfg = _build_base_cfg()

    from main.core import records_io

    call_count = {"value": 0}
    original = records_io.copy_file_controlled

    def _wrapped_copy(src_path: Path, dst_path: Path, kind: str = "artifact") -> None:
        call_count["value"] += 1
        original(src_path, dst_path, kind=kind)

    monkeypatch.setattr(records_io, "copy_file_controlled", _wrapped_copy)
    _run_embed_with_context(tmp_path, cfg, input_image)

    assert call_count["value"] >= 1

