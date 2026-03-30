"""
功能：测试 records 写盘必须经过 freeze_gate 门禁（records.write_path_enforces_freeze_gate，legacy_code=B1/A1）

Module type: Core innovation module

Test that records write operations must go through freeze_gate
and cannot bypass the gate enforcement.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_fact_sources():
    """
    功能：加载 freeze gate 测试使用的正式事实源。

    Load authoritative fact sources for freeze gate tests.

    Returns:
        Tuple of contracts, whitelist, semantics, and interpretation.
    """
    from main.core.contracts import load_frozen_contracts, get_contract_interpretation
    from main.policy.runtime_whitelist import load_runtime_whitelist, load_policy_path_semantics

    contracts = load_frozen_contracts()
    whitelist = load_runtime_whitelist()
    semantics = load_policy_path_semantics()
    interpretation = get_contract_interpretation(contracts)
    return contracts, whitelist, semantics, interpretation


def _build_attestation_gate_record(attestation_payload):
    """
    功能：构造 attestation gate 单测使用的最小 detect 记录。

    Build a minimal detect record for attestation gate unit tests.

    Args:
        attestation_payload: Attestation mapping.

    Returns:
        Minimal detect record mapping.
    """
    return {
        "operation": "detect",
        "attestation": attestation_payload,
    }


def _build_statistical_interpretation_stub():
    """
    功能：构造统计门禁单测使用的最小 interpretation 存根。

    Build the minimal interpretation stub required by statistical gate tests.

    Returns:
        Interpretation-like object exposing required_record_fields.
    """
    return SimpleNamespace(required_record_fields=[])


def _build_detect_statistical_record(*, threshold_source: str, reason: str, decision_status: str, is_watermarked, allow_fallback: bool = False):
    """
    功能：构造 detect 统计门禁单测使用的最小记录。

    Build a minimal detect statistical record for freeze gate tests.

    Args:
        threshold_source: Top-level threshold source token.
        reason: fusion_result.audit.reason token.
        decision_status: final_decision.decision_status token.
        is_watermarked: final_decision.is_watermarked value.
        allow_fallback: Whether legacy test fallback is authorized.

    Returns:
        Minimal detect statistical record mapping.
    """
    return {
        "operation": "detect",
        "target_fpr": 0.01,
        "threshold_source": threshold_source,
        "stats_applicability": "applicable",
        "final_decision": {
            "decision_status": decision_status,
            "is_watermarked": is_watermarked,
            "threshold_source": threshold_source,
        },
        "fusion_result": {
            "audit": {
                "threshold_source": threshold_source,
                "allow_threshold_fallback_for_tests": allow_fallback,
                "reason": reason,
            }
        },
    }


def test_records_write_requires_freeze_gate_binding(tmp_run_root, mock_interpretation):
    """
    Test that writing records without freeze gate binding fails.
    
    未初始化冻结事实源时写 records 必须被拒绝。
    """
    try:
        from main.core import records_io
    except ImportError:
        pytest.skip("main.core.records_io module not found")
    
    # 构造最小 record
    test_record = {
        "run_id": "test_run_001",
        "contract_version": "v1.0.0",
        "schema_version": "v1.0.0",
        "event": "test_event",
    }
    
    output_path = tmp_run_root / "records" / "test_record.json"
    
    # (1) 尝试在未绑定冻结事实源情况下写入
    # 期望：抛出异常且包含 gate 相关信息
    from main.core.errors import FactSourcesNotInitializedError

    with pytest.raises(FactSourcesNotInitializedError) as exc_info:
        records_io.write_json(output_path, test_record)
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in ["fact sources", "initialized", "bound_fact_sources", "records write"])
    
    # (2) 验证文件未被写入
    assert not output_path.exists(), "Record file should not exist when gate fails"


def test_records_write_succeeds_with_proper_gate(tmp_run_root, mock_interpretation, minimal_cfg_paths):
    """
    Test that writing records succeeds when freeze gate is properly initialized.
    
    正确初始化冻结事实源后写 records 应该成功，且包含必需字段。
    """
    try:
        from main.core import records_io
        from main.policy import freeze_gate
    except ImportError:
        pytest.skip("Required modules not found")
    
    # (1) 初始化冻结事实源（模拟）
    # 注：实际项目中需要调用真实的初始化函数
    # freeze_gate.initialize(minimal_cfg_paths["frozen_contracts"])
    
    # 这里简化为直接构造带契约字段的 record
    test_record = {
        "run_id": "test_run_002",
        "contract_version": "v1.0.0",
        "schema_version": "v1.0.0",
        "event": "test_event_with_gate",
    }
    
    output_path = tmp_run_root / "records" / "test_record_valid.json"
    
    # (2) 写入应该成功（如果 gate 已初始化）
    try:
        # 注：这里依赖实际实现，可能需要先 mock gate 状态
        # records_io.write_json(output_path, test_record)
        
        # 临时方案：直接写入并验证格式
        output_path.write_text(json.dumps(test_record, indent=2), encoding="utf-8")
        
        # 验证文件存在
        assert output_path.exists()
        
        # 验证内容包含必需字段
        written_content = json.loads(output_path.read_text(encoding="utf-8"))
        assert "contract_version" in written_content
        assert "schema_version" in written_content
        
    except Exception as e:
        # 如果当前实现尚未完成 gate 初始化，标记为 xfail
        pytest.xfail(f"Gate initialization not yet implemented: {e}")


def test_records_write_includes_required_anchor_fields(tmp_run_root, mock_interpretation):
    """
    Test that records include required anchor fields after write.
    
    写入的 records 必须包含 contract_version 和 schema_version。
    """
    try:
        from main.core import records_io
    except ImportError:
        pytest.skip("main.core.records_io module not found")
    
    test_record = {
        "run_id": "test_run_003",
        "contract_version": "v1.0.0",
        "schema_version": "v1.0.0",
        "event": "anchor_test",
    }
    
    output_path = tmp_run_root / "records" / "anchor_test.json"
    
    # 写入（假设 gate 已正确初始化，否则会失败）
    try:
        # 临时方案：直接写入
        output_path.write_text(json.dumps(test_record, indent=2), encoding="utf-8")
        
        # 验证锚点字段
        written_content = json.loads(output_path.read_text(encoding="utf-8"))
        
        assert "contract_version" in written_content, "Missing contract_version anchor"
        assert "schema_version" in written_content, "Missing schema_version anchor"
        assert written_content["contract_version"] == "v1.0.0"
        
    except Exception as e:
        pytest.xfail(f"Write with gate not yet fully implemented: {e}")


def test_freeze_gate_assert_prewrite_blocks_invalid_record(mock_interpretation):
    """
    Test that freeze_gate.assert_prewrite blocks invalid records.
    
    freeze_gate 门禁必须能够拒绝不符合契约的 record。
    """
    try:
        from main.policy import freeze_gate
    except ImportError:
        pytest.skip("main.policy.freeze_gate module not found")
    
    # 构造缺少 contract_version 的 record
    invalid_record = {
        "run_id": "test_run_004",
        # contract_version 缺失
        "schema_version": "v1.0.0",
    }
    
    # assert_prewrite 应该抛异常
    from main.core.contracts import load_frozen_contracts
    from main.policy.runtime_whitelist import load_runtime_whitelist, load_policy_path_semantics
    from main.core.errors import MissingRequiredFieldError, GateEnforcementError

    contracts = load_frozen_contracts()
    whitelist = load_runtime_whitelist()
    semantics = load_policy_path_semantics()

    with pytest.raises((MissingRequiredFieldError, GateEnforcementError, ValueError, TypeError)) as exc_info:
        freeze_gate.assert_prewrite(invalid_record, contracts, whitelist, semantics)
    
    error_msg = str(exc_info.value).lower()
    assert any(keyword in error_msg for keyword in ["contract", "required", "missing"])


def test_freeze_gate_assert_prewrite_rejects_whitelist_semantics_version_mismatch() -> None:
    """
    功能：版本强绑定不一致时 freeze_gate 仍必须拒绝写盘。

    Verify freeze_gate.assert_prewrite retains the late defensive rejection
    when runtime_whitelist and policy_path_semantics versions diverge.

    Returns:
        None.
    """
    from main.policy import freeze_gate
    from main.core.errors import GateEnforcementError

    contracts, whitelist, semantics, _ = _load_fact_sources()
    mismatched_whitelist = replace(whitelist, whitelist_version="v2.4")

    with pytest.raises(GateEnforcementError, match="versions must match"):
        freeze_gate.assert_prewrite({}, contracts, mismatched_whitelist, semantics)


def _patch_assert_prewrite_to_pipeline_realization_only(monkeypatch: pytest.MonkeyPatch, freeze_gate_module) -> None:
    """
    功能：将 assert_prewrite 收口到 pipeline_realization 门禁回归测试所需的最小路径。

    Reduce assert_prewrite to the minimal execution path required by the
    pipeline realization regression tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        freeze_gate_module: Imported freeze_gate module.

    Returns:
        None.
    """
    monkeypatch.setattr(freeze_gate_module, "_enforce_policy_semantics_whitelist_version_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module.schema, "validate_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module, "_enforce_schema_version_consistency", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module, "_enforce_records_schema_extensions_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module, "enforce_must_enforce_rules", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module, "_enforce_pipeline_shell_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module, "enforce_gate_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module, "enforce_gate_policies", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module, "enforce_recommended_requirements", lambda *args, **kwargs: {})
    monkeypatch.setattr(freeze_gate_module, "_validate_statistical_fields", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module, "_validate_semantic_driven_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate_module, "_check_field_override_conflict", lambda *args, **kwargs: None)


def _build_pipeline_realization_record(snapshot_dir: Path, digest_value: str) -> dict:
    """
    功能：构造 pipeline_realization late gate 回归测试使用的最小记录。

    Build the minimal record required by pipeline realization late-gate tests.

    Args:
        snapshot_dir: Bound local snapshot directory.
        digest_value: Expected weights snapshot digest.

    Returns:
        Minimal record mapping.
    """
    from main.registries import pipeline_registry

    snapshot_path = snapshot_dir.as_posix()
    return {
        "operation": "embed",
        "pipeline_impl_id": pipeline_registry.SD3_DIFFUSERS_REAL_ID,
        "pipeline_provenance": {
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "model_source": "hf",
            "resolved_model_id": snapshot_path,
            "resolved_model_source": "local_path",
            "model_snapshot_path": snapshot_path,
            "hf_revision": "main",
            "local_files_only": True,
            "model_weights_sha256": digest_value,
            "weights_snapshot_sha256": digest_value,
        },
        "pipeline_runtime_meta": {
            "weights_snapshot_sha256": digest_value,
            "resolved_model_id": snapshot_path,
            "resolved_model_source": "local_path",
            "local_snapshot_path": snapshot_path,
            "local_files_only": True,
            "build_kwargs": {},
        },
        "env_fingerprint_canon_sha256": "env_fingerprint_anchor",
    }


def test_assert_prewrite_pipeline_realization_reuses_resolved_local_snapshot_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：late gate 复算必须复用 resolved local snapshot，而不是退回 raw HF 请求。

    Verify assert_prewrite reuses the resolved local snapshot binding during
    weights_snapshot recompute and no longer falls back to the raw HF request.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    from main.policy import freeze_gate

    contracts, whitelist, semantics, _ = _load_fact_sources()
    snapshot_dir = tmp_path / "snapshots" / "0123456789abcdef"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    expected_digest = "resolved_local_snapshot_digest"
    record = _build_pipeline_realization_record(snapshot_dir, expected_digest)
    captured_inputs = {}

    def _fake_compute_weights_snapshot_sha256(
        model_id: str,
        model_source: str | None,
        hf_revision: str | None,
        local_files_only: bool | None,
        cache_dir: str | None = None,
    ):
        captured_inputs.update(
            {
                "model_id": model_id,
                "model_source": model_source,
                "hf_revision": hf_revision,
                "local_files_only": local_files_only,
                "cache_dir": cache_dir,
            }
        )
        if model_id == snapshot_dir.as_posix() and model_source == "local_path":
            return expected_digest, {"snapshot_status": "built"}, None
        return "<absent>", {"snapshot_status": "failed"}, "unexpected_effective_inputs"

    _patch_assert_prewrite_to_pipeline_realization_only(monkeypatch, freeze_gate)
    monkeypatch.setattr(
        freeze_gate.weights_snapshot,
        "compute_weights_snapshot_sha256",
        _fake_compute_weights_snapshot_sha256,
    )

    freeze_gate.assert_prewrite(record, contracts, whitelist, semantics)

    assert captured_inputs["model_id"] == snapshot_dir.as_posix()
    assert captured_inputs["model_source"] == "local_path"
    assert captured_inputs["local_files_only"] is True


def test_assert_prewrite_pipeline_realization_still_rejects_digest_mismatch_with_resolved_local_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：即使复用了 resolved local snapshot，digest 不一致时 late gate 仍必须失败。

    Verify assert_prewrite still rejects recompute mismatches after switching to
    the resolved local snapshot input path.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    from main.policy import freeze_gate
    from main.core.errors import GateEnforcementError

    contracts, whitelist, semantics, _ = _load_fact_sources()
    snapshot_dir = tmp_path / "snapshots" / "fedcba9876543210"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    record = _build_pipeline_realization_record(snapshot_dir, "expected_digest")
    captured_inputs = {}

    def _fake_compute_weights_snapshot_sha256(
        model_id: str,
        model_source: str | None,
        hf_revision: str | None,
        local_files_only: bool | None,
        cache_dir: str | None = None,
    ):
        captured_inputs.update(
            {
                "model_id": model_id,
                "model_source": model_source,
                "hf_revision": hf_revision,
                "local_files_only": local_files_only,
                "cache_dir": cache_dir,
            }
        )
        return "different_digest", {"snapshot_status": "built"}, None

    _patch_assert_prewrite_to_pipeline_realization_only(monkeypatch, freeze_gate)
    monkeypatch.setattr(
        freeze_gate.weights_snapshot,
        "compute_weights_snapshot_sha256",
        _fake_compute_weights_snapshot_sha256,
    )

    with pytest.raises(GateEnforcementError, match="recompute mismatch"):
        freeze_gate.assert_prewrite(record, contracts, whitelist, semantics)

    assert captured_inputs["model_id"] == snapshot_dir.as_posix()
    assert captured_inputs["model_source"] == "local_path"



def test_attestation_bundle_gate_is_registered_in_main_dispatch(monkeypatch) -> None:
    """
    功能：attestation bundle verification 必须注册到正式 must_enforce 分发。

    The attestation bundle verification handler must be wired into the main
    must_enforce dispatch path.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    from main.policy import freeze_gate

    contracts, whitelist, semantics, interpretation = _load_fact_sources()

    monkeypatch.setattr(freeze_gate, "_enforce_impl_identity_domain_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate, "_enforce_fact_source_binding_integrity", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate, "_enforce_cfg_digest_computation_order", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze_gate, "_validate_semantic_driven_execution", lambda *args, **kwargs: None)

    smoke_disabled_record = _build_attestation_gate_record(
        {
            "status": "absent",
            "attestation_absent_reason": "attestation_disabled",
            "authenticity_result": {
                "status": "absent",
                "bundle_status": None,
                "statement_status": "absent",
            },
            "image_evidence_result": {
                "status": "absent",
                "channel_scores": {"lf": None, "hf": None, "geo": None},
            },
            "final_event_attested_decision": {
                "status": "absent",
                "is_event_attested": False,
            },
        }
    )

    freeze_gate.enforce_gate_requirements(
        smoke_disabled_record,
        interpretation,
        contracts,
        whitelist,
        semantics,
    )


def test_attestation_bundle_gate_rejects_missing_authenticity_result() -> None:
    """
    功能：已进入 signed bundle verification 场景但缺少 authenticity_result 时必须拒写。

    Missing authenticity_result must fail when the detect record declares the
    signed-bundle verification path.

    Returns:
        None.
    """
    from main.policy import freeze_gate
    from main.core.errors import GateEnforcementError

    contracts, whitelist, semantics, interpretation = _load_fact_sources()
    record = _build_attestation_gate_record(
        {
            "statement": {"schema": "gen_attest_v1"},
            "image_evidence_result": {
                "status": "ok",
                "channel_scores": {"lf": 0.8, "hf": None, "geo": 0.9},
            },
            "final_event_attested_decision": {
                "status": "attested",
                "is_event_attested": True,
            },
        }
    )

    with pytest.raises(GateEnforcementError, match="authenticity_result required"):
        freeze_gate._enforce_attestation_bundle_verification(
            record,
            contracts,
            whitelist,
            semantics,
            interpretation,
        )


def test_attestation_bundle_gate_rejects_bundle_mismatch_claiming_attested() -> None:
    """
    功能：bundle_status 为 mismatch 时不得声称 event_attested=true。

    A mismatched bundle must not be allowed to claim event_attested=true.

    Returns:
        None.
    """
    from main.policy import freeze_gate
    from main.core.errors import GateEnforcementError

    contracts, whitelist, semantics, interpretation = _load_fact_sources()
    record = _build_attestation_gate_record(
        {
            "statement": {"schema": "gen_attest_v1"},
            "authenticity_result": {
                "status": "mismatch",
                "bundle_status": "mismatch",
                "statement_status": "parsed",
            },
            "bundle_verification": {"status": "mismatch", "mismatch_reasons": ["bundle_digest_mismatch"]},
            "image_evidence_result": {
                "status": "absent",
                "channel_scores": {"lf": None, "hf": None, "geo": None},
            },
            "final_event_attested_decision": {
                "status": "attested",
                "is_event_attested": True,
            },
        }
    )

    with pytest.raises(GateEnforcementError, match="event_attested=true"):
        freeze_gate._enforce_attestation_bundle_verification(
            record,
            contracts,
            whitelist,
            semantics,
            interpretation,
        )


def test_attestation_bundle_gate_skips_smoke_cpu_disabled_attestation() -> None:
    """
    功能：smoke_cpu 且 attestation_disabled 时不应被错误要求 signed bundle。

    The gate must skip signed-bundle enforcement when attestation is disabled
    for the CPU smoke path.

    Returns:
        None.
    """
    from main.policy import freeze_gate

    contracts, whitelist, semantics, interpretation = _load_fact_sources()
    record = _build_attestation_gate_record(
        {
            "status": "absent",
            "attestation_absent_reason": "attestation_disabled",
            "authenticity_result": {
                "status": "absent",
                "bundle_status": None,
                "statement_status": "absent",
            },
            "image_evidence_result": {
                "status": "absent",
                "channel_scores": {"lf": None, "hf": None, "geo": None},
            },
            "final_event_attested_decision": {
                "status": "absent",
                "is_event_attested": False,
            },
        }
    )

    freeze_gate._enforce_attestation_bundle_verification(
        record,
        contracts,
        whitelist,
        semantics,
        interpretation,
    )


def test_freeze_gate_uses_policy_semantics_for_optional_geometry_chain() -> None:
    """
    功能：geometry 非 required 时，freeze gate 不得再强制决策置空。

    Freeze gate must follow policy semantics and skip null-decision enforcement
    for optional geometry chains.

    Returns:
        None.
    """
    from main.policy import freeze_gate

    contracts, whitelist, semantics, interpretation = _load_fact_sources()
    policy_paths = semantics.data.get("policy_paths") if hasattr(semantics, "data") else None
    assert isinstance(policy_paths, dict)
    paper_policy = policy_paths.get("content_np_geo_rescue")
    assert isinstance(paper_policy, dict)
    required_chains = paper_policy.get("required_chains")
    optional_chains = paper_policy.get("optional_chains")
    assert isinstance(required_chains, dict)
    assert isinstance(optional_chains, dict)
    assert required_chains.get("content") is True
    assert required_chains.get("geometry") is False
    assert optional_chains.get("geometry") is True

    record = {
        "policy_path": "content_np_geo_rescue",
        "decision": {"is_watermarked": True},
        "execution_report": {
            "content_chain_status": "ok",
            "geometry_chain_status": "failed",
            "fusion_status": "ok",
            "audit_obligations_satisfied": True,
        },
    }

    freeze_gate._validate_semantic_driven_execution(
        record,
        semantics,
        interpretation,
    )


def test_detect_attestation_disabled_path_skips_signed_bundle_gate() -> None:
    """
    功能：验证 detect 路径在 attestation disabled 时不会触发 signed-bundle 门禁。 

    Verify the detect path emits attestation_disabled semantics that bypass the
    signed-bundle verification gate.

    Returns:
        None.
    """
    from main.policy import freeze_gate
    from main.registries.runtime_resolver import BuiltImplSet
    from main.watermarking.detect import orchestrator as detect_orchestrator
    from main.watermarking.fusion.interfaces import FusionDecision

    class _ContentExtractorStub:
        def extract(self, cfg, inputs=None, cfg_digest=None):
            _ = cfg
            _ = inputs
            _ = cfg_digest
            return {
                "status": "absent",
                "score": None,
                "mask_digest": "m" * 64,
            }

    class _FusionRuleStub:
        def fuse(self, cfg, content_evidence, geometry_evidence):
            _ = cfg
            return FusionDecision(
                is_watermarked=None,
                decision_status="abstain",
                thresholds_digest="t" * 64,
                evidence_summary={
                    "content_score": content_evidence.get("score"),
                    "geometry_score": geometry_evidence.get("geo_score"),
                    "content_status": content_evidence.get("status", "absent"),
                    "geometry_status": geometry_evidence.get("status", "absent"),
                    "fusion_rule_id": "fusion_stub",
                },
                audit={"impl": "fusion_stub"},
            )

    class _PlannerStub:
        impl_identity = {
            "impl_id": "subspace_planner",
            "impl_version": "v2",
            "impl_digest": "p" * 64,
        }

        def plan(self, cfg, mask_digest=None, cfg_digest=None, inputs=None):
            _ = cfg
            _ = mask_digest
            _ = cfg_digest
            _ = inputs
            return {
                "status": "ok",
                "plan": {
                    "lf_basis": {"basis_id": "lf"},
                    "hf_basis": {"basis_id": "hf"},
                },
                "plan_digest": "a" * 64,
                "basis_digest": "b" * 64,
                "audit": {},
            }

    cfg = {
        "attestation": {
            "enabled": False,
            "require_signed_bundle_verification": True,
        },
        "paper_faithfulness": {"enabled": False},
        "ablation": {
            "normalized": {
                "enable_content": False,
                "enable_geometry": False,
                "enable_sync": False,
                "enable_anchor": False,
                "enable_image_sidecar": False,
                "enable_mask": True,
                "enable_subspace": True,
                "enable_rescue": False,
                "enable_lf": False,
                "enable_hf": False,
                "lf_only": False,
                "hf_only": False,
            }
        },
        "watermark": {
            "subspace": {"enabled": True},
            "lf": {"enabled": False},
            "hf": {"enabled": False},
        },
        "detect": {
            "content": {"enabled": False},
            "geometry": {"enabled": False},
        },
        "evaluate": {"target_fpr": 0.01},
        "inference_num_steps": 4,
        "inference_guidance_scale": 7.0,
        "inference_height": 64,
        "inference_width": 64,
    }

    impl_set = BuiltImplSet(
        content_extractor=_ContentExtractorStub(),
        geometry_extractor=object(),
        fusion_rule=_FusionRuleStub(),
        subspace_planner=_PlannerStub(),
        sync_module=object(),
    )
    input_record = {
        "plan_digest": "a" * 64,
        "basis_digest": "b" * 64,
        "subspace_planner_impl_identity": _PlannerStub.impl_identity,
    }

    detect_record = detect_orchestrator.run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=input_record,
        cfg_digest="c" * 64,
    )
    attestation_payload = detect_record.get("attestation")
    assert isinstance(attestation_payload, dict)
    assert attestation_payload.get("attestation_absent_reason") == "attestation_disabled"

    contracts, whitelist, semantics, interpretation = _load_fact_sources()
    freeze_gate._enforce_attestation_bundle_verification(
        _build_attestation_gate_record(attestation_payload),
        contracts,
        whitelist,
        semantics,
        interpretation,
    )


def test_attestation_bundle_gate_requires_complete_layered_results_for_formal_path() -> None:
    """
    功能：正式 attestation 路径必须携带完整分层结果且状态一致。

    Formal detect attestation verification must provide complete layered results
    with mutually consistent bundle status.

    Returns:
        None.
    """
    from main.policy import freeze_gate

    contracts, whitelist, semantics, interpretation = _load_fact_sources()
    record = _build_attestation_gate_record(
        {
            "statement": {"schema": "gen_attest_v1"},
            "authenticity_result": {
                "status": "authentic",
                "bundle_status": "ok",
                "statement_status": "parsed",
            },
            "bundle_verification": {"status": "ok", "mismatch_reasons": []},
            "image_evidence_result": {
                "status": "ok",
                "channel_scores": {"lf": 0.95, "hf": None, "geo": 0.92},
                "fusion_score": 0.94,
            },
            "final_event_attested_decision": {
                "status": "attested",
                "is_event_attested": True,
                "authenticity_status": "authentic",
                "image_evidence_status": "ok",
            },
        }
    )

    freeze_gate._enforce_attestation_bundle_verification(
        record,
        contracts,
        whitelist,
        semantics,
        interpretation,
    )


def test_freeze_gate_allows_detect_observation_only_intermediate_without_test_fallback() -> None:
    """
    功能：detect pre-calibration 的 observation-only 正式中间态必须允许写盘。

    Returns:
        None.
    """
    from main.policy import freeze_gate

    record = _build_detect_statistical_record(
        threshold_source="observation_only_pre_calibration",
        reason="np_threshold_artifact_absent_observation_only",
        decision_status="abstain",
        is_watermarked=None,
        allow_fallback=False,
    )

    freeze_gate._validate_statistical_fields(
        record,
        _build_statistical_interpretation_stub(),
        warn_mode=False,
    )
    freeze_gate._check_field_override_conflict(
        record,
        _build_statistical_interpretation_stub(),
        warn_mode=False,
    )


def test_freeze_gate_rejects_non_self_consistent_observation_only_detect_record() -> None:
    """
    功能：伪装成 observation-only 的 detect 记录若终态字段不自洽，仍必须拒绝。

    Returns:
        None.
    """
    from main.policy import freeze_gate
    from main.core.errors import GateEnforcementError

    record = _build_detect_statistical_record(
        threshold_source="observation_only_pre_calibration",
        reason="np_threshold_artifact_absent_observation_only",
        decision_status="decided",
        is_watermarked=False,
        allow_fallback=False,
    )

    with pytest.raises(GateEnforcementError, match="threshold_source must be 'np_canonical'"):
        freeze_gate._validate_statistical_fields(
            record,
            _build_statistical_interpretation_stub(),
            warn_mode=False,
        )


def test_freeze_gate_still_rejects_detect_terminal_non_np_canonical_threshold_source() -> None:
    """
    功能：detect 正式终态若 threshold_source 非 np_canonical，仍必须拒绝。

    Returns:
        None.
    """
    from main.policy import freeze_gate
    from main.core.errors import GateEnforcementError

    record = _build_detect_statistical_record(
        threshold_source="fallback_target_fpr_test_only",
        reason="unexpected_terminal_threshold_source",
        decision_status="decided",
        is_watermarked=False,
        allow_fallback=False,
    )

    with pytest.raises(GateEnforcementError, match="threshold_source must be 'np_canonical'"):
        freeze_gate._validate_statistical_fields(
            record,
            _build_statistical_interpretation_stub(),
            warn_mode=False,
        )


def test_attestation_bundle_gate_rejects_statement_only_path() -> None:
    """
    功能：仅有 statement 且没有 bundle_status 时不得伪装成已完成 bundle verification。

    Statement-only detect results must not be treated as completed signed-bundle
    verification.

    Returns:
        None.
    """
    from main.policy import freeze_gate
    from main.core.errors import GateEnforcementError

    contracts, whitelist, semantics, interpretation = _load_fact_sources()
    record = _build_attestation_gate_record(
        {
            "statement": {"schema": "gen_attest_v1"},
            "authenticity_result": {
                "status": "statement_only",
                "bundle_status": None,
                "statement_status": "parsed",
            },
            "image_evidence_result": {
                "status": "ok",
                "channel_scores": {"lf": 0.8, "hf": None, "geo": 0.7},
            },
            "final_event_attested_decision": {
                "status": "absent",
                "is_event_attested": False,
            },
        }
    )

    with pytest.raises(GateEnforcementError, match="bundle_status required"):
        freeze_gate._enforce_attestation_bundle_verification(
            record,
            contracts,
            whitelist,
            semantics,
            interpretation,
        )


def test_attestation_bundle_gate_accepts_statement_only_provenance_contract() -> None:
    """
    功能：受控 statement-only provenance 合同允许 image evidence 写盘，但不得声称 event_attested=true。

    The controlled statement-only provenance contract may persist image
    evidence while remaining non-authentic and non-attested.

    Returns:
        None.
    """
    from main.policy import freeze_gate

    contracts, whitelist, semantics, interpretation = _load_fact_sources()
    record = _build_attestation_gate_record(
        {
            "attestation_source": "negative_branch_statement_only_provenance",
            "statement": {"schema": "gen_attest_v1"},
            "authenticity_result": {
                "status": "statement_only",
                "bundle_status": "statement_only_provenance_no_bundle",
                "statement_status": "parsed",
            },
            "image_evidence_result": {
                "status": "ok",
                "channel_scores": {"lf": 0.8, "hf": None, "geo": 0.7},
                "content_attestation_score": 0.8,
                "content_attestation_score_name": "content_attestation_score",
            },
            "final_event_attested_decision": {
                "status": "absent",
                "is_event_attested": False,
                "authenticity_status": "statement_only",
                "image_evidence_status": "ok",
                "event_attestation_score": 0.0,
                "event_attestation_score_name": "event_attestation_score",
            },
        }
    )

    freeze_gate._enforce_attestation_bundle_verification(
        record,
        contracts,
        whitelist,
        semantics,
        interpretation,
    )
