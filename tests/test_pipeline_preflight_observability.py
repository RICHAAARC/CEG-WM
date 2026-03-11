"""
File purpose: Validate pipeline preflight observability and digest invariance.
Module type: General module
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from main.cli.run_detect import assert_detect_runtime_dependencies
from main.core import config_loader
from main.core.contracts import get_contract_interpretation, load_frozen_contracts
from main.core.errors import RunFailureReason
from main.core.status import build_run_closure_payload
from main.diffusion.sd3 import pipeline_factory
from main.watermarking.common.plan_digest_flow import build_content_plan_and_digest
from main.watermarking.fusion.neyman_pearson import compute_thresholds_digest


def _build_cfg() -> Dict[str, Any]:
    return {
        "policy_path": "content_chain.enabled",
        "pipeline_impl_id": "sd3_diffusers_real_v1",
        "pipeline_build_enabled": True,
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "model_source": "local",
        "hf_revision": "main",
        "device": "cpu",
        "model": {"dtype": "float32", "height": 512, "width": 512},
        "watermark": {
            "plan_digest": "plan_cfg",
            "subspace": {"enabled": True},
        },
        "evaluate": {"target_fpr": 1e-6},
    }


def _mock_env_fp_digest(_obj: Dict[str, Any]) -> str:
    return "env_fp_digest_test"


def test_preflight_does_not_weaken_detect_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build_cfg()
    monkeypatch.setattr(
        pipeline_factory.env_fingerprint,
        "build_env_fingerprint",
        lambda: {
            "python_version": "test",
            "platform": "test",
            "sys_platform": "test",
            "executable": "test",
            "torch_version": "<absent>",
            "cuda_available": "<absent>",
        },
    )
    monkeypatch.setattr(
        pipeline_factory.env_fingerprint,
        "compute_env_fingerprint_canon_sha256",
        _mock_env_fp_digest,
    )
    monkeypatch.setattr(
        pipeline_factory.diffusers_loader,
        "try_import_diffusers",
        lambda: (
            False,
            {
                "diffusers_version": "<absent>",
                "transformers_version": "<absent>",
                "safetensors_version": "<absent>",
                "import_error": "simulated_import_error",
            },
        ),
    )
    result = pipeline_factory.build_pipeline_shell(cfg)

    assert result.get("pipeline_build_status") in {"ok", "failed"}
    assert result.get("pipeline_obj") is None

    with pytest.raises(RuntimeError, match="detect_missing_pipeline_dependency"):
        assert_detect_runtime_dependencies(cfg, {"test_mode": False}, pipeline_obj=result.get("pipeline_obj"))


def test_preflight_fields_do_not_affect_cfg_or_plan_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build_cfg()
    monkeypatch.setattr(
        pipeline_factory.env_fingerprint,
        "build_env_fingerprint",
        lambda: {
            "python_version": "test",
            "platform": "test",
            "sys_platform": "test",
            "executable": "test",
            "torch_version": "<absent>",
            "cuda_available": "<absent>",
        },
    )
    monkeypatch.setattr(
        pipeline_factory.env_fingerprint,
        "compute_env_fingerprint_canon_sha256",
        _mock_env_fp_digest,
    )
    monkeypatch.setattr(
        pipeline_factory.diffusers_loader,
        "try_import_diffusers",
        lambda: (
            False,
            {
                "diffusers_version": "<absent>",
                "transformers_version": "<absent>",
                "safetensors_version": "<absent>",
                "import_error": "simulated_import_error",
            },
        ),
    )

    contracts = load_frozen_contracts(config_loader.FROZEN_CONTRACTS_PATH)
    interpretation = get_contract_interpretation(contracts)
    include_paths = list(interpretation.config_loader.cfg_digest_include_paths)

    thresholds_spec: Dict[str, Any] = {
        "threshold_rule_id": "neyman_pearson_v1",
        "threshold_rule_version": "v1",
        "target_fpr": 1e-6,
        "calibration": {
            "method": "empirical_quantile",
            "direction": "higher",
        },
    }

    subspace_result = SimpleNamespace(
        plan={
            "planner_impl_identity": {
                "impl_id": "subspace_planner_v2",
                "impl_version": "v2",
                "impl_digest": "digest",
            },
            "verifiable_input_domain_spec": {
                "planner_input_digest": "planner_input_digest_v1",
            },
            "planner_params": {
                "alpha": 0.5,
            },
        },
        plan_digest="plan_digest_fixed",
    )

    def _run_with_preflight(reason: str) -> tuple[str, str, str]:
        def _mock_preflight(_cfg: Dict[str, Any]) -> tuple[bool, str | None, str | None]:
            return False, reason, "summary"

        monkeypatch.setattr(
            pipeline_factory,
            "preflight_pipeline_build",
            _mock_preflight,
        )
        _ = pipeline_factory.build_pipeline_shell(cfg)
        cfg_digest = config_loader.compute_cfg_digest(cfg, include_paths, include_override_applied=False)
        _, plan_digest, _, _ = build_content_plan_and_digest(cfg, subspace_result, mask_digest="mask_digest_fixed")
        thresholds_digest = compute_thresholds_digest(thresholds_spec)
        return cfg_digest, str(plan_digest), thresholds_digest

    digest_a = _run_with_preflight("missing_model_weights")
    digest_b = _run_with_preflight("dependency_version_mismatch")

    assert digest_a == digest_b


def test_preflight_fields_are_written_to_run_closure_append_only() -> None:
    run_meta: Dict[str, Any] = {
        "run_id": "run-test",
        "command": "detect",
        "created_at_utc": "2026-02-22T00:00:00Z",
        "cfg_digest": "cfg_digest_value",
        "policy_path": "content_chain.enabled",
        "impl_id": "impl_test",
        "impl_version": "v1",
        "manifest_rel_path": "<absent>",
        "status_ok": False,
        "status_reason": RunFailureReason.RUNTIME_ERROR,
        "status_details": {"message": "runtime_error"},
        "pipeline_build_status": "failed",
        "pipeline_build_failure_reason": "dependency_version_mismatch",
        "pipeline_build_failure_summary": "diffusers_import_failed",
    }

    payload = build_run_closure_payload(run_meta, None)
    assert payload["pipeline_build_status"] == "failed"
    assert payload["pipeline_build_failure_reason"] == "dependency_version_mismatch"
    assert payload["pipeline_build_failure_summary"] == "diffusers_import_failed"

