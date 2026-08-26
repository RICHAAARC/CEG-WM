from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import run_content_v6_iss_canary as canary

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_canary_identity_is_frozen_and_disjoint_from_all_bound_data() -> None:
    assert canary.CANARY_ID == "content-v6-iss-detector-domain-operational-canary-v1"
    assert canary.UNIT_ID == "content-v6-iss-canary-0001"
    assert canary.SOURCE_ID == "content-v6-iss-canary-source-0001"
    assert canary.PROMPT == "A simple white ceramic cup on a wooden table in soft daylight"
    assert canary.SEED == 2026082600
    assert (canary.HEIGHT, canary.WIDTH) == (512, 512)
    assert canary.CLAIM_CEILING == "full_non_roster_runtime_canary_only"
    canary._assert_non_roster_identity(_ROOT)


@pytest.mark.integration
def test_execution_identity_requires_exact_clean_checkout_without_named_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = "a" * 40
    calls: list[tuple[str, ...]] = []

    def run_git(repo_root: Path, *arguments: str) -> str:
        calls.append(arguments)
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repo_root)
        if arguments == ("status", "--porcelain"):
            return ""
        raise AssertionError(f"unexpected git query: {arguments}")

    monkeypatch.setattr(canary, "_run_git", run_git)
    monkeypatch.setattr(canary.torch.cuda, "is_available", lambda: True)
    canary._validate_execution_identity(_ROOT, exact, exact)
    assert calls == [
        ("rev-parse", "--show-toplevel"),
        ("status", "--porcelain"),
    ]
    with pytest.raises(RuntimeError, match="resolved revision differs"):
        canary._validate_execution_identity(_ROOT, "b" * 40, exact)
    monkeypatch.setattr(
        canary,
        "_run_git",
        lambda repo_root, *arguments: (
            str(repo_root)
            if arguments == ("rev-parse", "--show-toplevel")
            else " M changed.py"
        ),
    )
    with pytest.raises(RuntimeError, match="checkout must be clean"):
        canary._validate_execution_identity(_ROOT, exact, exact)


@pytest.mark.integration
def test_canary_uses_one_paired_runtime_and_reports_denominator_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exact = "a" * 40
    monkeypatch.setenv(canary.KEY_ENV, "canary-key-value-01")
    monkeypatch.setenv(canary.TOKEN_ENV, "private-token")
    monkeypatch.setattr(canary, "_resolve_exact", lambda root: exact)
    monkeypatch.setattr(canary, "_validate_execution_identity", lambda *args: None)
    monkeypatch.setattr(canary, "_assert_non_roster_identity", lambda root: None)
    assets = object()
    monkeypatch.setattr(canary, "_load_pipeline_and_assets", lambda model, token: (object(), assets))
    pair_calls: list[tuple[object, ...]] = []
    output = SimpleNamespace(image="joint", primary_null="pass1", measurement=object())
    monkeypatch.setattr(
        canary,
        "_run_pair",
        lambda *args, **kwargs: pair_calls.append((*args, kwargs)) or output,
    )
    monkeypatch.setattr(
        canary,
        "_registered_scores",
        lambda image, key, received_assets: {
            "lf": 0.8 if image == "joint" else 0.2,
            "hf": 0.7 if image == "joint" else 0.1,
            "joint": 0.7 if image == "joint" else 0.1,
        },
    )
    monkeypatch.setattr(
        canary,
        "_validated_metrics",
        lambda received: {
            "combined_actual_dtype_relative_l2": 0.012,
            "lf_effective_relative_l2": 0.007,
            "hf_effective_relative_l2": 0.005,
            "lf_branch_share": 0.4,
            "hf_branch_share": 0.6,
            "minimum_counterfactual_effect": 0.01,
            "probe_evaluation_count": 64,
            "paired_rgb_psnr_db": 31.0,
        },
    )
    args = argparse.Namespace(repo_root=str(_ROOT), expected_exact=exact)
    assert canary.execute(args) == 0
    assert len(pair_calls) == 1
    assert pair_calls[0][-1]["seed"] == canary.SEED
    line = capsys.readouterr().out.strip()
    assert line.startswith(canary.PREFIX + " ")
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["status"] == "operational_canary_pass"
    assert payload["formal_roster_member"] is False
    assert payload["scientific_denominator_units"] == 0
    assert payload["claim_ceiling"] == canary.CLAIM_CEILING
    assert payload["primary_null_registered_joint_score"] == 0.1
    assert canary.KEY_ENV not in canary.os.environ
    assert canary.TOKEN_ENV not in canary.os.environ


@pytest.mark.integration
def test_canary_missing_secrets_emits_only_sanitized_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(canary.KEY_ENV, raising=False)
    monkeypatch.delenv(canary.TOKEN_ENV, raising=False)
    args = argparse.Namespace(repo_root=str(_ROOT), expected_exact="a" * 40)
    assert canary.execute(args) == 1
    line = capsys.readouterr().out.strip()
    payload = json.loads(line.split(" ", 1)[1])
    assert payload == {
        "canary_id": canary.CANARY_ID,
        "error_class": "RuntimeError",
        "stage": "identity_validation",
        "status": "operational_failure",
    }
