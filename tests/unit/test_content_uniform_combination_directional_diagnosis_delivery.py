from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import inspect
import json
import os
import subprocess
import sys
from zipfile import ZipFile

import pytest

from experiments.protocol.content_uniform_combination_directional_diagnosis import (
    load_content_uniform_combination_directional_protocol,
)
from scripts.experiment_execution import content_uniform_combination_directional_diagnosis_entrypoint as entrypoint
from scripts.experiment_execution import content_uniform_combination_directional_diagnosis_server as server
from scripts.experiment_execution.development_exploration_entrypoint import (
    _build_or_verify_package,
)


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/content_uniform_combination_directional_diagnosis.json"
HISTORICAL_EXECUTION_REVISION = "7fb29a7c38e2975b1c3e1c76218bb1759f9f94cf"
BUDGET_LOCALIZATION_REVISION = "01ff7c897d660e295fa832e265eb87b287d37ac6"
EXECUTION_REVISION = "c30b8a75e69cb0ef7a8515ab9eeb5c75f4314c36"
RUN_ID = "ceg_wm_content_uniform_combination_arm_budget_field_localization"
NOTEBOOK = ROOT / "notebooks/colab/content_uniform_combination_directional_diagnosis.ipynb"
ARM_IDS = (
    "hf_only",
    "lf_only",
    "uniform_combined_quarter",
    "uniform_combined_half",
    "uniform_combined_three_quarters",
)
FIELD_IDENTITIES = (
    "clean_to_watermarked_rgb_relative_l2",
    "realized_relative_l2",
)


def test_server_help_imports_from_isolated_working_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/experiment_execution/content_uniform_combination_directional_diagnosis_server.py"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--whitening-asset-persistent-root" in result.stdout


def test_execution_chain_freezes_real_public_surfaces_and_fixed_denominator() -> None:
    protocol, _reference, _probes = load_content_uniform_combination_directional_protocol(CONFIG, repository_root=ROOT)
    assert len(protocol.unit_roster) == 41
    source = inspect.getsource(entrypoint.execute_content_uniform_combination_directional_diagnosis_session)
    assert "_replay_current_whitening_asset" in source
    assert "runner.execute_operational_unit" in source
    assert "runner.execute_reference_fit_unit" in source
    assert "runner.execute_probe_unit" in source
    assert "runner.replay_aggregate" in source
    assert "successful_references" in source
    assert "cursor.routing_reference_records" not in source


def test_whitening_producer_failure_is_the_only_replay_error_wrapped_for_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = entrypoint.LfWhiteningAssetProducerReplayError(
        "bounded producer replay failure"
    )
    monkeypatch.setattr(
        entrypoint,
        "_replay_verified_whitening_asset",
        lambda **_arguments: (_ for _ in ()).throw(error),
    )
    with pytest.raises(
        entrypoint.ContentUniformCombinationDirectionalStartupError
    ) as caught:
        entrypoint._replay_whitening_asset_for_startup(repository=ROOT)
    assert caught.value.failure_class == "implementation_failure"
    assert caught.value.failure_type.endswith("LfWhiteningAssetProducerReplayError")
    assert str(caught.value) == "content combination startup failed"
    assert "bounded producer replay failure" not in str(caught.value)

    monkeypatch.setattr(
        entrypoint,
        "_replay_verified_whitening_asset",
        lambda **_arguments: (_ for _ in ()).throw(ValueError("current authority")),
    )
    with pytest.raises(ValueError, match="current authority"):
        entrypoint._replay_whitening_asset_for_startup(repository=ROOT)


def test_current_whitening_authority_drift_is_not_a_producer_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _reference, _probes = load_content_uniform_combination_directional_protocol(
        CONFIG, repository_root=ROOT
    )
    calls = []
    monkeypatch.setattr(
        entrypoint,
        "_replay_whitening_asset_for_startup",
        lambda **arguments: calls.append(arguments) or "verified_asset",
    )
    drifted = replace(protocol, whitening_asset_digest="0" * 64)
    with pytest.raises(
        entrypoint.ContentUniformCombinationDirectionalEntrypointError,
        match="current combination whitening authority drifted",
    ) as caught:
        entrypoint._replay_current_whitening_asset(
            protocol=drifted, repository=ROOT
        )
    assert type(caught.value) is entrypoint.ContentUniformCombinationDirectionalEntrypointError
    assert calls == []

    result = entrypoint._replay_current_whitening_asset(
        protocol=protocol, repository=ROOT
    )
    assert result == "verified_asset"
    assert calls == [{"required_protocol": protocol, "repository": ROOT}]


def test_whitening_producer_startup_diagnostic_is_safe_and_pre_store(
    tmp_path: Path,
) -> None:
    protocol, reference, probes = load_content_uniform_combination_directional_protocol(
        CONFIG, repository_root=ROOT
    )
    failure = entrypoint.ContentUniformCombinationDirectionalStartupError(
        failure_type=(
            "scripts.experiment_execution.lf_whitened_directional_validation_entrypoint."
            "LfWhiteningAssetProducerReplayError"
        ),
        failure_class="implementation_failure",
    )
    worker = server._startup_failure_worker(
        error=failure,
        persistent_root=tmp_path,
        run_id=protocol.run_id,
        session_id="whitening_producer_startup_session",
        protocol=protocol,
        reference_manifest=reference,
        probe_manifest=probes,
        package_sha256="a" * 64,
    )
    assert worker["committed_unit_count"] == 0
    assert worker["content_uniform_combination_directional_aggregate"] is None
    with ZipFile(worker["diagnostic_zip"]) as archive:
        payload = json.loads(archive.read("diagnostic.json"))
    assert payload == {
        "failure_class": "implementation_failure",
        "failure_type": failure.failure_type,
        "scientific_claims_supported": False,
        "stage": "content_uniform_combination_directional_diagnosis_startup",
    }
    serialized = json.dumps(worker, sort_keys=True)
    for forbidden in ("bounded producer replay failure", "traceback", "root_secret"):
        assert forbidden not in serialized


@pytest.mark.parametrize("exit_code", (0, 3))
def test_server_receipt_contract_preserves_one_thirty_two_eight(monkeypatch, tmp_path: Path, exit_code: int) -> None:
    protocol, reference, probes = load_content_uniform_combination_directional_protocol(CONFIG, repository_root=ROOT)
    monkeypatch.setattr(server, "_verify_repository", lambda *_: None)
    monkeypatch.setattr(server, "_probe_resources", lambda **_: {"gpu": "bounded"})
    monkeypatch.setattr(server, "_install_frozen_dependencies", lambda *_: None)
    monkeypatch.setattr(server, "_download_configured_model", lambda **_: None)
    package = tmp_path / "package.zip"
    package.write_bytes(b"package")
    monkeypatch.setattr(server, "_build_or_verify_package", lambda *_: package)
    artifact = tmp_path / "persistent" / protocol.run_id / "session_results" / "session.zip"
    artifact.parent.mkdir(parents=True)
    with ZipFile(artifact, "w") as target:
        target.writestr("committed_unit_ids.json", b"[]")
    worker = {
        "artifact_kind": (
            "content_uniform_combination_directional_diagnosis_result"
            if exit_code == 0
            else "content_uniform_combination_directional_diagnosis_failure"
        ),
        ("result_zip" if exit_code == 0 else "diagnostic_zip"): str(artifact),
        "protocol_digest": protocol.digest(),
        "reference_manifest_digest": server.canonical_digest(server.asdict(reference)),
        "probe_manifest_digest": server.canonical_digest(server.asdict(probes)),
        "unit_roster_digest": protocol.unit_roster_digest,
        "claim_boundary": protocol.claim_boundary,
        "content_uniform_combination_directional_aggregate": (
            {"aggregate_identity": "a" * 64} if exit_code == 0 else None
        ),
        "termination_reason": (
            "frozen_roster_complete" if exit_code == 0 else "worker_execution_failure"
        ),
    }
    monkeypatch.setattr(server, "execute_content_uniform_combination_directional_diagnosis_session", lambda **_: (exit_code, worker))
    runtime = ROOT / "configs/runtime/runtime_sd35_flowmatch.json"
    monkeypatch.setattr(server, "RUNTIME_CONFIG_PATH", runtime.relative_to(ROOT))
    code, receipt = server.execute_content_uniform_combination_directional_diagnosis_server_session(
        repository_root=ROOT,
        expected_revision="9" * 40,
        persistent_root=tmp_path / "persistent",
        whitening_asset_persistent_root=tmp_path / "fit",
        cache_root=tmp_path / "cache",
        run_id=protocol.run_id,
        session_id="combination_delivery_session",
        environment={"HF_TOKEN": "hf_secret", "CEG_WM_ROOT_KEY": "root_secret"},
        install_dependencies=False,
    )
    assert code == exit_code
    assert receipt["committed_revision"] == "9" * 40
    assert receipt["run_id"] == RUN_ID
    assert receipt["execution_package_sha256"] == sha256(b"package").hexdigest()
    assert (receipt["operational_unit_count"], receipt["reference_fit_cluster_count"], receipt["directional_probe_cluster_count"], receipt["total_unit_count"]) == (1, 32, 8, 41)
    assert receipt["maximum_attempts_per_unit"] == 1
    serialized = str(receipt)
    assert "hf_secret" not in serialized and "root_secret" not in serialized


def test_server_and_worker_do_not_claim_selection_or_formal_threshold() -> None:
    source = inspect.getsource(server) + inspect.getsource(entrypoint)
    assert '"formal_tau_created": False' in source
    assert '"candidate_promoted": False' in source
    assert '"scientific_claims_supported": False' in source
    assert "content_uniform_combination_directional_aggregate" in source


def test_live_worker_exports_only_fail_closed_arm_observation_leaf_reasons() -> None:
    cases = (
        (
            entrypoint.ContentCombinationArmRoleInvalidRunnerError(),
            "content_combination_arm_role_invalid",
        ),
        (
            entrypoint.ContentCombinationArmMeasurementNonfiniteRunnerError(),
            "content_combination_arm_measurement_nonfinite",
        ),
        (
            entrypoint.ContentCombinationArmMaterializationRejectedRunnerError(),
            "content_combination_arm_materialization_rejected",
        ),
        (
            entrypoint.ContentCombinationArmImageDigestInvalidRunnerError(),
            "content_combination_arm_image_digest_invalid",
        ),
        (
            entrypoint.ContentCombinationArmObservationIdentityDriftRunnerError(),
            "content_combination_arm_observation_identity_drift",
        ),
    )
    assert tuple(
        entrypoint._content_combination_observation_failure_reason(error)
        for error, _reason in cases
    ) == tuple(reason for _error, reason in cases)
    assert entrypoint._content_combination_observation_failure_reason(
        RuntimeError("content_combination_arm_canonical_budget_exceeded")
    ) is None
    assert not hasattr(
        entrypoint, "ContentCombinationArmRgbQualityBudgetExceededRunnerError"
    )
    assert not hasattr(
        entrypoint, "ContentCombinationArmRealizedContentBudgetExceededRunnerError"
    )
    assert "canonical_budget_exceeded" not in inspect.getsource(entrypoint)

    class DerivedMeasurementFailure(
        entrypoint.ContentCombinationArmMeasurementNonfiniteRunnerError
    ):
        pass

    assert entrypoint._content_combination_observation_failure_reason(
        DerivedMeasurementFailure()
    ) is None


def test_exact_execution_package_imports_combination_chain(tmp_path: Path) -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("detached research copy lacks exact Git checkout capability")
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--no-checkout", str(ROOT), str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "checkout",
            "--detach",
            HISTORICAL_EXECUTION_REVISION,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    package = _build_or_verify_package(
        checkout, tmp_path / "package_root", HISTORICAL_EXECUTION_REVISION
    )
    extracted = tmp_path / "extracted_package"
    with ZipFile(package) as archive:
        names = set(archive.namelist())
        assert (
            "scripts/experiment_execution/lf_whitened_directional_validation_entrypoint.py"
            in names
        )
        assert (
            "scripts/experiment_execution/content_uniform_combination_directional_diagnosis_entrypoint.py"
            in names
        )
        producer_replay_source = archive.read(
            "scripts/experiment_execution/lf_whitened_directional_validation_entrypoint.py"
        ).decode("utf-8")
        combination_source = archive.read(
            "scripts/experiment_execution/content_uniform_combination_directional_diagnosis_entrypoint.py"
        ).decode("utf-8")
        assert "WHITENING_ASSET_PACKAGE_SHA256" in producer_replay_source
        assert "cat-file" not in producer_replay_source
        assert "_replay_current_whitening_asset" in combination_source
        safe_reasons = (
            "content_combination_arm_role_invalid",
            "content_combination_arm_measurement_nonfinite",
            "content_combination_arm_canonical_budget_exceeded",
            "content_combination_arm_materialization_rejected",
            "content_combination_arm_image_digest_invalid",
            "content_combination_arm_observation_identity_drift",
        )
        for safe_reason in safe_reasons:
            assert safe_reason in combination_source
        archive.extractall(extracted)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import experiments.protocol.content_uniform_combination_directional_diagnosis; "
                "import experiments.metrics.content_uniform_combination_directional_diagnosis; "
                "import experiments.runners.content_uniform_combination_directional_diagnosis; "
                "import scripts.experiment_execution.content_uniform_combination_directional_diagnosis_entrypoint; "
                "import scripts.experiment_execution.content_uniform_combination_directional_diagnosis_server"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(extracted)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "execution_revision",
    (BUDGET_LOCALIZATION_REVISION, EXECUTION_REVISION),
)
def test_budget_localization_exact_packages_map_each_arm_and_field_reason(
    tmp_path: Path,
    execution_revision: str,
) -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("detached research copy lacks exact Git checkout capability")
    checkout = tmp_path / "budget_localization_checkout"
    subprocess.run(
        ["git", "clone", "--no-checkout", str(ROOT), str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "checkout",
            "--detach",
            execution_revision,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    package = _build_or_verify_package(
        checkout,
        tmp_path / f"budget_localization_package_root_{execution_revision}",
        execution_revision,
    )
    extracted = tmp_path / "budget_localization_extracted_package"
    with ZipFile(package) as archive:
        combination_source = archive.read(
            "scripts/experiment_execution/content_uniform_combination_directional_diagnosis_entrypoint.py"
        ).decode("utf-8")
        expected_reasons = tuple(
            f"content_combination_{arm_id}_{field_identity}_canonical_budget_exceeded"
            for field_identity in FIELD_IDENTITIES
            for arm_id in ARM_IDS
        )
        assert all(reason in combination_source for reason in expected_reasons)
        assert "content_combination_arm_canonical_budget_exceeded" not in combination_source
        if execution_revision == EXECUTION_REVISION:
            protocol_source = archive.read(
                "experiments/protocol/content_uniform_combination_directional_diagnosis.py"
            ).decode("utf-8")
            assert RUN_ID in protocol_source
        archive.extractall(extracted)

    worker_probe = (
        "import json; "
        "from scripts.experiment_execution import "
        "content_uniform_combination_directional_diagnosis_entrypoint as e; "
        f"arms={ARM_IDS!r}; "
        "pairs=((e.ContentCombinationArmRgbQualityBudgetExceededRunnerError,"
        "'clean_to_watermarked_rgb_relative_l2'),"
        "(e.ContentCombinationArmRealizedContentBudgetExceededRunnerError,"
        "'realized_relative_l2')); "
        "reasons=[e._content_combination_observation_failure_reason(t(a)) "
        "for t,_f in pairs for a in arms]; "
        "expected=[f'content_combination_{a}_{f}_canonical_budget_exceeded' "
        "for _t,f in pairs for a in arms]; "
        "assert reasons == expected; "
        "assert e._content_combination_observation_failure_reason("
        "RuntimeError('root_secret traceback sensitive budget')) is None; "
        "Derived=type('DerivedBudgetFailure',"
        "(e.ContentCombinationArmRealizedContentBudgetExceededRunnerError,),{}); "
        "assert e._content_combination_observation_failure_reason("
        "Derived('hf_only')) is None; "
        "print(json.dumps(reasons))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import experiments.protocol.content_uniform_combination_directional_diagnosis; "
                "import experiments.metrics.content_uniform_combination_directional_diagnosis; "
                "import experiments.runners.content_uniform_combination_directional_diagnosis; "
                "import scripts.experiment_execution.content_uniform_combination_directional_diagnosis_entrypoint; "
                "import scripts.experiment_execution.content_uniform_combination_directional_diagnosis_server; "
                + worker_probe
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(extracted)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    package_reasons = json.loads(result.stdout)
    assert package_reasons == [
        f"content_combination_{arm_id}_{field_identity}_canonical_budget_exceeded"
        for field_identity in FIELD_IDENTITIES
        for arm_id in ARM_IDS
    ]
    assert "root_secret" not in result.stdout


def test_notebook_is_thin_exact_output_free_and_exports_before_failure() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in document["cells"])
    assert EXECUTION_REVISION in source and RUN_ID in source
    assert "fresh run-specific persistent namespace" in source
    assert "content_uniform_combination_directional_diagnosis_server.py" in source
    assert "--whitening-asset-persistent-root" in source
    assert "SHA256SUMS" in source and "execution_receipt.json" in source
    assert source.index("copy_to_drive_export") < source.index("if server_exit_code != 0")
    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == []
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )
    for forbidden in (
        "aggregate_content_uniform_combination_directional_diagnosis",
        "create_content_combination_score_row",
        "fit_content_combination_fold_reference",
        "CegWmExperimentAdapter",
        "DevelopmentPersistentStore",
    ):
        assert forbidden not in source
