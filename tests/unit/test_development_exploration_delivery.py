"""Static delivery checks for the thin development exploration Notebook."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from experiments.protocol.development_exploration import (
    create_frozen_development_execution_intent_authority,
    load_frozen_development_exploration_protocol,
)
from experiments.runners.development_inputs import (
    build_development_manifest_and_key_roster,
    load_development_prompt_roster,
)
from scripts.experiment_execution import development_exploration_entrypoint


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "notebooks/colab/development_exploration.ipynb"
PROTOCOL_PATH = ROOT / "configs/experiments/thirteen_module_mechanism_screening.json"
PROMPT_ROSTER_PATH = (
    ROOT
    / "configs/experiments/thirteen_module_mechanism_screening_prompt_roster.json"
)
PROTOCOL_ID = "ceg_wm_thirteen_module_mechanism_screening"
EXECUTION_REVISION = "b66cb04ebb41f0d5473c498ad5769b467ff26d7e"
EXPECTED_RUN_ID = (
    "ceg_wm_thirteen_module_mechanism_screening_operational_validation"
)
SUPERSEDED_EXECUTION_REVISION = "2ff836f45c4012010092f7075e749507ae2ad9ae"
SUPERSEDED_RUN_ID = "ceg_wm_thirteen_module_mechanism_screening"
SUPERSEDED_RECOVERY_REVISIONS = (
    "ce536f1ad66b5f45c05d7b0a08e5c83fb8fb4b29",
    "6c84cb121030a1190a183955dd4a27798a0eb975",
)
SUPERSEDED_RECOVERY_RUN_ID = (
    "ceg_wm_thirteen_module_mechanism_screening_preflight_recovery"
)
EXPECTED_PACKAGE_BYTES = 4_547_630
EXPECTED_PACKAGE_SHA256 = "9f2e4f322496412e8af39338791fded921c9d82f7b9972af44d7f2560285b1e2"
EXPECTED_NOTEBOOK_SHA256 = "da50e813784ac4a86127befa3286873ef4f9e526a8d53ea06a16c4ff1bf1e8cd"
TEST_ROOT_KEY = "development_exploration_delivery_non_secret_test_root_key"


def _notebook_constant(notebook: dict[str, object], name: str) -> object:
    values: list[object] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            ):
                assert isinstance(statement.value, ast.Constant)
                values.append(statement.value.value)
    assert len(values) == 1
    return values[0]


def _create_minimal_package_repository(root: Path) -> Path:
    repository = root / "repository"
    repository.mkdir()
    tracked = repository / "package_member.txt"
    tracked.write_text("development package member\n", encoding="utf-8")
    subprocess.run(
        ("git", "init"),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "add", tracked.name),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return repository


@pytest.mark.quick
def test_development_exploration_notebook_is_thin_and_output_free() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = tuple(
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(code_cells) == 5
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert sha256(NOTEBOOK_PATH.read_bytes()).hexdigest() == EXPECTED_NOTEBOOK_SHA256
    assert "https://github.com/RICHAAARC/CEG-WM.git" in source
    assert f"EXECUTION_REVISION = '{EXECUTION_REVISION}'" in source
    assert _notebook_constant(notebook, "RUN_ID") == EXPECTED_RUN_ID
    assert "drive.mount('/content/drive')" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "userdata.get('CEG_WM_ROOT_KEY')" in source
    assert "git', '-C', str(CHECKOUT_ROOT), 'fetch'" in source
    assert "checkout', '--detach', 'FETCH_HEAD'" in source
    assert "status', '--porcelain'" in source
    assert "development_exploration_server.py" in source
    assert source.count("'--maximum-wiring-clusters', '2'") == 1
    assert source.count("'--stop-before-scientific-units'") == 1
    assert "subprocess.Popen" in source and "stderr=subprocess.STDOUT" in source
    assert "server_receipts' / SESSION_ID / 'execution_receipt.json'" in source
    assert "server_failures' / SESSION_ID" in source
    assert "execution_failure_receipt_*.json" in source
    assert "SHA256SUMS" in source
    assert "copy_to_drive_export" in source
    assert "Drive export SHA-256 mismatch" in source
    assert source.index("process = subprocess.Popen") < source.index(
        "EXPORT_ROOT.mkdir"
    )
    assert source.index("copy_to_drive_export(artifact_source") < source.index(
        "if server_exit_code != 0"
    )
    assert "mutable branch must never replace" in source
    assert "only development entrypoint currently authorized" in source
    assert "experiment_execution.ipynb" in source
    assert "runtime_qualification.ipynb" in source
    assert "paused and are not authorized to run" in source
    assert "240 scientific units plus 42 operational units, 282 total" in source
    assert "846 maximum attempts" in source
    assert "4 operational units and 0 scientific units" in source
    assert "all 10 operational screening units" in source
    assert "stops before unit 10" in source
    assert "Repeating Run all after those 10 units creates no new commit" in source
    assert "does not count toward module science" in source
    assert "Agent2 and Agent3 approve that separate boundary" in source
    assert SUPERSEDED_EXECUTION_REVISION in source
    assert SUPERSEDED_RUN_ID in source
    assert SUPERSEDED_RECOVERY_RUN_ID in source
    assert all(revision in source for revision in SUPERSEDED_RECOVERY_REVISIONS)
    assert "dangling intent are immutable diagnostics" in source
    assert (
        "never reads, resumes, migrates, rewrites, or deletes any of those namespaces"
        in source
    )
    assert "scientific completion is determined only" in source
    assert "COMMITTED" in source
    assert "ceg_wm_development_exploration_detector_crossfit_execution" in source
    assert "ceg_wm_development_exploration_scientific_execution" in source
    assert "ceg_wm_development_exploration_science_first_v42" in source
    assert "ceg_wm_development_exploration_joint_record_execution" in source
    assert "two operational commits, zero scientific commits" in source
    assert "dangling unit 0002 attempt 0" in source
    assert "builtins.AssertionError" in source
    assert "Existing records, dangling attempts, and full artifacts" in source
    assert "never read, migrated, rewritten, or deleted" in source
    for readme_path in (
        ROOT / "notebooks/colab/README.md",
        ROOT / "scripts/experiment_execution/README.md",
    ):
        readme = readme_path.read_text(encoding="utf-8")
        assert f"{EXPECTED_PACKAGE_BYTES:,} bytes" in readme
        assert EXPECTED_PACKAGE_SHA256 in readme
    for readme_path in (
        ROOT / "notebooks/README.md",
        ROOT / "notebooks/colab/README.md",
        ROOT / "scripts/experiment_execution/README.md",
    ):
        readme = readme_path.read_text(encoding="utf-8")
        assert "240 scientific" in readme
        assert "42 operational" in readme
        assert "282" in readme
        assert "846" in readme
        assert "paused" in readme or "暂停" in readme
        assert "not authorized" in readme or "未授权" in readme
        assert "506" in readme
        assert EXECUTION_REVISION in readme
        assert EXPECTED_RUN_ID in readme
        assert SUPERSEDED_EXECUTION_REVISION in readme
        assert SUPERSEDED_RUN_ID in readme
        assert SUPERSEDED_RECOVERY_RUN_ID in readme
        assert all(revision in readme for revision in SUPERSEDED_RECOVERY_REVISIONS)
    for forbidden in (
        "pip install",
        "snapshot_download(",
        "from_pretrained(",
        "DevelopmentExplorationRunner(",
        "DevelopmentScientificRecord(",
        "execute_development_exploration_session(",
        "hf_only_threshold_fit",
        "4096",
        "--skip-dependency-install",
        "zipfile",
    ):
        assert forbidden not in source


@pytest.mark.quick
def test_development_exploration_notebook_run_id_crosses_execution_intent_boundary() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    run_id = _notebook_constant(notebook, "RUN_ID")
    assert run_id == EXPECTED_RUN_ID
    assert run_id not in {SUPERSEDED_RUN_ID, SUPERSEDED_RECOVERY_RUN_ID}

    protocol = load_frozen_development_exploration_protocol(PROTOCOL_PATH)
    assert protocol.protocol_id == PROTOCOL_ID
    assert len(protocol.unit_roster) == 282
    assert protocol.study_budget.maximum_scientific_units == 240
    assert protocol.study_budget.maximum_operational_units == 42
    assert protocol.study_budget.maximum_total_units == 282
    assert protocol.study_budget.maximum_total_record_attempts == 846
    assert protocol.study_budget.wiring_counts_as_scientific_coverage is False
    prompts = load_development_prompt_roster(PROMPT_ROSTER_PATH)
    manifest, public_key_roster = build_development_manifest_and_key_roster(
        protocol,
        prompts,
        TEST_ROOT_KEY,
    )
    authority = create_frozen_development_execution_intent_authority(
        protocol,
        run_id=run_id,
        seed_namespace=prompts.seed_namespace,
        input_manifest=manifest,
        public_key_roster=public_key_roster,
    )

    assert authority.run_id == EXPECTED_RUN_ID
    assert authority.validate() == ()


@pytest.mark.quick
def test_development_package_create_only_write_does_not_require_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _create_minimal_package_repository(tmp_path)
    persistent = tmp_path / "persistent"

    def reject_hardlink(*_arguments: object, **_keywords: object) -> None:
        raise AssertionError("development package must not use hardlink publication")

    monkeypatch.setattr(
        development_exploration_entrypoint.os,
        "link",
        reject_hardlink,
    )
    package = development_exploration_entrypoint._build_or_verify_package(
        repository,
        persistent,
        EXECUTION_REVISION,
    )

    with zipfile.ZipFile(package) as archive:
        assert archive.namelist() == ["package_member.txt"]
        assert archive.read("package_member.txt") == b"development package member\n"
        assert archive.testzip() is None
    assert not tuple(package.parent.glob("*.building.zip"))


@pytest.mark.quick
def test_development_package_invalid_existing_destination_is_not_overwritten(
    tmp_path: Path,
) -> None:
    repository = _create_minimal_package_repository(tmp_path)
    persistent = tmp_path / "persistent"
    package_root = persistent / "development_execution_packages"
    package_root.mkdir(parents=True)
    package = package_root / f"ceg_wm_development_{EXECUTION_REVISION}.zip"
    invalid_existing_bytes = b"preexisting invalid package"
    package.write_bytes(invalid_existing_bytes)

    with pytest.raises(zipfile.BadZipFile):
        development_exploration_entrypoint._build_or_verify_package(
            repository,
            persistent,
            EXECUTION_REVISION,
        )

    assert package.read_bytes() == invalid_existing_bytes
    assert not tuple(package.parent.glob("*.building.zip"))
