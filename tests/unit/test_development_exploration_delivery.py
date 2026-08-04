"""Static delivery checks for the thin development exploration Notebook."""

from __future__ import annotations

import ast
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
PROTOCOL_PATH = ROOT / "configs/experiments/development_module_exploration.json"
PROMPT_ROSTER_PATH = (
    ROOT / "configs/experiments/development_exploration_prompt_roster.json"
)
EXECUTION_REVISION = "66cea01b5b78904af8545bf47fd1c1b6d133b381"
EXPECTED_RUN_ID = "ceg_wm_development_exploration_receipt_recovery"
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
    assert 4 <= len(code_cells) <= 6
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
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
    assert "scientific completion is determined only" in source
    assert "COMMITTED" in source
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

    protocol = load_frozen_development_exploration_protocol(PROTOCOL_PATH)
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
