from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.experiment_execution import runtime_qualification_bootstrap as bootstrap
from scripts.experiment_execution.build_runtime_qualification_package import (
    REQUIRED_FILES,
)


pytestmark = pytest.mark.unit
REVISION = "a" * 40


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _package_payloads() -> dict[str, bytes]:
    dependency_lock = [
        {"package_name": "python", "version_specifier": ">=3.12"},
        {"package_name": "synthetic-runtime", "version_specifier": "1.2.3"},
    ]
    return {
        "README.md": b"execution package\n",
        "configs/runtime/runtime_sd35_flowmatch.json": (
            json.dumps({"dependency_lock": dependency_lock}) + "\n"
        ).encode(),
        "pyproject.toml": b"[project]\nname='synthetic-runtime'\n",
        "requirements_runtime_qualification.txt": b"synthetic-runtime==1.2.3\n",
        "scripts/experiment_execution/__init__.py": b"",
        "scripts/experiment_execution/runtime_qualification_runner.py": b"# fake\n",
        "main/__init__.py": b"# method\n",
        "runtime/__init__.py": b"# runtime\n",
    }


def _write_package(
    path: Path,
    *,
    mutate_manifest=None,
    extra_member: tuple[str, bytes, bool] | None = None,
) -> str:
    payloads = _package_payloads()
    manifest = {
        "package_schema_version": 1,
        "profile_name": "experiment_execution_package",
        "runtime_candidate_revision": REVISION,
        "copied_files": [
            {
                "path": name,
                "sha256": _digest(payload),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(payloads.items())
        ],
        "excluded_parts": sorted(bootstrap.PACKAGE_EXCLUDED_PARTS),
        "package_ready": True,
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr(
            "runtime_execution_manifest.json",
            json.dumps(manifest),
        )
        if extra_member is not None:
            name, payload, symlink = extra_member
            info = zipfile.ZipInfo(name)
            if symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, payload)
    return bootstrap._sha256(path)


def _summary(profile: str, run_id: str, exit_code: int) -> dict[str, object]:
    passed = exit_code == 0
    incomplete = exit_code == 2
    failures = [] if passed else (["incomplete"] if incomplete else ["runtime_failure"])
    value: dict[str, object] = {
        field: None for field in bootstrap.SUMMARY_FIELDS
    }
    value.update(
        {
            "result_schema_version": 2,
            "profile": profile,
            "run_id": run_id,
            "result_zip_filename": f"ceg_wm_runtime_qualification_{run_id}.zip",
            "run_status": "passed" if passed else "failed",
            "runtime_candidate_revision": REVISION,
            "failure_count": len(failures),
            "failure_classes": failures,
            "checks": [],
        }
    )
    return value


def _write_result(path: Path, profile: str, run_id: str, exit_code: int) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "run_summary.json",
            json.dumps(_summary(profile, run_id, exit_code)),
        )
        archive.writestr("environment_summary.json", "{}")
        archive.writestr("runtime_checks.jsonl", "")
        archive.writestr("failures.jsonl", "" if exit_code == 0 else "{}\n")


class FakeCommands:
    def __init__(
        self,
        runner_exit_code: int = 0,
        *,
        produce_result: bool = True,
        pip_failure: bool = False,
        runner_os_error: bool = False,
    ):
        self.runner_exit_code = runner_exit_code
        self.produce_result = produce_result
        self.pip_failure = pip_failure
        self.runner_os_error = runner_os_error
        self.calls: list[tuple[str, ...]] = []
        self.environments: list[dict[str, str]] = []

    def __call__(self, command, **kwargs):
        command_tuple = tuple(command)
        self.calls.append(command_tuple)
        self.environments.append(dict(kwargs.get("env", {})))
        if command_tuple[1:4] == ("-m", "pip", "install"):
            if self.pip_failure:
                raise subprocess.CalledProcessError(1, command_tuple)
            return subprocess.CompletedProcess(command_tuple, 0, "", "")
        if self.runner_os_error:
            raise OSError("runner launch failed")
        result_path = Path(command_tuple[command_tuple.index("--result-zip") + 1])
        profile = command_tuple[command_tuple.index("--profile") + 1]
        run_id = command_tuple[command_tuple.index("--run-id") + 1]
        if self.produce_result:
            _write_result(
                result_path,
                profile,
                run_id,
                self.runner_exit_code,
            )
        return subprocess.CompletedProcess(
            command_tuple,
            self.runner_exit_code,
            "",
            "",
        )


def _run(
    tmp_path: Path,
    archive: Path,
    expected_digest: str,
    commands: FakeCommands,
    *,
    profile: str = "smoke",
    replay_source: Path | None = None,
    run_id: str = "run-001",
) -> tuple[int, dict[str, object]]:
    return bootstrap.run_bootstrap(
        profile=profile,
        package_zip=archive,
        expected_package_sha256=expected_digest,
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=tmp_path / "persistent",
        replay_source=replay_source,
        run_id=run_id,
        environment={
            "HF_TOKEN": "memory-only-token",
            "CEG_WM_ROOT_KEY": "memory-only-root-key",
        },
        command_runner=commands,
    )


def test_archive_digest_mismatch_precedes_extraction_install_and_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "package.zip"
    _write_package(archive)
    commands = FakeCommands()
    extract_calls: list[Path] = []
    monkeypatch.setattr(
        bootstrap,
        "_safe_extract",
        lambda path, _destination: extract_calls.append(path),
    )
    exit_code, result = _run(tmp_path, archive, "0" * 64, commands)
    assert exit_code == 3
    assert result["failure_stage"] == "archive_digest"
    assert commands.calls == []
    assert extract_calls == []
    assert not list((tmp_path / "ephemeral").rglob("*.zip"))
    assert Path(result["diagnostic_zip"]).is_file()


def test_archive_source_replacement_after_digest_uses_verified_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "package.zip"
    expected_digest = _write_package(archive)
    original_extract = bootstrap._safe_extract
    observed_snapshot: list[Path] = []

    def replace_source_then_extract(snapshot: Path, destination: Path) -> None:
        observed_snapshot.append(snapshot)
        assert snapshot != archive
        assert bootstrap._sha256(snapshot) == expected_digest
        archive.write_bytes(b"replacement after trusted snapshot")
        original_extract(snapshot, destination)

    monkeypatch.setattr(bootstrap, "_safe_extract", replace_source_then_extract)
    commands = FakeCommands()
    code, result = _run(tmp_path, archive, expected_digest, commands)
    assert code == 0
    assert result["package_sha256"] == expected_digest
    assert len(observed_snapshot) == 1
    assert observed_snapshot[0].is_file()
    assert bootstrap._sha256(archive) != expected_digest


@pytest.mark.parametrize(
    ("member_name", "symlink"),
    (
        ("../escape.py", False),
        ("/absolute.py", False),
        ("C:/escape.py", False),
        ("C:\\escape.py", False),
        ("link.py", True),
    ),
)
def test_archive_safety_rejects_hostile_paths_before_install(
    tmp_path: Path,
    member_name: str,
    symlink: bool,
) -> None:
    archive = tmp_path / "package.zip"
    digest = _write_package(
        archive,
        extra_member=(member_name, b"payload", symlink),
    )
    commands = FakeCommands()
    exit_code, result = _run(tmp_path, archive, digest, commands)
    assert exit_code == 3
    assert result["failure_stage"] == "archive_safety"
    assert commands.calls == []
    assert not (tmp_path / "escape.py").exists()


def test_archive_safety_rejects_duplicate_and_oversized_members(
    tmp_path: Path,
    monkeypatch,
) -> None:
    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("runtime_execution_manifest.json", "{}")
            archive.writestr("runtime_execution_manifest.json", "{}")
    commands = FakeCommands()
    code, result = _run(
        tmp_path,
        duplicate,
        bootstrap._sha256(duplicate),
        commands,
    )
    assert code == 3 and result["failure_stage"] == "archive_safety"
    assert commands.calls == []


def test_archive_safety_rejects_total_size_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "package.zip"
    _write_package(archive)
    monkeypatch.setattr(bootstrap, "MAX_TOTAL_BYTES", 1)
    commands = FakeCommands()
    code, result = _run(
        tmp_path,
        archive,
        bootstrap._sha256(archive),
        commands,
    )
    assert code == 3 and result["failure_stage"] == "archive_safety"
    assert commands.calls == []

    oversized = tmp_path / "oversized.zip"
    _write_package(oversized)
    monkeypatch.setattr(bootstrap, "MAX_MEMBER_BYTES", 1)
    code, result = _run(
        tmp_path,
        oversized,
        bootstrap._sha256(oversized),
        commands,
        run_id="run-002",
    )
    assert code == 3 and result["failure_stage"] == "archive_safety"
    assert commands.calls == []


@pytest.mark.parametrize(
    "mutation",
    (
        lambda manifest: manifest.update(package_schema_version=2),
        lambda manifest: manifest.update(package_ready=False),
        lambda manifest: manifest.update(unexpected=True),
        lambda manifest: manifest["copied_files"][0].update(path=".env"),
        lambda manifest: manifest["copied_files"][0].update(sha256="0" * 64),
    ),
)
def test_manifest_and_file_identity_drift_precedes_dependency_install(
    tmp_path: Path,
    mutation,
) -> None:
    archive = tmp_path / "package.zip"
    digest = _write_package(archive, mutate_manifest=mutation)
    commands = FakeCommands()
    code, result = _run(tmp_path, archive, digest, commands)
    assert code == 3
    assert result["failure_stage"] == "manifest"
    assert commands.calls == []


def test_all_pretrust_checks_precede_install_and_runner(tmp_path: Path) -> None:
    archive = tmp_path / "package.zip"
    digest = _write_package(archive)
    commands = FakeCommands()
    code, result = _run(tmp_path, archive, digest, commands)
    assert code == 0
    assert len(commands.calls) == 2
    assert commands.calls[0][0:4] == (
        sys.executable,
        "-m",
        "pip",
        "install",
    )
    assert commands.calls[1][0:3] == (
        sys.executable,
        "-m",
        "scripts.experiment_execution.runtime_qualification_runner",
    )
    assert "HF_TOKEN" not in commands.environments[0]
    assert "CEG_WM_ROOT_KEY" not in commands.environments[0]
    assert commands.environments[1]["HF_TOKEN"] == "memory-only-token"
    assert commands.environments[1]["CEG_WM_ROOT_KEY"] == "memory-only-root-key"
    assert result["artifact_kind"] == "qualification_result"


@pytest.mark.parametrize("runner_exit_code", (0, 1, 2))
def test_runner_exit_codes_preserve_validated_formal_result(
    tmp_path: Path,
    runner_exit_code: int,
) -> None:
    archive = tmp_path / "package.zip"
    digest = _write_package(archive)
    commands = FakeCommands(runner_exit_code)
    code, result = _run(tmp_path, archive, digest, commands)
    assert code == runner_exit_code
    assert result["runner_exit_code"] == runner_exit_code
    destination = Path(result["result_zip"])
    assert destination.is_file()
    assert destination.parent == (
        tmp_path / "persistent" / "runs" / REVISION / "run-001"
    )
    assert result["run_status"] == (
        "passed" if runner_exit_code == 0 else "failed"
    )


def test_runner_without_result_creates_runner_result_diagnostic(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "package.zip"
    digest = _write_package(archive)
    commands = FakeCommands(produce_result=False)
    code, result = _run(tmp_path, archive, digest, commands)
    assert code == 3
    diagnostic = Path(result["diagnostic_zip"])
    assert "bootstrap_failure" in diagnostic.name
    with zipfile.ZipFile(diagnostic) as artifact:
        assert artifact.namelist() == ["bootstrap_failure.json"]
        payload = json.loads(artifact.read("bootstrap_failure.json"))
    assert payload["artifact_kind"] == "bootstrap_failure"
    assert payload["bootstrap_failure_schema_version"] == 1
    assert payload["failure_stage"] == "runner_result"
    text = json.dumps(payload)
    assert "memory-only-token" not in text
    assert "memory-only-root-key" not in text


def test_runner_process_start_oserror_creates_runner_start_diagnostic(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "package.zip"
    digest = _write_package(archive)
    commands = FakeCommands(runner_os_error=True)
    code, result = _run(tmp_path, archive, digest, commands)
    assert code == 3
    assert result["failure_stage"] == "runner_start"
    with zipfile.ZipFile(result["diagnostic_zip"]) as artifact:
        payload = json.loads(artifact.read("bootstrap_failure.json"))
    assert payload["failure_stage"] == "runner_start"
    assert len(commands.calls) == 2


def test_pip_failure_creates_bootstrap_failure_without_runner(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "package.zip"
    digest = _write_package(archive)
    commands = FakeCommands(pip_failure=True)
    code, result = _run(tmp_path, archive, digest, commands)
    assert code == 3
    assert result["failure_stage"] == "dependency_install"
    assert len(commands.calls) == 1
    assert commands.calls[0][1:4] == ("-m", "pip", "install")
    assert Path(result["diagnostic_zip"]).is_file()


def test_profile_and_replay_switching_are_cli_inputs(tmp_path: Path) -> None:
    archive = tmp_path / "package.zip"
    digest = _write_package(archive)
    persistent = tmp_path / "persistent"
    replay_source = persistent / "runs" / REVISION / "source" / "source.zip"
    replay_source.parent.mkdir(parents=True)
    replay_source.write_bytes(b"source")
    commands = FakeCommands()
    code, _result = bootstrap.run_bootstrap(
        profile="replay",
        package_zip=archive,
        expected_package_sha256=digest,
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=persistent,
        replay_source=replay_source,
        run_id="replay-001",
        environment={
            "HF_TOKEN": "token",
            "CEG_WM_ROOT_KEY": "key",
        },
        command_runner=commands,
    )
    assert code == 0
    assert "--replay-source" in commands.calls[-1]


def test_bootstrap_is_not_an_execution_package_member() -> None:
    assert (
        "scripts/experiment_execution/runtime_qualification_bootstrap.py"
        not in REQUIRED_FILES
    )
    assert not any(
        path.startswith("scripts/experiment_execution/runtime_qualification_bootstrap")
        for path in REQUIRED_FILES
    )
