"""验证分层测试入口不会重新混合治理与方法 pytest。"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

import governance.tools.run_validation_profile as validation_profile
from governance.tools.run_validation_profile import (
    VALIDATION_ENVIRONMENT_OVERRIDES,
    VALIDATION_TEMPORARY_DIRECTORY_DEFAULTS,
    commands_for_profile,
    run_profile,
)


@pytest.mark.unit
def test_governance_profile_excludes_project_pytest() -> None:
    commands = commands_for_profile("governance", "python")

    assert commands == (
        (
            "python",
            "-m",
            "pytest",
            "-q",
            "-s",
            "-c",
            "governance/pytest.ini",
            "governance/tests",
        ),
        ("python", "governance/harness/run_all_audits.py"),
    )


@pytest.mark.unit
def test_method_profile_excludes_governance_pytest() -> None:
    commands = commands_for_profile("method", "python")

    assert commands == (
        ("python", "-m", "pytest", "-q", "-s", "tests"),
        ("python", "governance/harness/run_all_audits.py"),
    )


@pytest.mark.unit
def test_full_profile_runs_both_pytest_suites_before_harness() -> None:
    commands = commands_for_profile("full", "python")

    assert commands == (
        ("python", "-m", "pytest", "-q", "-s", "tests"),
        (
            "python",
            "-m",
            "pytest",
            "-q",
            "-s",
            "-c",
            "governance/pytest.ini",
            "governance/tests",
        ),
        ("python", "governance/harness/run_all_audits.py"),
    )


@pytest.mark.unit
def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown validation profile"):
        commands_for_profile("combined", "python")


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing_variable_name",
    (None, "TMPDIR", "TMP", "TEMP"),
    ids=("all_supplied", "tmpdir_missing", "tmp_missing", "temp_missing"),
)
def test_full_profile_preserves_present_temporary_directories_and_defaults_each_missing_variable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_variable_name: str | None,
) -> None:
    supplied_directories = {
        "TMPDIR": str(tmp_path / "tmpdir_supplied"),
        "TMP": str(tmp_path / "tmp_supplied"),
        "TEMP": str(tmp_path / "temp_supplied"),
    }
    for variable_name, directory in supplied_directories.items():
        monkeypatch.setenv(variable_name, directory)
    if missing_variable_name is not None:
        monkeypatch.delenv(missing_variable_name)
    observed_environments: list[dict[str, str]] = []

    def pass_command(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert cwd == tmp_path
        assert check is False
        observed_environments.append(dict(env))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", pass_command)

    assert run_profile("full", tmp_path) == 0
    assert len(observed_environments) == len(commands_for_profile("full"))
    for environment in observed_environments:
        for variable_name, supplied_directory in supplied_directories.items():
            expected_directory = (
                VALIDATION_TEMPORARY_DIRECTORY_DEFAULTS[variable_name]
                if variable_name == missing_variable_name
                else supplied_directory
            )
            assert environment[variable_name] == expected_directory
        assert {
            key: environment[key] for key in VALIDATION_ENVIRONMENT_OVERRIDES
        } == VALIDATION_ENVIRONMENT_OVERRIDES


@pytest.mark.unit
def test_run_profile_propagates_first_failure_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, ...]] = []
    observed_environment: dict[str, str] = {}
    for variable_name in VALIDATION_TEMPORARY_DIRECTORY_DEFAULTS:
        monkeypatch.delenv(variable_name, raising=False)

    def fail_first(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert cwd == tmp_path
        assert check is False
        calls.append(command)
        observed_environment.update(env)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", fail_first)
    elapsed = iter((10.0, 12.5))
    monkeypatch.setattr(validation_profile, "perf_counter", lambda: next(elapsed))

    assert run_profile("full", tmp_path) == 7
    assert calls == [commands_for_profile("full")[0]]
    assert {
        key: observed_environment[key]
        for key in VALIDATION_ENVIRONMENT_OVERRIDES
    } == VALIDATION_ENVIRONMENT_OVERRIDES
    assert {
        key: observed_environment[key]
        for key in VALIDATION_TEMPORARY_DIRECTORY_DEFAULTS
    } == VALIDATION_TEMPORARY_DIRECTORY_DEFAULTS
    output = capsys.readouterr().out
    assert "command_identity=project_pytest" in output
    assert "walltime_seconds=2.500 returncode=7" in output


@pytest.mark.unit
def test_run_profile_stops_before_harness_after_second_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    return_codes = iter((0, 9))
    for variable_name in VALIDATION_TEMPORARY_DIRECTORY_DEFAULTS:
        monkeypatch.delenv(variable_name, raising=False)

    def fail_second(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert cwd == tmp_path
        assert check is False
        assert {
            key: env[key] for key in VALIDATION_ENVIRONMENT_OVERRIDES
        } == VALIDATION_ENVIRONMENT_OVERRIDES
        assert {
            key: env[key] for key in VALIDATION_TEMPORARY_DIRECTORY_DEFAULTS
        } == VALIDATION_TEMPORARY_DIRECTORY_DEFAULTS
        calls.append(command)
        return subprocess.CompletedProcess(command, next(return_codes))

    monkeypatch.setattr(subprocess, "run", fail_second)

    full_commands = commands_for_profile("full")
    assert run_profile("full", tmp_path) == 9
    assert calls == list(full_commands[:2])
