"""验证分层测试入口不会重新混合治理与方法 pytest。"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from governance.tools.run_validation_profile import (
    GOVERNANCE_PARALLEL_TEST_PATHS,
    PARALLEL_WORKER_COUNT,
    PROJECT_PARALLEL_TEST_PATHS,
    VALIDATION_ENVIRONMENT_OVERRIDES,
    commands_for_profile,
    run_profile,
)


def _parallel_command(
    paths: tuple[str, ...],
    *,
    configuration: str | None = None,
) -> tuple[str, ...]:
    prefix = ("python", "-m", "pytest", "-q", "-s", "-x")
    if configuration is not None:
        prefix = (*prefix, "-c", configuration)
    return (
        *prefix,
        "-n",
        str(PARALLEL_WORKER_COUNT),
        "--dist=loadfile",
        *paths,
    )


def _serial_command(
    paths: tuple[str, ...],
    *,
    test_root: str,
    configuration: str | None = None,
) -> tuple[str, ...]:
    prefix = ("python", "-m", "pytest", "-q", "-s", "-x")
    if configuration is not None:
        prefix = (*prefix, "-c", configuration)
    return (*prefix, *(f"--ignore={path}" for path in paths), test_root)


@pytest.mark.unit
def test_governance_profile_excludes_project_pytest() -> None:
    commands = commands_for_profile("governance", "python")

    assert commands == (
        _parallel_command(
            GOVERNANCE_PARALLEL_TEST_PATHS,
            configuration="governance/pytest.ini",
        ),
        _serial_command(
            GOVERNANCE_PARALLEL_TEST_PATHS,
            test_root="governance/tests",
            configuration="governance/pytest.ini",
        ),
        ("python", "governance/harness/run_all_audits.py"),
    )


@pytest.mark.unit
def test_method_profile_excludes_governance_pytest() -> None:
    commands = commands_for_profile("method", "python")

    assert commands == (
        _parallel_command(PROJECT_PARALLEL_TEST_PATHS),
        _serial_command(PROJECT_PARALLEL_TEST_PATHS, test_root="tests"),
        ("python", "governance/harness/run_all_audits.py"),
    )


@pytest.mark.unit
def test_full_profile_runs_both_pytest_suites_before_harness() -> None:
    commands = commands_for_profile("full", "python")

    assert commands == (
        _parallel_command(PROJECT_PARALLEL_TEST_PATHS),
        _serial_command(PROJECT_PARALLEL_TEST_PATHS, test_root="tests"),
        _parallel_command(
            GOVERNANCE_PARALLEL_TEST_PATHS,
            configuration="governance/pytest.ini",
        ),
        _serial_command(
            GOVERNANCE_PARALLEL_TEST_PATHS,
            test_root="governance/tests",
            configuration="governance/pytest.ini",
        ),
        ("python", "governance/harness/run_all_audits.py"),
    )


@pytest.mark.unit
def test_parallel_allowlists_are_positive_existing_file_sets() -> None:
    assert len(PROJECT_PARALLEL_TEST_PATHS) == 28
    assert len(GOVERNANCE_PARALLEL_TEST_PATHS) == 6
    assert len(set(PROJECT_PARALLEL_TEST_PATHS)) == 28
    assert len(set(GOVERNANCE_PARALLEL_TEST_PATHS)) == 6
    assert all(Path(path).is_file() for path in PROJECT_PARALLEL_TEST_PATHS)
    assert all(Path(path).is_file() for path in GOVERNANCE_PARALLEL_TEST_PATHS)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("parallel_paths", "test_root", "configuration"),
    (
        (PROJECT_PARALLEL_TEST_PATHS, "tests", None),
        (
            GOVERNANCE_PARALLEL_TEST_PATHS,
            "governance/tests",
            "governance/pytest.ini",
        ),
    ),
)
def test_serial_shard_collects_the_directory_complement_without_repetition(
    parallel_paths: tuple[str, ...],
    test_root: str,
    configuration: str | None,
) -> None:
    profile = "method" if test_root == "tests" else "governance"
    parallel, serial = commands_for_profile(profile, "python")[:2]

    assert parallel == _parallel_command(
        parallel_paths,
        configuration=configuration,
    )
    assert serial == _serial_command(
        parallel_paths,
        test_root=test_root,
        configuration=configuration,
    )
    assert set(path for path in parallel if path.endswith(".py")) == set(
        parallel_paths
    )
    assert set(
        argument.removeprefix("--ignore=")
        for argument in serial
        if argument.startswith("--ignore=")
    ) == set(parallel_paths)
    assert test_root in serial
    assert "-n" not in serial
    assert "--dist=loadfile" not in serial
    assert "-x" in parallel and "-x" in serial


@pytest.mark.unit
def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown validation profile"):
        commands_for_profile("combined", "python")


@pytest.mark.unit
def test_run_profile_propagates_first_failure_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fail_first(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert cwd == tmp_path
        assert check is False
        assert all(
            env[key] == value
            for key, value in VALIDATION_ENVIRONMENT_OVERRIDES.items()
        )
        calls.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", fail_first)

    assert run_profile("full", tmp_path) == 7
    assert calls == [commands_for_profile("full")[0]]


@pytest.mark.unit
def test_run_profile_stops_before_harness_after_second_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    return_codes = iter((0, 9))

    def fail_second(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert cwd == tmp_path
        assert check is False
        assert all(
            env[key] == value
            for key, value in VALIDATION_ENVIRONMENT_OVERRIDES.items()
        )
        calls.append(command)
        return subprocess.CompletedProcess(command, next(return_codes))

    monkeypatch.setattr(subprocess, "run", fail_second)

    full_commands = commands_for_profile("full")
    assert run_profile("full", tmp_path) == 9
    assert calls == list(full_commands[:2])


@pytest.mark.unit
def test_run_profile_runs_harness_only_after_all_selected_pytest_shards_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def pass_command(
        command: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert cwd == tmp_path
        assert check is False
        assert all(
            env[key] == value
            for key, value in VALIDATION_ENVIRONMENT_OVERRIDES.items()
        )
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", pass_command)

    full_commands = commands_for_profile("full")
    assert run_profile("full", tmp_path) == 0
    assert calls == list(full_commands)
    assert calls[-1][1:] == ("governance/harness/run_all_audits.py",)
