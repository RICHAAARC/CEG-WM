"""验证分层测试入口不会重新混合治理与方法 pytest。"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from governance.tools.run_validation_profile import commands_for_profile, run_profile


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
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert cwd == tmp_path
        assert check is False
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
    ) -> subprocess.CompletedProcess[tuple[str, ...]]:
        assert cwd == tmp_path
        assert check is False
        calls.append(command)
        return subprocess.CompletedProcess(command, next(return_codes))

    monkeypatch.setattr(subprocess, "run", fail_second)

    full_commands = commands_for_profile("full")
    assert run_profile("full", tmp_path) == 9
    assert calls == list(full_commands[:2])
