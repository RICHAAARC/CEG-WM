"""按变更范围运行 CEG-WM 的登记验证档位。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import shlex
import subprocess
import sys

PROFILE_NAMES = ("governance", "method", "full")
PARALLEL_WORKER_COUNT = 4
PROJECT_PARALLEL_TEST_PATHS = (
    "tests/functional/test_governed_artifact_structures.py",
    "tests/functional/test_lf_null_whitened_detector.py",
    "tests/unit/test_comparison_preflight.py",
    "tests/unit/test_development_exploration_metrics.py",
    "tests/unit/test_development_inputs.py",
    "tests/unit/test_development_module_exploration.py",
    "tests/unit/test_development_worker_persistence.py",
    "tests/unit/test_geometry_chain.py",
    "tests/unit/test_hf_content_backbone.py",
    "tests/unit/test_hf_only_detector_directional_validation.py",
    "tests/unit/test_hf_only_reference_metrics.py",
    "tests/unit/test_hf_only_reference_protocol.py",
    "tests/unit/test_hf_only_threshold_fit_gpu_execution.py",
    "tests/unit/test_hf_transmission_diagnostic.py",
    "tests/unit/test_internal_experiment_components.py",
    "tests/unit/test_internal_governed_runner.py",
    "tests/unit/test_internal_scientific_validation_protocol.py",
    "tests/unit/test_joint_decision.py",
    "tests/unit/test_key_schedule.py",
    "tests/unit/test_lf_null_whitened_detector.py",
    "tests/unit/test_lf_routing_combination.py",
    "tests/unit/test_lf_transmission_diagnostic.py",
    "tests/unit/test_lf_whitened_score_screening.py",
    "tests/unit/test_runtime_configuration_and_adapter.py",
    "tests/unit/test_runtime_content_write_and_vae.py",
    "tests/unit/test_runtime_qk_observation.py",
    "tests/unit/test_runtime_qualification_bootstrap.py",
    "tests/unit/test_runtime_routing_observation.py",
)
GOVERNANCE_PARALLEL_TEST_PATHS = (
    "governance/tests/test_extended_naming_audit.py",
    "governance/tests/test_governance_policy.py",
    "governance/tests/test_harness_registry.py",
    "governance/tests/test_naming_and_field_rules.py",
    "governance/tests/test_notebook_governance.py",
    "governance/tests/test_validation_profiles.py",
)
VALIDATION_ENVIRONMENT_OVERRIDES = {
    "TMPDIR": "/tmp",
    "TEMP": "/tmp",
    "TMP": "/tmp",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def _pytest_commands(
    python_executable: str,
    *,
    configuration: str | None,
    test_root: str,
    parallel_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Build one bounded parallel shard and its directory-selected complement."""
    prefix = (
        python_executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        "-x",
    )
    if configuration is not None:
        prefix = (*prefix, "-c", configuration)
    parallel = (
        *prefix,
        "-n",
        str(PARALLEL_WORKER_COUNT),
        "--dist=loadfile",
        *parallel_paths,
    )
    serial = (
        *prefix,
        *(f"--ignore={path}" for path in parallel_paths),
        test_root,
    )
    return parallel, serial


def commands_for_profile(
    profile: str,
    python_executable: str = sys.executable,
) -> tuple[tuple[str, ...], ...]:
    """返回指定验证档位的顺序命令。"""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown validation profile: {profile}")

    project_parallel, project_serial = _pytest_commands(
        python_executable,
        configuration=None,
        test_root="tests",
        parallel_paths=PROJECT_PARALLEL_TEST_PATHS,
    )
    governance_parallel, governance_serial = _pytest_commands(
        python_executable,
        configuration="governance/pytest.ini",
        test_root="governance/tests",
        parallel_paths=GOVERNANCE_PARALLEL_TEST_PATHS,
    )
    harness = (
        python_executable,
        "governance/harness/run_all_audits.py",
    )

    if profile == "governance":
        return governance_parallel, governance_serial, harness
    if profile == "method":
        return project_parallel, project_serial, harness
    return (
        project_parallel,
        project_serial,
        governance_parallel,
        governance_serial,
        harness,
    )


def run_profile(
    profile: str,
    repository_root: Path,
    *,
    dry_run: bool = False,
) -> int:
    """从仓库根目录运行档位；首个失败立即停止。"""
    commands = commands_for_profile(profile)
    environment = os.environ.copy()
    environment.update(VALIDATION_ENVIRONMENT_OVERRIDES)
    for command in commands:
        print(f"[{profile}] {shlex.join(command)}", flush=True)
        if dry_run:
            continue
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="运行 CEG-WM governance、method 或 full 验证档位。",
    )
    parser.add_argument("profile", choices=PROFILE_NAMES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将运行的命令。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并从项目根目录运行验证。"""
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    return run_profile(args.profile, repository_root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
