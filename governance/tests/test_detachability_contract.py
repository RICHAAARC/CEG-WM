"""验证外层护栏可整体移除而不破坏研究项目。"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from governance.harness.lib.project_policy import load_root_policy

OUTER_KINDS = {"project_skills", "governance_guidance", "control_plane"}
LOCAL_OR_GENERATED_KINDS = {
    "local_environment",
    "version_control",
    "generated_cache",
    "editor_state",
    "generated_asset",
}
RESEARCH_KINDS = {
    "governed_configuration",
    "governed_documentation",
    "governed_code",
    "entrypoint",
    "derived_artifact_code",
    "support_code",
    "framework_asset",
    "external_dependency",
    "test_code",
}
DETACHED_SMOKE_NODE_IDS = (
    "tests/unit/test_key_schedule.py::test_key_schedule_root_and_domain_separation",
    "tests/unit/test_runtime_configuration_and_adapter.py::test_mock_backend_initialization_preserves_frozen_identity",
    "tests/unit/test_internal_scientific_validation_protocol.py::test_internal_record_contains_all_scientific_trace_groups",
    "tests/unit/test_internal_governed_runner.py::test_runner_composes_real_adapter_attack_and_metric_replay_once",
    "tests/functional/test_lf_null_whitened_detector.py::test_lf_whitened_candidate_crosses_real_public_adapter_without_raw_fallback",
    "tests/unit/test_lf_whitened_score_screening_delivery.py::test_lf_whitened_screening_server_help_imports_from_isolated_cwd",
    "tests/unit/test_experiment_execution_delivery.py::test_builder_path_scanners_preserve_behavior_without_source_local_paths",
    "tests/functional/test_governed_artifact_structures.py::test_artifact_manifest_records_rebuild_provenance",
)


def _collected_project_node_ids(result: subprocess.CompletedProcess[str]) -> tuple[str, ...]:
    assert result.returncode == 0, result.stdout + result.stderr
    node_ids = []
    for output_line in result.stdout.splitlines():
        stripped_line = output_line.strip()
        if not stripped_line.startswith("tests/") or "::" not in stripped_line:
            continue
        relative_path, separator, node_suffix = stripped_line.partition("::")
        normalized_relative_path = relative_path.replace("\\", "/")
        node_ids.append(f"{normalized_relative_path}{separator}{node_suffix}")
    assert node_ids
    assert len(node_ids) == len(set(node_ids))
    return tuple(sorted(node_ids))


@pytest.mark.constraint
def test_research_project_runs_after_outer_guard_is_removed(tmp_path: Path) -> None:
    source_root = Path.cwd()
    detached_root = tmp_path / "research_project"
    detached_root.mkdir()
    root_policy = load_root_policy(source_root)
    root_registry = root_policy["root_registry"]
    registered_kinds = {metadata["kind"] for metadata in root_registry.values()}
    assert registered_kinds == OUTER_KINDS | LOCAL_OR_GENERATED_KINDS | RESEARCH_KINDS
    research_roots = {
        root_name for root_name, metadata in root_registry.items() if metadata["kind"] in RESEARCH_KINDS
    }
    classified_roots = {
        root_name
        for root_name, metadata in root_registry.items()
        if metadata["kind"] in OUTER_KINDS | LOCAL_OR_GENERATED_KINDS | RESEARCH_KINDS
    }
    assert classified_roots == set(root_registry)
    assert "third_party" in research_roots

    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
    for root_name in sorted(research_roots):
        source = source_root / root_name
        if source.exists():
            shutil.copytree(source, detached_root / root_name, ignore=ignore)
    for file_name in (
        "pyproject.toml",
        "requirements_hf_only_threshold_fit_gpu_execution.txt",
    ):
        shutil.copy2(source_root / file_name, detached_root / file_name)
    assert (detached_root / "requirements_hf_only_threshold_fit_gpu_execution.txt").is_file()

    assert not (detached_root / ".agents").exists()
    assert not (detached_root / ".codex").exists()
    assert not (detached_root / "governance").exists()
    assert not (detached_root / "docs" / "governance").exists()
    if (source_root / "third_party").exists():
        assert (detached_root / "third_party").exists()

    broken_links = []
    for document_path in (detached_root / "docs").rglob("*.md"):
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document_path.read_text(encoding="utf-8")):
            local_target = target.split("#", 1)[0].strip()
            if not local_target or local_target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (document_path.parent / local_target).resolve().exists():
                broken_links.append((document_path.relative_to(detached_root).as_posix(), target))
    assert broken_links == []

    import_result = subprocess.run(
        [sys.executable, "-c", "import main, runtime, experiments.protocol, paper_artifacts"],
        cwd=detached_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr

    artifact_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from paper_artifacts.digest import build_stable_digest; "
                "first = build_stable_digest({'sample': 1}); "
                "assert first == build_stable_digest({'sample': 1}) and len(first) == 64"
            ),
        ],
        cwd=detached_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert artifact_result.returncode == 0, artifact_result.stderr

    collect_command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
    ]
    source_collection = subprocess.run(
        collect_command,
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    detached_collection = subprocess.run(
        collect_command,
        cwd=detached_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert _collected_project_node_ids(detached_collection) == _collected_project_node_ids(
        source_collection
    )

    smoke_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            *DETACHED_SMOKE_NODE_IDS,
        ],
        cwd=detached_root,
        check=False,
        capture_output=True,
        text=True,
    )
    smoke_output = smoke_result.stdout + smoke_result.stderr
    assert smoke_result.returncode == 0, smoke_output
    assert re.search(r"\b8 passed\b", smoke_output)
    assert not re.search(r"\b(?:skipped|xfailed|xpassed)\b", smoke_output)
