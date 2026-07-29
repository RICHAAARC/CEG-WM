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
        "requirements_runtime_qualification.txt",
    ):
        shutil.copy2(source_root / file_name, detached_root / file_name)
    assert (
        detached_root / "requirements_runtime_qualification.txt"
    ).is_file()

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

    test_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "-p",
            "no:cacheprovider",
        ],
        cwd=detached_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert test_result.returncode == 0, test_result.stdout + test_result.stderr
