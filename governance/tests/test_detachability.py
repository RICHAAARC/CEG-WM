from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.constraint
def test_research_tree_runs_after_outer_guard_is_removed(tmp_path: Path) -> None:
    source = Path.cwd()
    detached = tmp_path / "detached"
    detached.mkdir()
    for root_name in ("src", "experiments", "configs", "tests", "notebooks", "docs"):
        candidate = source / root_name
        if candidate.exists():
            shutil.copytree(candidate, detached / root_name)
    for file_name in ("README.md", "pyproject.toml"):
        shutil.copy2(source / file_name, detached / file_name)

    assert not (detached / "governance").exists()
    assert not (detached / ".codex").exists()
    assert not (detached / ".agents").exists()

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(detached / "src")
    imported = subprocess.run(
        [sys.executable, "-c", "import cegwm, cegwm.protocol"],
        cwd=detached,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr

    tested = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=detached,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tested.returncode == 0, tested.stdout + tested.stderr
