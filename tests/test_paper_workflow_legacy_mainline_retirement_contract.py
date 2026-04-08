"""
File purpose: Guard active paper_workflow files against legacy mainline references.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
_TEXT_FILE_SUFFIXES = {".py", ".ipynb", ".yaml", ".yml", ".json", ".md"}
_LEGACY_MARKERS = [
    "notebook/00_main",
    "01_Paper_Full_Cuda.ipynb",
    "01_Paper_Full_Cuda_Parallel.ipynb",
    "02_Parallel_Attestation_Statistics.ipynb",
    "03_Experiment_Matrix_Full.ipynb",
    "04_Release_And_Signoff.ipynb",
    "scripts/01_Paper_Full_Cuda.py",
    "scripts/01_Paper_Full_Cuda_Parallel.py",
    "scripts/01_run_paper_full_cuda.py",
    "scripts/01_run_paper_full_cuda_parallel.py",
    "scripts/01_run_paper_full_cuda_parallel_worker.py",
    "scripts/02_Parallel_Attestation_Statistics.py",
    "scripts/03_Experiment_Matrix_Full.py",
    "scripts/04_Release_And_Signoff.py",
    "01_Paper_Full_Cuda_mainline",
    "01_Paper_Full_Cuda_Parallel_mainline",
    "01_Paper_Full_Cuda_Parallel",
]


def _iter_active_text_files() -> List[Path]:
    """
    Collect active text files that must stay detached from the legacy mainline.

    Args:
        None.

    Returns:
        Sorted active file paths.
    """
    active_files: List[Path] = []
    for root in [REPO_ROOT / "paper_workflow", REPO_ROOT / "tests"]:
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix not in _TEXT_FILE_SUFFIXES:
                continue
            if file_path == Path(__file__).resolve():
                continue
            active_files.append(file_path)
    return active_files


def test_active_paths_detach_from_legacy_mainline() -> None:
    """
    Verify active paper_workflow and active tests do not reference retired mainline files.

    Args:
        None.

    Returns:
        None.
    """
    violations: List[dict[str, str]] = []
    for file_path in _iter_active_text_files():
        file_text = file_path.read_text(encoding="utf-8")
        for marker in _LEGACY_MARKERS:
            if marker in file_text:
                violations.append(
                    {
                        "file": file_path.relative_to(REPO_ROOT).as_posix(),
                        "marker": marker,
                    }
                )

    assert not violations, json.dumps(violations, ensure_ascii=False, indent=2)