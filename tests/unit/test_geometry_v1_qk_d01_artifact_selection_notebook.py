"""Narrow static contract for the D0.1 Drive-only Colab handoff."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "geometry_v1_qk_d01_artifact_selection_colab.ipynb"


def test_d01_notebook_binds_source_artifact_and_runner_checkout_separately() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8")); cells = notebook["cells"]
    assert notebook["nbformat"] == 4 and cells[0]["cell_type"] == "code"
    assert cells[0]["source"] == ["from google.colab import drive\n", "drive.mount('/content/drive')\n"]
    assert all(cell.get("execution_count") is None and not cell.get("outputs", []) for cell in cells if cell["cell_type"] == "code")
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)

    assert "SOURCE_ARTIFACT_EXACT = '4732211beefbeface95cb842c117b9719e362f1a'" in source
    assert "D01_RUNNER_EXACT = 'ccfb7bcefbb18f9812a4e800bbea18b91b031ebb'" in source
    assert "SOURCE_RUN_ID = 'geometry-v1-qk-d0-4732211beefb'" in source
    assert "SOURCE_PROTOCOL = 'geometry-v1-qk-d0-all-layer-discovery-v1'" in source
    assert "SOURCE_PLAN_DIGEST = '96e1e5ae6fb8ae66a545b1b10d6c896176989272c81ef1fd737184dcdfaea7b8'" in source
    assert "/content/drive/MyDrive/CEG-WM/Geometry-V1/D0/Geometry-V1-QK-D0-4732211beefb-20260827T064555Z" in source
    assert "'execution_exact': SOURCE_ARTIFACT_EXACT" in source
    assert "'run_id': SOURCE_RUN_ID" in source and "'protocol': SOURCE_PROTOCOL" in source
    assert "'plan_digest': SOURCE_PLAN_DIGEST" in source and "'path': str(SOURCE_ROOT)" in source

    assert "subprocess.run(['git', 'clone', '--no-checkout', REPO_URL, str(repo)], check=True)" in source
    assert "subprocess.run(['git', 'checkout', '--detach', D01_RUNNER_EXACT], cwd=repo, check=True)" in source
    assert "execution_commit != D01_RUNNER_EXACT or not checkout_clean" in source
    assert "SOURCE_ARTIFACT_EXACT" not in source.split("subprocess.run(['git', 'checkout', '--detach',", 1)[1].split("\n", 1)[0]
    assert "RUNNER_PATH = 'experiments/run_geometry_v1_qk_d01_artifact_selection_operational.py'" in source
    assert "runner_path = repo / RUNNER_PATH" in source and "str(runner_path)" in source
    assert "'--repo-root', str(repo)" in source and "'--expected-exact', execution_commit" in source
    assert "'--source-root', str(SOURCE_ROOT)" in source and "'--output-root', str(run_dir)" in source
    assert "'--control-fd', str(control_write)" in source and "/content/drive/MyDrive/CEG-WM/Geometry-V1/D01" in source
    assert "'runner_execution_identity': {'commit': execution_commit}" in source
    assert "'source_artifact_identity': source_artifact_identity" in source

    assert source.count("subprocess.Popen(") == 1 and "RUNNER_ATTEMPTED" in source and "timeout=7200" in source
    assert "MAX_CONTROL_BYTES = 1024" in source and "stderr=subprocess.DEVNULL" in source
    for prohibited in ("HF_TOKEN", "diffusers", "torch", "cuda", "zipfile", "read_bytes", "hashlib", "google.colab.secrets"):
        assert prohibited not in source
    assert "does not display source ZIP contents, retry, fall back, switch layers, tune, or choose per sample" in source
