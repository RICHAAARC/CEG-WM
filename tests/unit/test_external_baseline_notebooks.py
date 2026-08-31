import json
from pathlib import Path


def test_external_baseline_notebooks_are_thin_and_pinned() -> None:
    exacts = {"tree_ring": "3015283d9cf82e90b628f02ad2121bd37408ca9a", "gaussian_shading": "09c678fadc7545acf7be12647ddf2a5e66f6a9dc", "shallow_diffuse": "c80c553fdf66fda8db735d77a9d56538b7a0ade8"}
    for method, exact in exacts.items():
        book = json.loads(Path(f"notebooks/baseline_v1_{method}_colab_canary.ipynb").read_text())
        code = [c for c in book["cells"] if c["cell_type"] == "code"]
        assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
        text = "\n".join("".join(c["source"]) for c in code)
        for required in ("Baseline-V1", exact, "--detach", "status", "porcelain", "userdata", "child_env", "sys.executable", "pip','install','-e", "FORCE_RERUN_ALL", "--force-rerun-all", "HF_TOKEN", "RUN_ID", "final_manifest.json", "engineering-only"):
            assert required in text
        assert "force_remount" not in text
