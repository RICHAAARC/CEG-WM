"""Static contract checks for the thin unexecuted Colab launcher."""
import json
from pathlib import Path

NOTEBOOK = Path("notebooks/baseline_v1_t2smark_colab_canary.ipynb")

def test_thin_launcher_contract() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code)
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for marker in ("https://github.com/RICHAAARC/CEG-WM.git", "'Baseline-V1'",
                   "checkout','--detach',resolved_exact", "status','--porcelain",
                   "cegwm.baselines.t2smark_canary", "RUN_ID='t2smark_sd35_one_unit_v1'",
                   "FORCE_RERUN_ALL=False", "HF_TOKEN", "cwd=checkout", "0xD009/T2SMark.git",
                   "--official-source", "OFFICIAL_EXACT=", "official_head==OFFICIAL_EXACT",
                   "symbolic-ref','-q','--short','HEAD", "official_dirty"):
        assert marker in text
    assert "force_remount" not in text and "T2SMarkCodec" not in text
