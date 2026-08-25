import json
from pathlib import Path


def test_notebook_is_prepared_and_bound_to_k() -> None:
    value = json.loads(Path("notebooks/geometry_v1_qk_operational_preflight_colab.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in value["cells"])
    assert value["nbformat"] == 4
    assert "6af140b770a5d065d8af54c4042448854056ee87" in source
    assert "PREPARED_NOT_EXECUTED" in source
    assert "force_remount=False" in source
    assert "CEG_WM_ROOT_KEY" in source and "process.kill" in source
