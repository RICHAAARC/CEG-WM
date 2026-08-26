from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


NOTEBOOK = Path("notebooks/geometry_v1_qk_operational_preflight_colab.ipynb")


def _source() -> tuple[dict[str, object], list[str], str]:
    value = json.loads(NOTEBOOK.read_text())
    codes = ["".join(cell["source"]) for cell in value["cells"] if cell["cell_type"] == "code"]
    return value, codes, "\n".join("".join(cell["source"]) for cell in value["cells"])


def _fixed_helper(codes: list[str]):
    module = ast.parse("\n".join(codes))
    helper = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "build_fixed_operational_rgb")
    namespace = {"np": np, "Image": Image}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "fixed_helper", "exec"), namespace)
    return namespace["build_fixed_operational_rgb"]


def test_notebook_is_complete_prepared_create_only_handoff_with_fixed_rgb_input() -> None:
    value, codes, source = _source()
    assert value["nbformat"] == 4
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in value["cells"] if cell["cell_type"] == "code")
    assert "drive.mount('/content/drive', force_remount=False)" in codes[0]
    assert "from google.colab import files" not in source and "files.upload" not in source
    assert "from google.colab import userdata" in source
    assert "--single-branch','--branch',BRANCH" in source
    assert "checkout_commit=checkout_identity()" in source
    assert "--detach" not in source and "EXECUTION_EXACT" not in source
    assert "Geometry-V1-'+checkout_commit[:12]+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')" in source
    assert "PREPARED_NOT_EXECUTED" in source and "try: pass" not in source
    assert "pip','install'" in source
    assert source.index("try:") < source.index("input_dir.mkdir()") < source.index("userdata.get('HF_TOKEN')")
    assert "CEGWM_GEOMETRY_V1_OPERATIONAL_PREFLIGHT " in source and "CEGWM_GEOMETRY_V1_OPERATIONAL_FAILURE " in source
    assert "control/return-code mismatch" in source and "artifact-unavailable" in source
    assert "pass_fds=(control_write,)" in source and "stdout=subprocess.DEVNULL" in source and "stderr=subprocess.DEVNULL" in source
    assert "MAX_CONTROL_BYTES+1" in source and "receipt_sha256" in source and "sidecar_value" in source
    assert "--output-root',str(run_dir)" in source and "exclusive_copy" not in source
    assert "copy2" not in source and "write_bytes" not in source and "content_v" not in source and "Content" not in source
    assert "process.kill" in source and "runner_env.pop('HF_TOKEN'" in source and "input_dir.rmdir" in source and "shutil.rmtree(run_dir)" not in source
    assert "geometry_v1_batch2b_fixed_rgb.png" in source and "fixed_path.open('xb')" in source
    assert "str(fixed_path)" in source and "input_paths.append(fixed_path)" in source
    assert source.index("input_paths.append(fixed_path)") < source.index("fixed_path.open('xb')")
    assert "fixed_image=None; fixed_array=None" in source
    assert "zipfile" not in source and "archive.read_bytes" not in source
    assert "re.fullmatch(r'[0-9a-f]{64}',payload['receipt_sha256'])" in source
    assert "payload['archive_filename']!=run_id+'.zip'" in source
    assert "payload['sidecar_filename']!=run_id+'.zip.sha256'" in source
    assert "Artifact-only inspection" in codes[-1] and "Popen" not in codes[-1]
    assert "checkpoint=run_dir/'checkpoint.json'" in codes[-1]


def test_fixed_operational_rgb_is_exact_deterministic_non_degenerate_input() -> None:
    _, codes, source = _source()
    helper = _fixed_helper(codes)
    first, second = helper(), helper()
    first_array, second_array = np.asarray(first), np.asarray(second)
    assert first.mode == second.mode == "RGB"
    assert first.size == second.size == (512, 512)
    assert first_array.shape == (512, 512, 3) and first_array.dtype == np.uint8
    assert first_array.tobytes() == second_array.tobytes()
    assert hashlib.sha256(first_array.tobytes()).hexdigest() == "46d0b1bc4a7ff14709db1513f501ca5a64b6b7ec59670d72e372e14fd321f122"
    assert np.array_equal(first_array[40, 55], (241, 67, 31))
    assert np.array_equal(first_array[300, 335], (19, 181, 223))
    assert not np.array_equal(first_array[0, 0], first_array[511, 511])
    assert len(np.unique(first_array.reshape(-1, 3), axis=0)) > 100
    helper_source = ast.unparse(next(node for node in ast.parse("\n".join(codes)).body if isinstance(node, ast.FunctionDef) and node.name == "build_fixed_operational_rgb")).lower()
    assert "image.fromarray(array)" in helper_source
    assert "mode=" not in helper_source
    for forbidden in ("random", "seed", "http", "prompt", "content", "private", "upload"):
        assert forbidden not in helper_source
