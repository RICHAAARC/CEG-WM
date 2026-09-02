from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from experiments import run_geometry_v7_r0 as runner
from cegwm.protocol.content_chain import load_content_chain_contract


_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_real_blind_score_adapter_preserves_all_registered_and_wrong_key_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_content_chain_contract(_REPO_ROOT)
    final_rgb = Image.new("RGB", (512, 512), "gray")
    seen: list[object] = []

    def raw_scores(image, detection_key, wrong_keys, assets, calibration_asset):
        seen.extend((image, detection_key, wrong_keys, assets, calibration_asset))
        return {
            branch: {
                "registered": float(branch_index + 1),
                **{
                    f"wrong_{index:02d}": float(branch_index + 1) - 0.1 - index
                    for index in range(16)
                },
            }
            for branch_index, branch in enumerate(("lf", "hf", "weighted_joint"))
        }

    monkeypatch.setattr(runner, "blind_weighted_scores", raw_scores)
    assets = object()
    detection_key = b"k" * 32
    wrong_keys = tuple(bytes([index]) * 32 for index in range(16))
    score = runner._content_scorer(
        detection_key=detection_key,
        wrong_keys=wrong_keys,
        assets=assets,
        contract=contract,
    )(final_rgb)

    assert seen == [
        final_rgb,
        detection_key,
        wrong_keys,
        assets,
        contract.calibration_asset,
    ]
    assert (score.lf, score.hf, score.weighted_joint) == (1.0, 2.0, 3.0)
    assert len(score.wrong_key_lf) == len(score.wrong_key_hf) == 16
    assert len(score.wrong_key_weighted_joint) == 16
    assert score.gate_a_margin == pytest.approx(0.1)


@pytest.mark.integration
def test_global_real_setup_failure_projects_complete_fixed_development_grid(
    tmp_path: Path,
) -> None:
    contract = load_content_chain_contract(_REPO_ROOT)
    result = runner._setup_failure_result(
        repo_root=_REPO_ROOT,
        exact="2" * 40,
        key_digest="3" * 64,
        contract=contract,
        result_root=tmp_path,
        failure_stage="content_runtime_setup",
        error=RuntimeError("model load stopped"),
    )
    assert result["status"] == "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR"
    assert len(result["development_aggregates"]) == 4
    assert len(result["raw_unit_records"]) == 4 * 4
    assert result["image_files"] == []
    assert result["evaluation_aggregate"] is None
    assert result["selection"]["complete"] is True
    assert result["selection"]["selected_residual_strength_multiplier"] is None
    assert len(result["failures"]) == 4 * 4 * 4
    assert "model load stopped" not in json.dumps(result)
    for record in result["raw_unit_records"]:
        assert record["failure_arm_denominator"] == 4
        assert record["failed_arm_count"] == 4
        assert all(arm["image_file"] is None for arm in record["arms"])
        assert all(arm["errors"] for arm in record["arms"])


@pytest.mark.integration
def test_result_json_and_sidecar_are_create_only(tmp_path: Path) -> None:
    result = {
        "schema": runner.RESULT_SCHEMA,
        "status": "OPERATIONAL_FAILURE_RETAINED_FIXED_DENOMINATOR",
    }
    runner._write_result(tmp_path, result)
    payload = (tmp_path / "result.json").read_bytes()
    assert payload.endswith(b"\n")
    assert (tmp_path / "result.json.sha256").read_text(encoding="ascii").endswith(
        "  result.json\n"
    )
    with pytest.raises(FileExistsError):
        runner._write_result(tmp_path, result)


@pytest.mark.integration
def test_execute_removes_unpublished_partial_package_on_terminal_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "result"
    args = SimpleNamespace(result_dir=str(result_root))

    def fake_run(_args):
        result_root.mkdir()
        (result_root / "partial").write_text("unpublished", encoding="utf-8")
        return {"status": "PAIRED_COMPATIBILITY_CANARY_FAILED"}

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(
        runner,
        "_write_result",
        lambda *_args: (_ for _ in ()).throw(OSError("sidecar stopped")),
    )
    with pytest.raises(OSError, match="sidecar stopped"):
        runner.execute(args)
    assert not result_root.exists()


@pytest.mark.integration
def test_runner_source_binds_real_routes_and_exact_eval_sync_semantics() -> None:
    source = inspect.getsource(runner)
    ast.parse(source)
    scorer_source = inspect.getsource(runner._content_scorer)
    assert "blind_weighted_scores(" in scorer_source
    assert all(
        forbidden not in scorer_source
        for forbidden in ("prompt", "measurement", "original", "latent", "truth")
    )
    producer_source = inspect.getsource(runner._produce_pairs)
    assert "run_content_iss_evaluation_pair(" in producer_source
    quality_source = inspect.getsource(runner._quality_scorer)
    assert "peak_signal_noise_ratio(" in quality_source
    assert "watermarked_tensor, base_tensor, data_range=1.0" in quality_source
    assert "structural_similarity_index_measure(" in quality_source
    assert 'lpips.LPIPS(net="alex")' in quality_source
    assert "perceptual(watermarked_tensor, base_tensor)" in quality_source
    assert "normalize=True" not in quality_source and "mul(2" not in quality_source
    assert "SyncSealTorchScript.from_file" in source
    assert "download_official_syncseal_torchscript" in source


@pytest.mark.integration
def test_bound_notebook_is_unexecuted_real_runner_wiring_and_fail_closed() -> None:
    path = _REPO_ROOT / "notebooks" / "geometry_v7_r0.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code[0]["source"] == [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')",
    ]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code)
    for index, cell in enumerate(code):
        ast.parse("".join(cell["source"]), filename=f"{path}:code-cell-{index}")
    source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])
    assert re.search(
        r"^APPROVED_EXACT = '[0-9a-f]{40}'$", source, flags=re.MULTILINE
    )
    assert "re.fullmatch(r'[0-9a-f]{40}', APPROVED_EXACT)" in source
    assert "'checkout', '--detach'" in source
    assert source.count("'experiments.run_geometry_v7_r0'") == 1
    assert "userdata.get('CEG_WM_ROOT_KEY')" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "'torchmetrics', 'lpips'" in source
    assert "if LOCAL_RESULT_DIR.exists() or SYNCSEAL_CHECKPOINT.exists():" in source
    assert "if DRIVE_RESULT_DIR.exists():" in source
    assert "shutil.copytree(LOCAL_RESULT_DIR, DRIVE_RESULT_DIR)" in source
    assert "force_remount" not in source
    assert "git', 'pull'" not in source and "branch --show-current" not in source
    assert "/content/drive" not in source.replace(
        "drive.mount('/content/drive')", ""
    ).replace("Path('/content/drive/MyDrive/CEG-WM/Geometry-V7')", "")


@pytest.mark.integration
def test_runner_cli_help_imports_without_optional_quality_dependencies() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(_REPO_ROOT / "src"), environment.get("PYTHONPATH", "")),
        )
    )
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.run_geometry_v7_r0", "--help"],
        cwd=_REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--expected-exact" in completed.stdout
    assert "--syncseal-checkpoint" in completed.stdout
