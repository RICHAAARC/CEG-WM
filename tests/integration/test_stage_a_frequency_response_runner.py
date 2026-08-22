from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from cegwm.shared.keys import normalize_detection_key
from experiments.stage_a_frequency_response import run_colab as runner

_ROOT = Path(__file__).resolve().parents[2]
_KEY = "frequency-response-integration-detection-key"
_TOKEN = "hf_frequency_response_test_token"


class _Generator:
    def __init__(self, device: str) -> None:
        assert device == "cuda"
        self.seed = 0

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    shutil.copytree(_ROOT / "configs", repo / "configs")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Frequency Response Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "configs"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True)
    return repo, subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _pattern(seed: int, offset: int) -> Image.Image:
    yy, xx = np.mgrid[:24, :24]
    pixels = np.stack(((xx * 3 + yy * 5 + seed) % 100 + 20 + offset, (xx * 7 + yy * 2 + seed) % 100 + 20 + offset, (xx + yy * 11 + seed) % 100 + 20 + offset), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, fail_hf_call: int | None = None) -> dict[str, object]:
    calls: dict[str, object] = {"hf": 0, "lf": 0, "plain": 0, "seeds": [], "scores": 0}
    registered = normalize_detection_key(_KEY)
    hf_assets, lf_assets = SimpleNamespace(method="hf"), SimpleNamespace(method="lf")

    def load(model_id: str, token: str) -> tuple[object, object, object]:
        assert model_id == "stabilityai/stable-diffusion-3.5-medium" and token == _TOKEN
        return object(), hf_assets, lf_assets

    def hf(_: object, __: str, ___: bytes, assets: object, **kwargs: object) -> SimpleNamespace:
        calls["hf"] = int(calls["hf"]) + 1
        if calls["hf"] == fail_hf_call:
            raise RuntimeError("sensitive runtime detail")
        assert assets is hf_assets
        seed = kwargs["generator"].seed
        calls["seeds"].append(("hf", seed))
        return SimpleNamespace(image=_pattern(seed, 14), injection_budget=SimpleNamespace(relative_l2=0.01199))

    def lf(_: object, __: str, ___: bytes, assets: object, **kwargs: object) -> SimpleNamespace:
        calls["lf"] = int(calls["lf"]) + 1
        assert assets is lf_assets
        seed = kwargs["generator"].seed
        calls["seeds"].append(("lf", seed))
        return SimpleNamespace(image=_pattern(seed, 24), injection_budget=SimpleNamespace(relative_l2=0.01198))

    def plain(_: object, __: str, **kwargs: object) -> Image.Image:
        calls["plain"] = int(calls["plain"]) + 1
        seed = kwargs["generator"].seed
        calls["seeds"].append(("plain", seed))
        return _pattern(seed, 0)

    def scores(image: Image.Image, key: bytes, wrong_keys: tuple[bytes, ...], assets: object) -> dict[str, float]:
        calls["scores"] = int(calls["scores"]) + 1
        mean = float(np.asarray(image, dtype=np.float64).mean() / 255.0)
        return {"registered": mean + (0.5 if key == registered else 0.0), **{f"wrong_{index:02d}": mean + wrong[0] / 8192.0 for index, wrong in enumerate(wrong_keys)}}

    monkeypatch.setattr(runner, "_load_pipeline_and_assets", load)
    monkeypatch.setattr(runner.torch, "Generator", _Generator)
    monkeypatch.setattr(runner, "run_sd35_hf", hf)
    monkeypatch.setattr(runner, "run_sd35_lf", lf)
    monkeypatch.setattr(runner, "run_sd35_plain", plain)
    monkeypatch.setattr(runner, "_scores", scores)
    return calls


def _args(repo: Path, exact: str, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(repo_root=str(repo), expected_exact=exact, output_dir=str(output_dir))


@pytest.mark.integration
def test_runner_exports_only_complete_fixed_320_descriptive_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    calls = _install_fakes(monkeypatch)
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _TOKEN)
    output = tmp_path / "output"
    assert runner.execute(_args(repo, exact, output)) == 0
    result = json.loads((output / "frequency_response_evidence.json").read_text(encoding="utf-8"))
    assert result["evidence_contract"] == "STANDALONE_LF_HF_FREQUENCY_RESPONSE_EVIDENCE"
    assert result["complete"] is True and result["rc"] == 0 and len(result["records"]) == 320
    assert [tuple((record["condition"], record["arm"])) for record in result["records"][:40]] == list(runner.expected_pairs())
    assert calls["hf"] == calls["lf"] == calls["plain"] == 8 and calls["scores"] == 320
    seeds = calls["seeds"]
    for index in range(8):
        triple = seeds[index * 3:index * 3 + 3]
        assert [name for name, _ in triple] == ["hf", "lf", "plain"] and len({seed for _, seed in triple}) == 1
    assert set(result["descriptive_per_method_response"]) == {"hf", "lf"}
    assert not any(term in result for term in ("winner", "complementarity", "joint", "robustness"))


@pytest.mark.integration
def test_runner_retains_failed_unit_and_withholds_complete_rc0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, exact = _repo(tmp_path)
    _install_fakes(monkeypatch, fail_hf_call=3)
    monkeypatch.setenv(runner.KEY_ENV, _KEY)
    monkeypatch.setenv(runner.TOKEN_ENV, _TOKEN)
    output = tmp_path / "output"
    assert runner.execute(_args(repo, exact, output)) == 2
    result = json.loads((output / "frequency_response_evidence.json").read_text(encoding="utf-8"))
    failed = [record for record in result["records"] if record["status"] == "operational_failure"]
    assert result["complete"] is False and result["rc"] == 2 and len(result["records"]) == 320
    assert len(failed) == 40 and {record["unit_id"] for record in failed} == {"frequency-response-0003"}
    assert {record["failure_reason"] for record in failed} == {"unit_execution_failure"}
