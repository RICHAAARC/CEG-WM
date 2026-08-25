from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

from experiments import content_texture_stratification_v1_adapter as adapter
from experiments import run_content_texture_stratification_v1 as runner


class _FailingBinaryFile:
    def __init__(self, handle, operation: str) -> None:
        self._handle = handle
        self._operation = operation

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def write(self, payload: bytes):
        if self._operation == "write":
            self._handle.write(payload[:1])
            raise OSError("controlled partial write")
        return self._handle.write(payload)

    def flush(self):
        self._handle.flush()
        if self._operation == "flush":
            raise OSError("controlled flush failure")

    def close(self):
        self._handle.close()
        if self._operation == "close":
            raise OSError("controlled close failure")


def _roster(prefix: str, seed: int) -> bytes:
    rows = [
        {"unit_id": f"{prefix}-{index:04d}", "split": prefix, "source_id": f"{prefix}-source-{index:04d}", "prompt": f"private prompt {index}", "seed": seed + index, "height": 512, "width": 512}
        for index in range(1, 9)
    ]
    return b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows)


def _score_payload(offset: float) -> dict[str, float]:
    result = {}
    for branch_index, branch in enumerate(("lf", "hf", "joint")):
        result[f"{branch}__registered"] = offset + branch_index * 0.001 + 0.05
        for index in range(16):
            result[f"{branch}__wrong_{index:02d}"] = offset + branch_index * 0.001 + index * 0.0001
    return result


class TextureRunnerTests(unittest.TestCase):
 def test_v2_adapter_calls_the_frozen_entrypoint_without_a_variant_proxy(self) -> None:
    fake_runner = types.ModuleType("experiments.run_content_adaptive_dual_branch_v2_clean")
    fake_runtime = types.ModuleType("cegwm.runtime.diffusers_sd35")
    fake_keys = types.ModuleType("cegwm.shared.keys")
    calls = []
    pipeline = object()
    assets = types.SimpleNamespace(hf_public_assets=object(), lf_public_assets=object())
    protocol = types.SimpleNamespace()
    candidate = object()
    primary_null = object()

    fake_runner._load_protocol = lambda root: protocol
    fake_runner._load_pipeline_and_assets = lambda model_id, token: (pipeline, assets)
    fake_runner._wrong_keys = lambda key, loaded_protocol: (b"wrong",)
    fake_runner._blind_scores = lambda image, key, wrong, hf, lf: image
    fake_runner._flat_scores = lambda image: {"image": image}

    def run_sd35_content_adaptive(loaded_pipeline, prompt, key, loaded_assets, *, height, width, generator):
        calls.append((loaded_pipeline, prompt, key, loaded_assets, height, width, generator))
        return types.SimpleNamespace(image=candidate)

    fake_runner.run_sd35_content_adaptive = run_sd35_content_adaptive
    fake_runtime.run_sd35_plain = lambda loaded_pipeline, prompt, *, height, width, generator: primary_null
    fake_keys.normalize_detection_key = lambda text: b"registered"
    unit = {"prompt": "frozen V2 prompt", "seed": 1213061, "unit_id": "content-adaptive-v2-0001", "global_ordinal": 1}
    emitted = []

    with mock.patch.dict(
        sys.modules,
        {
            "experiments.run_content_adaptive_dual_branch_v2_clean": fake_runner,
            "cegwm.runtime.diffusers_sd35": fake_runtime,
            "cegwm.shared.keys": fake_keys,
        },
    ), mock.patch.object(adapter, "_modules_inside"), mock.patch.object(adapter, "_verify_protocol"), mock.patch.object(
        adapter, "_generator", side_effect=lambda seed: f"generator-{seed}"
    ), mock.patch.object(adapter, "_emit_success", side_effect=lambda *args: emitted.append(args)):
        adapter._v234(Path("/detached-v2"), "v2", [unit], "secret key", "token")

    self.assertFalse(hasattr(fake_runner, "V2_RUNNER_VARIANT"))
    self.assertFalse(hasattr(fake_runner, "ContentRunnerVariant"))
    self.assertFalse(hasattr(fake_runner, "run_joint"))
    self.assertEqual(calls, [(pipeline, unit["prompt"], b"registered", assets, 512, 512, "generator-1213061")])
    self.assertEqual(emitted[0][:4], ("v2", unit, candidate, primary_null))
    self.assertEqual(emitted[0][4:], ({"image": candidate}, {"image": primary_null}))

 def test_checkpoint_cleans_only_files_created_by_this_call(self) -> None:
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    original_open = type(root).open
    identity = {"exact": "0" * 40}
    state = {"phase": "common_plain"}

    for target in ("json", "sidecar"):
        for operation in ("write", "flush", "close"):
            with self.subTest(target=target, operation=operation):
                stage = root / f"{target}-{operation}"
                stage.mkdir()

                def controlled_open(path, mode="r", *args, **kwargs):
                    handle = original_open(path, mode, *args, **kwargs)
                    is_target = path.name.endswith(".json") if target == "json" else path.name.endswith(".json.sha256")
                    return _FailingBinaryFile(handle, operation) if mode == "xb" and is_target else handle

                with mock.patch.object(type(root), "open", new=controlled_open):
                    with self.assertRaises(OSError):
                        runner._checkpoint(stage, 1, identity, state)
                self.assertFalse((stage / "checkpoint-0001.json").exists())
                self.assertFalse((stage / "checkpoint-0001.json.sha256").exists())

    preexisting_json = root / "preexisting-json"
    preexisting_json.mkdir()
    json_path = preexisting_json / "checkpoint-0001.json"
    json_path.write_bytes(b"existing-json")
    with self.assertRaises(FileExistsError):
        runner._checkpoint(preexisting_json, 1, identity, state)
    self.assertEqual(json_path.read_bytes(), b"existing-json")
    self.assertFalse((preexisting_json / "checkpoint-0001.json.sha256").exists())

    preexisting_sidecar = root / "preexisting-sidecar"
    preexisting_sidecar.mkdir()
    sidecar_path = preexisting_sidecar / "checkpoint-0001.json.sha256"
    sidecar_path.write_bytes(b"existing-sidecar")
    with self.assertRaises(FileExistsError):
        runner._checkpoint(preexisting_sidecar, 1, identity, state)
    self.assertFalse((preexisting_sidecar / "checkpoint-0001.json").exists())
    self.assertEqual(sidecar_path.read_bytes(), b"existing-sidecar")

 def test_fake_coordinator_preserves_counts_separation_and_create_only_artifact(self) -> None:
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    tmp_path = Path(temporary.name)
    repo = Path(__file__).parents[2]
    protocol = runner.load_protocol(repo)
    fake = tmp_path / "fake"
    paths = {name: fake / name for name in protocol.config["sources"]}
    for path in paths.values():
        path.mkdir(parents=True)
    old = _roster("content-adaptive-v2", 1213060)
    current = _roster("content-v6-iss-eval", 2026082499)
    old_path = paths["v2"] / protocol.config["rosters_in_order"][0]["path"]
    current_path = paths["v6"] / protocol.config["rosters_in_order"][1]["path"]
    old_path.parent.mkdir(parents=True)
    current_path.parent.mkdir(parents=True)
    old_path.write_bytes(old)
    current_path.write_bytes(current)
    protocol.config["rosters_in_order"][0]["sha256"] = hashlib.sha256(old).hexdigest()
    protocol.config["rosters_in_order"][1]["sha256"] = hashlib.sha256(current).hexdigest()
    for relative in ("experiments/run_content_v4_clean.py", "src/cegwm/method/content_whitening_v4.py", "src/cegwm/runtime/content_adaptive_sd35_v3.py"):
        for name in ("v4", "v5"):
            target = paths[name] / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("same", encoding="ascii")

    def fake_child(adapter, source, exact, phase, units_path, output, cache, bindings_path, env, **kwargs):
        units = json.loads(units_path.read_text())
        if phase == "asset_prefetch":
            (output / "model_bindings.json").write_text(json.dumps({"root": str(cache), "files": [], "manifest_sha256": "0" * 64}), encoding="ascii")
            return [{"event": "source_validated"}, {"event": "asset_prefetch"}, {"event": "phase_complete", "phase": phase}]
        if phase == "v5_validate":
            return [{"event": "source_validated"}, {"event": "v5_validated"}, {"event": "phase_complete", "phase": phase}]
        if phase == "common_plain_v2":
            events = []
            for unit in units:
                raw = bytearray(512 * 512 * 3)
                raw[3] = unit["global_ordinal"]
                ppm = b"P6\n512 512\n255\n" + raw
                relative = f"plain_rgb/{unit['global_ordinal']:02d}-{unit['unit_id']}.ppm"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(ppm)
                events.append({"event": "plain", "global_ordinal": unit["global_ordinal"], "unit_id": unit["unit_id"], "status": "success", "relative_path": relative, "plain_ppm_sha256": hashlib.sha256(ppm).hexdigest(), "plain_rgb_sha256": hashlib.sha256(raw).hexdigest()})
            return [{"event": "source_validated"}, *events, {"event": "phase_complete", "phase": phase}]
        events = []
        for unit in units:
            raw = bytearray(512 * 512 * 3)
            raw[3] = unit["global_ordinal"]
            digest = hashlib.sha256(raw).hexdigest()
            events.append({"event": "unit", "method": phase, "global_ordinal": unit["global_ordinal"], "unit_id": unit["unit_id"], "status": "success", "candidate_rgb_sha256": hashlib.sha256(f"{phase}-{unit['global_ordinal']}".encode()).hexdigest(), "primary_null_rgb_sha256": digest, "scores": _score_payload(unit["global_ordinal"] * 0.001), "primary_null_scores": _score_payload(unit["global_ordinal"] * 0.001 - 0.01)})
        return [{"event": "source_validated"}, *events, {"event": "phase_complete", "phase": phase}]

    local = tmp_path / "local"
    sink = tmp_path / "sink"
    args = Namespace(repo_root=str(repo), expected_exact="19ed1a351dc860ea4446309a475eaa74fc976df5", local_work_root=str(local), artifact_sink=str(sink), provenance_root=str(tmp_path / "provenance"))
    shared = types.ModuleType("cegwm.shared")
    keys = types.ModuleType("cegwm.shared.keys")
    keys.normalize_detection_key = lambda value: b"key"
    keys.public_key_digest = lambda value: protocol.config["public_key_digest"]
    shared.keys = keys
    with mock.patch.object(runner, "load_protocol", lambda root: protocol), mock.patch.object(runner, "_identity", lambda *args: None), mock.patch.object(runner, "_create_checkouts", lambda *args: paths), mock.patch.object(runner, "_stage_asset", lambda provenance, output, spec: output), mock.patch.object(runner, "_child", fake_child), mock.patch.dict(sys.modules, {"cegwm.shared": shared, "cegwm.shared.keys": keys}), mock.patch.dict(os.environ, {"CEG_WM_ROOT_KEY": "secret", "HF_TOKEN": "token"}, clear=False):
        self.assertEqual(runner.execute(args), 0)
    run_id = f"{protocol.run_id_prefix}-{protocol.config['public_key_digest'][:12]}"
    archive_path = sink / args.expected_exact / run_id / "terminal" / f"{run_id}.zip"
    with zipfile.ZipFile(archive_path) as archive:
        self.assertIsNone(archive.testzip())
        names = archive.namelist()
        self.assertEqual(names[:7], ["receipt.json", "bindings.json", "environment_record.json", "records.json", "result.json", "per_unit.csv", "associations.csv"])
        self.assertEqual(len([name for name in names if name.startswith("plain_rgb/")]), 16)
        result = json.loads(archive.read("result.json"))
        records = json.loads(archive.read("records.json"))
        self.assertEqual(result["status"], "analysis_complete")
        self.assertEqual(result["fixed_method_rows"], 112)
        self.assertEqual(len(records), 112)
        self.assertNotIn(b"private prompt", archive.read("records.json"))
    checkpoints = sorted((local / "checkpoints").glob("checkpoint-*.json"))
    self.assertEqual(len(checkpoints), 9)
    self.assertEqual(len(list((local / "checkpoints").glob("checkpoint-*.json.sha256"))), 9)
    self.assertEqual([json.loads(path.read_text())["state"]["phase"] for path in checkpoints], ["common_plain", "v2", "v3", "v4", "v5_derived", "v6", "v7", "v8", "analysis"])


if __name__ == "__main__":
    unittest.main()
