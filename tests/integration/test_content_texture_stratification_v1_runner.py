from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stdout
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
 def test_domain_null_cache_and_c3_join_fail_closed(self) -> None:
    labels = {"registered": .5, **{f"wrong_{i:02d}": .25 for i in range(16)}}
    null = {"registered": .1, **{f"wrong_{i:02d}": 0. for i in range(16)}}
    cache = {}; plain = "a" * 64
    for domain in ("ordinary_lf", "v4_lf", "hf"): runner._null_cache_put(cache, domain, 1, plain, null)
    self.assertEqual(set(runner._join_domain_maps("c3", {d: labels for d in ("ordinary_lf", "v4_lf", "hf")}, cache, 1, plain)), {"ordinary_lf", "v4_lf", "hf"})
    with self.assertRaises(ValueError): runner._join_domain_maps("c3", {d: labels for d in ("ordinary_lf", "v4_lf", "hf")}, {}, 1, plain)
    self.assertEqual(len(runner._required_association_matrix()), 14)
 def test_runner_normalizes_real_adapter_canonical_score_serialization(self) -> None:
    class Image:
        mode, size = "RGB", (512, 512)

        def tobytes(self, *args):
            if args != ("raw", "RGB"):
                raise AssertionError(args)
            return bytes(512 * 512 * 3)

    scores = _score_payload(0.1)
    unit = {"global_ordinal": 1, "unit_id": "unit-0001"}
    with redirect_stdout(io.StringIO()) as captured:
        adapter._emit_success("c6", unit, Image(), Image(), scores, None)
    line = captured.getvalue()
    event = runner._adapter_event(line)
    self.assertEqual(tuple(event["candidate_scores"]), ("v4_lf", "hf"))
    self.assertEqual(event["candidate_scores"]["v4_lf"]["registered"], scores["lf__registered"])
    with self.assertRaises(ValueError): runner._adapter_event(runner.EVENT_PREFIX + json.dumps({"event":"unit","method":"c3","global_ordinal":1,"unit_id":"unit-0001","status":"success","candidate_rgb_sha256":"a","primary_null_rgb_sha256":"b","scores":scores,"primary_null_scores":scores}))

 def test_v4_rescore_calls_only_public_lf_scorer_and_emits_no_paths_or_keys(self) -> None:
    temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
    root = Path(temporary.name); bindings = root / "bindings.json"
    bindings.write_text(json.dumps({"1": {"candidate_relative":"c3/001.ppm","candidate_ppm_sha256":"a" * 64,"candidate_rgb_sha256":"b" * 64,"plain_relative":"plain/001.ppm","plain_ppm_sha256":"c" * 64,"plain_rgb_sha256":"d" * 64}}), encoding="utf-8")
    v4 = types.ModuleType("experiments.run_content_v4_clean"); engine = types.ModuleType("experiments.run_content_adaptive_dual_branch_v2_clean"); keys = types.ModuleType("cegwm.shared.keys")
    v4._load_protocol = lambda _root: object(); v4._load_pipeline_and_assets = lambda *_: (object(), types.SimpleNamespace(lf_public_assets=object()))
    calls = []
    v4.score_content_v4_lf_image = lambda image, key, assets: calls.append((image, key, assets)) or .5
    engine._wrong_keys = lambda key, protocol: tuple(f"wrong-{index}" for index in range(16)); keys.normalize_detection_key = lambda value: "registered"
    unit = {"global_ordinal": 1, "unit_id": "u1"}; emitted = []
    with mock.patch.dict(sys.modules, {"experiments.run_content_v4_clean": v4, "experiments.run_content_adaptive_dual_branch_v2_clean": engine, "cegwm.shared.keys": keys}), mock.patch.object(adapter, "_modules_inside"), mock.patch.object(adapter, "_verify_protocol"), mock.patch.object(adapter, "_safe_transient_ppm", side_effect=(object(), object())), mock.patch.object(adapter, "_event", side_effect=emitted.append):
        adapter._c3_v4_lf_rescore(root, [unit], "secret", "token", bindings, root)
    self.assertEqual(len(calls), 34)
    event = emitted[0]; self.assertEqual(event["event"], "v4_lf_rescore")
    self.assertEqual(tuple(event["candidate_scores"]["v4_lf"]), ("registered", *(f"wrong_{i:02d}" for i in range(16))))
    self.assertNotIn("path", json.dumps(event)); self.assertNotIn("secret", json.dumps(event))

 def test_failure_events_and_n96_spearman_fail_closed(self) -> None:
    failure = runner._adapter_event(runner.EVENT_PREFIX + json.dumps({"event":"v4_lf_rescore","method":"c3","global_ordinal":1,"unit_id":"u1","status":"operational_failure","failure_class":"RuntimeError"}))
    self.assertEqual(failure["status"], "operational_failure")
    monotone = runner._n96_spearman(list(range(96)), list(range(96)))
    self.assertEqual(monotone["interpretability"], "available"); self.assertAlmostEqual(monotone["rho"], 1.0)
    ties = runner._n96_spearman([index // 2 for index in range(96)], [index // 2 for index in range(96)])
    self.assertEqual(ties["interpretability"], "available")
    self.assertEqual(runner._n96_spearman([0.0] * 96, list(range(96)))["interpretability"], "unavailable_zero_rank_variance")

 def test_attributable_unit_failures_preserve_public_class(self) -> None:
    c3 = {"event":"unit", "method":"c3", "global_ordinal":1, "unit_id":"u1", "status":"success"}
    upstream = {"status":"operational_failure", "failure_class":"OSError"}
    v4 = {"status":"operational_failure", "failure_class":"TimeoutError"}
    runner._retain_unit_failure(c3, upstream, v4)
    self.assertEqual((c3["status"], c3["failure_class"]), ("operational_failure", "OSError"))
    c6 = {"status":"success"}; runner._retain_unit_failure(c6, {"status":"success"}, fallback="ValueError")
    self.assertEqual((c6["status"], c6["failure_class"]), ("operational_failure", "ValueError"))
    event = runner._adapter_event(runner.EVENT_PREFIX + json.dumps({"event":"unit","method":"c2","global_ordinal":1,"unit_id":"u1","status":"operational_failure","failure_class":"TimeoutError"}))
    self.assertEqual(event["failure_class"], "TimeoutError")

 def test_failure_stage_enum_fails_closed_and_failure_line_is_bounded(self) -> None:
    original = runner._failure_stage
    self.addCleanup(runner._set_failure_stage, original)
    runner._set_failure_stage("v6")
    with self.assertRaises(ValueError):
        runner._set_failure_stage("unknown_stage")
    runtime = runner._failure_line(RuntimeError("private diagnostic text"))
    unknown = runner._failure_line(LookupError("/private/path"))
    for line, failure_class in ((runtime, "RuntimeError"), (unknown, "OtherOperationalError")):
        self.assertLessEqual(len(line.encode("utf-8")), 4096)
        self.assertTrue(line.startswith(runner.RESULT_PREFIX + " "))
        payload = json.loads(line.split(" ", 1)[1])
        self.assertEqual(payload, {
            "status": "analysis_incomplete",
            "failure_class": failure_class,
            "failure_stage": "v6",
        })
        self.assertNotIn("private", line)
        self.assertNotIn("/", line)

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

 def test_prefetch_is_record_only_without_model_load_or_cache_byte_hashing(self) -> None:
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    cache = root / "cache"
    cache.mkdir()
    (cache / "metadata-only").write_bytes(b"not read")
    output = root / "output"
    with mock.patch.dict(os.environ, {"HF_HOME": str(cache)}, clear=False), mock.patch.object(adapter, "_load_v2", side_effect=AssertionError("prefetch must not load V2")), mock.patch.object(type(cache), "open", side_effect=AssertionError("cache bytes must not be read")):
        observation = adapter._cache_observation(cache)
    self.assertEqual(observation, {"status": "available", "file_count": 1, "record_only": True})
    failure = {"status": "unavailable", "failure_class": "RuntimeError", "record_only": True}
    with mock.patch.dict(os.environ, {"HF_HOME": str(cache)}, clear=False), mock.patch.object(adapter, "_load_v2", side_effect=AssertionError("prefetch must not load V2")), mock.patch.object(adapter, "_cache_observation", return_value=failure), redirect_stdout(io.StringIO()) as captured:
        adapter._prefetch(root, "token", output, cache)
    binding = json.loads((output / "model_bindings.json").read_text(encoding="ascii"))
    self.assertEqual(binding["cache_observation"], failure)
    self.assertNotIn("manifest_sha256", binding)
    self.assertNotIn("model_bindings_path", captured.getvalue())
    self.assertFalse(hasattr(adapter, "_verify_cache"))

 def test_hf_home_mismatch_is_a_record_and_child_is_not_forced_offline(self) -> None:
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    expected = root / "expected"
    actual = root / "actual"
    actual.mkdir()
    with mock.patch.dict(os.environ, {"HF_HOME": str(actual)}, clear=False):
        bound, record = adapter._hf_home_binding(expected)
    self.assertEqual(bound, actual.resolve())
    self.assertEqual(record, {"status": "mismatched", "record_only": True})

    seen = {}
    class Process:
        stdout = [runner.EVENT_PREFIX + json.dumps({"event": "phase_complete", "phase": "v2"}) + "\n"]
        def wait(self):
            return 0
        def kill(self):
            raise AssertionError("unexpected kill")
    def popen(*args, **kwargs):
        seen.update(kwargs["env"])
        self.assertIs(kwargs["stderr"], runner.subprocess.DEVNULL)
        return Process()
    with mock.patch.object(runner.subprocess, "Popen", side_effect=popen):
        with self.assertRaises(RuntimeError):
            runner._child(root / "adapter.py", root, "0" * 40, "v2", root / "units.json", root, expected, None, {})
    self.assertEqual(seen["HF_HOME"], str(expected))
    self.assertNotIn("HF_HUB_OFFLINE", seen)
    self.assertNotIn("TRANSFORMERS_OFFLINE", seen)

 def test_hf_home_observation_resolve_failure_does_not_block_prefetch_or_dispatch(self) -> None:
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    expected = root / "expected"
    output = root / "output"
    with mock.patch.object(Path, "resolve", side_effect=OSError("private path failure")):
        actual, record = adapter._hf_home_binding(expected)
        with redirect_stdout(io.StringIO()) as captured:
            adapter._prefetch(root, "token", output, expected)
    self.assertIsNone(actual)
    self.assertEqual(record, {"status": "unavailable", "failure_class": "OSError", "record_only": True})
    binding = json.loads((output / "model_bindings.json").read_text(encoding="ascii"))
    self.assertEqual(binding["hf_home"], "")
    self.assertEqual(binding["hf_home_binding"], record)
    self.assertEqual(binding["cache_observation"], record)
    self.assertIn('"hf_home_status":"unavailable"', captured.getvalue())
    self.assertNotIn("private", captured.getvalue())

    args = Namespace(source_root=str(root), expected_exact="0" * 40, local_output_root=str(root / "dispatch-output"), hf_cache_root=str(expected), units_json=None, model_bindings_json=None, phase="v5_validate", v7_asset_root=None, v8_asset_root=None)
    with mock.patch.object(adapter, "_identity"), mock.patch.object(adapter, "_hf_home_binding", return_value=(None, record)), mock.patch.object(adapter, "_secrets", return_value=("", "token")), mock.patch.object(adapter, "_v5_validate") as v5_validate, mock.patch.object(adapter, "_event"):
        self.assertEqual(adapter.execute(args), 0)
    v5_validate.assert_called_once_with(root, {"hf_home": "", "hf_home_binding": record})

 def test_common_plain_uses_only_sd35_pipeline_and_frozen_plain_runner(self) -> None:
    calls = []
    image = object()
    class Pipeline:
        def __call__(self, **kwargs):
            return object()
        def to(self, device):
            calls.append(("to", device))
            return self
    pipeline = Pipeline()
    class PipelineClass:
        @staticmethod
        def from_pretrained(model_id, *, torch_dtype, token):
            calls.append(("from_pretrained", model_id, torch_dtype, token))
            return pipeline
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.StableDiffusion3Pipeline = PipelineClass
    fake_runtime = types.ModuleType("cegwm.runtime.diffusers_sd35")
    fake_runtime.run_sd35_plain = lambda loaded, prompt, *, height, width, generator: calls.append(("plain", loaded, prompt, height, width, generator)) or image
    unit = {"prompt": "frozen plain prompt", "seed": 17, "unit_id": "unit-0001", "global_ordinal": 1}
    with mock.patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers, "cegwm.runtime.diffusers_sd35": fake_runtime}), mock.patch.object(adapter, "_load_v2", side_effect=AssertionError("plain must not load V2")), mock.patch.object(adapter, "_generator", return_value="generator"), mock.patch.object(adapter, "_write_plain", return_value=("plain_rgb/01-unit-0001.ppm", "p", "r")), mock.patch.object(adapter, "_event"):
        adapter._common_plain(Path("/detached"), [unit], "token", Path("/output"))
    self.assertEqual(calls, [("from_pretrained", "stabilityai/stable-diffusion-3.5-medium", "float16", "token"), ("to", "cuda"), ("plain", pipeline, unit["prompt"], 512, 512, "generator")])

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

 def test_terminal_publish_cleans_partial_pair_and_refuses_existing_run(self) -> None:
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    run_root = root / "sink" / "exact" / "run"
    with mock.patch.object(runner.shutil, "copyfileobj", side_effect=OSError("partial terminal")):
        with self.assertRaises(OSError):
            runner._publish_terminal(run_root, "run", [("receipt.json", b"{}")], root)
    self.assertFalse(run_root.exists())
    original_mkdir = type(root).mkdir
    def controlled_mkdir(path, *args, **kwargs):
        if path == run_root / "terminal":
            raise OSError("terminal directory creation failed")
        return original_mkdir(path, *args, **kwargs)
    with mock.patch.object(type(root), "mkdir", new=controlled_mkdir):
        with self.assertRaises(OSError):
            runner._publish_terminal(run_root, "run", [("receipt.json", b"{}")], root)
    self.assertFalse(run_root.exists())
    run_root.mkdir(parents=True)
    with self.assertRaises(FileExistsError):
        runner._publish_terminal(run_root, "run", [("receipt.json", b"{}")], root)

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

    def fake_child(adapter, source, exact, phase, units_path, output, cache, bindings_path, env, **kwargs):
        units = json.loads(units_path.read_text())
        if phase == "asset_prefetch":
            (output / "model_bindings.json").write_text(json.dumps({"cache_observation": {"status": "unavailable", "failure_class": "RuntimeError", "record_only": True}}), encoding="ascii")
            return [{"event": "source_validated"}, {"event": "asset_prefetch"}, {"event": "phase_complete", "phase": phase}]
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
        if phase == "c3_v4_lf_rescore":
            events = []
            for unit in units:
                raw = bytearray(512 * 512 * 3); raw[3] = unit["global_ordinal"]
                digest = hashlib.sha256(raw).hexdigest()
                scores = {"registered": .5, **{f"wrong_{index:02d}": .25 for index in range(16)}}
                null = {"registered": .1, **{f"wrong_{index:02d}": .0 for index in range(16)}}
                events.append({"event": "v4_lf_rescore", "method": "c3", "global_ordinal": unit["global_ordinal"], "unit_id": unit["unit_id"], "status": "success", "candidate_rgb_sha256": hashlib.sha256(f"c3-{unit['global_ordinal']}".encode()).hexdigest(), "plain_rgb_sha256": digest, "candidate_ppm_sha256": "a" * 64, "plain_ppm_sha256": "b" * 64, "candidate_scores": {"v4_lf": scores}, "null_scores": {"v4_lf": null}})
            return [{"event": "source_validated"}, *events, {"event": "phase_complete", "phase": phase}]
        events = []
        for unit in units:
            raw = bytearray(512 * 512 * 3)
            raw[3] = unit["global_ordinal"]
            digest = hashlib.sha256(raw).hexdigest()
            score = _score_payload(unit["global_ordinal"] * 0.001); null = _score_payload(unit["global_ordinal"] * 0.001 - 0.01)
            domains = ("v4_lf", "hf") if phase == "c6" else ("ordinary_lf", "hf")
            branches = {"ordinary_lf": "lf", "v4_lf": "lf", "hf": "hf"}
            event = {"event": "unit", "method": phase, "global_ordinal": unit["global_ordinal"], "unit_id": unit["unit_id"], "status": "success", "candidate_rgb_sha256": hashlib.sha256(f"{phase}-{unit['global_ordinal']}".encode()).hexdigest(), "primary_null_rgb_sha256": digest, "candidate_scores": {domain: {label: score[f"{branches[domain]}__{label}"] for label in ("registered", *(f"wrong_{index:02d}" for index in range(16)))} for domain in domains}, "null_scores": {domain: {label: null[f"{branches[domain]}__{label}"] for label in ("registered", *(f"wrong_{index:02d}" for index in range(16)))} for domain in domains}}
            if phase == "c3":
                event.update({"candidate_ppm_sha256": "a" * 64, "candidate_raw_rgb_sha256": event["candidate_rgb_sha256"]})
            events.append(event)
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
        self.assertEqual(len([name for name in names if name.startswith("plain_rgb/")]), 96)
        result = json.loads(archive.read("result.json"))
        records = json.loads(archive.read("records.json"))
        self.assertEqual(result["status"], "analysis_complete")
        self.assertEqual(result["fixed_method_rows"], 288)
        self.assertEqual(len(records), 288)
        self.assertNotIn(b"private prompt", archive.read("records.json"))
        self.assertNotIn("model_manifest_sha256", json.loads(archive.read("bindings.json")))
    checkpoints = sorted((local / "checkpoints").glob("checkpoint-*.json"))
    self.assertEqual(len(checkpoints), 5)
    self.assertEqual(len(list((local / "checkpoints").glob("checkpoint-*.json.sha256"))), 5)
    self.assertEqual([json.loads(path.read_text())["state"]["phase"] for path in checkpoints], ["common_plain", "c2", "c3", "c6", "analysis"])

    failure_local, failure_sink = tmp_path / "failure-local", tmp_path / "failure-sink"
    failure_args = Namespace(repo_root=str(repo), expected_exact="19ed1a351dc860ea4446309a475eaa74fc976df5", local_work_root=str(failure_local), artifact_sink=str(failure_sink), provenance_root=str(tmp_path / "provenance"))
    with mock.patch.object(runner, "load_protocol", lambda root: protocol), mock.patch.object(runner, "_identity", lambda *args: None), mock.patch.object(runner, "_create_checkouts", lambda *args: paths), mock.patch.object(runner, "_stage_asset", lambda provenance, output, spec: output), mock.patch.object(runner, "_child", fake_child), mock.patch.object(runner, "_derive_row", side_effect=ValueError("private analysis failure")), mock.patch.dict(sys.modules, {"cegwm.shared": shared, "cegwm.shared.keys": keys}), mock.patch.dict(os.environ, {"CEG_WM_ROOT_KEY": "secret", "HF_TOKEN": "token"}, clear=False), redirect_stdout(io.StringIO()) as captured:
        self.assertEqual(runner.execute(failure_args), 2)
    operational = json.loads(captured.getvalue().split(" ", 1)[1])
    self.assertLessEqual(len(captured.getvalue().encode("utf-8")), 4096)
    self.assertEqual(set(operational), set(runner.OPERATIONAL_RESULT_FIELDS))
    self.assertEqual((operational["artifact_kind"], operational["status"], operational["failure_class"], operational["failure_stage"], operational["last_completed_checkpoint"], operational["result_member"]), ("operational_terminal", "operational_failure", "ValueError", "analysis", 4, "failure.json"))
    failure_root = failure_sink / failure_args.expected_exact / run_id
    failure_archive = failure_root / "terminal" / f"{run_id}.zip"
    failure_sidecar = failure_root / "terminal" / f"{run_id}.zip.sha256"
    self.assertEqual(failure_sidecar.read_text(encoding="ascii"), f"{hashlib.sha256(failure_archive.read_bytes()).hexdigest()}  {failure_archive.name}\n")
    with zipfile.ZipFile(failure_archive) as archive:
        self.assertIsNone(archive.testzip())
        names = archive.namelist()
        self.assertEqual(names[:2], ["receipt.json", "failure.json"])
        self.assertEqual([name for name in names if name.startswith("checkpoints/")], [part for index in range(1, 5) for part in (f"checkpoints/checkpoint-{index:04d}.json", f"checkpoints/checkpoint-{index:04d}.json.sha256")])
        self.assertIn("audit/model_bindings.json", names)
        self.assertIn("audit/plain_bindings.json", names)
        receipt, failure = json.loads(archive.read("receipt.json")), json.loads(archive.read("failure.json"))
        self.assertEqual((receipt["status"], receipt["failure_class"], receipt["failure_stage"], receipt["last_completed_checkpoint"], receipt["resume_allowed"]), ("operational_failure", "ValueError", "analysis", 4, False))
        self.assertEqual(receipt["result_member"], "failure.json")
        self.assertIn(receipt["result_member"], names)
        self.assertEqual(archive.read(receipt["result_member"]), runner.stable_json_bytes(failure))
        self.assertEqual(failure["exact"], failure_args.expected_exact)
        public_bytes = b"".join(archive.read(name) for name in names)
        for forbidden in (b"private prompt", b"secret", b"token", str(tmp_path).encode()):
            self.assertNotIn(forbidden, public_bytes)
    with mock.patch.object(runner, "load_protocol", lambda root: protocol), mock.patch.object(runner, "_identity", lambda *args: None), mock.patch.dict(sys.modules, {"cegwm.shared": shared, "cegwm.shared.keys": keys}), mock.patch.dict(os.environ, {"CEG_WM_ROOT_KEY": "secret", "HF_TOKEN": "token"}, clear=False):
        with self.assertRaises(FileExistsError):
            runner.execute(failure_args)
    self.assertTrue(failure_archive.is_file())


if __name__ == "__main__":
    unittest.main()
