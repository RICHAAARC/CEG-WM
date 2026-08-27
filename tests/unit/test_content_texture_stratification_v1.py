from __future__ import annotations

import hashlib
import ast
import contextlib
import io
import json
import math
import subprocess
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path

from cegwm.protocol.content_texture_stratification_v1 import (
    average_ranks,
    encode_p6_rgb,
    exact_spearman,
    f64_from_hex,
    f64_hex,
    load_protocol,
    margins,
    parse_p6_texture,
    require_scores,
    stable_json_bytes,
    stratified_exact,
)

ROOT = Path(__file__).parents[2]


class _Image:
    mode = "RGB"
    size = (512, 512)

    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def tobytes(self, *args: str) -> bytes:
        assert args == ("raw", "RGB")
        return self.raw


def _scores(registered: float, wrong: float) -> dict[str, float]:
    return {
        f"{branch}__{label}": registered if label == "registered" else wrong
        for branch in ("lf", "hf", "joint")
        for label in ("registered", *(f"wrong_{index:02d}" for index in range(16)))
    }


class TextureProtocolTests(unittest.TestCase):
    def _run_notebook_dispatch(self, *, runner_rc: int, payload: dict[str, str], terminal_bytes: bytes | None = None, terminal_binding: str | None = None):
        notebook = json.loads((ROOT / "notebooks/content_texture_stratification_v1_colab.ipynb").read_text())
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        runner_cell = code[2]
        tree = ast.parse(runner_cell)
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "google.colab")]
        fail_function = next(node for node in ast.parse(code[0]).body if isinstance(node, ast.FunctionDef) and node.name == "fail")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        terminal_zip = root / "terminal.zip"
        terminal_sha = root / "terminal.zip.sha256"
        if terminal_bytes is not None:
            terminal_zip.write_bytes(terminal_bytes)
            terminal_sha.write_text(terminal_binding, encoding="ascii")
        devnull = object()
        capture_at_dispatch: list[bytes] = []
        stdout_chunks: list[str] = []

        class RecordingEnv(dict[str, str]):
            def __init__(self, value: dict[str, str]):
                super().__init__(value)
                self.popped: list[str] = []

            def pop(self, key, *default):
                self.popped.append(key)
                return super().pop(key, *default)

        class TrackRunnerEnv(ast.NodeTransformer):
            def visit_Assign(self, node):
                self.generic_visit(node)
                if any(isinstance(target, ast.Name) and target.id == "runner_env" for target in node.targets) and isinstance(node.value, ast.DictComp):
                    node.value = ast.Call(func=ast.Name(id="RecordingEnv", ctx=ast.Load()), args=[node.value], keywords=[])
                return node

        tree = ast.fix_missing_locations(TrackRunnerEnv().visit(tree))

        class DispatchStdout:
            def write(self, text):
                stdout_chunks.append(text)
                capture_at_dispatch.append(bytes(namespace["captured"]))
                return len(text)

            def flush(self):
                return None

        class Process:
            def __init__(self, command, *, cwd, env, stdout, stderr):
                self.stdout = io.BytesIO(("CEGWM_TEXTURE_RESULT " + json.dumps(payload) + "\n").encode())
                self._rc = runner_rc
                self.env = env
                self.launch_env = dict(env)
                self.stderr = stderr

            def poll(self):
                return self._rc

            def kill(self):
                raise AssertionError("bounded fixture must not kill")

            def wait(self):
                return self._rc

        namespace = {
            "json": json, "re": __import__("re"), "Path": Path,
            "RecordingEnv": RecordingEnv,
            "subprocess": types.SimpleNamespace(Popen=Process, PIPE=object(), DEVNULL=devnull),
            "sys": types.SimpleNamespace(executable="python"),
            "os": types.SimpleNamespace(environ=RecordingEnv({"PUBLIC": "ok", "AWS_SECRET": "private"})),
            "userdata": types.SimpleNamespace(get=lambda name: {"CEG_WM_ROOT_KEY": "root-key", "HF_TOKEN": "hf-token"}[name]),
            "RESULT_PREFIX": "CEGWM_TEXTURE_RESULT", "FAILURE_PREFIX": "CEGWM_TEXTURE_HANDOFF_FAILURE", "CAPTURE_LIMIT": 4096,
            "TERMINAL_RESULT_FIELDS": {"status", "claim_ceiling", "exact", "protocol_digest", "run_id", "terminal_sha256"},
            "OPERATIONAL_RESULT_FIELDS": {"artifact_kind", "status", "claim_ceiling", "exact", "protocol_digest", "run_id", "terminal_sha256", "failure_class", "failure_stage", "last_completed_checkpoint", "result_member"},
            "RUNNER_PUBLIC_FAILURES": {"FileExistsError", "FileNotFoundError", "ImportError", "MemoryError", "OSError", "OutOfMemoryError", "RuntimeError", "TimeoutError", "TypeError", "ValueError"},
            "RUNNER_FAILURE_STAGES": {"identity", "protocol", "secrets", "checkouts", "rosters", "assets", "prefetch", "common_plain", "v2", "v3", "v4", "v5_validate", "v6", "v7", "v8", "analysis", "terminal_publication"},
            "_ALLOWED_ERRORS": {"CalledProcessError", "FileExistsError", "FileNotFoundError", "ImportError", "MemoryError", "ModuleNotFoundError", "OSError", "OutOfMemoryError", "RuntimeError", "TimeoutError", "TypeError", "UnicodeDecodeError", "ValueError"},
            "HANDOFF_FAILED": False, "RUNNER_ATTEMPTED": False, "ACCEPTED_ARTIFACT": None,
            "ANALYSIS_ID": "content_texture_stratification_v1", "TARGET_BRANCH": "stage-a-content-texture-stratification-v1", "EXPECTED_EXACT": "3ed674236e9f562a1e5a537ae0e4bef7080d4853",
            "CLAIM_CEILING": "exploratory_prospective_texture_stratification_only", "PROTOCOL_DIGEST": "3bf6552daa78ea11b3038d682f2ec623d011f4cbb5709233b1702fae1437a70e", "RUN_ID": "content-texture-stratification-v1-3bf6552daa78-805bc21e173a",
            "SOURCE": root / "source", "LOCAL": root / "local", "RUN_ROOT": root / "run-root", "TERMINAL_ZIP": terminal_zip, "TERMINAL_SHA": terminal_sha,
            "RUNNER_MODULE": "experiments.run_content_texture_stratification_v1", "SINK": root / "sink", "DRIVE_TARGET": root / "sink", "PROVENANCE": root / "provenance",
            "git": lambda *args: {("branch", "--show-current"): "stage-a-content-texture-stratification-v1", ("rev-parse", "HEAD"): "3ed674236e9f562a1e5a537ae0e4bef7080d4853", ("status", "--porcelain"): ""}[args],
        }
        exec(compile(ast.Module(body=[fail_function], type_ignores=[]), "<notebook-fail>", "exec"), namespace)
        with contextlib.redirect_stdout(DispatchStdout()):
            exec(compile(tree, "<notebook-runner-cell>", "exec"), namespace)
        return namespace, stdout_chunks, capture_at_dispatch, devnull

    def test_frozen_v2_exposes_the_real_content_adaptive_entrypoint(self) -> None:
        exact = "4d8b0df5bf7840d242115669f1d3115cdf6810cc"
        engine_source = subprocess.run(
            ["git", "show", f"{exact}:experiments/run_content_adaptive_dual_branch_v2_clean.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        runtime_source = subprocess.run(
            ["git", "show", f"{exact}:src/cegwm/runtime/content_adaptive_sd35_v2.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        engine_tree = ast.parse(engine_source)
        runtime_tree = ast.parse(runtime_source)
        imported = {
            alias.name
            for node in engine_tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "cegwm.runtime.content_adaptive_sd35_v2"
            for alias in node.names
        }
        self.assertIn("run_sd35_content_adaptive", imported)
        self.assertNotIn("V2_RUNNER_VARIANT", {node.id for node in ast.walk(engine_tree) if isinstance(node, ast.Name)})
        function = next(
            node
            for node in runtime_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_sd35_content_adaptive"
        )
        self.assertEqual([argument.arg for argument in function.args.args], ["pipeline", "prompt", "detection_key", "assets"])
        self.assertEqual([argument.arg for argument in function.args.kwonlyargs], ["height", "width", "generator"])

    def test_protocol_freezes_sources_rosters_counts_and_claim_ceiling(self) -> None:
        protocol = load_protocol(ROOT)
        self.assertEqual(list(protocol.config["sources"]), ["v2", "v3", "v4", "v5", "v6", "v7", "v8"])
        self.assertEqual([item["sha256"] for item in protocol.config["rosters_in_order"]], ["dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88", "20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f"])
        self.assertEqual(protocol.config["execution"], {"plain_generations": 16, "paired_method_generations": 192, "total_diffusion_calls": 208, "callback_writes": 96, "probe_evaluations": 6144, "fixed_analysis_rows": 112, "checkpoint_count": 9, "checkpoint_stages": ["common_plain", "v2", "v3", "v4", "v5_derived", "v6", "v7", "v8", "analysis"], "checkpoint_scope": "local_transient", "resume_allowed": False})
        self.assertEqual(protocol.config["claim_ceiling"], "exploratory_prospective_texture_stratification_only")
        self.assertEqual(len(protocol.protocol_digest), 64)

    def test_p6_and_texture_use_exact_rgb_forward_gradient(self) -> None:
        raw = bytearray(512 * 512 * 3)
        raw[3] = 255
        ppm, ppm_sha, raw_sha = encode_p6_rgb(_Image(bytes(raw)))
        self.assertEqual(ppm_sha, hashlib.sha256(ppm).hexdigest())
        self.assertEqual(raw_sha, hashlib.sha256(raw).hexdigest())
        expected = (255.0 + 255.0 + 255.0) / (512.0 * 512.0 * 3.0)
        self.assertEqual(parse_p6_texture(ppm), expected)
        self.assertEqual(parse_p6_texture(b"P6\n512 512\n255\n" + bytes(len(raw))), 0.0)
        with self.assertRaises(ValueError):
            parse_p6_texture(ppm[:-1])

    def test_score_margins_are_strict_branch_local(self) -> None:
        candidate, null = _scores(0.5, 0.25), _scores(0.1, 0.0)
        self.assertEqual(margins(candidate, null, "lf"), (0.25, 0.4))
        self.assertEqual(margins(candidate, null, "hf"), (0.25, 0.4))
        malformed = dict(candidate); malformed["lf__registered"] = True
        with self.assertRaises(TypeError): require_scores(malformed)
        malformed["lf__registered"] = math.inf
        with self.assertRaises(ValueError): require_scores(malformed)

    def test_average_ranks_and_exact_two_sided_permutations(self) -> None:
        self.assertEqual(average_ranks([1.0, 1.0, 3.0]), (Fraction(3, 2), Fraction(3, 2), Fraction(3)))
        increasing = [float(index) for index in range(8)]
        result = exact_spearman(increasing, increasing)
        self.assertEqual((result["rho"], result["permutation_extreme_count"], result["permutation_total_count"]), (1.0, 2, 40320))
        self.assertEqual(result["permutation_p_value"], 2 / 40320)
        combined = stratified_exact((increasing, increasing), (increasing, increasing))
        self.assertEqual((combined["rho"], combined["permutation_extreme_count"], combined["permutation_total_count"]), (1.0, 2, 40320**2))

    def test_binary64_and_canonical_json(self) -> None:
        for value in (-1.25, 0.0, 3.5): self.assertEqual(f64_from_hex(f64_hex(value)), value)
        with self.assertRaises(ValueError): f64_hex(float("nan"))
        self.assertEqual(stable_json_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}\n')

    def test_notebook_is_thin_drive_first_and_single_runner(self) -> None:
        notebook = json.loads((ROOT / "notebooks/content_texture_stratification_v1_colab.ipynb").read_text())
        self.assertEqual((notebook["nbformat"], notebook["nbformat_minor"]), (4, 5))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code), 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"])); self.assertIsNone(cell["execution_count"]); self.assertEqual(cell["outputs"], [])
        self.assertLess(code[0].index("drive.mount('/content/drive')"), code[0].index("REPO_URL"))
        joined = "\n".join(code)
        self.assertEqual(joined.count("3ed674236e9f562a1e5a537ae0e4bef7080d4853"), 2)
        self.assertNotIn("18716f2b68f7916585e3fd50951ca2b4a384f3f8", joined)
        self.assertNotIn("ac7883dddced981ba4e7b6067c5e437b9ff7c1b3", joined)
        self.assertIn("SHORT_COMMIT = EXECUTION_COMMIT[:7]", joined)
        self.assertIn("SHORT_COMMIT != '3ed6742'", joined)
        self.assertIn("datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')", joined)
        self.assertEqual(joined.count("datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')"), 1)
        self.assertIn("DRIVE_CONTENT_ROOT = Path('/content/drive/MyDrive/CEG-WM/Content')", joined)
        self.assertIn("Content-Texture-{SHORT_COMMIT}-{RUN_UTC}", joined)
        self.assertIn("LOCAL = Path('/content') / (DRIVE_TARGET.name + '-local')", joined)
        self.assertIn("'--artifact-sink', str(DRIVE_TARGET)", joined)
        self.assertNotIn("/CEG-WM/content_texture_stratification_v1", joined)
        self.assertEqual(joined.count("subprocess.Popen("), 1)
        self.assertEqual(joined.count("experiments.run_content_texture_stratification_v1"), 1)
        for flag in ("--repo-root", "--expected-exact", "--local-work-root", "--artifact-sink", "--provenance-root"):
            self.assertEqual(joined.count(flag), 1)
        self.assertNotIn("files.download", joined)
        self.assertNotIn("--resume", joined)
        self.assertNotIn("retry", joined)
        self.assertNotIn("fallback", joined)
        self.assertIn("stderr=subprocess.DEVNULL", joined)
        self.assertLess(joined.index("process = subprocess.Popen"), joined.index("runner_env.pop('CEG_WM_ROOT_KEY', None)"))
        self.assertNotIn("subprocess.Popen", code[3])
        self.assertIn("globals().get('ACCEPTED_ARTIFACT')", code[3])
        self.assertNotIn("Gate", joined)

    def test_notebook_parses_terminal_and_operational_runner_schemas(self) -> None:
        notebook = json.loads((ROOT / "notebooks/content_texture_stratification_v1_colab.ipynb").read_text())
        runner_cell = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"][2]
        tree = ast.parse(runner_cell)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_runner_result")
        namespace = {
            "json": json,
            "RESULT_PREFIX": "CEGWM_TEXTURE_RESULT",
            "CAPTURE_LIMIT": 4096,
            "TERMINAL_RESULT_FIELDS": {"status", "claim_ceiling", "exact", "protocol_digest", "run_id", "terminal_sha256"},
            "OPERATIONAL_RESULT_FIELDS": {"artifact_kind", "status", "claim_ceiling", "exact", "protocol_digest", "run_id", "terminal_sha256", "failure_class", "failure_stage", "last_completed_checkpoint", "result_member"},
            "RUNNER_PUBLIC_FAILURES": {"FileExistsError", "FileNotFoundError", "ImportError", "MemoryError", "OSError", "OutOfMemoryError", "RuntimeError", "TimeoutError", "TypeError", "ValueError"},
            "RUNNER_FAILURE_STAGES": {"identity", "protocol", "secrets", "checkouts", "rosters", "assets", "prefetch", "common_plain", "v2", "v3", "v4", "v5_validate", "v6", "v7", "v8", "analysis", "terminal_publication"},
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook-parser>", "exec"), namespace)
        parse = namespace["parse_runner_result"]
        terminal = {"status": "analysis_complete", "claim_ceiling": "ceiling", "exact": "a" * 40, "protocol_digest": "b" * 64, "run_id": "run", "terminal_sha256": "c" * 64}
        self.assertEqual(parse(0, ("CEGWM_TEXTURE_RESULT " + json.dumps(terminal) + "\n").encode())[0], "terminal")
        terminal["status"] = "not_interpretable"
        self.assertEqual(parse(2, ("CEGWM_TEXTURE_RESULT " + json.dumps(terminal) + "\n").encode())[0], "terminal")
        operational = {"artifact_kind": "operational_terminal", "status": "operational_failure", "claim_ceiling": "ceiling", "exact": "a" * 40, "protocol_digest": "b" * 64, "run_id": "run", "terminal_sha256": "c" * 64, "failure_class": "MemoryError", "failure_stage": "v6", "last_completed_checkpoint": 8, "result_member": "failure.json"}
        self.assertEqual(parse(2, ("CEGWM_TEXTURE_RESULT " + json.dumps(operational) + "\n").encode()), ("operational_terminal", operational))
        for bad in ({**operational, "failure_class": "UnknownError"}, {**operational, "failure_stage": "unknown"}, {**operational, "last_completed_checkpoint": 10}, {**operational, "result_member": "result.json"}):
            with self.assertRaises(RuntimeError):
                parse(2, ("CEGWM_TEXTURE_RESULT " + json.dumps(bad) + "\n").encode())

    def test_notebook_accepts_operational_terminal_before_clearing_capture(self) -> None:
        terminal_bytes = b"operational-terminal-fixture"
        terminal_sha = hashlib.sha256(terminal_bytes).hexdigest()
        payload = {"artifact_kind": "operational_terminal", "status": "operational_failure", "claim_ceiling": "exploratory_prospective_texture_stratification_only", "exact": "3ed674236e9f562a1e5a537ae0e4bef7080d4853", "protocol_digest": "3bf6552daa78ea11b3038d682f2ec623d011f4cbb5709233b1702fae1437a70e", "run_id": "content-texture-stratification-v1-3bf6552daa78-805bc21e173a", "terminal_sha256": terminal_sha, "failure_class": "MemoryError", "failure_stage": "v6", "last_completed_checkpoint": 8, "result_member": "failure.json"}
        namespace, stdout_chunks, capture_at_dispatch, devnull = self._run_notebook_dispatch(runner_rc=2, payload=payload, terminal_bytes=terminal_bytes, terminal_binding=terminal_sha + "  terminal.zip\n")
        self.assertEqual(stdout_chunks, [])
        self.assertEqual(capture_at_dispatch, [])
        self.assertEqual(namespace["ACCEPTED_ARTIFACT"]["artifact_kind"], "operational_terminal")
        self.assertEqual(namespace["ACCEPTED_ARTIFACT"]["result_member"], "failure.json")
        self.assertEqual(namespace["captured"], bytearray())
        runner_env = namespace["process"].env
        self.assertEqual(namespace["process"].launch_env["CEG_WM_ROOT_KEY"], "root-key")
        self.assertEqual(namespace["process"].launch_env["HF_TOKEN"], "hf-token")
        self.assertNotIn("AWS_SECRET", namespace["process"].launch_env)
        self.assertEqual(namespace["process"].launch_env["PUBLIC"], "ok")
        self.assertNotIn("CEG_WM_ROOT_KEY", runner_env)
        self.assertNotIn("HF_TOKEN", runner_env)
        self.assertEqual(runner_env.popped[:2], ["CEG_WM_ROOT_KEY", "HF_TOKEN"])
        self.assertEqual(namespace["root_key"], "")
        self.assertEqual(namespace["hf_token"], "")
        self.assertIsNone(namespace["runner_env"])
        self.assertEqual(namespace["process"].stderr, devnull)

    def test_notebook_terminal_dispatch_validates_pairs_and_fails_closed(self) -> None:
        terminal_bytes = b"terminal-fixture"
        terminal_sha = hashlib.sha256(terminal_bytes).hexdigest()
        for runner_rc, status in ((0, "analysis_complete"), (2, "not_interpretable")):
            payload = {"status": status, "claim_ceiling": "exploratory_prospective_texture_stratification_only", "exact": "3ed674236e9f562a1e5a537ae0e4bef7080d4853", "protocol_digest": "3bf6552daa78ea11b3038d682f2ec623d011f4cbb5709233b1702fae1437a70e", "run_id": "content-texture-stratification-v1-3bf6552daa78-805bc21e173a", "terminal_sha256": terminal_sha}
            namespace, stdout_chunks, _, _ = self._run_notebook_dispatch(runner_rc=runner_rc, payload=payload, terminal_bytes=terminal_bytes, terminal_binding=terminal_sha + "  terminal.zip\n")
            self.assertEqual(stdout_chunks, [])
            self.assertEqual(namespace["ACCEPTED_ARTIFACT"]["status"], status)
        payload = {"status": "analysis_complete", "claim_ceiling": "exploratory_prospective_texture_stratification_only", "exact": "3ed674236e9f562a1e5a537ae0e4bef7080d4853", "protocol_digest": "3bf6552daa78ea11b3038d682f2ec623d011f4cbb5709233b1702fae1437a70e", "run_id": "content-texture-stratification-v1-3bf6552daa78-805bc21e173a", "terminal_sha256": terminal_sha}
        namespace, stdout_chunks, _, _ = self._run_notebook_dispatch(runner_rc=0, payload=payload)
        self.assertIsNone(namespace["ACCEPTED_ARTIFACT"])
        self.assertEqual(json.loads("".join(stdout_chunks).split(" ", 1)[1])["stage"], "runner_result_or_artifact_validation")
        namespace, stdout_chunks, _, _ = self._run_notebook_dispatch(runner_rc=0, payload=payload, terminal_bytes=terminal_bytes, terminal_binding=("0" * 64) + "  terminal.zip\n")
        self.assertIsNone(namespace["ACCEPTED_ARTIFACT"])
        self.assertEqual(json.loads("".join(stdout_chunks).split(" ", 1)[1])["stage"], "runner_result_or_artifact_validation")


if __name__ == "__main__":
    unittest.main()
