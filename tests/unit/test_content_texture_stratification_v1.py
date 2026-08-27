from __future__ import annotations

import hashlib
import ast
import json
import math
import subprocess
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
    N96_FAMILIES,
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

    def test_protocol_freezes_n96_roster_counts_and_claim_ceiling(self) -> None:
        protocol = load_protocol(ROOT)
        self.assertEqual(list(protocol.config["sources"]), ["v2", "v3", "v4", "v5", "v6", "v7", "v8"])
        self.assertEqual(protocol.config["execution"], {"plain_generations": 96, "candidate_generations": 288, "total_diffusion_calls": 384, "callback_writes": 288, "probe_evaluations": 18432, "score_vectors": 960, "primitive_blind_scorer_calls": 16320, "candidate_unit_rows": 288, "checkpoint_scope": "local_transient", "resume_allowed": False})
        self.assertEqual(protocol.config["evaluation"]["calibration_manifest_binding_status"], "not_applicable_until_E1")
        self.assertEqual(protocol.config["claim_ceiling"], "exploratory_prospective_texture_stratification_only")
        self.assertEqual(len(protocol.protocol_digest), 64)

    def test_n96_manifest_has_fixed_slots_seeds_and_new_prompt_bytes(self) -> None:
        protocol = load_protocol(ROOT)
        spec = protocol.config["rosters_in_order"][0]
        rows = [json.loads(line) for line in (ROOT / spec["path"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({row["prompt"].encode("utf-8") for row in rows}), 96)
        self.assertEqual([row["seed"] for row in rows], list(range(2026100000, 2026100096)))
        self.assertEqual([row["semantic_family"] for row in rows[:8]], list(N96_FAMILIES))
        old_prompts = set()
        for path in (ROOT / "configs").rglob("*.json*"):
            if path == ROOT / spec["path"]:
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            stack = [value]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    if isinstance(item.get("prompt"), str): old_prompts.add(item["prompt"].encode("utf-8"))
                    stack.extend(item.values())
                elif isinstance(item, list): stack.extend(item)
        self.assertFalse(old_prompts & {row["prompt"].encode("utf-8") for row in rows})

    def test_thin_notebook_has_drive_first_mount_and_single_runner(self) -> None:
        notebook = json.loads((ROOT / "notebooks/content_texture_stratification_v1_colab.ipynb").read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(cells[0]["source"], ["from google.colab import drive\n", "drive.mount('/content/drive')\n"])
        source = "".join(line for cell in cells for line in cell["source"])
        self.assertEqual(source.count("subprocess.Popen("), 1)
        self.assertNotIn("force_remount", source)
        self.assertNotIn("retry", source.lower())
        self.assertTrue(all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells))

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

if __name__ == "__main__":
    unittest.main()
