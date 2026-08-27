from __future__ import annotations

import hashlib
import ast
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

if __name__ == "__main__":
    unittest.main()
