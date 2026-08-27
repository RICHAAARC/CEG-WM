"""Frozen public analysis contract for prospective Content texture stratification."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROTOCOL_CONFIG = "configs/content_chain/content_texture_stratification_v1.json"
PUBLIC_STATUS = frozenset({"analysis_complete", "analysis_incomplete", "not_interpretable"})
SCORE_LABELS = ("registered", *(f"wrong_{index:02d}" for index in range(16)))
N96_FAMILIES = (
    "indoor_still_life", "natural_landscape", "urban_architecture", "people_activity",
    "animals", "food_material_closeup", "abstract_geometry", "low_light_weather",
)
_PROMPT_FORBIDDEN = re.compile(r"\b(?:watermark|lf|hf|texture|frequency|robustness|expected[ _-]?outcome|content[-_ ]?texture|content_v\d+)\b", re.I)


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def f64_hex(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("binary64 value must be finite")
    return struct.pack(">d", value).hex()


def f64_from_hex(word: str) -> float:
    if not isinstance(word, str) or len(word) != 16 or word.lower() != word:
        raise ValueError("binary64 word must be lowercase 16-hex")
    try:
        value = struct.unpack(">d", bytes.fromhex(word))[0]
    except (ValueError, struct.error) as error:
        raise ValueError("binary64 word differs") from error
    if not math.isfinite(value):
        raise ValueError("binary64 word must decode finite")
    return value


def rational(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


@dataclass(frozen=True)
class TextureProtocol:
    root: Path
    config: Mapping[str, Any]
    protocol_digest: str
    run_id_prefix: str


def load_protocol(repo_root: str | Path) -> TextureProtocol:
    root = Path(repo_root).resolve()
    path = root / PROTOCOL_CONFIG
    payload = path.read_bytes()
    config = json.loads(payload)
    required = {"schema_version", "protocol_id", "analysis_id", "claim_ceiling", "public_key_digest", "model_id", "dino_asset_id", "generation", "rosters_in_order", "sources", "assets", "execution", "statistics", "status_values", "evaluation"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("texture protocol config fields differ")
    if config["claim_ceiling"] != "exploratory_prospective_texture_stratification_only":
        raise ValueError("texture claim ceiling differs")
    if list(config["sources"]) != ["v2", "v3", "v4", "v5", "v6", "v7", "v8"]:
        raise ValueError("texture method order differs")
    execution = config["execution"]
    if (execution["total_diffusion_calls"], execution["callback_writes"], execution["probe_evaluations"], execution["score_vectors"], execution["primitive_blind_scorer_calls"], execution["candidate_unit_rows"], execution["checkpoint_scope"], execution["resume_allowed"]) != (384, 288, 18432, 960, 16320, 288, "local_transient", False):
        raise ValueError("texture execution identity differs")
    roster = config["rosters_in_order"]
    if len(roster) != 1 or roster[0].get("unit_count") != 96 or roster[0].get("blocks") != 12 or roster[0].get("slots_per_block") != 8:
        raise ValueError("texture N96 roster identity differs")
    manifest = (root / roster[0]["path"]).read_bytes()
    if sha256_bytes(manifest) != roster[0]["sha256"]:
        raise ValueError("texture N96 manifest digest differs")
    units = [json.loads(line) for line in manifest.splitlines()]
    if len(units) != 96 or len({(u["unit_id"], u["prompt"].encode(), u["seed"]) for u in units}) != 96:
        raise ValueError("texture N96 units differ")
    for ordinal, unit in enumerate(units):
        block, slot = divmod(ordinal, 8)
        if (set(unit) != {"unit_id", "block_id", "slot_index", "semantic_family", "source_id", "prompt", "seed", "height", "width"}
                or unit["unit_id"] != f"content-texture-n96-b{block + 1:02d}-s{slot + 1:02d}"
                or unit["block_id"] != f"b{block + 1:02d}"
                or unit["source_id"] != f"content-texture-n96-b{block + 1:02d}-s{slot + 1:02d}"
                or unit["slot_index"] != slot + 1
                or unit["semantic_family"] != N96_FAMILIES[slot]
                or unit["seed"] != 2026100000 + ordinal
                or unit["height"] != 512 or unit["width"] != 512
                or not isinstance(unit["prompt"], str) or not unit["prompt"].strip()
                or _PROMPT_FORBIDDEN.search(unit["prompt"])):
            raise ValueError("texture N96 unit fields differ")
    if config["evaluation"].get("calibration_manifest_binding_status") != "pending_separate_C1":
        raise ValueError("texture calibration binding differs")
    digest = sha256_bytes(stable_json_bytes(config))
    return TextureProtocol(root, config, digest, f"content-texture-stratification-v1-{digest[:12]}")


def encode_p6_rgb(image: Any) -> tuple[bytes, str, str]:
    if getattr(image, "mode", None) != "RGB" or getattr(image, "size", None) != (512, 512):
        raise ValueError("plain image must be RGB 512x512")
    raw = image.tobytes("raw", "RGB")
    if not isinstance(raw, bytes) or len(raw) != 512 * 512 * 3:
        raise ValueError("plain RGB bytes differ")
    ppm = b"P6\n512 512\n255\n" + raw
    return ppm, sha256_bytes(ppm), sha256_bytes(raw)


def parse_p6_texture(ppm: bytes) -> float:
    header = b"P6\n512 512\n255\n"
    if not isinstance(ppm, bytes) or not ppm.startswith(header) or len(ppm) != len(header) + 512 * 512 * 3:
        raise ValueError("P6 payload differs")
    raw = ppm[len(header):]
    total = 0.0
    for row in range(512):
        base = row * 512 * 3
        for col in range(512):
            pos = base + col * 3
            for channel in range(3):
                value = raw[pos + channel]
                dx = 0 if col == 0 else value - raw[pos - 3 + channel]
                dy = 0 if row == 0 else value - raw[pos - 512 * 3 + channel]
                total += math.sqrt(float(dx * dx + dy * dy))
    value = total / float(512 * 512 * 3)
    if not math.isfinite(value) or not 0.0 <= value <= 255.0 * math.sqrt(2.0):
        raise ValueError("texture value differs")
    return value


def require_scores(scores: Mapping[str, Any]) -> dict[str, float]:
    expected = tuple(f"{branch}__{label}" for branch in ("lf", "hf", "joint") for label in SCORE_LABELS)
    if tuple(scores) != expected:
        raise ValueError("score fields must be exact ordered 3-by-17")
    result: dict[str, float] = {}
    for name, item in scores.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("score must be a real scalar")
        value = float(item)
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        result[name] = value
    return result


def margins(candidate: Mapping[str, Any], primary_null: Mapping[str, Any], branch: str) -> tuple[float, float]:
    if branch not in {"lf", "hf"}:
        raise ValueError("analysis branch differs")
    left, right = require_scores(candidate), require_scores(primary_null)
    registered = left[f"{branch}__registered"]
    wrong = max(left[f"{branch}__wrong_{index:02d}"] for index in range(16))
    return registered - wrong, registered - right[f"{branch}__registered"]


def average_ranks(values: Sequence[float]) -> tuple[Fraction, ...]:
    if not values or any(not math.isfinite(float(item)) for item in values):
        raise ValueError("ranks require finite nonempty values")
    result = [Fraction(0) for _ in values]
    order = sorted(range(len(values)), key=lambda index: values[index])
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = Fraction((cursor + 1) + end, 2)
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return tuple(result)


def _centered_products(left: Sequence[Fraction], right: Sequence[Fraction]) -> tuple[Fraction, Fraction, Fraction]:
    n = len(left)
    if n != len(right) or n == 0:
        raise ValueError("rank vectors differ")
    lm = sum(left, Fraction()) / n
    rm = sum(right, Fraction()) / n
    c = sum((x - lm) * (y - rm) for x, y in zip(left, right))
    sl = sum((x - lm) ** 2 for x in left)
    sr = sum((y - rm) ** 2 for y in right)
    return c, sl, sr


@lru_cache(maxsize=256)
def _permutation_counts(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Counter[Fraction]:
    counter: Counter[Fraction] = Counter()
    for permutation in itertools.permutations(right):
        counter[_centered_products(left, permutation)[0]] += 1
    return counter


def exact_spearman(texture: Sequence[float], response: Sequence[float]) -> dict[str, Any]:
    if len(texture) != 8 or len(response) != 8:
        raise ValueError("exact Spearman requires n=8")
    x, y = average_ranks(texture), average_ranks(response)
    c, st, sm = _centered_products(x, y)
    if st <= 0 or sm <= 0:
        return {"interpretability": "unavailable_zero_rank_variance", "c": rational(c), "st": rational(st), "sm": rational(sm)}
    rho = float(c) / math.sqrt(float(st * sm))
    extreme = sum(count for value, count in _permutation_counts(x, y).items() if abs(value) >= abs(c))
    p_value = extreme / 40320.0
    return {"interpretability": "available", "rho": rho, "rho_be_hex": f64_hex(rho), "c": rational(c), "st": rational(st), "sm": rational(sm), "permutation_extreme_count": extreme, "permutation_total_count": 40320, "permutation_p_value": p_value, "permutation_p_be_hex": f64_hex(p_value)}


def stratified_exact(texture_groups: Sequence[Sequence[float]], response_groups: Sequence[Sequence[float]]) -> dict[str, Any]:
    if tuple(map(len, texture_groups)) != (8, 8) or tuple(map(len, response_groups)) != (8, 8):
        raise ValueError("stratified analysis requires two n=8 rosters")
    observed = Fraction()
    counters: list[Counter[Fraction]] = []
    st_total = Fraction()
    sm_total = Fraction()
    for texture, response in zip(texture_groups, response_groups):
        x, y = average_ranks(texture), average_ranks(response)
        c, st, sm = _centered_products(x, y)
        observed += c
        st_total += st
        sm_total += sm
        counters.append(_permutation_counts(x, y))
    if st_total <= 0 or sm_total <= 0:
        return {"interpretability": "unavailable_zero_rank_variance", "c": rational(observed)}
    extreme = sum(left_count * right_count for left, left_count in counters[0].items() for right, right_count in counters[1].items() if abs(left + right) >= abs(observed))
    total = 40320 * 40320
    rho = float(observed) / math.sqrt(float(st_total * sm_total))
    p_value = extreme / float(total)
    return {"interpretability": "available", "rho": rho, "rho_be_hex": f64_hex(rho), "c": rational(observed), "st": rational(st_total), "sm": rational(sm_total), "permutation_extreme_count": extreme, "permutation_total_count": total, "permutation_p_value": p_value, "permutation_p_be_hex": f64_hex(p_value)}


def median(values: Iterable[float]) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered or any(not math.isfinite(item) for item in ordered):
        raise ValueError("median requires finite values")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


__all__ = ["N96_FAMILIES", "PROTOCOL_CONFIG", "PUBLIC_STATUS", "SCORE_LABELS", "TextureProtocol", "average_ranks", "encode_p6_rgb", "exact_spearman", "f64_from_hex", "f64_hex", "load_protocol", "margins", "median", "parse_p6_texture", "rational", "require_scores", "sha256_bytes", "stable_json_bytes", "stratified_exact"]
