"""Frozen data and decision contract for Content V9 stability evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CONTENT_V9_STABILITY_SOURCE_EXACT = "c38522dcab6cb173cedf8415cee2fd30998222ba"
CONTENT_V9_STABILITY_METHOD_ID = (
    "content_v9_v6_calibrated_weighted_joint_multi_cohort_stability_v1"
)
CONTENT_V9_STABILITY_EVALUATED_CANDIDATE_ID = (
    "content_v9_v6_calibrated_weighted_joint_multi_cohort_stability_semantic_gate_v1"
)
CONTENT_V9_STABILITY_PROTOCOL_ID = (
    "cegwm-stage-a-content-v9-calibrated-weighted-joint-multi-cohort-stability-v1"
)
CONTENT_V9_STABILITY_PROTOCOL_DIGEST = (
    "9bc8a94c1d022cfaaf3c36018422b245e42764571314ee048d612e58a19ca031"
)
CONTENT_V9_STABILITY_RECORD_CONTRACT_ID = (
    "content_v9_calibrated_weighted_joint_stability_record_v1"
)
CONTENT_V9_STABILITY_RUN_TEMPLATE = (
    "content-v9-stability-{protocol_digest_12}-{calibration_asset_sha256_12}-"
    "{public_key_digest_12}"
)

CONTENT_V9_STABILITY_OLD_MANIFEST = "content_adaptive_dual_branch_v2_clean.jsonl"
CONTENT_V9_STABILITY_OLD_MANIFEST_SHA256 = (
    "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88"
)
CONTENT_V9_STABILITY_CURRENT_MANIFEST = "content_v6_iss_clean.jsonl"
CONTENT_V9_STABILITY_CURRENT_MANIFEST_SHA256 = (
    "20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f"
)
CONTENT_V9_STABILITY_NOVEL_MANIFEST = "content_v9_novel_seed_stability_v1.jsonl"
CONTENT_V9_STABILITY_NOVEL_MANIFEST_SHA256 = (
    "d9dd998fb3e7e3e0c4693d1188fd18cc7df32424f47ee378b385cabcee51ceb2"
)
CONTENT_V9_STABILITY_NOVEL_PROMPT_LIST_SHA256 = (
    "4691ebd78a05f3ab617dd83a9ee94b9632bfc4ca9ffc8483f25e71082ba38618"
)
CONTENT_V9_STABILITY_SEED_01_SLICE_SHA256 = (
    "222b39f97c55cec48201f6782539c9bb8aabe27f27bcfb1f495361e870a77fef"
)
CONTENT_V9_STABILITY_SEED_02_SLICE_SHA256 = (
    "af7c58c3be501715b484977b3888d19a13ef29f7fce7be7f1a3a4e8461fbb83d"
)
CONTENT_V9_UNUSED_EVALUATION_MANIFEST = "content_v9_clean_evaluation_v1.jsonl"
CONTENT_V9_UNUSED_EVALUATION_MANIFEST_SHA256 = (
    "dd9d9c60974f07f3727b0c46b08c1678dfb9b57339a662735ad4178b9473849d"
)

_CONFIG_NAME = "content_v9_multi_cohort_stability_v1.json"
_FIELDS = ("unit_id", "split", "source_id", "prompt", "seed", "height", "width")
_HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ContentV9StabilityUnit:
    unit_id: str
    split: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ContentV9StabilityContract:
    old_roster_reference: tuple[ContentV9StabilityUnit, ...]
    current_v6_roster_reference: tuple[ContentV9StabilityUnit, ...]
    novel_seed_01: tuple[ContentV9StabilityUnit, ...]
    novel_seed_02: tuple[ContentV9StabilityUnit, ...]
    config: Mapping[str, Any]
    protocol_digest: str


def _stable_line(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load_rows(path: Path, *, expected_sha256: str, count: int) -> tuple[dict[str, Any], ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256 or not raw.endswith(b"\n"):
        raise ValueError("Content V9 stability manifest bytes differ")
    lines = raw.splitlines()
    if len(lines) != count:
        raise ValueError("Content V9 stability manifest count differs")
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Content V9 stability manifest must be UTF-8 JSONL") from error
        if (
            not isinstance(value, dict)
            or tuple(value) != _FIELDS
            or _stable_line(value) != line
        ):
            raise ValueError("Content V9 stability manifest fields, order, or encoding differ")
        if any(not isinstance(value[name], str) or not value[name].strip() for name in _FIELDS[:4]):
            raise ValueError("Content V9 stability text identity must be non-empty")
        if (
            isinstance(value["seed"], bool)
            or not isinstance(value["seed"], int)
            or value["height"] != 512
            or value["width"] != 512
        ):
            raise ValueError("Content V9 stability seed or dimensions differ")
        rows.append(value)
    return tuple(rows)


def _unit(row: Mapping[str, Any]) -> ContentV9StabilityUnit:
    return ContentV9StabilityUnit(**{name: row[name] for name in _FIELDS})


def _identity_sets(units: Iterable[ContentV9StabilityUnit]) -> tuple[set[Any], ...]:
    received = tuple(units)
    return (
        {unit.unit_id for unit in received},
        {unit.source_id for unit in received},
        {unit.prompt for unit in received},
        {unit.seed for unit in received},
        {(unit.prompt, unit.seed) for unit in received},
    )


def _validate_novel(rows: tuple[dict[str, Any], ...]) -> tuple[
    tuple[ContentV9StabilityUnit, ...], tuple[ContentV9StabilityUnit, ...]
]:
    blocks: list[tuple[ContentV9StabilityUnit, ...]] = []
    for seed_index, block in enumerate((rows[:32], rows[32:]), 1):
        raw = b"".join(_stable_line(row) + b"\n" for row in block)
        expected_slice = (
            CONTENT_V9_STABILITY_SEED_01_SLICE_SHA256
            if seed_index == 1
            else CONTENT_V9_STABILITY_SEED_02_SLICE_SHA256
        )
        if hashlib.sha256(raw).hexdigest() != expected_slice:
            raise ValueError("Content V9 stability seed stratum bytes differ")
        units = tuple(_unit(row) for row in block)
        for ordinal, unit in enumerate(units, 1):
            suffix = f"{seed_index:02d}"
            if (
                unit.unit_id != f"content-v9-stability-seed-{suffix}-{ordinal:04d}"
                or unit.split != f"content_v9_novel_seed_stability_seed_{suffix}_v1"
                or unit.source_id
                != f"content-v9-stability-source-{ordinal:04d}-seed-{suffix}"
                or unit.seed
                != (2026101000 if seed_index == 1 else 2026102000) + ordinal - 1
            ):
                raise ValueError("Content V9 stability ordered novel identity differs")
        blocks.append(units)
    if tuple(unit.prompt for unit in blocks[0]) != tuple(unit.prompt for unit in blocks[1]):
        raise ValueError("Content V9 stability seed strata prompts differ")
    prompt_bytes = b"".join(unit.prompt.encode("utf-8") + b"\n" for unit in blocks[0])
    if hashlib.sha256(prompt_bytes).hexdigest() != CONTENT_V9_STABILITY_NOVEL_PROMPT_LIST_SHA256:
        raise ValueError("Content V9 stability ordered prompt identity differs")
    for units in blocks:
        if any(len(values) != 32 for values in _identity_sets(units)):
            raise ValueError("Content V9 stability stratum identities must be unique")
    return blocks[0], blocks[1]


def _canonical_digest(
    config: Mapping[str, Any],
    old_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    novel_rows: Sequence[Mapping[str, Any]],
) -> str:
    canonical = json.dumps(
        {
            "config": config,
            "old_roster_reference": list(old_rows),
            "current_v6_roster_reference": list(current_rows),
            "novel_seed_stability": list(novel_rows),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_content_v9_stability_contract(repo_root: str | Path) -> ContentV9StabilityContract:
    root = Path(repo_root) / "configs" / "content_chain"
    old_rows = _load_rows(
        root / CONTENT_V9_STABILITY_OLD_MANIFEST,
        expected_sha256=CONTENT_V9_STABILITY_OLD_MANIFEST_SHA256,
        count=8,
    )
    current_rows = _load_rows(
        root / CONTENT_V9_STABILITY_CURRENT_MANIFEST,
        expected_sha256=CONTENT_V9_STABILITY_CURRENT_MANIFEST_SHA256,
        count=8,
    )
    novel_rows = _load_rows(
        root / CONTENT_V9_STABILITY_NOVEL_MANIFEST,
        expected_sha256=CONTENT_V9_STABILITY_NOVEL_MANIFEST_SHA256,
        count=64,
    )
    novel_seed_01, novel_seed_02 = _validate_novel(novel_rows)
    old = tuple(_unit(row) for row in old_rows)
    current = tuple(_unit(row) for row in current_rows)
    section_sets = (_identity_sets(old), _identity_sets(current), _identity_sets(novel_seed_01))
    for left_index, left in enumerate(section_sets):
        for right in section_sets[left_index + 1 :]:
            if any(a & b for a, b in zip(left, right, strict=True)):
                raise ValueError("Content V9 stability section identities overlap")
    novel_01_sets = _identity_sets(novel_seed_01)
    novel_02_sets = _identity_sets(novel_seed_02)
    if any(
        left & right
        for index, (left, right) in enumerate(
            zip(novel_01_sets, novel_02_sets, strict=True)
        )
        if index != 2  # The same 32 prompts intentionally define both seed strata.
    ):
        raise ValueError("Content V9 stability novel seed identities overlap")
    unused_path = root / CONTENT_V9_UNUSED_EVALUATION_MANIFEST
    if hashlib.sha256(unused_path.read_bytes()).hexdigest() != CONTENT_V9_UNUSED_EVALUATION_MANIFEST_SHA256:
        raise ValueError("Content V9 unused evaluation provenance differs")
    try:
        config = json.loads((root / _CONFIG_NAME).read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Content V9 stability config must be UTF-8 JSON") from error
    if not isinstance(config, dict):
        raise ValueError("Content V9 stability config must be an object")
    protocol_digest = _canonical_digest(config, old_rows, current_rows, novel_rows)
    if protocol_digest != CONTENT_V9_STABILITY_PROTOCOL_DIGEST:
        raise ValueError("Content V9 stability canonical protocol digest differs")
    return ContentV9StabilityContract(
        old, current, novel_seed_01, novel_seed_02, config, protocol_digest
    )


def strict_weighted_gate(margins: Sequence[float], *, required: int) -> tuple[int, bool]:
    if isinstance(required, bool) or not isinstance(required, int) or required <= 0:
        raise ValueError("required gate count must be a positive integer")
    values = tuple(margins)
    if len(values) < required:
        raise ValueError("gate denominator is smaller than its required count")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("gate margins must be real numbers")
    numeric = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("gate margins must be finite")
    count = sum(value > 0.0 for value in numeric)
    return count, count >= required


def deterministic_stability_run_id(
    protocol_digest: str, calibration_asset_sha256: str, public_key_digest: str
) -> str:
    for value in (protocol_digest, calibration_asset_sha256, public_key_digest):
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            raise ValueError("run identity inputs must be lowercase 64-hex")
    return CONTENT_V9_STABILITY_RUN_TEMPLATE.format(
        protocol_digest_12=protocol_digest[:12],
        calibration_asset_sha256_12=calibration_asset_sha256[:12],
        public_key_digest_12=public_key_digest[:12],
    )


__all__ = [name for name in globals() if name.startswith("CONTENT_V9_")] + [
    "ContentV9StabilityContract",
    "ContentV9StabilityUnit",
    "deterministic_stability_run_id",
    "load_content_v9_stability_contract",
    "strict_weighted_gate",
]
