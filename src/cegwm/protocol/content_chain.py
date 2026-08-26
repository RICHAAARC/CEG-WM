"""Frozen data and decision contract for content chain evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cegwm.method.content_weighted_joint import (
    WeightedJointAsset,
    load_calibration_asset,
)
from cegwm.protocol.content_adaptive import ContentChainProtocol
from cegwm.protocol.content_iss import load_content_iss_protocol

CONTENT_CHAIN_METHOD_ID = (
    "content_calibrated_weighted_joint_stability_v1"
)
CONTENT_CHAIN_EVALUATED_CANDIDATE_ID = (
    "content_calibrated_weighted_joint_stability_semantic_gate_v1"
)
CONTENT_CHAIN_PROTOCOL_ID = (
    "cegwm-content-calibrated-weighted-joint-stability-v1"
)
CONTENT_CHAIN_PROTOCOL_DIGEST = (
    "4b749a31346901c8a78b3512a68a335bd84aa17f4c8769dbbef9995c16cff529"
)
CONTENT_CHAIN_RECORD_CONTRACT_ID = (
    "content_calibrated_weighted_joint_stability_record_v1"
)
CONTENT_CHAIN_STATE_SCHEMA_ID = "content_chain_state_v1"
CONTENT_CHAIN_ARTIFACT_CONTRACT_ID = (
    "content_chain_artifact_v1"
)
CONTENT_CHAIN_TERMINAL_RECEIPT_ID = (
    "content_chain_terminal_receipt_v1"
)
CONTENT_CHAIN_EXECUTION_SCOPE_ID = (
    "content_chain_evaluation_v1"
)
CONTENT_CHAIN_CALIBRATION_ASSET = (
    "assets/content_v9_calibrated_weighted_joint_v1.json"
)
CONTENT_CHAIN_CALIBRATION_ASSET_SHA256 = (
    "63c17e8200a92383b061541fc234dfef36e4b7356954c160ce5f048f820cde96"
)
CONTENT_CHAIN_CALIBRATION_ASSET_SIDECAR_FILE_SHA256 = (
    "d543d604e5d9226ddb4c378e160fa389abce223fe8adbb54562c3e6666537301"
)
CONTENT_CHAIN_CALIBRATION_PRODUCER_EXACT = (
    "c38522dcab6cb173cedf8415cee2fd30998222ba"
)
CONTENT_CHAIN_CALIBRATION_PROTOCOL_DIGEST = (
    "68f37585eb6eab123bad7c1703767df08404718ce4771f73fbbec236491a1e01"
)
CONTENT_CHAIN_CALIBRATION_PUBLIC_KEY_DIGEST = (
    "a82b191410993cc2619ab239b62e5f58040bba0affde8e56b43844e58edaebb3"
)
CONTENT_CHAIN_PUBLIC_KEY_DIGEST = (
    "805bc21e173a83898f3b7034d75e6ed02f65894a6885377d9659ee3091b4dd77"
)
CONTENT_CHAIN_RUN_TEMPLATE = (
    "content-chain-{protocol_digest_12}-{calibration_asset_sha256_12}-"
    "{public_key_digest_12}"
)

CONTENT_CHAIN_REFERENCE_MANIFEST = "content_adaptive_dual_branch_v2_clean.jsonl"
CONTENT_CHAIN_REFERENCE_MANIFEST_SHA256 = (
    "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88"
)
CONTENT_CHAIN_EVALUATION_MANIFEST = "content_v6_iss_clean.jsonl"
CONTENT_CHAIN_EVALUATION_MANIFEST_SHA256 = (
    "20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f"
)
CONTENT_CHAIN_NOVEL_MANIFEST = "content_chain_novel_seed_stability.jsonl"
CONTENT_CHAIN_NOVEL_MANIFEST_SHA256 = (
    "33613cb24de87c86a573ac0dda80523912e001c922494051f5d89a9e2851831b"
)
CONTENT_CHAIN_NOVEL_PROMPT_LIST_SHA256 = (
    "4691ebd78a05f3ab617dd83a9ee94b9632bfc4ca9ffc8483f25e71082ba38618"
)
CONTENT_CHAIN_SEED_01_SLICE_SHA256 = (
    "db7671e3214af784b43ba2344c03a38ceb793fc49bb631f3439cb177dfde5916"
)
CONTENT_CHAIN_SEED_02_SLICE_SHA256 = (
    "462724d7f793398f20346c65eeeaa7110d385c984bfdd0ba1ba1f25a179c620c"
)
_CONFIG_NAME = "content_chain_stability.json"
_FIELDS = ("unit_id", "split", "source_id", "prompt", "seed", "height", "width")
_HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ContentChainUnit:
    unit_id: str
    split: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ContentChainContract:
    reference_roster: tuple[ContentChainUnit, ...]
    evaluation_roster: tuple[ContentChainUnit, ...]
    novel_seed_01: tuple[ContentChainUnit, ...]
    novel_seed_02: tuple[ContentChainUnit, ...]
    config: Mapping[str, Any]
    protocol_digest: str
    runtime_protocol: ContentChainProtocol
    calibration_asset: WeightedJointAsset


def _stable_line(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load_rows(path: Path, *, expected_sha256: str, count: int) -> tuple[dict[str, Any], ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256 or not raw.endswith(b"\n"):
        raise ValueError("content chain manifest bytes differ")
    lines = raw.splitlines()
    if len(lines) != count:
        raise ValueError("content chain manifest count differs")
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("content chain manifest must be UTF-8 JSONL") from error
        if (
            not isinstance(value, dict)
            or tuple(value) != _FIELDS
            or _stable_line(value) != line
        ):
            raise ValueError("content chain manifest fields, order, or encoding differ")
        if any(not isinstance(value[name], str) or not value[name].strip() for name in _FIELDS[:4]):
            raise ValueError("content chain text identity must be non-empty")
        if (
            isinstance(value["seed"], bool)
            or not isinstance(value["seed"], int)
            or value["height"] != 512
            or value["width"] != 512
        ):
            raise ValueError("content chain seed or dimensions differ")
        rows.append(value)
    return tuple(rows)


def _unit(row: Mapping[str, Any]) -> ContentChainUnit:
    return ContentChainUnit(**{name: row[name] for name in _FIELDS})


def _identity_sets(units: Iterable[ContentChainUnit]) -> tuple[set[Any], ...]:
    received = tuple(units)
    return (
        {unit.unit_id for unit in received},
        {unit.source_id for unit in received},
        {unit.prompt for unit in received},
        {unit.seed for unit in received},
        {(unit.prompt, unit.seed) for unit in received},
    )


def _validate_novel(rows: tuple[dict[str, Any], ...]) -> tuple[
    tuple[ContentChainUnit, ...], tuple[ContentChainUnit, ...]
]:
    blocks: list[tuple[ContentChainUnit, ...]] = []
    for seed_index, block in enumerate((rows[:32], rows[32:]), 1):
        raw = b"".join(_stable_line(row) + b"\n" for row in block)
        expected_slice = (
            CONTENT_CHAIN_SEED_01_SLICE_SHA256
            if seed_index == 1
            else CONTENT_CHAIN_SEED_02_SLICE_SHA256
        )
        if hashlib.sha256(raw).hexdigest() != expected_slice:
            raise ValueError("content chain seed stratum bytes differ")
        units = tuple(_unit(row) for row in block)
        for ordinal, unit in enumerate(units, 1):
            suffix = f"{seed_index:02d}"
            if (
                unit.unit_id != f"content-chain-seed-{suffix}-{ordinal:04d}"
                or unit.split != f"content_chain_novel_seed_stability_seed_{suffix}_v1"
                or unit.source_id
                != f"content-chain-source-{ordinal:04d}-seed-{suffix}"
                or unit.seed
                != (2026101000 if seed_index == 1 else 2026102000) + ordinal - 1
            ):
                raise ValueError("content chain ordered novel identity differs")
        blocks.append(units)
    if tuple(unit.prompt for unit in blocks[0]) != tuple(unit.prompt for unit in blocks[1]):
        raise ValueError("content chain seed strata prompts differ")
    prompt_bytes = b"".join(unit.prompt.encode("utf-8") + b"\n" for unit in blocks[0])
    if hashlib.sha256(prompt_bytes).hexdigest() != CONTENT_CHAIN_NOVEL_PROMPT_LIST_SHA256:
        raise ValueError("content chain ordered prompt identity differs")
    for units in blocks:
        if any(len(values) != 32 for values in _identity_sets(units)):
            raise ValueError("content chain stratum identities must be unique")
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
            "reference_roster": list(old_rows),
            "evaluation_roster": list(current_rows),
            "novel_seed_stability": list(novel_rows),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_content_chain_contract(repo_root: str | Path) -> ContentChainContract:
    repo = Path(repo_root)
    root = repo / "configs" / "content_chain"
    old_rows = _load_rows(
        root / CONTENT_CHAIN_REFERENCE_MANIFEST,
        expected_sha256=CONTENT_CHAIN_REFERENCE_MANIFEST_SHA256,
        count=8,
    )
    current_rows = _load_rows(
        root / CONTENT_CHAIN_EVALUATION_MANIFEST,
        expected_sha256=CONTENT_CHAIN_EVALUATION_MANIFEST_SHA256,
        count=8,
    )
    novel_rows = _load_rows(
        root / CONTENT_CHAIN_NOVEL_MANIFEST,
        expected_sha256=CONTENT_CHAIN_NOVEL_MANIFEST_SHA256,
        count=64,
    )
    novel_seed_01, novel_seed_02 = _validate_novel(novel_rows)
    old = tuple(_unit(row) for row in old_rows)
    current = tuple(_unit(row) for row in current_rows)
    section_sets = (_identity_sets(old), _identity_sets(current), _identity_sets(novel_seed_01))
    for left_index, left in enumerate(section_sets):
        for right in section_sets[left_index + 1 :]:
            if any(a & b for a, b in zip(left, right, strict=True)):
                raise ValueError("content chain section identities overlap")
    novel_01_sets = _identity_sets(novel_seed_01)
    novel_02_sets = _identity_sets(novel_seed_02)
    if any(
        left & right
        for index, (left, right) in enumerate(
            zip(novel_01_sets, novel_02_sets, strict=True)
        )
        if index != 2  # The same 32 prompts intentionally define both seed strata.
    ):
        raise ValueError("content chain novel seed identities overlap")
    try:
        config = json.loads((root / _CONFIG_NAME).read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("content chain config must be UTF-8 JSON") from error
    if not isinstance(config, dict):
        raise ValueError("content chain config must be an object")
    protocol_digest = _canonical_digest(config, old_rows, current_rows, novel_rows)
    if protocol_digest != CONTENT_CHAIN_PROTOCOL_DIGEST:
        raise ValueError("content chain canonical protocol digest differs")
    asset_path = root / CONTENT_CHAIN_CALIBRATION_ASSET
    sidecar_path = asset_path.with_name(f"{asset_path.name}.sha256")
    asset_bytes = asset_path.read_bytes()
    sidecar_bytes = sidecar_path.read_bytes()
    if (
        hashlib.sha256(asset_bytes).hexdigest()
        != CONTENT_CHAIN_CALIBRATION_ASSET_SHA256
        or hashlib.sha256(sidecar_bytes).hexdigest()
        != CONTENT_CHAIN_CALIBRATION_ASSET_SIDECAR_FILE_SHA256
    ):
        raise ValueError("content chain accepted calibration asset bytes differ")
    calibration_asset = load_calibration_asset(asset_path, sidecar_path)
    payload = calibration_asset.payload
    if (
        payload["producer_exact"]
        != CONTENT_CHAIN_CALIBRATION_PRODUCER_EXACT
        or payload["calibration_protocol_digest"]
        != CONTENT_CHAIN_CALIBRATION_PROTOCOL_DIGEST
        or payload["calibration_public_key_digest"]
        != CONTENT_CHAIN_CALIBRATION_PUBLIC_KEY_DIGEST
    ):
        raise ValueError("content chain accepted calibration identity differs")
    runtime_protocol = load_content_iss_protocol(repo)
    return ContentChainContract(
        old,
        current,
        novel_seed_01,
        novel_seed_02,
        config,
        protocol_digest,
        runtime_protocol,
        calibration_asset,
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
    return CONTENT_CHAIN_RUN_TEMPLATE.format(
        protocol_digest_12=protocol_digest[:12],
        calibration_asset_sha256_12=calibration_asset_sha256[:12],
        public_key_digest_12=public_key_digest[:12],
    )


__all__ = [name for name in globals() if name.startswith("CONTENT_CHAIN_")] + [
    "ContentChainContract",
    "ContentChainUnit",
    "deterministic_stability_run_id",
    "load_content_chain_contract",
    "strict_weighted_gate",
]
