"""Frozen Phase-1 data contract for Content V6 detector-domain ISS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

V6_PERSONAL_SPEC_ID = "CEGWM_V6_PERSONAL_SPEC_V1"
V6_PERSONAL_SPEC_SHA256 = (
    "770a0d79cfdb9d98156f6b8d585ae0c0554313f5dfd745ceb5e228d7f3fc02ce"
)
V6_DEVELOPMENT_SPLIT = "content_v6_iss_development_v1"
V6_EVALUATION_SPLIT = "content_v6_iss_clean_v1"
V6_DEVELOPMENT_MANIFEST = "content_v6_iss_development_v1.jsonl"
V6_EVALUATION_MANIFEST = "content_v6_iss_clean.jsonl"
V6_DEVELOPMENT_MANIFEST_SHA256 = (
    "4ff3efa6b98efb62d542b210ebf00f3fc624513342475ce417e9099e334066ea"
)
V6_EVALUATION_MANIFEST_SHA256 = (
    "20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f"
)
V6_DEVELOPMENT_PROMPT_LIST_SHA256 = (
    "fd2120c0ed9be832687a30de85d38dac5fb2abb23b7bd372c7d327d004cbc9ba"
)
V6_EVALUATION_PROMPT_LIST_SHA256 = (
    "ec1b29c673fa109c6078b3dc070d3dd42aa93f834aaaf387d282aa475bd2b219"
)

_FIELDS = ("unit_id", "split", "source_id", "prompt", "seed", "height", "width")
_V4_FORMAL_MANIFEST = "content_adaptive_dual_branch_v2_clean.jsonl"
_V4_FIT_MANIFEST = "content_v4_clean_null_whitening_fit_v1.json"


@dataclass(frozen=True, slots=True)
class ContentV6Unit:
    unit_id: str
    split: str
    source_id: str
    prompt: str
    seed: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ContentV6DataContract:
    development: tuple[ContentV6Unit, ...]
    evaluation: tuple[ContentV6Unit, ...]
    development_manifest_sha256: str
    evaluation_manifest_sha256: str


def _stable_line(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _parse_unit(value: Any, *, split: str, ordinal: int, count: int) -> ContentV6Unit:
    if not isinstance(value, dict) or tuple(value) != _FIELDS:
        raise ValueError("Content V6 manifest fields or order differ")
    expected_prefix = "content-v6-iss-dev" if split == V6_DEVELOPMENT_SPLIT else "content-v6-iss-eval"
    expected_source = f"{expected_prefix}-source-{ordinal:04d}"
    expected_unit = f"{expected_prefix}-{ordinal:04d}"
    expected_seed = (2026082400 if count == 32 else 2026082500) + ordinal - 1
    for field in ("unit_id", "split", "source_id", "prompt"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"Content V6 {field} must be non-empty text")
    if (
        value["unit_id"] != expected_unit
        or value["split"] != split
        or value["source_id"] != expected_source
        or isinstance(value["seed"], bool)
        or value["seed"] != expected_seed
        or value["height"] != 512
        or value["width"] != 512
    ):
        raise ValueError("Content V6 ordered unit identity differs")
    return ContentV6Unit(**value)


def _load_jsonl(
    path: Path,
    *,
    split: str,
    count: int,
    manifest_sha256: str,
    prompt_sha256: str,
) -> tuple[ContentV6Unit, ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256 or not raw.endswith(b"\n"):
        raise ValueError("Content V6 manifest bytes differ")
    lines = raw.splitlines()
    if len(lines) != count:
        raise ValueError("Content V6 manifest unit count differs")
    units: list[ContentV6Unit] = []
    for ordinal, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Content V6 manifest must be UTF-8 JSONL") from error
        if not isinstance(value, dict) or _stable_line(value) != line:
            raise ValueError("Content V6 manifest line must use stable JSON")
        units.append(_parse_unit(value, split=split, ordinal=ordinal, count=count))
    prompt_bytes = b"".join(unit.prompt.encode("utf-8") + b"\n" for unit in units)
    if hashlib.sha256(prompt_bytes).hexdigest() != prompt_sha256:
        raise ValueError("Content V6 ordered prompt identity differs")
    return tuple(units)


def _identity_sets(units: Iterable[ContentV6Unit]) -> tuple[set[Any], ...]:
    received = tuple(units)
    return (
        {unit.unit_id for unit in received},
        {unit.source_id for unit in received},
        {unit.prompt for unit in received},
        {unit.seed for unit in received},
        {(unit.prompt, unit.seed) for unit in received},
    )


def _historical_units(root: Path) -> tuple[ContentV6Unit, ...]:
    historical: list[ContentV6Unit] = []
    formal_path = root / _V4_FORMAL_MANIFEST
    for line in formal_path.read_bytes().splitlines():
        value = json.loads(line)
        historical.append(ContentV6Unit(
            value["unit_id"], value["split"], value["source_id"], value["prompt"],
            value["seed"], value["height"], value["width"],
        ))
    fit_value = json.loads((root / _V4_FIT_MANIFEST).read_bytes())
    for value in fit_value["entries"]:
        historical.append(ContentV6Unit(
            value["unit_id"], "content_v4_fit", value["unit_id"], value["prompt"],
            value["generation_seed"], 512, 512,
        ))
    return tuple(historical)


def _require_unique_and_disjoint(
    development: tuple[ContentV6Unit, ...],
    evaluation: tuple[ContentV6Unit, ...],
    historical: tuple[ContentV6Unit, ...],
) -> None:
    dev_sets = _identity_sets(development)
    eval_sets = _identity_sets(evaluation)
    historical_sets = _identity_sets(historical)
    for values, expected in ((dev_sets, 32), (eval_sets, 8)):
        if any(len(field) != expected for field in values):
            raise ValueError("Content V6 manifest identities must be unique")
    for dev, evaluation_field, old in zip(dev_sets, eval_sets, historical_sets, strict=True):
        if dev & evaluation_field or dev & old or evaluation_field & old:
            raise ValueError("Content V6 development/evaluation data identities overlap")


def load_content_v6_data_contract(repo_root: str | Path) -> ContentV6DataContract:
    """Load the fixed dev/eval manifests without constructing a final run identity."""

    root = Path(repo_root) / "configs" / "content_chain"
    development = _load_jsonl(
        root / V6_DEVELOPMENT_MANIFEST,
        split=V6_DEVELOPMENT_SPLIT,
        count=32,
        manifest_sha256=V6_DEVELOPMENT_MANIFEST_SHA256,
        prompt_sha256=V6_DEVELOPMENT_PROMPT_LIST_SHA256,
    )
    evaluation = _load_jsonl(
        root / V6_EVALUATION_MANIFEST,
        split=V6_EVALUATION_SPLIT,
        count=8,
        manifest_sha256=V6_EVALUATION_MANIFEST_SHA256,
        prompt_sha256=V6_EVALUATION_PROMPT_LIST_SHA256,
    )
    _require_unique_and_disjoint(development, evaluation, _historical_units(root))
    return ContentV6DataContract(
        development,
        evaluation,
        V6_DEVELOPMENT_MANIFEST_SHA256,
        V6_EVALUATION_MANIFEST_SHA256,
    )


__all__ = [
    "ContentV6DataContract",
    "ContentV6Unit",
    "V6_DEVELOPMENT_MANIFEST",
    "V6_DEVELOPMENT_MANIFEST_SHA256",
    "V6_DEVELOPMENT_PROMPT_LIST_SHA256",
    "V6_EVALUATION_MANIFEST",
    "V6_EVALUATION_MANIFEST_SHA256",
    "V6_EVALUATION_PROMPT_LIST_SHA256",
    "V6_PERSONAL_SPEC_ID",
    "V6_PERSONAL_SPEC_SHA256",
    "load_content_v6_data_contract",
]
