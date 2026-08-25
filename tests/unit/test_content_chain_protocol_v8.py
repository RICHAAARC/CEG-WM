from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from cegwm.protocol import content_chain_v8 as v8

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_v8_binds_byte_exact_dev32_and_two_evaluation_rosters_in_order() -> None:
    protocol = v8.load_content_v8_protocol(_ROOT)
    assert hashlib.sha256(
        (_ROOT / "configs/content_chain/content_v6_iss_development_v1.jsonl").read_bytes()
    ).hexdigest() == v8.V8_DEVELOPMENT_MANIFEST_SHA256
    assert tuple(len(roster.units) for roster in protocol.evaluation_rosters) == (8, 8)
    assert len(protocol.development) == 32
    assert tuple(roster.manifest_sha256 for roster in protocol.evaluation_rosters) == (
        "dd30c719ae5a48b2a9a652420a3237adb74ffd26af8bac90e25c1d03fe845b88",
        "20058788bfe7d75878e7263efda2b8de94c6fdcd3a963f64368f2ba4d594868f",
    )
    assert protocol.development[0].unit_id == "content-v6-iss-dev-0001"
    assert protocol.development[-1].unit_id == "content-v6-iss-dev-0032"
    assert tuple(roster.role for roster in protocol.evaluation_rosters) == (
        "content_v2_reference", "content_v6_current",
    )
    assert protocol.config["publication"]["resume"] is False


@pytest.mark.unit
def test_v8_identity_sets_are_unique_and_pairwise_disjoint() -> None:
    protocol = v8.load_content_v8_protocol(_ROOT)
    groups = (protocol.development, *(roster.units for roster in protocol.evaluation_rosters))
    for field in ("unit_id", "source_id", "prompt", "seed"):
        sets = tuple({getattr(unit, field) for unit in group} for group in groups)
        assert tuple(map(len, sets)) == (32, 8, 8)
        assert not sets[0] & sets[1]
        assert not sets[0] & sets[2]
        assert not sets[1] & sets[2]


@pytest.mark.unit
def test_v8_loader_fails_closed_on_manifest_or_order_drift(tmp_path: Path) -> None:
    source = _ROOT / "configs/content_chain"
    target = tmp_path / "configs/content_chain"
    target.mkdir(parents=True)
    for name in (
        v8.V8_CONFIG,
        v8.V8_DEVELOPMENT_MANIFEST,
        *v8.V8_EVALUATION_MANIFESTS_IN_ORDER,
    ):
        shutil.copyfile(source / name, target / name)
    path = target / v8.V8_DEVELOPMENT_MANIFEST
    rows = path.read_text(encoding="utf-8").splitlines()
    value = json.loads(rows[0])
    value["seed"] += 1
    rows[0] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest bytes differ"):
        v8.load_content_v8_protocol(tmp_path)
