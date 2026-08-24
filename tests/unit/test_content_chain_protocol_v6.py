from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cegwm.protocol.content_chain_v6 import (
    V6_DEVELOPMENT_MANIFEST_SHA256,
    V6_DEVELOPMENT_PROMPT_LIST_SHA256,
    V6_EVALUATION_MANIFEST_SHA256,
    V6_EVALUATION_PROMPT_LIST_SHA256,
    V6_PERSONAL_SPEC_SHA256,
    load_content_v6_data_contract,
)

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_v6_manifests_bind_exact_order_hashes_and_disjoint_data_roles() -> None:
    contract = load_content_v6_data_contract(_ROOT)
    assert V6_PERSONAL_SPEC_SHA256 == (
        "770a0d79cfdb9d98156f6b8d585ae0c0554313f5dfd745ceb5e228d7f3fc02ce"
    )
    assert len(contract.development) == 32
    assert len(contract.evaluation) == 8
    assert contract.development_manifest_sha256 == V6_DEVELOPMENT_MANIFEST_SHA256
    assert contract.evaluation_manifest_sha256 == V6_EVALUATION_MANIFEST_SHA256
    assert tuple(unit.seed for unit in contract.development) == tuple(range(2026082400, 2026082432))
    assert tuple(unit.seed for unit in contract.evaluation) == tuple(range(2026082500, 2026082508))
    assert contract.development[0].unit_id == "content-v6-iss-dev-0001"
    assert contract.development[-1].unit_id == "content-v6-iss-dev-0032"
    assert contract.evaluation[0].unit_id == "content-v6-iss-eval-0001"
    assert contract.evaluation[-1].unit_id == "content-v6-iss-eval-0008"
    dev_prompts = b"".join(unit.prompt.encode() + b"\n" for unit in contract.development)
    eval_prompts = b"".join(unit.prompt.encode() + b"\n" for unit in contract.evaluation)
    assert hashlib.sha256(dev_prompts).hexdigest() == V6_DEVELOPMENT_PROMPT_LIST_SHA256
    assert hashlib.sha256(eval_prompts).hexdigest() == V6_EVALUATION_PROMPT_LIST_SHA256


@pytest.mark.unit
def test_v6_manifest_loader_fails_closed_on_data_or_serialization_drift(tmp_path: Path) -> None:
    source = _ROOT / "configs" / "content_chain"
    target = tmp_path / "configs" / "content_chain"
    target.mkdir(parents=True)
    for name in (
        "content_v6_iss_development_v1.jsonl",
        "content_v6_iss_clean.jsonl",
        "content_adaptive_dual_branch_v2_clean.jsonl",
        "content_v4_clean_null_whitening_fit_v1.json",
    ):
        (target / name).write_bytes((source / name).read_bytes())
    rows = (target / "content_v6_iss_clean.jsonl").read_text().splitlines()
    changed = json.loads(rows[0])
    changed["prompt"] = "A watchmaker sorting steel springs beneath a magnifying lamp"
    rows[0] = json.dumps(changed, separators=(",", ":"))
    (target / "content_v6_iss_clean.jsonl").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="manifest bytes differ"):
        load_content_v6_data_contract(tmp_path)
