"""
File purpose: Contract tests for compact JSON writing in notebook runtime helpers.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

from scripts.notebook_runtime_common import write_json_atomic_compact


def test_write_json_atomic_compact_preserves_json_semantics(tmp_path: Path) -> None:
    """
    Verify the compact JSON writer preserves payload semantics while avoiding
    pretty-print formatting overhead.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    output_path = tmp_path / "compact.json"
    payload: Dict[str, Any] = {
        "artifact_type": "paper_workflow_pw04_quality_shard",
        "schema_version": "pw_stage_04_v1",
        "family_id": "family_compact_writer",
        "quality_shard_index": 0,
        "clean_quality_summary": {
            "count": 1,
            "pair_rows": [{"pair_id": "clean_pair_0001", "psnr": 32.5, "status": "ok"}],
        },
        "attack_quality_summary": {
            "count": 2,
            "pair_rows": [{"pair_id": "attack_pair_0001", "psnr": 28.1, "status": "ok"}],
        },
    }

    write_json_atomic_compact(output_path, payload)

    file_text = output_path.read_text(encoding="utf-8")
    loaded_payload = json.loads(file_text)

    assert cast(Dict[str, Any], loaded_payload) == payload
    assert "\n" not in file_text
    assert ": " not in file_text
    assert ", " not in file_text
