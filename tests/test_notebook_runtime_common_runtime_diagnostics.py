"""
File purpose: Contract tests for notebook runtime diagnostics helper payloads.
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from scripts.notebook_runtime_common import (
    build_stage_runtime_workload_summary,
    build_stage_runtime_diagnostics_payload,
    write_stage_runtime_diagnostics,
)


def test_build_stage_runtime_diagnostics_payload_minimal_fields(tmp_path: Path) -> None:
    """
    Verify runtime diagnostics payload includes required fields and truncates stdio tails.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    expected_output_path = tmp_path / "runtime_state" / "pw00_summary.json"
    expected_output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_output_path.write_text("{}", encoding="utf-8")
    stdout_text = "stdout-prefix\n" + ("A" * 5000)
    stderr_text = "stderr-prefix\n" + ("B" * 5000)

    payload = build_stage_runtime_diagnostics_payload(
        stage_name="PW00_Paper_Eval_Family_Manifest",
        family_id="family_runtime_diag",
        expected_output_label="pw00_summary",
        expected_output_path=expected_output_path,
        started_at_utc="2026-04-20T00:00:00+00:00",
        finished_at_utc="2026-04-20T00:00:05+00:00",
        elapsed_seconds=5.0,
        return_code=0,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        count_summary={
            "prompt_count": 8,
            "seed_count": 8,
            "total_event_count": 64,
        },
        workload_summary=build_stage_runtime_workload_summary(
            unit_label="source_events",
            unit_count=64,
            elapsed_seconds=5.0,
        ),
    )

    assert payload["stage_name"] == "PW00_Paper_Eval_Family_Manifest"
    assert payload["family_id"] == "family_runtime_diag"
    assert payload["expected_output_label"] == "pw00_summary"
    assert payload["expected_output_exists"] is True
    assert payload["return_code"] == 0
    assert payload["started_at_utc"] == "2026-04-20T00:00:00+00:00"
    assert payload["finished_at_utc"] == "2026-04-20T00:00:05+00:00"
    assert payload["elapsed_seconds"] == 5.0
    assert payload["schema_version"] == "pw_runtime_diagnostics_v1"
    assert payload["count_summary"] == {
        "prompt_count": 8,
        "seed_count": 8,
        "total_event_count": 64,
    }
    assert payload["workload_summary"] == {
        "unit_label": "source_events",
        "unit_count": 64,
        "elapsed_seconds_per_unit": 5.0 / 64.0,
    }
    assert isinstance(payload["stdout_tail"], str)
    assert isinstance(payload["stderr_tail"], str)
    assert len(str(payload["stdout_tail"])) < len(stdout_text)
    assert len(str(payload["stderr_tail"])) < len(stderr_text)
    assert str(payload["stdout_tail"]).endswith("A" * 4000)
    assert str(payload["stderr_tail"]).endswith("B" * 4000)


def test_build_stage_runtime_workload_summary_handles_zero_unit_count() -> None:
    """
    Verify workload summary uses None for per-unit elapsed time when unit count is zero.

    Args:
        None.

    Returns:
        None.
    """
    payload = build_stage_runtime_workload_summary(
        unit_label="blocking_reasons",
        unit_count=0,
        elapsed_seconds=3.5,
    )

    assert payload == {
        "unit_label": "blocking_reasons",
        "unit_count": 0,
        "elapsed_seconds_per_unit": None,
    }


def test_write_stage_runtime_diagnostics_writes_json_object(tmp_path: Path) -> None:
    """
    Verify runtime diagnostics helper writes one UTF-8 JSON object file.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    diagnostics_path = tmp_path / "runtime_state" / "pw02_runtime_diagnostics.json"
    payload: Dict[str, Any] = {
        "stage_name": "PW02_Source_Merge_And_Global_Thresholds",
        "family_id": "family_runtime_diag",
        "expected_output_exists": True,
        "return_code": 0,
    }

    written_path = write_stage_runtime_diagnostics(
        diagnostics_path=diagnostics_path,
        payload=payload,
    )

    assert written_path == diagnostics_path
    loaded_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert isinstance(loaded_payload, dict)
    assert loaded_payload["stage_name"] == "PW02_Source_Merge_And_Global_Thresholds"
    assert loaded_payload["family_id"] == "family_runtime_diag"
