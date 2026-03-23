"""
文件目的：workflow 输入准备主代码回归测试。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path

from main.evaluation.workflow_inputs import ensure_minimal_ground_truth_records


def test_ensure_minimal_ground_truth_records_generates_labelled_pairs(tmp_path: Path) -> None:
    """
    功能：验证主代码可直接生成 calibrate 所需的最小标签样本对。

    Verify main-code workflow input preparation generates labelled detect-record
    pairs for calibration without script-side patching.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_record = {
        "operation": "detect",
        "content_evidence_payload": {
            "status": "ok",
            "score": 0.82,
        },
    }
    source_path = records_dir / "detect_record.json"
    source_path.write_text(json.dumps(source_record, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = {
        "calibration": {
            "detect_records_glob": str(source_path),
            "score_name": "content_score",
            "minimal_ground_truth_pair_count": 2,
        }
    }

    summary = ensure_minimal_ground_truth_records(cfg, run_root, "calibrate")

    assert summary["generated"] is True
    generated_glob = cfg["calibration"]["detect_records_glob"]
    generated_paths = sorted((run_root / "artifacts" / "workflow_inputs" / "calibration").glob("*.json"))
    assert generated_glob.endswith("*.json")
    assert len(generated_paths) == 4

    generated_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in generated_paths]
    positive_payload = next(payload for payload in generated_payloads if payload["label"] is True)
    negative_payload = next(payload for payload in generated_payloads if payload["label"] is False)
    assert positive_payload["label"] is True
    assert negative_payload["label"] is False
    assert positive_payload["content_evidence_payload"]["status"] == "ok"
    assert negative_payload["content_evidence_payload"]["status"] == "ok"
    assert positive_payload["content_evidence_payload"]["score"] > negative_payload["content_evidence_payload"]["score"]