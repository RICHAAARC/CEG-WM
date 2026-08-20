from __future__ import annotations

from pathlib import Path

import pytest

from scripts.experiment_execution.contrastive_lf_branch_attribution_server import (
    CHECKSUMS_FILENAME,
    ContrastiveLfDeliveryError,
    finalize_contrastive_lf_delivery,
    finalize_contrastive_lf_preexecution_failure,
)


@pytest.mark.unit
def test_preexecution_failure_is_bounded_create_only_and_sha_last(tmp_path: Path) -> None:
    output = tmp_path / "run"
    code, receipt = finalize_contrastive_lf_preexecution_failure(
        observed_repository_revision="1" * 40,
        run_id="contrastive-lf-branch-attribution-" + "2" * 32,
        output_root=output,
        failure_reason="RuntimeError",
    )
    assert code == 2
    assert receipt["result_classification"] == "operational_failure"
    names = {path.name for path in output.iterdir()}
    assert CHECKSUMS_FILENAME in names
    sums = (output / CHECKSUMS_FILENAME).read_text(encoding="ascii").splitlines()
    assert {row.split("  ", 1)[1] for row in sums} == names - {CHECKSUMS_FILENAME}
    assert (output / CHECKSUMS_FILENAME).stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns for path in output.iterdir() if path.name != CHECKSUMS_FILENAME
    )
    with pytest.raises(ContrastiveLfDeliveryError, match="already exists"):
        finalize_contrastive_lf_preexecution_failure(
            observed_repository_revision="1" * 40,
            run_id="contrastive-lf-branch-attribution-" + "2" * 32,
            output_root=output,
            failure_reason="RuntimeError",
        )


@pytest.mark.unit
def test_preexecution_failure_rejects_unbounded_reason(tmp_path: Path) -> None:
    with pytest.raises(ContrastiveLfDeliveryError, match="bounded"):
        finalize_contrastive_lf_preexecution_failure(
            observed_repository_revision="1" * 40,
            run_id="contrastive-lf-branch-attribution-" + "3" * 32,
            output_root=tmp_path / "run",
            failure_reason="x" * 121,
        )
