"""One production Stage-A null-fit plus candidate-selection entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
from typing import Callable, Sequence

from experiments.protocol.contrastive_lf_branch_attribution import (
    MANIFEST_PATHS,
    NULL_FIT_ROLE,
    SELECTION_ROLE,
    load_manifest,
)
from experiments.runners.contrastive_lf_branch_attribution import (
    StageAOperations,
    create_adapter_backed_stage_a_operations,
    execute_stage_a_resumable,
    execute_stage_a_null_fit_and_selection,
)
from experiments.runners.development_persistence import StageACommittedUnitStore
from scripts.experiment_execution.contrastive_lf_branch_attribution_server import (
    finalize_contrastive_lf_delivery,
    finalize_contrastive_lf_preexecution_failure,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def execute_contrastive_lf_entrypoint(
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
    operations_factory: Callable[[], StageAOperations],
) -> tuple[int, dict[str, object]]:
    operations: StageAOperations | None = None
    result = None
    try:
        null_manifest = load_manifest(
            PACKAGE_ROOT / MANIFEST_PATHS[NULL_FIT_ROLE], expected_role=NULL_FIT_ROLE
        )
        selection_manifest = load_manifest(
            PACKAGE_ROOT / MANIFEST_PATHS[SELECTION_ROLE], expected_role=SELECTION_ROLE
        )
        operations = operations_factory()
        result = execute_stage_a_null_fit_and_selection(
            null_manifest, selection_manifest, operations
        )
        try:
            operations.close()
        except Exception as exc:
            result = replace(
                result,
                selection_result=None,
                result_classification="operational_failure",
                failure_reason=type(exc).__name__[:120],
            )
        operations = None
        return finalize_contrastive_lf_delivery(
            result,
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            output_root=output_root,
        )
    except Exception as exc:
        if operations is not None:
            try:
                operations.close()
            except Exception:
                pass
        return finalize_contrastive_lf_preexecution_failure(
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            output_root=output_root,
            failure_reason=type(exc).__name__[:120],
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--observed-repository-revision", required=True)
    parser.add_argument("--new-run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--package-sha256", required=True)
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        parser.error("--execute is required")
    operations = None
    resolved_store: StageACommittedUnitStore | None = None
    stop_requested = False

    def bind_resolved_store(store: StageACommittedUnitStore) -> None:
        nonlocal resolved_store
        if resolved_store is not None and resolved_store.run_root != store.run_root:
            raise RuntimeError("Stage-A resolved run identity changed")
        resolved_store = store

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    def hard_stop(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt("Stage-A hard session cap reached")

    prior_term = signal.signal(signal.SIGTERM, request_stop)
    prior_alarm = signal.signal(signal.SIGALRM, hard_stop)
    signal.alarm(24 * 60 * 60)
    try:
        null_manifest = load_manifest(
            PACKAGE_ROOT / MANIFEST_PATHS[NULL_FIT_ROLE], expected_role=NULL_FIT_ROLE
        )
        selection_manifest = load_manifest(
            PACKAGE_ROOT / MANIFEST_PATHS[SELECTION_ROLE], expected_role=SELECTION_ROLE
        )
        operations = create_adapter_backed_stage_a_operations(
            implementation_revision=arguments.observed_repository_revision
        )
        outcome = execute_stage_a_resumable(
            null_manifest,
            selection_manifest,
            operations,
            runs_root=arguments.runs_root,
            new_run_id=arguments.new_run_id,
            session_id=arguments.session_id,
            package_sha256=arguments.package_sha256,
            stop_requested=lambda: stop_requested,
            resolved_run_callback=bind_resolved_store,
        )
        if outcome.execution_result is None:
            receipt = {
                "cache_diagnostics": outcome.cache_diagnostics,
                "completed_null_fit_units": outcome.completed_null_fit_units,
                "completed_selection_units": outcome.completed_selection_units,
                "most_recent_snapshot_path": outcome.most_recent_snapshot_path,
                "producer_revisions": list(outcome.producer_revisions),
                "run_id": outcome.run_id,
                "session_id": outcome.session_id,
                "session_status": outcome.session_status,
            }
            print(json.dumps(receipt, sort_keys=True))
            return 3
        code, receipt = finalize_contrastive_lf_delivery(
            outcome.execution_result,
            observed_repository_revision=arguments.observed_repository_revision,
            run_id=outcome.run_id,
            output_root=Path(outcome.run_root) / "final",
            session_provenance={
                "cache_diagnostics": outcome.cache_diagnostics,
                "heterogeneous_revisions": len(outcome.producer_revisions) > 1,
                "producer_revisions": list(outcome.producer_revisions),
                "session_id": outcome.session_id,
            },
        )
        print(json.dumps(receipt, sort_keys=True))
        return code
    except (Exception, KeyboardInterrupt) as exc:
        run_parent = (
            resolved_store.run_root
            if resolved_store is not None
            else Path(arguments.runs_root) / arguments.new_run_id
        )
        if resolved_store is not None:
            units = resolved_store.committed_units()
            revisions = tuple(
                sorted({str(unit["producer_revision"]) for unit in units})
            )
            snapshot_index = len(
                tuple((resolved_store.run_root / "snapshots").glob("*.zip"))
            )
            cache_diagnostics = (
                operations.cache_diagnostics()
                if operations is not None
                else {
                    "cache_entry_count": 0,
                    "cache_hit_count": 0,
                    "cache_miss_count": 0,
                    "vae_encode_count": 0,
                }
            )
            snapshot = resolved_store.write_snapshot(
                session_id=arguments.session_id,
                snapshot_index=snapshot_index,
                payload={
                    "behavior_identity_digest": resolved_store.behavior_identity_digest,
                    "cache_diagnostics": cache_diagnostics,
                    "committed_unit_count": len(units),
                    "failure_reason": type(exc).__name__[:120],
                    "producer_revisions": list(revisions),
                    "reason": "resolved_run_operational_failure",
                    "run_id": resolved_store.run_root.name,
                    "session_id": arguments.session_id,
                },
            )
            session_status = (
                "interrupted_resumable" if units else "operational_failure"
            )
            session_payload = {
                "behavior_identity_digest": resolved_store.behavior_identity_digest,
                "cache_diagnostics": cache_diagnostics,
                "committed_unit_count": len(units),
                "ended_at_utc": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "failure_reason": type(exc).__name__[:120],
                "heterogeneous_revisions": len(revisions) > 1,
                "most_recent_snapshot_path": snapshot["archive_path"],
                "producer_revision": arguments.observed_repository_revision,
                "producer_revisions": list(revisions),
                "result_classification": "operational_failure",
                "run_id": resolved_store.run_root.name,
                "session_id": arguments.session_id,
                "session_status": session_status,
            }
            resolved_store.write_session_receipt(arguments.session_id, session_payload)
            if units:
                print(json.dumps(session_payload, sort_keys=True))
                return 3
            code, receipt = finalize_contrastive_lf_preexecution_failure(
                observed_repository_revision=arguments.observed_repository_revision,
                run_id=resolved_store.run_root.name,
                output_root=resolved_store.run_root / "final",
                failure_reason=type(exc).__name__[:120],
            )
            print(json.dumps(receipt, sort_keys=True))
            return code
        run_root = run_parent / "final"
        if not run_root.exists():
            code, receipt = finalize_contrastive_lf_preexecution_failure(
                observed_repository_revision=arguments.observed_repository_revision,
                run_id=arguments.new_run_id,
                output_root=run_root,
                failure_reason=type(exc).__name__[:120],
            )
            sessions = run_parent / "sessions"
            sessions.mkdir(exist_ok=True)
            session_payload = {
                "committed_unit_count": 0,
                "ended_at_utc": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "heterogeneous_revisions": False,
                "most_recent_snapshot_path": None,
                "producer_revision": arguments.observed_repository_revision,
                "producer_revisions": [arguments.observed_repository_revision],
                "result_classification": "operational_failure",
                "run_id": arguments.new_run_id,
                "session_id": arguments.session_id,
                "session_status": "completed",
            }
            with (sessions / f"{arguments.session_id}.json").open("xb") as handle:
                handle.write(
                    (json.dumps(session_payload, sort_keys=True) + "\n").encode("utf-8")
                )
            print(json.dumps(receipt, sort_keys=True))
            return code
        print(
            json.dumps(
                {
                    "failure_reason": type(exc).__name__[:120],
                    "science_started": False,
                    "scientific_unit_count": 0,
                    "status": "blocked",
                },
                sort_keys=True,
            )
        )
        return 2
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, prior_term)
        signal.signal(signal.SIGALRM, prior_alarm)
        if operations is not None:
            try:
                operations.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
