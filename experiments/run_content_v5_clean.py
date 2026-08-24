"""Explicit paired-cohort entrypoint with authentic V4 runtime wiring for Content V5."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v4_clean as v4_runner
from cegwm.protocol.content_chain_v5 import (
    CONTENT_V5_ARMS,
    CONTENT_V5_CONTROL_EXECUTION_SCOPE_ID,
    CONTENT_V5_CONTROL_RUN_PREFIX,
    CONTENT_V5_PRIMARY_EXECUTION_SCOPE_ID,
    CONTENT_V5_PRIMARY_RUN_PREFIX,
    CONTENT_V5_RECORD_CONTRACT_ID,
    CONTENT_V5_STATE_SCHEMA_ID,
    ContentChainProtocol,
    evaluate_content_v5_decision,
    load_content_v5_clean_protocol,
)

COMPLETE_EXECUTION = (
    "complete_for_content_v5_whitened_lf_adaptive_hf_branchwise_or_evaluation"
)


def _load_protocol(repo_root: Path, cohort_id: str) -> ContentChainProtocol:
    root = repo_root / "configs" / "content_chain"
    return load_content_v5_clean_protocol(
        root / "content_v5_lf_or_hf_clean_v1.json",
        root / "content_v5_primary_evaluation_v1.jsonl",
        root / "content_adaptive_dual_branch_v2_clean.jsonl",
        cohort_id=cohort_id,
    )


def _load_primary_protocol(repo_root: Path) -> ContentChainProtocol:
    return _load_protocol(repo_root, "primary_1")


def _load_control_protocol(repo_root: Path) -> ContentChainProtocol:
    return _load_protocol(repo_root, "control_1")


def _variant(
    *,
    name: str,
    execution_scope_id: str,
    run_prefix: str,
    load_protocol: Callable[[Path], ContentChainProtocol],
) -> engine.ContentRunnerVariant:
    return engine.ContentRunnerVariant(
        name=name,
        execution_scope_id=execution_scope_id,
        complete_execution=COMPLETE_EXECUTION,
        arms=CONTENT_V5_ARMS,
        record_contract_id=CONTENT_V5_RECORD_CONTRACT_ID,
        state_schema_id=CONTENT_V5_STATE_SCHEMA_ID,
        run_prefix=run_prefix,
        load_protocol=load_protocol,
        load_pipeline_and_assets=v4_runner._load_pipeline_and_assets,
        run_joint=v4_runner._run_joint,
        lf_scorer=v4_runner.score_content_v4_lf_image,
        decision_evaluator=evaluate_content_v5_decision,
    )


CONTENT_V5_PRIMARY_RUNNER_VARIANT = _variant(
    name="Content V5 primary_1",
    execution_scope_id=CONTENT_V5_PRIMARY_EXECUTION_SCOPE_ID,
    run_prefix=CONTENT_V5_PRIMARY_RUN_PREFIX,
    load_protocol=_load_primary_protocol,
)
CONTENT_V5_CONTROL_RUNNER_VARIANT = _variant(
    name="Content V5 control_1",
    execution_scope_id=CONTENT_V5_CONTROL_EXECUTION_SCOPE_ID,
    run_prefix=CONTENT_V5_CONTROL_RUN_PREFIX,
    load_protocol=_load_control_protocol,
)
CONTENT_V5_RUNNER_VARIANTS = {
    "primary_1": CONTENT_V5_PRIMARY_RUNNER_VARIANT,
    "control_1": CONTENT_V5_CONTROL_RUNNER_VARIANT,
}


def _selected_variant(cohort: object) -> engine.ContentRunnerVariant:
    if not isinstance(cohort, str) or cohort not in CONTENT_V5_RUNNER_VARIANTS:
        raise ValueError("Content V5 requires explicit primary_1 or control_1 cohort")
    return CONTENT_V5_RUNNER_VARIANTS[cohort]


def execute(args: argparse.Namespace) -> int:
    return engine.execute(args, variant=_selected_variant(getattr(args, "cohort", None)))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-exact", required=True)
    parser.add_argument("--local-work-root", required=True)
    parser.add_argument("--artifact-sink", required=True)
    parser.add_argument("--cohort", required=True, choices=tuple(CONTENT_V5_RUNNER_VARIANTS))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
