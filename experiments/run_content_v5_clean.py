"""Blocked formal entrypoint and authentic V4 runtime wiring for Content V5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v4_clean as v4_runner
from cegwm.protocol.content_chain_v5 import (
    CONTENT_V5_ARMS,
    CONTENT_V5_EXECUTION_SCOPE_ID,
    CONTENT_V5_RECORD_CONTRACT_ID,
    CONTENT_V5_RUN_PREFIX,
    CONTENT_V5_STATE_SCHEMA_ID,
    evaluate_content_v5_decision,
    load_content_v5_method_definition,
    require_content_v5_manifest_binding,
)

COMPLETE_EXECUTION = (
    "complete_for_content_v5_whitened_lf_adaptive_hf_branchwise_or_evaluation"
)


def _load_protocol(repo_root: Path) -> NoReturn:
    """Validate the V5 method definition, then stop before any V4 roster fallback."""

    load_content_v5_method_definition(
        repo_root / "configs" / "content_chain" / "content_v5_lf_or_hf_clean_v1.json"
    )
    require_content_v5_manifest_binding()


CONTENT_V5_RUNNER_VARIANT = engine.ContentRunnerVariant(
    name="Content V5",
    execution_scope_id=CONTENT_V5_EXECUTION_SCOPE_ID,
    complete_execution=COMPLETE_EXECUTION,
    arms=CONTENT_V5_ARMS,
    record_contract_id=CONTENT_V5_RECORD_CONTRACT_ID,
    state_schema_id=CONTENT_V5_STATE_SCHEMA_ID,
    run_prefix=CONTENT_V5_RUN_PREFIX,
    load_protocol=_load_protocol,
    load_pipeline_and_assets=v4_runner._load_pipeline_and_assets,
    run_joint=v4_runner._run_joint,
    lf_scorer=v4_runner.score_content_v4_lf_image,
    decision_evaluator=evaluate_content_v5_decision,
)


def execute(args: argparse.Namespace) -> int:
    _load_protocol(Path(args.repo_root).resolve())
    return engine.execute(args, variant=CONTENT_V5_RUNNER_VARIANT)


def _arguments() -> argparse.Namespace:
    return engine._arguments()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
