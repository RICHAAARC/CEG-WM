"""Thin reference-only V6 runner over the frozen Content V2 roster."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v6_clean as v6_runner
from cegwm.protocol.content_chain_v2 import ContentChainProtocol
from cegwm.protocol.content_chain_v6_reference_oldroster import (
    CONTENT_V6_REFERENCE_OLDROSTER_ARMS,
    CONTENT_V6_REFERENCE_OLDROSTER_EXECUTION_SCOPE_ID,
    CONTENT_V6_REFERENCE_OLDROSTER_RECORD_CONTRACT_ID,
    CONTENT_V6_REFERENCE_OLDROSTER_RUN_PREFIX,
    CONTENT_V6_REFERENCE_OLDROSTER_STATE_SCHEMA_ID,
    load_content_v6_reference_oldroster_protocol,
)

COMPLETE_EXECUTION = "complete_for_content_v6_reference_oldroster_evaluation"


def _load_protocol(repo_root: Path) -> ContentChainProtocol:
    return load_content_v6_reference_oldroster_protocol(repo_root)


CONTENT_V6_REFERENCE_OLDROSTER_RUNNER_VARIANT = replace(
    v6_runner.CONTENT_V6_RUNNER_VARIANT,
    name="Content V6 reference old roster",
    execution_scope_id=CONTENT_V6_REFERENCE_OLDROSTER_EXECUTION_SCOPE_ID,
    complete_execution=COMPLETE_EXECUTION,
    arms=CONTENT_V6_REFERENCE_OLDROSTER_ARMS,
    record_contract_id=CONTENT_V6_REFERENCE_OLDROSTER_RECORD_CONTRACT_ID,
    state_schema_id=CONTENT_V6_REFERENCE_OLDROSTER_STATE_SCHEMA_ID,
    run_prefix=CONTENT_V6_REFERENCE_OLDROSTER_RUN_PREFIX,
    load_protocol=_load_protocol,
)


def execute(args: argparse.Namespace) -> int:
    return engine.execute(args, variant=CONTENT_V6_REFERENCE_OLDROSTER_RUNNER_VARIANT)


def _arguments() -> argparse.Namespace:
    return engine._arguments()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
