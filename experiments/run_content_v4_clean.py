"""Thin formal runner entrypoint for the clean Content V4 candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v3_clean as v3_runner
from cegwm.method.content_whitening_v4 import (
    FrozenContentV4LFPublicAssets,
    load_frozen_content_v4_whitening_asset,
    score_content_v4_lf_image,
)
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.protocol.content_chain_v4 import (
    CONTENT_V4_ARMS,
    CONTENT_V4_EXECUTION_SCOPE_ID,
    CONTENT_V4_RECORD_CONTRACT_ID,
    CONTENT_V4_RUN_PREFIX,
    CONTENT_V4_STATE_SCHEMA_ID,
    ContentChainProtocol,
    load_content_v4_clean_protocol,
)
from cegwm.runtime.content_adaptive_sd35_v3 import (
    ContentV3EmbedAssets,
    run_sd35_content_v3,
)

COMPLETE_EXECUTION = "complete_for_content_v4_whitened_lf_adaptive_hf_evaluation"
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ContentV4RunnerAssets:
    """Unchanged V3 embed assets plus the V4-only public LF detector asset."""

    embed_assets: ContentV3EmbedAssets
    lf_public_assets: FrozenContentV4LFPublicAssets

    def __post_init__(self) -> None:
        if not isinstance(self.embed_assets, ContentV3EmbedAssets):
            raise TypeError("Content V4 runner requires Content V3 embed assets")
        if not isinstance(self.lf_public_assets, FrozenContentV4LFPublicAssets):
            raise TypeError("Content V4 runner requires frozen V4 LF public assets")
        if self.lf_public_assets.carrier_assets is not self.embed_assets.lf_public_assets:
            raise ValueError("Content V4 embed and detector must share the LF public carrier assets")

    @property
    def hf_public_assets(self) -> FrozenHFPublicAssets:
        return self.embed_assets.hf_public_assets


def _load_protocol(repo_root: Path) -> ContentChainProtocol:
    root = repo_root / "configs" / "content_chain"
    return load_content_v4_clean_protocol(
        root / "content_v4_clean_v1.json",
        root / "content_adaptive_dual_branch_v2_clean.jsonl",
    )


def _load_pipeline_and_assets(model_id: str, token: str) -> tuple[Any, ContentV4RunnerAssets]:
    pipeline, embed_assets = v3_runner._load_pipeline_and_assets(model_id, token)
    whitening_asset = load_frozen_content_v4_whitening_asset(_REPO_ROOT)
    return pipeline, ContentV4RunnerAssets(
        embed_assets,
        FrozenContentV4LFPublicAssets(embed_assets.lf_public_assets, whitening_asset),
    )


def _run_joint(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentV4RunnerAssets,
    **kwargs: Any,
) -> Any:
    if not isinstance(assets, ContentV4RunnerAssets):
        raise TypeError("Content V4 joint runner requires ContentV4RunnerAssets")
    return run_sd35_content_v3(
        pipeline,
        prompt,
        detection_key,
        assets.embed_assets,
        **kwargs,
    )


CONTENT_V4_RUNNER_VARIANT = engine.ContentRunnerVariant(
    name="Content V4",
    execution_scope_id=CONTENT_V4_EXECUTION_SCOPE_ID,
    complete_execution=COMPLETE_EXECUTION,
    arms=CONTENT_V4_ARMS,
    record_contract_id=CONTENT_V4_RECORD_CONTRACT_ID,
    state_schema_id=CONTENT_V4_STATE_SCHEMA_ID,
    run_prefix=CONTENT_V4_RUN_PREFIX,
    load_protocol=_load_protocol,
    load_pipeline_and_assets=_load_pipeline_and_assets,
    run_joint=_run_joint,
    lf_scorer=score_content_v4_lf_image,
)


def execute(args: argparse.Namespace) -> int:
    return engine.execute(args, variant=CONTENT_V4_RUNNER_VARIANT)


def _arguments() -> argparse.Namespace:
    return engine._arguments()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
