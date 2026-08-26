"""Thin formal runner entrypoint for the clean content-whitening candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments import content_adaptive_engine as engine
from experiments import content_unweighted_engine as v3_runner
from cegwm.method.content_whitening import (
    FrozenContentWhiteningLFPublicAssets,
    load_frozen_content_whitening_asset,
    score_content_whitened_lf_image,
)
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.protocol.content_whitening import (
    CONTENT_WHITENING_ARMS,
    CONTENT_WHITENING_EXECUTION_SCOPE_ID,
    CONTENT_WHITENING_RECORD_CONTRACT_ID,
    CONTENT_WHITENING_RUN_PREFIX,
    CONTENT_WHITENING_STATE_SCHEMA_ID,
    ContentChainProtocol,
    load_content_whitening_protocol,
)
from cegwm.runtime.content_unweighted_sd35 import (
    ContentUnweightedEmbedAssets,
    run_sd35_content_unweighted,
)

COMPLETE_EXECUTION = "complete_for_content_v4_whitened_lf_adaptive_hf_evaluation"
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ContentWhiteningRunnerAssets:
    """Content-unweighted embed assets plus the content-whitening public LF detector asset."""

    embed_assets: ContentUnweightedEmbedAssets
    lf_public_assets: FrozenContentWhiteningLFPublicAssets

    def __post_init__(self) -> None:
        if not isinstance(self.embed_assets, ContentUnweightedEmbedAssets):
            raise TypeError("content-whitening runner requires content-unweighted embed assets")
        if not isinstance(self.lf_public_assets, FrozenContentWhiteningLFPublicAssets):
            raise TypeError("content-whitening runner requires frozen content-whitening LF public assets")
        if self.lf_public_assets.carrier_assets is not self.embed_assets.lf_public_assets:
            raise ValueError("content-whitening embed and detector must share the LF public carrier assets")

    @property
    def hf_public_assets(self) -> FrozenHFPublicAssets:
        return self.embed_assets.hf_public_assets


def _load_protocol(repo_root: Path) -> ContentChainProtocol:
    root = repo_root / "configs" / "content_chain"
    return load_content_whitening_protocol(
        root / "content_v4_clean_v1.json",
        root / "content_adaptive_dual_branch_v2_clean.jsonl",
    )


def _load_pipeline_and_assets(model_id: str, token: str) -> tuple[Any, ContentWhiteningRunnerAssets]:
    pipeline, embed_assets = v3_runner._load_pipeline_and_assets(model_id, token)
    whitening_asset = load_frozen_content_whitening_asset(_REPO_ROOT)
    return pipeline, ContentWhiteningRunnerAssets(
        embed_assets,
        FrozenContentWhiteningLFPublicAssets(embed_assets.lf_public_assets, whitening_asset),
    )


def _run_joint(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentWhiteningRunnerAssets,
    **kwargs: Any,
) -> Any:
    if not isinstance(assets, ContentWhiteningRunnerAssets):
        raise TypeError("content-whitening joint runner requires ContentWhiteningRunnerAssets")
    return run_sd35_content_unweighted(
        pipeline,
        prompt,
        detection_key,
        assets.embed_assets,
        **kwargs,
    )


CONTENT_WHITENING_RUNNER_VARIANT = engine.ContentRunnerVariant(
    name="content-whitening",
    execution_scope_id=CONTENT_WHITENING_EXECUTION_SCOPE_ID,
    complete_execution=COMPLETE_EXECUTION,
    arms=CONTENT_WHITENING_ARMS,
    record_contract_id=CONTENT_WHITENING_RECORD_CONTRACT_ID,
    state_schema_id=CONTENT_WHITENING_STATE_SCHEMA_ID,
    run_prefix=CONTENT_WHITENING_RUN_PREFIX,
    load_protocol=_load_protocol,
    load_pipeline_and_assets=_load_pipeline_and_assets,
    run_joint=_run_joint,
    lf_scorer=score_content_whitened_lf_image,
)


def execute(args: argparse.Namespace) -> int:
    return engine.execute(args, variant=CONTENT_WHITENING_RUNNER_VARIANT)


def _arguments() -> argparse.Namespace:
    return engine._arguments()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
