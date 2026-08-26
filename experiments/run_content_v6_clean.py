"""Thin formal runner entrypoint for the clean Content V6 ISS candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments import run_content_adaptive_dual_branch_v2_clean as engine
from experiments import run_content_v4_clean as v4_runner
from cegwm.method.content_iss_v6 import load_frozen_content_v6_iss_asset
from cegwm.method.content_whitening_v4 import (
    FrozenContentV4LFPublicAssets,
    score_content_v4_lf_image,
)
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.protocol.content_chain_v6 import (
    CONTENT_V6_ARMS,
    CONTENT_V6_EXECUTION_SCOPE_ID,
    CONTENT_V6_RECORD_CONTRACT_ID,
    CONTENT_V6_RUN_PREFIX,
    CONTENT_V6_STATE_SCHEMA_ID,
    ContentChainProtocol,
    load_content_v6_clean_protocol,
)
from cegwm.runtime.content_adaptive_sd35_v3 import ContentV3EmbedAssets
from cegwm.runtime.content_iss_sd35_v6 import (
    ContentV6EvaluationAssets,
    run_content_v6_evaluation_pair,
)

COMPLETE_EXECUTION = "complete_for_content_v6_detector_domain_iss_evaluation"
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ContentV6RunnerAssets:
    evaluation_assets: ContentV6EvaluationAssets

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_assets, ContentV6EvaluationAssets):
            raise TypeError("Content V6 runner requires evaluation assets")

    @property
    def embed_assets(self) -> ContentV3EmbedAssets:
        return self.evaluation_assets.embed_assets

    @property
    def hf_public_assets(self) -> FrozenHFPublicAssets:
        return self.evaluation_assets.hf_public_assets

    @property
    def lf_public_assets(self) -> FrozenContentV4LFPublicAssets:
        return self.evaluation_assets.lf_public_assets


def _load_protocol(repo_root: Path) -> ContentChainProtocol:
    return load_content_v6_clean_protocol(repo_root)


def _load_pipeline_and_assets(model_id: str, token: str) -> tuple[Any, ContentV6RunnerAssets]:
    pipeline, v4_assets = v4_runner._load_pipeline_and_assets(model_id, token)
    evaluation_assets = ContentV6EvaluationAssets(
        v4_assets.embed_assets,
        v4_assets.lf_public_assets,
        load_frozen_content_v6_iss_asset(_REPO_ROOT),
    )
    return pipeline, ContentV6RunnerAssets(evaluation_assets)


def _run_pair(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentV6RunnerAssets,
    *,
    height: int,
    width: int,
    seed: int,
) -> Any:
    if not isinstance(assets, ContentV6RunnerAssets):
        raise TypeError("Content V6 paired runner requires ContentV6RunnerAssets")
    return run_content_v6_evaluation_pair(
        pipeline,
        prompt,
        detection_key,
        assets.evaluation_assets,
        height=height,
        width=width,
        seed=seed,
    )


def _unpaired_forbidden(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError("Content V6 must use its sole paired pass1/pass2 runtime")


CONTENT_V6_RUNNER_VARIANT = engine.ContentRunnerVariant(
    name="Content V6",
    execution_scope_id=CONTENT_V6_EXECUTION_SCOPE_ID,
    complete_execution=COMPLETE_EXECUTION,
    arms=CONTENT_V6_ARMS,
    record_contract_id=CONTENT_V6_RECORD_CONTRACT_ID,
    state_schema_id=CONTENT_V6_STATE_SCHEMA_ID,
    run_prefix=CONTENT_V6_RUN_PREFIX,
    load_protocol=_load_protocol,
    load_pipeline_and_assets=_load_pipeline_and_assets,
    run_joint=_unpaired_forbidden,
    lf_scorer=score_content_v4_lf_image,
    run_pair=_run_pair,
)


def execute(args: argparse.Namespace) -> int:
    return engine.execute(args, variant=CONTENT_V6_RUNNER_VARIANT)


def _arguments() -> argparse.Namespace:
    return engine._arguments()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
