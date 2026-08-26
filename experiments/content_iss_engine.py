"""Thin formal runner entrypoint for the clean content ISS ISS candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments import content_adaptive_engine as engine
from experiments import content_whitening_engine as v4_runner
from cegwm.method.content_iss import load_frozen_content_iss_asset
from cegwm.method.content_whitening import (
    FrozenContentWhiteningLFPublicAssets,
    score_content_whitened_lf_image,
)
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.protocol.content_iss import (
    CONTENT_ISS_ARMS,
    CONTENT_ISS_EXECUTION_SCOPE_ID,
    CONTENT_ISS_RECORD_CONTRACT_ID,
    CONTENT_ISS_RUN_PREFIX,
    CONTENT_ISS_STATE_SCHEMA_ID,
    ContentChainProtocol,
    load_content_iss_protocol,
)
from cegwm.runtime.content_unweighted_sd35 import ContentUnweightedEmbedAssets
from cegwm.runtime.content_iss_sd35 import (
    ContentISSEvaluationAssets,
    run_content_iss_evaluation_pair,
)

COMPLETE_EXECUTION = "complete_for_content_v6_detector_domain_iss_evaluation"
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ContentISSRunnerAssets:
    evaluation_assets: ContentISSEvaluationAssets

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_assets, ContentISSEvaluationAssets):
            raise TypeError("content ISS runner requires evaluation assets")

    @property
    def embed_assets(self) -> ContentUnweightedEmbedAssets:
        return self.evaluation_assets.embed_assets

    @property
    def hf_public_assets(self) -> FrozenHFPublicAssets:
        return self.evaluation_assets.hf_public_assets

    @property
    def lf_public_assets(self) -> FrozenContentWhiteningLFPublicAssets:
        return self.evaluation_assets.lf_public_assets


def _load_protocol(repo_root: Path) -> ContentChainProtocol:
    return load_content_iss_protocol(repo_root)


def _load_pipeline_and_assets(model_id: str, token: str) -> tuple[Any, ContentISSRunnerAssets]:
    pipeline, v4_assets = v4_runner._load_pipeline_and_assets(model_id, token)
    evaluation_assets = ContentISSEvaluationAssets(
        v4_assets.embed_assets,
        v4_assets.lf_public_assets,
        load_frozen_content_iss_asset(_REPO_ROOT),
    )
    return pipeline, ContentISSRunnerAssets(evaluation_assets)


def _run_pair(
    pipeline: Any,
    prompt: str,
    detection_key: str | bytes | bytearray | memoryview,
    assets: ContentISSRunnerAssets,
    *,
    height: int,
    width: int,
    seed: int,
) -> Any:
    if not isinstance(assets, ContentISSRunnerAssets):
        raise TypeError("content ISS paired runner requires ContentISSRunnerAssets")
    return run_content_iss_evaluation_pair(
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
    raise RuntimeError("content ISS must use its sole paired pass1/pass2 runtime")


CONTENT_ISS_RUNNER_VARIANT = engine.ContentRunnerVariant(
    name="content ISS",
    execution_scope_id=CONTENT_ISS_EXECUTION_SCOPE_ID,
    complete_execution=COMPLETE_EXECUTION,
    arms=CONTENT_ISS_ARMS,
    record_contract_id=CONTENT_ISS_RECORD_CONTRACT_ID,
    state_schema_id=CONTENT_ISS_STATE_SCHEMA_ID,
    run_prefix=CONTENT_ISS_RUN_PREFIX,
    load_protocol=_load_protocol,
    load_pipeline_and_assets=_load_pipeline_and_assets,
    run_joint=_unpaired_forbidden,
    lf_scorer=score_content_whitened_lf_image,
    run_pair=_run_pair,
)


def execute(args: argparse.Namespace) -> int:
    return engine.execute(args, variant=CONTENT_ISS_RUNNER_VARIANT)


def _arguments() -> argparse.Namespace:
    return engine._arguments()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
