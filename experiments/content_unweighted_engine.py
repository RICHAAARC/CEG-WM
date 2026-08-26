"""Local formal runner entrypoint for the clean content-unweighted candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from experiments import content_adaptive_engine as engine
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
)
from cegwm.method.hf import FrozenHFPublicAssets
from cegwm.protocol.content_unweighted import (
    CONTENT_UNWEIGHTED_ARMS,
    CONTENT_UNWEIGHTED_EXECUTION_SCOPE_ID,
    CONTENT_UNWEIGHTED_RECORD_CONTRACT_ID,
    CONTENT_UNWEIGHTED_RUN_PREFIX,
    CONTENT_UNWEIGHTED_STATE_SCHEMA_ID,
    ContentChainProtocol,
    load_content_unweighted_clean_protocol,
)
from cegwm.runtime.content_unweighted_sd35 import (
    ContentUnweightedEmbedAssets,
    load_dino_content_assets,
    run_sd35_content_unweighted,
)
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline

COMPLETE_EXECUTION = "complete_for_content_unweighted_evaluation"


def _load_protocol(repo_root: Path) -> ContentChainProtocol:
    root = repo_root / "configs" / "content_chain"
    return load_content_unweighted_clean_protocol(
        root / "content_v3_clean_v1.json",
        root / "content_adaptive_dual_branch_v2_clean.jsonl",
    )


def _load_pipeline_and_assets(
    model_id: str,
    token: str,
) -> tuple[Any, ContentUnweightedEmbedAssets]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_content_unweighted_execution")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=token)
    pipeline.to("cuda")
    vae = getattr(pipeline, "vae", None)
    processor = getattr(pipeline, "image_processor", None)
    hf = FrozenHFPublicAssets(
        vae=vae,
        image_processor=processor,
        image_processor_id=f"{model_id}:image_processor",
    )
    lf = FrozenLFPublicAssets(
        vae=vae,
        image_processor=processor,
        image_processor_id=f"{model_id}:image_processor",
        candidate_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        evaluated_candidate_id=LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    dino_model, dino_processor = load_dino_content_assets(token=token)
    dino_model.to("cuda")
    dino_model.eval()
    return pipeline, ContentUnweightedEmbedAssets(dino_model, dino_processor, hf, lf)


CONTENT_UNWEIGHTED_RUNNER_VARIANT = engine.ContentRunnerVariant(
    name="content-unweighted",
    execution_scope_id=CONTENT_UNWEIGHTED_EXECUTION_SCOPE_ID,
    complete_execution=COMPLETE_EXECUTION,
    arms=CONTENT_UNWEIGHTED_ARMS,
    record_contract_id=CONTENT_UNWEIGHTED_RECORD_CONTRACT_ID,
    state_schema_id=CONTENT_UNWEIGHTED_STATE_SCHEMA_ID,
    run_prefix=CONTENT_UNWEIGHTED_RUN_PREFIX,
    load_protocol=_load_protocol,
    load_pipeline_and_assets=_load_pipeline_and_assets,
    run_joint=run_sd35_content_unweighted,
)


def execute(args: argparse.Namespace) -> int:
    return engine.execute(args, variant=CONTENT_UNWEIGHTED_RUNNER_VARIANT)


def _arguments() -> argparse.Namespace:
    return engine._arguments()


if __name__ == "__main__":
    raise SystemExit(execute(_arguments()))
