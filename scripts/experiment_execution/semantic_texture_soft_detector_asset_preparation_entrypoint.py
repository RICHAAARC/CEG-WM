"""Repository-local Phase-B clean primary-null asset-preparation entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Sequence

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.methods import CegWmExperimentAdapter, load_ceg_wm_experiment_adapter_configuration
from experiments.runners.semantic_texture_soft_detector_asset_preparation import (
    SemanticTextureSoftDetectorAssetPreparationConfiguration,
    prepare_semantic_texture_soft_detector_assets,
)
from runtime import InspyrenetSemanticRuntime, Sd35PipelineBackend, create_runtime_adapter
from scripts.experiment_execution import semantic_texture_soft_detector_asset_preparation_server as delivery_server


CONFIGURATION_PATH = PACKAGE_ROOT / "configs/experiments/semantic_texture_soft_detector_asset_preparation.json"
RUNTIME_CONFIGURATION_PATH = PACKAGE_ROOT / "configs/runtime/runtime_sd35_flowmatch.json"
ADAPTER_CONFIGURATION_PATH = PACKAGE_ROOT / "configs/experiments/internal_execution_components.json"
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class SemanticTextureSoftDetectorAssetEntrypointError(RuntimeError):
    def __init__(self, blocked_class: str) -> None:
        super().__init__(blocked_class)
        self.blocked_class = blocked_class


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SemanticTextureSoftDetectorAssetEntrypointError("environment_blocked")
    path = Path(value)
    if not path.is_absolute():
        raise SemanticTextureSoftDetectorAssetEntrypointError("integrity_blocked")
    return path.resolve()


def _load_configuration() -> SemanticTextureSoftDetectorAssetPreparationConfiguration:
    try:
        raw = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticTextureSoftDetectorAssetEntrypointError("integrity_blocked") from exc
    expected = {
        "schema_version", "profile_id", "asset_bundle_namespace", "whitening_fit_manifest_path",
        "branch_null_manifest_path", "whitening_fit_source_cluster_count", "branch_null_source_cluster_count",
        "diagnostic_only", "zero_science_boundary",
    }
    if (
        type(raw) is not dict or set(raw) != expected or raw["schema_version"] != 1
        or raw["profile_id"] != "semantic_texture_soft_detector_asset_preparation"
        or raw["asset_bundle_namespace"] != "semantic_texture_soft_detector_assets_primary_null_v1"
        or raw["whitening_fit_source_cluster_count"] != 32 or raw["branch_null_source_cluster_count"] != 32
        or raw["diagnostic_only"] is not True
        or raw["zero_science_boundary"] != {"aggregate": None, "candidate_promoted": False, "formal_tau_created": False, "science_started": False, "scientific_claims_supported": False, "scientific_unit_count": 0}
    ):
        raise SemanticTextureSoftDetectorAssetEntrypointError("integrity_blocked")
    return SemanticTextureSoftDetectorAssetPreparationConfiguration(
        whitening_fit_manifest_path=raw["whitening_fit_manifest_path"],
        branch_null_manifest_path=raw["branch_null_manifest_path"],
        target_prompt="a red cube", target_seed=2026081701,
    )


def execute_semantic_texture_soft_detector_asset_preparation_entrypoint(*, observed_repository_revision: str, run_id: str, output_root: str | Path) -> tuple[int, dict[str, object]]:
    """Prepare one exact 32+32 bundle and persist a bounded terminal receipt."""
    if _REVISION.fullmatch(observed_repository_revision) is None:
        raise SemanticTextureSoftDetectorAssetEntrypointError("integrity_blocked")
    blocked_class = "implementation_blocked"
    delivery_parent = Path(output_root).resolve()
    try:
        if delivery_parent.exists():
            raise SemanticTextureSoftDetectorAssetEntrypointError("integrity_blocked")
        delivery_parent.mkdir(parents=True)
        configuration = _load_configuration()
        hf_token, root_key = os.environ.get("HF_TOKEN"), os.environ.get("CEG_WM_ROOT_KEY")
        if not hf_token or not root_key:
            raise SemanticTextureSoftDetectorAssetEntrypointError("environment_blocked")
        checkpoint_path = _required_path("CEG_WM_INSPYRENET_CHECKPOINT_PATH")
        cache_root = _required_path("CEG_WM_CACHE_ROOT")
        persistent_root = _required_path("CEG_WM_PERSISTENT_ROOT")
        backend = Sd35PipelineBackend(cache_root=cache_root, persistent_root=persistent_root, hf_token=hf_token, prompt="bond", negative_prompt="")
        runtime_adapter = create_runtime_adapter(backend, RUNTIME_CONFIGURATION_PATH)
        session = runtime_adapter.initialize("cuda")
        semantic_runtime = InspyrenetSemanticRuntime(checkpoint_path, selected_device=session.selected_device)
        adapter = CegWmExperimentAdapter(load_ceg_wm_experiment_adapter_configuration(ADAPTER_CONFIGURATION_PATH), runtime_adapter)
        def base_latent_for_seed(seed: int) -> torch.Tensor:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            return torch.randn((1, 16, session.image_height // 8, session.image_width // 8), dtype=torch.float32, device="cpu", generator=generator).to(device=session.selected_device, dtype=torch.float16)
        bundle = prepare_semantic_texture_soft_detector_assets(
            adapter, configuration, repository_root=PACKAGE_ROOT, detection_key=root_key,
            semantic_runtime=semantic_runtime, base_latent_for_seed=base_latent_for_seed,
            set_generation_prompt=backend.set_development_generation_prompts,
        )
        return delivery_server.finalize_semantic_texture_soft_detector_asset_delivery(
            bundle,
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            output_root=delivery_parent,
        )
    except Exception as error:
        if isinstance(error, SemanticTextureSoftDetectorAssetEntrypointError):
            blocked_class = error.blocked_class
        elif isinstance(error, (MemoryError, OSError)):
            blocked_class = "resource_blocked"
        return delivery_server.finalize_semantic_texture_soft_detector_asset_blocked_delivery(
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            blocked_class=blocked_class,
            output_root=delivery_parent,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--observed-repository-revision")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    arguments = parser.parse_args(argv)
    if not arguments.execute or any(value is None for value in (arguments.observed_repository_revision, arguments.run_id, arguments.output_root)):
        parser.error("production execution arguments are incomplete")
    code, receipt = execute_semantic_texture_soft_detector_asset_preparation_entrypoint(observed_repository_revision=arguments.observed_repository_revision, run_id=arguments.run_id, output_root=arguments.output_root)
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
