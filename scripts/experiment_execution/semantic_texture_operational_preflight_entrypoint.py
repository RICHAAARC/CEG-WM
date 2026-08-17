"""Repository-local production entrypoint for semantic-texture preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from runtime import (
    InspyrenetSemanticRuntime,
    Sd35PipelineBackend,
    create_runtime_adapter,
)
from scripts.experiment_execution import (
    semantic_texture_operational_preflight_server as delivery_server,
)


CONFIGURATION_PATH = (
    PACKAGE_ROOT / "configs/experiments/semantic_texture_operational_preflight.json"
)
RUNTIME_CONFIGURATION_PATH = (
    PACKAGE_ROOT / "configs/runtime/runtime_sd35_flowmatch.json"
)
ADAPTER_CONFIGURATION_PATH = (
    PACKAGE_ROOT / "configs/experiments/internal_execution_components.json"
)


def _load_runner_module():
    runner_path = (
        PACKAGE_ROOT
        / "experiments/runners/semantic_texture_operational_preflight.py"
    )
    specification = importlib.util.spec_from_file_location(
        "ceg_wm_semantic_texture_operational_preflight_runner",
        runner_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("operational runner module is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_RUNNER = _load_runner_module()


class SemanticTextureOperationalEntrypointError(RuntimeError):
    """A classified trusted-package failure before method execution."""

    def __init__(self, blocked_class: str) -> None:
        if blocked_class not in _RUNNER.ALLOWED_BLOCKED_CLASSES:
            raise ValueError("entrypoint blocked class is not registered")
        super().__init__(blocked_class)
        self.blocked_class = blocked_class


def _required_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SemanticTextureOperationalEntrypointError("environment_blocked")
    path = Path(value)
    if not path.is_absolute():
        raise SemanticTextureOperationalEntrypointError("integrity_blocked")
    return path.resolve()


def _pre_execution_blocked_class(error: BaseException) -> str:
    if isinstance(error, SemanticTextureOperationalEntrypointError):
        return error.blocked_class
    if isinstance(error, (MemoryError, OSError)):
        return "resource_blocked"
    return "implementation_blocked"


def execute_semantic_texture_operational_preflight_entrypoint(
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
) -> tuple[int, dict[str, object]]:
    """Construct only registered public runtime objects and run both-unit roster."""

    configuration = _RUNNER.load_semantic_texture_operational_configuration(
        CONFIGURATION_PATH
    )
    try:
        hf_token = os.environ.get("HF_TOKEN")
        root_key = os.environ.get("CEG_WM_ROOT_KEY")
        if not hf_token or not root_key:
            raise SemanticTextureOperationalEntrypointError(
                "environment_blocked"
            )
        source_root = _required_environment_path(
            "CEG_WM_INSPYRENET_SOURCE_ROOT"
        )
        checkpoint_path = _required_environment_path(
            "CEG_WM_INSPYRENET_CHECKPOINT_PATH"
        )
        cache_root = _required_environment_path("CEG_WM_CACHE_ROOT")
        persistent_root = _required_environment_path("CEG_WM_PERSISTENT_ROOT")
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        backend = Sd35PipelineBackend(
            cache_root=cache_root,
            persistent_root=persistent_root,
            hf_token=hf_token,
            prompt=configuration.generation_prompt,
            negative_prompt=configuration.generation_negative_prompt,
        )
        runtime_adapter = create_runtime_adapter(
            backend,
            RUNTIME_CONFIGURATION_PATH,
        )
        runtime_session = runtime_adapter.initialize("cuda")
        semantic_runtime = InspyrenetSemanticRuntime(
            checkpoint_path,
            selected_device=runtime_session.selected_device,
        )
        adapter_configuration = load_ceg_wm_experiment_adapter_configuration(
            ADAPTER_CONFIGURATION_PATH
        )
        adapter = CegWmExperimentAdapter(
            adapter_configuration,
            runtime_adapter,
        )
        latent_generator = torch.Generator(device="cpu")
        latent_generator.manual_seed(configuration.generation_seed)
        base_latent_cpu = torch.randn(
            (
                1,
                16,
                runtime_session.image_height // 8,
                runtime_session.image_width // 8,
            ),
            dtype=torch.float32,
            device="cpu",
            generator=latent_generator,
        )
        base_latent = base_latent_cpu.to(
            device=runtime_session.selected_device,
            dtype=torch.float16,
        )
        result = _RUNNER.execute_semantic_texture_operational_preflight(
            adapter,
            configuration,
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            base_latent=base_latent,
            detection_key=root_key,
            semantic_runtime=semantic_runtime,
        )
    except Exception as error:
        result = _RUNNER.create_semantic_texture_operational_pre_execution_failure(
            configuration,
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            blocked_class=_pre_execution_blocked_class(error),
        )
    return delivery_server.finalize_semantic_texture_operational_preflight_delivery(
        result,
        output_root=output_root,
    )


def _boundary_description() -> dict[str, object]:
    return {
        "aggregate": None,
        "asset_authority_status": "identity_blocked",
        "candidate_promoted": False,
        "formal_tau_created": False,
        "profile_id": "semantic_texture_operational_preflight",
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "unit_roster": [
            "semantic_texture_write_operational",
            "semantic_texture_blind_detection_operational",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe-boundary", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--observed-repository-revision")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    arguments = parser.parse_args(argv)
    if arguments.describe_boundary:
        print(json.dumps(_boundary_description(), indent=2, sort_keys=True))
        return 0
    if not arguments.execute or any(
        value is None
        for value in (
            arguments.observed_repository_revision,
            arguments.run_id,
            arguments.output_root,
        )
    ):
        parser.error("production execution arguments are incomplete")
    exit_code, receipt = execute_semantic_texture_operational_preflight_entrypoint(
        observed_repository_revision=arguments.observed_repository_revision,
        run_id=arguments.run_id,
        output_root=arguments.output_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
