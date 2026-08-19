"""Repository-local production entrypoint for semantic-texture preflight."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import stat
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
from experiments.methods.ceg_wm import (
    materialize_semantic_texture_soft_detector_asset_bundle,
)
from experiments.protocol.semantic_texture_soft_detector_assets import (
    SemanticTextureSoftDetectorAssetBundle,
)
from runtime import (
    InspyrenetSemanticRuntime,
    Sd35PipelineBackend,
    create_runtime_adapter,
)
from runtime.routing_observation import (
    InspyrenetSemanticRuntimeInitializationError,
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


def _load_authenticated_detector_assets(
    path: str | Path,
    configuration: object,
) -> tuple[object, object, object]:
    """Read one configured bundle without path following or replacement."""

    bundle_path = Path(path)
    try:
        before = bundle_path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise OSError
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(bundle_path, flags)
        try:
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
            ):
                raise OSError
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                blob = handle.read()
        finally:
            os.close(descriptor)
        if sha256(blob).hexdigest() != configuration.detector_asset_bundle_sha256:
            raise ValueError
        bundle = SemanticTextureSoftDetectorAssetBundle.from_mapping(json.loads(blob))
        if bundle.bundle_digest != configuration.detector_asset_bundle_digest:
            raise ValueError
        return materialize_semantic_texture_soft_detector_asset_bundle(bundle)
    except MemoryError:
        raise SemanticTextureOperationalEntrypointError("resource_blocked") from None
    except Exception:
        raise SemanticTextureOperationalEntrypointError("integrity_blocked") from None


def execute_semantic_texture_operational_preflight_entrypoint(
    *,
    observed_repository_revision: str,
    run_id: str,
    output_root: str | Path,
    detector_asset_bundle: str | Path,
) -> tuple[int, dict[str, object]]:
    """Construct only registered public runtime objects and run both-unit roster."""

    configuration = _RUNNER.load_semantic_texture_operational_configuration(
        CONFIGURATION_PATH
    )
    pre_execution_stage = "required_environment"
    semantic_runtime_initialization_step: str | None = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
        root_key = os.environ.get("CEG_WM_ROOT_KEY")
        if not hf_token or not root_key:
            raise SemanticTextureOperationalEntrypointError(
                "environment_blocked"
            )
        checkpoint_path = _required_environment_path(
            "CEG_WM_INSPYRENET_CHECKPOINT_PATH"
        )
        cache_root = _required_environment_path("CEG_WM_CACHE_ROOT")
        persistent_root = _required_environment_path("CEG_WM_PERSISTENT_ROOT")
        pre_execution_stage = "detector_asset_loading"
        whitening_asset, hf_null, lf_null = _load_authenticated_detector_assets(
            detector_asset_bundle,
            configuration,
        )
        pre_execution_stage = "runtime_backend_construction"
        backend = Sd35PipelineBackend(
            cache_root=cache_root,
            persistent_root=persistent_root,
            hf_token=hf_token,
            prompt=configuration.generation_prompt,
            negative_prompt=configuration.generation_negative_prompt,
        )
        pre_execution_stage = "runtime_configuration"
        runtime_adapter = create_runtime_adapter(
            backend,
            RUNTIME_CONFIGURATION_PATH,
        )
        pre_execution_stage = "runtime_initialization"
        runtime_session = runtime_adapter.initialize("cuda")
        pre_execution_stage = "semantic_runtime_initialization"
        semantic_runtime = InspyrenetSemanticRuntime(
            checkpoint_path,
            selected_device=runtime_session.selected_device,
        )
        pre_execution_stage = "experiment_adapter_initialization"
        adapter_configuration = load_ceg_wm_experiment_adapter_configuration(
            ADAPTER_CONFIGURATION_PATH
        )
        adapter = CegWmExperimentAdapter(
            adapter_configuration,
            runtime_adapter,
        )
        pre_execution_stage = "latent_preparation"
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
        pre_execution_stage = "runner_admission"
        result = _RUNNER.execute_semantic_texture_operational_preflight(
            adapter,
            configuration,
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            base_latent=base_latent,
            detection_key=root_key,
            semantic_runtime=semantic_runtime,
            whitening_asset=whitening_asset,
            hf_null=hf_null,
            lf_null=lf_null,
        )
    except Exception as error:
        if (
            pre_execution_stage == "semantic_runtime_initialization"
            and isinstance(error, InspyrenetSemanticRuntimeInitializationError)
        ):
            semantic_runtime_initialization_step = error.step
        result = _RUNNER.create_semantic_texture_operational_pre_execution_failure(
            configuration,
            observed_repository_revision=observed_repository_revision,
            run_id=run_id,
            blocked_class=_pre_execution_blocked_class(error),
            pre_execution_stage=pre_execution_stage,
            semantic_runtime_initialization_step=semantic_runtime_initialization_step,
        )
    return delivery_server.finalize_semantic_texture_operational_preflight_delivery(
        result,
        output_root=output_root,
    )


def _boundary_description() -> dict[str, object]:
    return {
        "aggregate": None,
        "asset_authority_status": "diagnostic_bundle_authenticated",
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
    parser.add_argument("--detector-asset-bundle")
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
            arguments.detector_asset_bundle,
            arguments.detector_asset_bundle,
        )
    ):
        parser.error("production execution arguments are incomplete")
    exit_code, receipt = execute_semantic_texture_operational_preflight_entrypoint(
        observed_repository_revision=arguments.observed_repository_revision,
        run_id=arguments.run_id,
        output_root=arguments.output_root,
        detector_asset_bundle=arguments.detector_asset_bundle,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
