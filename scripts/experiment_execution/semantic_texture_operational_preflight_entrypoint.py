"""Git-less package entrypoint for the zero-science operational runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


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
    """The package entrypoint could not preserve its Phase A boundary."""


def execute_semantic_texture_operational_preflight_entrypoint(
    *,
    adapter: object,
    configuration_path: str | Path,
    source_revision: str,
    run_id: str,
    package_identity: str,
    base_latent: object,
    detection_key: str,
    semantic_runtime: object,
):
    """Invoke only the package-local runner with live write inputs."""

    configuration = _RUNNER.load_semantic_texture_operational_configuration(
        configuration_path
    )
    return _RUNNER.execute_semantic_texture_operational_preflight(
        adapter,
        configuration,
        source_revision=source_revision,
        run_id=run_id,
        package_identity=package_identity,
        base_latent=base_latent,
        detection_key=detection_key,
        semantic_runtime=semantic_runtime,
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
    parser.add_argument(
        "--describe-boundary",
        action="store_true",
        help="print the fixed Phase A boundary without loading a model",
    )
    arguments = parser.parse_args(argv)
    if arguments.describe_boundary:
        print(json.dumps(_boundary_description(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
