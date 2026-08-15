"""Frozen CPU tests for the salient-local-LF mask/write pilot."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from struct import pack, unpack
import subprocess
import sys

import pytest
import torch

from experiments.methods import CegWmExperimentAdapter, load_ceg_wm_experiment_adapter_configuration
from experiments.metrics.salient_local_lf_mask_write_validation import (
    SalientLocalLfMaskWriteMetricError,
    SalientLocalLfTerminalFailure,
    aggregate_salient_local_lf_mask_write_validation,
    create_mask_write_observation,
    observe_public_rgb8_quality,
)
from experiments.protocol.development_records import DevelopmentScientificRecord
from experiments.protocol.salient_local_lf_mask_write_validation import (
    CANONICAL_CONTENT_RELATIVE_L2_LIMIT,
    SCIENTIFIC_ROSTER_AUTHORITY_DIGEST,
    SalientLocalLfMaskWriteProtocolError,
    canonical_digest,
    _collect_deny_axes,
    load_salient_local_lf_mask_write_validation_protocol,
)
from experiments.runners.salient_local_lf_mask_write_validation import (
    SalientLocalLfMaskWriteIdentityError,
    SalientLocalLfMaskWriteIntegrityError,
    SalientLocalLfMaskWriteRunnerError,
    SalientLocalLfMaskWriteValidationRunner,
    aggregate_supports_scientific_claim,
    _actual_dtype_budget_pass,
)
from experiments.runners.development_persistence import (
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
)
from main import identify_root_key, rgb8_image_digest
from runtime import InspyrenetSaliencyRuntime, Sd35RuntimeAdapter
from runtime import (
    RuntimeBackendIdentity,
    RuntimeDeviceCapabilities,
    RuntimeVaeFactors,
    create_runtime_adapter,
)
from scripts.experiment_execution.salient_local_lf_mask_write_validation_entrypoint import (
    _classify_scientific_failure,
    _safe_failure,
)
from scripts.experiment_execution.build_salient_local_lf_mask_write_validation_package import (
    SalientLocalLfPackageBuildError,
    build_salient_local_lf_mask_write_validation_package,
    resolve_required_git_authority_revisions,
    verify_extracted_salient_local_lf_mask_write_validation_package,
    verify_salient_local_lf_mask_write_validation_package,
)
from scripts.experiment_execution.salient_local_lf_mask_write_validation_server import (
    _extract_verified_execution_package,
    _verify_locked_dependencies,
    hydrate_required_git_authority_revisions,
)
from tests.helpers.historical_repository import materialize_historical_repository


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/salient_local_lf_mask_write_validation.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"
SERVER_MODULE = "scripts.experiment_execution.salient_local_lf_mask_write_validation_server"
SERVER_STARTUP_MEMBERS = {
    "scripts/experiment_execution/__init__.py",
    "scripts/experiment_execution/build_salient_local_lf_mask_write_validation_package.py",
    "scripts/experiment_execution/development_exploration_entrypoint.py",
    "scripts/experiment_execution/development_exploration_server.py",
    "scripts/experiment_execution/salient_local_lf_mask_write_validation_entrypoint.py",
    "scripts/experiment_execution/salient_local_lf_mask_write_validation_server.py",
}
HISTORICAL_CANDIDATE_OVERLAY_REVISION = (
    "19fe5e42ba782ee604c801875fd2e330f9312abb"
)
CANDIDATE_OVERLAY_PATH = (
    ".codex/research_state/salient_local_lf_candidate_readiness.yaml"
)
CANDIDATE_OVERLAY_STATUS_FIELDS = (
    "source_cpu_api_implementation_ready",
    "candidate_runtime_qualified",
    "experiment_protocol_admitted",
    "masked_lf_whitening_asset_ready",
    "rgb_quality_gate_defined",
    "scientific_mechanism_validated",
    "promoted",
    "formal_detector",
    "diagnostic_only",
)


def struct_binary32(value: float) -> float:
    return unpack(">f", pack(">f", value))[0]


def next_binary32(value: float) -> float:
    bits = unpack(">I", pack(">f", value))[0]
    return unpack(">f", pack(">I", bits + 1))[0]


def _protocol():
    return load_salient_local_lf_mask_write_validation_protocol(CONFIG, repository_root=ROOT)


def _exact_commit_for_current_strict_sources(tmp_path: Path) -> str:
    index = tmp_path / "salient_local_lf_exact_package.index"
    environment = {
        **os.environ,
        "GIT_INDEX_FILE": str(index),
        "GIT_AUTHOR_NAME": "CEG-WM package test",
        "GIT_AUTHOR_EMAIL": "ceg-wm-package-test@example.invalid",
        "GIT_COMMITTER_NAME": "CEG-WM package test",
        "GIT_COMMITTER_EMAIL": "ceg-wm-package-test@example.invalid",
    }
    subprocess.run(("git", "read-tree", "HEAD"), cwd=ROOT, env=environment, check=True)
    subprocess.run(
        (
            "git", "add", "--",
            "configs/experiments/salient_local_lf_mask_write_validation.json",
            "configs/experiments/salient_local_lf_mask_write_validation_manifest.json",
            "experiments/protocol/salient_local_lf_mask_write_validation.py",
            "experiments/runners/salient_local_lf_mask_write_validation.py",
            "scripts/experiment_execution/build_salient_local_lf_mask_write_validation_package.py",
            "scripts/experiment_execution/salient_local_lf_mask_write_validation_server.py",
            "tests/unit/test_salient_local_lf_mask_write_validation.py",
        ),
        cwd=ROOT, env=environment, check=True,
    )
    tree = subprocess.run(
        ("git", "write-tree"), cwd=ROOT, env=environment,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    parent = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, env=environment,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return subprocess.run(
        ("git", "commit-tree", tree, "-p", parent), cwd=ROOT, env=environment,
        input="test: materialize exact salient local LF package authority\n",
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _quality(*, over_limit: bool = False):
    clean = torch.full((1, 3, 512, 512), 100, dtype=torch.uint8)
    marked = torch.full_like(clean, 101)
    if over_limit:
        marked.reshape(-1)[0] = 102
    return observe_public_rgb8_quality(
        clean,
        marked,
        clean_image_digest=rgb8_image_digest(clean),
        marked_image_digest=rgb8_image_digest(marked),
    )


def _observation(
    cluster: int,
    *,
    mechanism: bool = True,
    quality_pass: bool = True,
    source_cluster_id: str | None = None,
    identity_pass: bool = True,
    integrity_pass: bool = True,
):
    quality = _quality(over_limit=not quality_pass)
    return create_mask_write_observation(
        cluster_ordinal=cluster,
        source_cluster_id=source_cluster_id or f"{cluster + 1:064x}",
        clean_image_digest=quality.clean_image_digest,
        marked_image_digest=quality.marked_image_digest,
        embed_saliency_observation_identity=f"{cluster + 11:064x}",
        detect_saliency_observation_identity=f"{cluster + 21:064x}",
        embed_mask_identity=f"{cluster + 31:064x}",
        detect_mask_identity=f"{cluster + 41:064x}",
        embed_mask_coverage=256 if mechanism else 32,
        detect_mask_coverage=256,
        mask_intersection_over_union=0.75,
        nominal_masked_lf_outside_bitwise_zero=True,
        nominal_masked_lf_inside_nonzero=True,
        nominal_masked_lf_consumed_by_materialization=True,
        accepted_materialization_replay_identity=f"{cluster + 51:064x}",
        realized_relative_l2=0.01,
        actual_dtype_budget_pass=True,
        identity_pass=identity_pass,
        integrity_pass=integrity_pass,
        quality=quality,
    )


def _runner() -> SalientLocalLfMaskWriteValidationRunner:
    protocol = _protocol()
    runtime = object.__new__(Sd35RuntimeAdapter)
    saliency = object.__new__(InspyrenetSaliencyRuntime)
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS),
        runtime_adapter=runtime,
    )
    return SalientLocalLfMaskWriteValidationRunner(
        protocol=protocol, adapter=adapter, runtime_adapter=runtime,
        saliency_runtime=saliency, method_code_revision="1" * 40,
        registered_root_key="salient-mask-write-test-root",
        protocol_digest=protocol.digest(), execution_intent_authority_digest="2" * 64,
        candidate_config_digest="3" * 64, package_identity="4" * 64,
    )


class _PublicRunnerPosterior:
    def __init__(self, mode_value: torch.Tensor) -> None:
        self._mode_value = mode_value

    def mode(self) -> torch.Tensor:
        return self._mode_value.detach().clone()


class _PublicRunnerBackend:
    def __init__(self) -> None:
        self.configuration = None
        self.run_calls = 0
        self.decode_calls = 0
        self.encode_calls = 0

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(cpu_available=True, cuda_device_count=0)

    def prepare(self, configuration, selected_device: str) -> RuntimeBackendIdentity:
        self.configuration = configuration
        return RuntimeBackendIdentity(
            candidate_id=configuration.candidate_id,
            runtime_config_digest=configuration.runtime_config_digest,
            runtime_backend_name="salient_local_lf_public_runner_cpu_fixture",
            selected_device=selected_device,
            model_id=configuration.model_id,
            model_revision=configuration.model_revision,
            pipeline_class=configuration.pipeline_class,
            scheduler_class=configuration.scheduler_class,
            inference_steps=configuration.inference_steps,
            guidance_scale=configuration.guidance_scale,
            image_height=configuration.image_height,
            image_width=configuration.image_width,
            generation_seed_device=configuration.generation_seed_device,
            latent_dtype=configuration.latent_dtype,
            template_dtype=configuration.template_dtype,
            score_dtype=configuration.score_dtype,
            callback_index=configuration.callback_index,
            callback_hold_scheduler_intervals=configuration.callback_hold_scheduler_intervals,
            vae_decode_protocol=configuration.vae_decode_protocol,
            vae_encode_protocol=configuration.vae_encode_protocol,
            vae_scaling_factor_source=configuration.vae_scaling_factor_source,
            vae_shift_factor_source=configuration.vae_shift_factor_source,
            detection_schedule_index=configuration.detection_schedule_index,
            detection_conditioning_protocol=configuration.detection_conditioning_protocol,
            qk_layer_names=configuration.qk_layer_names,
            dependency_lock=configuration.dependency_lock,
        )

    def close(self) -> None:
        return None

    def run_generation(self, initial_latent, callback):
        assert self.configuration is not None
        self.run_calls += 1
        state = initial_latent.detach().clone()
        for callback_index in range(self.configuration.inference_steps):
            state = callback(callback_index, state)
        return state

    def vae_factors(self) -> RuntimeVaeFactors:
        return RuntimeVaeFactors(scaling_factor=0.5, shift_factor=0.25)

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        image = torch.sigmoid(latent.to(torch.float32).mean(dim=1, keepdim=True))
        return torch.nn.functional.interpolate(
            image.repeat(1, 3, 1, 1), size=(512, 512), mode="bilinear", align_corners=False,
        )

    def vae_encode(self, image: torch.Tensor) -> _PublicRunnerPosterior:
        self.encode_calls += 1
        downsampled = torch.nn.functional.interpolate(
            image.to(torch.float32).mean(dim=1, keepdim=True),
            size=(64, 64), mode="bilinear", align_corners=False,
        )
        return _PublicRunnerPosterior(downsampled.repeat(1, 16, 1, 1))


class _PublicRunnerSaliencyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_count = 0

    def forward_inspyre(self, model_input: torch.Tensor) -> dict[str, object]:
        self.forward_count += 1
        assert tuple(model_input.shape) == (1, 3, 1024, 1024)
        raw = torch.full((1, 1, 64, 64), -10.0, dtype=torch.float32)
        raw[:, :, 16:48, 16:48] = 10.0
        return {
            "saliency": [
                torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                raw,
            ]
        }


def _public_runner() -> tuple[
    SalientLocalLfMaskWriteValidationRunner,
    _PublicRunnerBackend,
    _PublicRunnerSaliencyModel,
]:
    protocol = _protocol()
    backend = _PublicRunnerBackend()
    runtime = create_runtime_adapter(backend)
    runtime.initialize("cpu")
    model = _PublicRunnerSaliencyModel()
    saliency = object.__new__(InspyrenetSaliencyRuntime)
    saliency._device = torch.device("cpu")
    saliency._model = model
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS),
        runtime_adapter=runtime,
    )
    runner = SalientLocalLfMaskWriteValidationRunner(
        protocol=protocol, adapter=adapter, runtime_adapter=runtime,
        saliency_runtime=saliency, method_code_revision="1" * 40,
        registered_root_key="salient-local-lf-public-runner-root",
        protocol_digest=protocol.digest(), execution_intent_authority_digest="2" * 64,
        candidate_config_digest="3" * 64, package_identity="4" * 64,
    )
    return runner, backend, model


def test_authored_roster_and_historical_producer_authorities_are_exact() -> None:
    protocol = _protocol()
    assert protocol.manifest.scientific_roster_authority_digest == SCIENTIFIC_ROSTER_AUTHORITY_DIGEST
    assert canonical_digest(protocol.manifest.authority_payload()) == SCIENTIFIC_ROSTER_AUTHORITY_DIGEST
    assert len(protocol.unit_roster) == 10
    assert tuple(item.generation_seed for item in protocol.manifest.entries) == tuple(range(202608150100, 202608150108))
    assert tuple(item.source_cluster_id for item in protocol.manifest.entries) == (
        "dd32b622fef8f72ec34ab75821f07d4f6aac09357e3e9ae64ba1dd3b088841b9",
        "134b101f00c67ff3f7a572599f48b9e08375ea4b99bd0758f0cbdda6372dcace",
        "9e80a52b9811bd818d4fc54e737b5d14352f07ab28fcf1ffb4e58b9a43efa3cb",
        "05ece93f211d8bfe08ccccfe58afc75318f780e32b833a5432a0e819a2384882",
        "cf5e80a5f458747470b6f3b31432eb06113650790a56204796f286e3c7ac26f7",
        "5650dc2956ed60e33f1fca9126f57dd8d1cd5c83f01847a4ddc0fb620917ebf1",
        "47a08bd4378ae98c781671f1494b16dd6793c47212327f3b5335a1ad0395d2e8",
        "e13d99fe6a37a22328de208908e81fc53a4ffaa8bd1d49b8bb8261ef9a298d91",
    )
    assert tuple(item.producer_revision for item in protocol.historical_prior_authorities) == (
        "925c2cbc727e3b18e91c0b3981eeed1b470a955a",
        "7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da",
    )
    assert tuple(len(item.paths) for item in protocol.historical_prior_authorities) == (3, 3)
    assert protocol.current_experiment_authority.tracked_path_count == 27
    assert protocol.current_experiment_authority.current_unique_prompt_digest_count == 1724
    assert protocol.manifest.future_split_deny_authority.exclusion_roles == (
        "masked_lf_whitening_fit", "independent_confirmation", "candidate_selection",
        "calibration", "evaluation",
    )
    assert canonical_digest(asdict(protocol.manifest.future_split_deny_authority)) == (
        protocol.manifest.future_split_deny_authority_digest
    )


def test_roster_authority_and_derived_identity_tamper_fail_closed(tmp_path: Path) -> None:
    manifest = json.loads((ROOT / "configs/experiments/salient_local_lf_mask_write_validation_manifest.json").read_text())
    manifest["entries"][0]["generation_seed"] += 1
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    config = json.loads(CONFIG.read_text())
    config["manifest_path"] = str(target.relative_to(ROOT)) if ROOT in target.parents else str(target)
    config["manifest_file_sha256"] = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SalientLocalLfMaskWriteProtocolError):
        load_salient_local_lf_mask_write_validation_protocol(config_path, repository_root=ROOT)


def test_future_split_deny_axis_and_digest_tamper_fail_closed(tmp_path: Path) -> None:
    manifest = json.loads((ROOT / "configs/experiments/salient_local_lf_mask_write_validation_manifest.json").read_text())
    manifest["future_split_deny_authority"]["key_lineage_digests"] = manifest[
        "future_split_deny_authority"
    ]["key_lineage_digests"][:-1]
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    config = json.loads(CONFIG.read_text())
    config["manifest_path"] = str(target)
    config["manifest_file_sha256"] = sha256(target.read_bytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SalientLocalLfMaskWriteProtocolError):
        load_salient_local_lf_mask_write_validation_protocol(config_path, repository_root=ROOT)


def test_current_authority_inventory_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["current_experiment_authority"]["paths"][0]["raw_sha256"] = "0" * 64
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SalientLocalLfMaskWriteProtocolError):
        load_salient_local_lf_mask_write_validation_protocol(config_path, repository_root=ROOT)


def test_required_git_authority_revision_resolution_is_exact_and_fail_closed() -> None:
    config_payload = CONFIG.read_bytes()
    execution_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert resolve_required_git_authority_revisions(
        execution_revision=execution_revision,
        config_payload=config_payload,
    ) == (
        execution_revision,
        "061991c67bb0ceb3fbfe3359a2d86b78f301f171",
        "925c2cbc727e3b18e91c0b3981eeed1b470a955a",
        "7c0d86d6eac5ffcfc4a30f2f5fb22884aaa848da",
    )

    invalid_documents = []
    missing = json.loads(config_payload)
    del missing["current_experiment_authority"]
    invalid_documents.append(missing)
    duplicated = json.loads(config_payload)
    duplicated["historical_prior_authorities"][1]["producer_revision"] = (
        duplicated["historical_prior_authorities"][0]["producer_revision"]
    )
    invalid_documents.append(duplicated)
    malformed = json.loads(config_payload)
    malformed["current_experiment_authority"]["producer_revision"] = "x" * 40
    invalid_documents.append(malformed)
    drifted = json.loads(config_payload)
    drifted["historical_prior_authorities"][0]["authority_identity"] = (
        "historical_authority_drifted"
    )
    invalid_documents.append(drifted)
    for document in invalid_documents:
        with pytest.raises(SalientLocalLfPackageBuildError):
            resolve_required_git_authority_revisions(
                execution_revision=execution_revision,
                config_payload=json.dumps(document),
            )


def test_shallow_checkout_hydrates_exact_authorities_before_protocol_package_load(
    tmp_path: Path,
) -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("local Git authority objects unavailable")
    execution_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    required = resolve_required_git_authority_revisions(
        execution_revision=execution_revision,
        config_payload=CONFIG.read_bytes(),
    )
    remote = tmp_path / "exact-authority-remote.git"
    subprocess.run(("git", "init", "--bare", str(remote)), check=True, capture_output=True)
    subprocess.run(
        (
            "git",
            "push",
            f"file://{remote}",
            f"{execution_revision}:refs/heads/execution",
            *(f"{revision}:refs/heads/authority-{index}"
              for index, revision in enumerate(required[1:], start=1)),
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    shallow = tmp_path / "shallow-repository"
    subprocess.run(
        (
            "git",
            "clone",
            "--no-local",
            "--no-hardlinks",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            "execution",
            f"file://{remote}",
            str(shallow),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    observed_execution_revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=shallow, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert observed_execution_revision == execution_revision
    for revision in required[1:]:
        missing = subprocess.run(
            ("git", "cat-file", "-e", f"{revision}^{{commit}}"),
            cwd=shallow,
            check=False,
            capture_output=True,
        )
        assert missing.returncode != 0

    assert hydrate_required_git_authority_revisions(shallow, required) == required
    for revision in required:
        observed = subprocess.run(
            ("git", "rev-parse", "--verify", f"{revision}^{{commit}}"),
            cwd=shallow,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert observed == revision
    protocol = load_salient_local_lf_mask_write_validation_protocol(
        shallow / CONFIG.relative_to(ROOT),
        repository_root=shallow,
    )
    assert protocol.run_id == "ceg_wm_salient_local_lf_mask_write_validation"
    package = tmp_path / "shallow-execution-package.zip"
    built = build_salient_local_lf_mask_write_validation_package(
        shallow,
        package,
        execution_revision,
    )
    assert built["committed_revision"] == execution_revision
    assert package.is_file()


def test_exact_package_replays_current_and_historical_authority_without_git(
    tmp_path: Path,
) -> None:
    revision = _exact_commit_for_current_strict_sources(tmp_path)
    package = tmp_path / "execution-package.zip"
    built = build_salient_local_lf_mask_write_validation_package(ROOT, package, revision)
    verified = verify_salient_local_lf_mask_write_validation_package(ROOT, package, revision)
    assert verified == built

    extracted = _extract_verified_execution_package(package, tmp_path / "gitless-repository")
    assert not (extracted / ".git").exists()
    extracted_manifest = verify_extracted_salient_local_lf_mask_write_validation_package(
        extracted, revision,
    )
    assert extracted_manifest["committed_revision"] == revision
    extracted_paths = {entry["path"] for entry in extracted_manifest["entries"]}
    assert SERVER_STARTUP_MEMBERS <= extracted_paths
    replayed = load_salient_local_lf_mask_write_validation_protocol(
        extracted / "configs/experiments/salient_local_lf_mask_write_validation.json",
        repository_root=extracted,
    )
    assert replayed.digest() == _protocol().digest()
    assert replayed.current_experiment_authority.tracked_path_count == 27
    assert tuple(len(authority.paths) for authority in replayed.historical_prior_authorities) == (3, 3)
    isolated_environment = dict(os.environ)
    isolated_environment.pop("PYTHONPATH", None)
    isolated_environment["PYTHONNOUSERSITE"] = "1"
    isolated_import = subprocess.run(
        (
            sys.executable, "-I", "-c",
            "import sys; "
            f"sys.path.insert(0, {str(extracted)!r}); "
            f"import {SERVER_MODULE} as server; "
            "assert callable(server.main)",
        ),
        cwd=tmp_path, env=isolated_environment, check=False,
        capture_output=True, text=True,
    )
    assert isolated_import.returncode == 0, isolated_import.stderr
    isolated_help = subprocess.run(
        (sys.executable, "-m", SERVER_MODULE, "--help"),
        cwd=extracted, env=isolated_environment, check=False,
        capture_output=True, text=True,
    )
    assert isolated_help.returncode == 0, isolated_help.stderr
    assert "--expected-revision" in isolated_help.stdout

    corrupted_package = tmp_path / "corrupted-execution-package.zip"
    corrupted_bytes = bytearray(package.read_bytes())
    corrupted_bytes[-1] ^= 1
    corrupted_package.write_bytes(corrupted_bytes)
    with pytest.raises(SalientLocalLfPackageBuildError):
        verify_salient_local_lf_mask_write_validation_package(
            ROOT, corrupted_package, revision,
        )

    builder_tampered = _extract_verified_execution_package(
        package, tmp_path / "builder-tampered-repository",
    )
    builder_member = builder_tampered / (
        "scripts/experiment_execution/build_salient_local_lf_mask_write_validation_package.py"
    )
    builder_member.write_bytes(builder_member.read_bytes() + b"tampered")
    with pytest.raises(SalientLocalLfPackageBuildError):
        verify_extracted_salient_local_lf_mask_write_validation_package(
            builder_tampered, revision,
        )

    manifest_tampered = _extract_verified_execution_package(
        package, tmp_path / "manifest-tampered-repository",
    )
    manifest_member = manifest_tampered / "PACKAGE_MANIFEST.json"
    manifest_payload = json.loads(manifest_member.read_text("utf-8"))
    manifest_payload["committed_revision"] = "0" * 40
    manifest_member.write_text(
        json.dumps(
            manifest_payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(SalientLocalLfPackageBuildError):
        verify_extracted_salient_local_lf_mask_write_validation_package(
            manifest_tampered, revision,
        )

    current_tampered = _extract_verified_execution_package(
        package, tmp_path / "current-authority-tampered-repository",
    )
    current_member = current_tampered / replayed.current_experiment_authority.paths[0].package_member_path
    current_member.write_bytes(current_member.read_bytes() + b"tampered")
    with pytest.raises(SalientLocalLfPackageBuildError):
        verify_extracted_salient_local_lf_mask_write_validation_package(
            current_tampered, revision,
        )
    with pytest.raises(SalientLocalLfMaskWriteProtocolError):
        load_salient_local_lf_mask_write_validation_protocol(
            current_tampered / "configs/experiments/salient_local_lf_mask_write_validation.json",
            repository_root=current_tampered,
        )

    historical_tampered = _extract_verified_execution_package(
        package, tmp_path / "historical-authority-tampered-repository",
    )
    historical_member = historical_tampered / replayed.historical_prior_authorities[0].paths[0].package_member_path
    historical_member.write_bytes(historical_member.read_bytes() + b"tampered")
    with pytest.raises(SalientLocalLfPackageBuildError):
        verify_extracted_salient_local_lf_mask_write_validation_package(
            historical_tampered, revision,
        )
    with pytest.raises(SalientLocalLfMaskWriteProtocolError):
        load_salient_local_lf_mask_write_validation_protocol(
            historical_tampered / "configs/experiments/salient_local_lf_mask_write_validation.json",
            repository_root=historical_tampered,
        )


def test_prior_authority_collision_collector_covers_all_eight_axes() -> None:
    protocol = _protocol()
    entry = protocol.manifest.entries[0]
    axes = {
        name: set() for name in (
            "prompt_digests", "generation_seeds", "cluster_identities",
            "source_cluster_ids", "key_lineage_digests", "image_lineage_digests",
            "namespaces", "lineage_authorities",
        )
    }
    _collect_deny_axes({
        "prompt_digest": entry.prompt_digest,
        "generation_seed": entry.generation_seed,
        "cluster_identity": entry.cluster_identity,
        "source_cluster_id": entry.source_cluster_id,
        "key_lineage_digest": entry.key_lineage_digest,
        "image_lineage_digest": entry.image_lineage_digest,
        "seed_namespace": protocol.manifest.seed_namespace,
        "registered_key_derivation_identity": (
            protocol.manifest.registered_key_derivation_identity
        ),
    }, axes)
    assert all(axes[name] for name in axes)
    assert protocol.manifest.seed_namespace in axes["namespaces"]
    assert protocol.manifest.registered_key_derivation_identity in axes[
        "lineage_authorities"
    ]


def test_actual_dtype_budget_uses_exact_binary32_boundary() -> None:
    assert CANONICAL_CONTENT_RELATIVE_L2_LIMIT == struct_binary32(3 / 250)
    assert _actual_dtype_budget_pass(
        budget_status="accepted",
        realized_relative_l2=CANONICAL_CONTENT_RELATIVE_L2_LIMIT,
    ) is True
    assert _actual_dtype_budget_pass(
        budget_status="accepted",
        realized_relative_l2=next_binary32(CANONICAL_CONTENT_RELATIVE_L2_LIMIT),
    ) is False


def test_signed_integer_quality_accepts_exact_boundary_and_rejects_next() -> None:
    accepted = _quality()
    rejected = _quality(over_limit=True)
    assert accepted.squared_code_delta_sum == 786432
    assert accepted.quality_pass is True
    assert rejected.squared_code_delta_sum == 786435
    assert rejected.quality_pass is False
    assert accepted.normalized_mean_squared_error == 1 / 65025
    assert accepted.root_mean_squared_code_delta == 1.0
    with pytest.raises(SalientLocalLfMaskWriteMetricError):
        observe_public_rgb8_quality(
            torch.zeros((1, 3, 512, 512), dtype=torch.uint8),
            torch.zeros((1, 3, 512, 512), dtype=torch.uint8),
            clean_image_digest="0" * 64,
            marked_image_digest="0" * 64,
        )


def test_quality_violation_is_complete_scientific_negative_not_failure() -> None:
    observations = [_observation(index, quality_pass=index != 7) for index in range(8)]
    aggregate = aggregate_salient_local_lf_mask_write_validation(observations, ())
    assert aggregate.successful_observation_count == 8
    assert aggregate.quality_success_count == 7
    assert aggregate.module_outcome == "mechanism_signal_not_observed"
    assert aggregate.candidate_recommendation == "candidate_not_recommended"
    assert aggregate.allow_request_for_independent_masked_lf_null_fit is False


def test_mechanism_requires_seven_of_eight_and_quality_requires_eight_of_eight() -> None:
    passing = [_observation(index, mechanism=index != 7) for index in range(8)]
    assert aggregate_salient_local_lf_mask_write_validation(passing, ()).allow_request_for_independent_masked_lf_null_fit
    failing = [_observation(index, mechanism=index not in {6, 7}) for index in range(8)]
    assert not aggregate_salient_local_lf_mask_write_validation(failing, ()).allow_request_for_independent_masked_lf_null_fit
    with pytest.raises(SalientLocalLfMaskWriteMetricError):
        aggregate_salient_local_lf_mask_write_validation(passing[:7], ())


def test_failure_priority_and_fixed_denominator_are_stable() -> None:
    observations = [_observation(index) for index in range(6)]
    failures = (
        SalientLocalLfTerminalFailure(6, "resource_failure", "resource_failure"),
        SalientLocalLfTerminalFailure(7, "implementation_failure", "implementation_failure"),
    )
    aggregate = aggregate_salient_local_lf_mask_write_validation(observations, failures)
    assert aggregate.module_outcome == "implementation_blocked"
    assert aggregate.scientific_denominator == 8


def test_scientific_failure_classification_is_exact_and_prioritized() -> None:
    assert _classify_scientific_failure(SalientLocalLfMaskWriteIdentityError("private")) == (
        "identity_failure", "salient_local_lf_public_observation_identity_drift",
    )
    assert _classify_scientific_failure(SalientLocalLfMaskWriteIntegrityError("private")) == (
        "integrity_failure", "salient_local_lf_public_materialization_integrity_drift",
    )
    assert _classify_scientific_failure(RuntimeError("private"))[0] == "implementation_failure"
    assert _classify_scientific_failure(MemoryError("private"))[0] == "resource_failure"
    assert _classify_scientific_failure(OSError("private"))[0] == "environment_failure"


def test_scientific_record_roundtrip_preserves_typed_observation() -> None:
    runner = _runner()
    source_cluster_id = runner.protocol.manifest.entries[0].source_cluster_id
    record = runner._scientific_record(
        unit_index=2, attempt_index=0, elapsed=0.25,
        observation=_observation(0, source_cluster_id=source_cluster_id),
    )
    replay = DevelopmentScientificRecord.from_payload(json.loads(json.dumps(record.payload())))
    assert replay == record
    assert replay.operation_result_payload["mask_write_observation"]["quality"]["squared_code_delta_sum"] == 786432
    assert replay.metric_observation["source_cluster_id"] == runner.protocol.analysis_identity(2).source_cluster_id


def test_precommit_scientific_observation_authority_fails_closed() -> None:
    runner = _runner()
    source_cluster_id = runner.protocol.manifest.entries[0].source_cluster_id
    with pytest.raises(SalientLocalLfMaskWriteIdentityError):
        runner._scientific_record(
            unit_index=1, attempt_index=0, elapsed=0.25,
            observation=_observation(0, source_cluster_id=source_cluster_id),
        )
    with pytest.raises(SalientLocalLfMaskWriteIdentityError):
        runner._scientific_record(
            unit_index=2, attempt_index=0, elapsed=0.25,
            observation=_observation(1, source_cluster_id=source_cluster_id),
        )
    with pytest.raises(SalientLocalLfMaskWriteIdentityError):
        runner._scientific_record(
            unit_index=2, attempt_index=0, elapsed=0.25,
            observation=_observation(0, source_cluster_id="f" * 64),
        )
    with pytest.raises(SalientLocalLfMaskWriteIdentityError):
        runner._scientific_record(
            unit_index=2, attempt_index=0, elapsed=0.25,
            observation=_observation(
                0, source_cluster_id=source_cluster_id, identity_pass=False,
            ),
        )
    with pytest.raises(SalientLocalLfMaskWriteIntegrityError):
        runner._scientific_record(
            unit_index=2, attempt_index=0, elapsed=0.25,
            observation=_observation(
                0, source_cluster_id=source_cluster_id, integrity_pass=False,
            ),
        )


def test_real_persistent_store_commits_recovers_and_replays_fixed_ten(
    tmp_path: Path,
) -> None:
    runner = _runner()
    worker = FrozenWorkerIdentity(
        revision=runner.method_code_revision,
        protocol_digest=runner.protocol_digest,
        execution_intent_authority_digest=runner.execution_intent_authority_digest,
        input_manifest_digest=runner.protocol.manifest.digest(),
        candidate_config_digest=runner.candidate_config_digest,
        unit_roster_digest=runner.protocol.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        tmp_path, run_id=runner.protocol.run_id, worker_identity=worker,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    lease = store.acquire_lease(
        session_id="salient_local_lf_persistence_session",
        now_epoch_seconds=100, lease_duration_seconds=1000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=100)
    for unit_index in range(10):
        intent = store.create_session_intent(cursor, lease, now_epoch_seconds=101 + unit_index * 2)
        if unit_index < 2:
            record = runner._operational_record(
                unit_index=unit_index,
                operation={
                    "operational_role": "environment_runtime_throughput_preflight",
                    "case_ids": [f"salient_local_lf_operational_preflight_{unit_index}"],
                    "responsibility_result_digests": [["content_embedder", f"{unit_index + 80:064x}"]],
                    "runtime_config_digest": f"{unit_index + 90:064x}",
                    "counts_as_scientific_coverage": False,
                    "scientific_claims_supported": False,
                },
                elapsed=0.25, attempt_index=0,
            )
        else:
            entry = runner.protocol.manifest.entries[unit_index - 2]
            record = runner._scientific_record(
                unit_index=unit_index, attempt_index=0, elapsed=0.25,
                observation=_observation(
                    unit_index - 2, source_cluster_id=entry.source_cluster_id,
                ),
            )
        marker = store.commit_session_unit(
            cursor, lease, intent, record=record,
            raw_secret_values=("salient-mask-write-test-root", runner.registered_root_key),
            now_epoch_seconds=102 + unit_index * 2,
        )
        assert marker.unit_index == unit_index
        assert marker.attempt_index == 0
    reopened = DevelopmentPersistentStore(
        tmp_path, run_id=runner.protocol.run_id, worker_identity=worker,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    recovery = reopened.recover(now_epoch_seconds=500)
    evidence = reopened.verified_terminal_scientific_evidence_for_unit_indexes(
        tuple(range(2, 10)), now_epoch_seconds=500,
    )
    aggregate = runner.replay_aggregate(evidence)
    assert len(recovery.committed_units) == 10
    assert tuple(item.unit_index for item in recovery.committed_units) == tuple(range(10))
    assert aggregate.successful_observation_count == 8
    assert aggregate_supports_scientific_claim(aggregate) is True
    first_record, first_marker = evidence[0]
    with pytest.raises(SalientLocalLfMaskWriteRunnerError):
        runner.replay_aggregate(((first_record, replace(first_marker, unit_id="development_unit_0003")), *evidence[1:]))


def test_public_cpu_runner_executes_all_units_through_store_without_record_proxies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, backend, model = _public_runner()
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "public_cpu_fixture")
    latent = torch.linspace(
        -1.0, 1.0, steps=16 * 64 * 64, dtype=torch.float32,
    ).reshape((1, 16, 64, 64)).to(torch.float16)
    worker = FrozenWorkerIdentity(
        revision=runner.method_code_revision,
        protocol_digest=runner.protocol_digest,
        execution_intent_authority_digest=runner.execution_intent_authority_digest,
        input_manifest_digest=runner.protocol.manifest.digest(),
        candidate_config_digest=runner.candidate_config_digest,
        unit_roster_digest=runner.protocol.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        tmp_path, run_id=runner.protocol.run_id, worker_identity=worker,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    lease = store.acquire_lease(
        session_id="salient_local_lf_public_runner_session",
        now_epoch_seconds=1000, lease_duration_seconds=1000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=1000)
    for unit_index in range(10):
        intent = store.create_session_intent(cursor, lease, now_epoch_seconds=1001 + unit_index * 2)
        if unit_index == 0:
            record = runner.execute_checkpoint_runtime_preflight(attempt_index=0)
        elif unit_index == 1:
            record = runner.execute_public_runtime_preflight(base_latent=latent, attempt_index=0)
        else:
            record = runner.execute_scientific_unit(
                unit_index=unit_index, base_latent=latent, attempt_index=0,
            )
        store.commit_session_unit(
            cursor, lease, intent, record=record,
            raw_secret_values=(
                "salient-local-lf-public-runner-root", runner.registered_root_key,
            ),
            now_epoch_seconds=1002 + unit_index * 2,
        )
    reopened = DevelopmentPersistentStore(
        tmp_path, run_id=runner.protocol.run_id, worker_identity=worker,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    recovery = reopened.recover(now_epoch_seconds=1200)
    evidence = reopened.verified_terminal_scientific_evidence_for_unit_indexes(
        tuple(range(2, 10)), now_epoch_seconds=1200,
    )
    aggregate = runner.replay_aggregate(evidence)
    assert tuple(item.unit_index for item in recovery.committed_units) == tuple(range(10))
    assert aggregate.successful_observation_count == 8
    assert aggregate_supports_scientific_claim(aggregate) is True
    assert backend.run_calls == 18
    assert backend.encode_calls == 9
    assert model.forward_count == 19


def test_safe_failure_is_package_relative_bounded_and_secret_free() -> None:
    secret = "registered-root-secret-value"
    try:
        raise RuntimeError(secret + " /content/drive/private/checkpoint")
    except RuntimeError as exc:
        diagnostic = _safe_failure(exc, repository=ROOT,
                                   operation_identity="salient_local_lf_test_execution", unit_index=2)
    encoded = json.dumps(diagnostic)
    assert secret not in encoded
    assert "/content/" not in encoded
    assert len(diagnostic["failure_message_redacted"].encode()) <= 512
    assert len(diagnostic["package_relative_frames"]) <= 8


def test_historical_candidate_overlay_preserves_pre_delivery_status_authority(
    tmp_path: Path,
) -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("local Git producer objects unavailable")
    historical_root = materialize_historical_repository(
        source_root=ROOT,
        revision=HISTORICAL_CANDIDATE_OVERLAY_REVISION,
        destination=tmp_path / "historical-candidate-overlay",
        paths=(CANDIDATE_OVERLAY_PATH,),
    )
    historical_state = json.loads(
        (historical_root / CANDIDATE_OVERLAY_PATH).read_text(encoding="utf-8")
    )

    assert {
        field: historical_state[field] for field in CANDIDATE_OVERLAY_STATUS_FIELDS
    } == {
        "source_cpu_api_implementation_ready": True,
        "candidate_runtime_qualified": False,
        "experiment_protocol_admitted": False,
        "masked_lf_whitening_asset_ready": False,
        "rgb_quality_gate_defined": False,
        "scientific_mechanism_validated": False,
        "promoted": False,
        "formal_detector": False,
        "diagnostic_only": True,
    }


def test_current_candidate_overlay_matches_policy_status_authority() -> None:
    current_state = json.loads(
        (ROOT / CANDIDATE_OVERLAY_PATH).read_text(encoding="utf-8")
    )
    policy = json.loads(
        (ROOT / "governance/policies/method_readiness_rules.yaml").read_text(
            encoding="utf-8"
        )
    )
    current_status = {
        field: current_state[field] for field in CANDIDATE_OVERLAY_STATUS_FIELDS
    }

    assert current_status == {
        "source_cpu_api_implementation_ready": True,
        "candidate_runtime_qualified": False,
        "experiment_protocol_admitted": True,
        "masked_lf_whitening_asset_ready": False,
        "rgb_quality_gate_defined": True,
        "scientific_mechanism_validated": False,
        "promoted": False,
        "formal_detector": False,
        "diagnostic_only": True,
    }
    assert policy["salient_local_lf_candidate_readiness_overlay"]["status"] == current_status


def test_sixty_seven_distribution_lock_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = (ROOT / "requirements_inspyrenet_salient_local_lf_gpu_execution.txt").read_text().splitlines()
    versions = dict(line.split("==", 1) for line in lock if line and not line.startswith("#"))
    monkeypatch.setattr(
        "scripts.experiment_execution.salient_local_lf_mask_write_validation_server.metadata.version",
        lambda name: versions[name],
    )
    assert len(versions) == 67
    assert _verify_locked_dependencies(ROOT) == "855f73f7cb79cc9b9ec5f4d5a62b17cafc336866836601360882bd1cbaa3568b"
