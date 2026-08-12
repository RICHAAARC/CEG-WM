"""Production worker for frozen LF whitened directional validation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import stat
from tempfile import TemporaryDirectory
import time
from time import monotonic
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from experiments.methods import CegWmExperimentAdapter, load_ceg_wm_experiment_adapter_configuration
from experiments.metrics.lf_whitened_score_screening import fit_lf_null_whitening_asset
from experiments.protocol.development_records import DevelopmentScientificRecord
from experiments.protocol.internal_splits import AnalysisUnitIdentity, derive_source_cluster_id
from experiments.protocol.lf_whitened_directional_validation import (
    OPERATIONAL_UNIT_COUNT,
    canonical_digest,
    load_authority_deny_axes,
    load_lf_whitened_directional_validation_protocol,
)
from experiments.protocol.lf_whitened_score_screening import (
    derive_lf_whitening_analysis_identity,
    load_lf_whitened_score_screening_protocol,
)
from experiments.runners.development_persistence import (
    DIAGNOSTIC_SCHEMA_VERSION,
    GPU_MIX_POLICY,
    HARD_SESSION_CAP_SECONDS,
    SOFT_STOP_SECONDS,
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
    SessionReceipt,
    canonical_json_bytes,
    create_frozen_development_unit_binding,
)
from experiments.runners.lf_whitened_directional_validation import (
    LfWhitenedDirectionalEvidenceViolation,
    LfWhitenedDirectionalValidationRunner,
)
from main import LfNullWhiteningAsset, identify_root_key, key_schedule_sha256_counter
from runtime import Sd35PipelineBackend, create_runtime_adapter
from scripts.experiment_execution.development_exploration_entrypoint import (
    _base_latent,
    _canonical_bytes,
    _environment_digest,
    _session_runtime_identity,
)
from scripts.experiment_execution.lf_whitened_score_screening_entrypoint import (
    _derive_registered_experiment_root as _derive_screening_registered_root,
)


PROTOCOL_PATH = Path("configs/experiments/lf_whitened_directional_validation.json")
WHITENING_FIT_PROTOCOL_PATH = Path("configs/experiments/lf_whitened_score_screening.json")
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")
WHITENING_ASSET_PRODUCER_REVISION = "a78c47184cf83ad351bb4442ebd31c218726de25"
WHITENING_ASSET_PACKAGE = Path(
    "development_execution_packages/"
    "ceg_wm_development_a78c47184cf83ad351bb4442ebd31c218726de25.zip"
)
WHITENING_ASSET_PACKAGE_SHA256 = (
    "7d50f476e7e01f664b6fd1fd48220e2618379cc5de6b5e582474e33815142210"
)
WHITENING_ASSET_DIGEST = (
    "d15601a1e58e33bc2a90b9fb56aaab6faea0f80b732ba920466573f4621fb7a4"
)
WHITENING_ASSET_REQUIRED_CLOSURE = frozenset(
    {
        "configs/experiments/internal_execution_components.json",
        "configs/experiments/lf_whitened_score_screening.json",
        "configs/experiments/lf_whitened_score_screening_manifest.json",
        "configs/experiments/lf_whitening_null_fit_manifest.json",
        "configs/runtime/runtime_sd35_flowmatch.json",
        "experiments/metrics/lf_whitened_score_screening.py",
        "experiments/methods/ceg_wm.py",
        "experiments/protocol/development_records.py",
        "experiments/protocol/internal_splits.py",
        "experiments/protocol/lf_whitened_score_screening.py",
        "experiments/runners/development_persistence.py",
        "experiments/runners/lf_whitened_score_screening.py",
        "main/content_chain/embedder.py",
        "main/content_chain/__init__.py",
        "main/content_chain/lf_carrier.py",
        "main/content_chain/lf_detector.py",
        "main/content_chain/lf_whitening.py",
        "main/shared/key_schedule.py",
        "main/__init__.py",
        "main/shared/__init__.py",
        "runtime/__init__.py",
        "runtime/adapter.py",
        "runtime/backend.py",
        "runtime/configuration.py",
        "runtime/content_write.py",
        "runtime/sd35_backend.py",
    }
)


class LfWhitenedDirectionalEntrypointError(RuntimeError):
    """The LF directional worker could not preserve its frozen boundary."""


class LfWhiteningAssetProducerReplayError(RuntimeError):
    """The frozen LF whitening producer evidence could not be replayed."""


def _derive_registered_experiment_root(
    base_root_key: str,
    *,
    protocol_digest: str,
    manifest_digest: str,
    key_family_namespace: str,
) -> str:
    stream = key_schedule_sha256_counter(
        base_root_key,
        {
            "candidate_id": "lf_low_pass",
            "operator": "carrier_template",
            "responsibility_domain": "lf_carrier",
            "model_revision": canonical_digest({
                "derivation_identity": "lf_whitened_directional_registered_key_derivation",
                "key_family_namespace": key_family_namespace,
                "manifest_digest": manifest_digest,
                "protocol_digest": protocol_digest,
            }),
            "tensor_role": "base_gaussian",
        },
        (8,),
    )
    return "ceg-wm-lf-whitened-directional-registered:" + stream.domain_digest


def _is_resource_failure(error: BaseException) -> bool:
    resource_types = tuple(dict.fromkeys((
        MemoryError,
        getattr(torch, "OutOfMemoryError", MemoryError),
        getattr(torch.cuda, "OutOfMemoryError", MemoryError),
    )))
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, resource_types):
            return True
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _safe_json(path: Path, *, role: str, canonical: bool = False) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise LfWhiteningAssetProducerReplayError(f"{role} is unavailable")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LfWhiteningAssetProducerReplayError(f"{role} is unreadable") from exc
    if type(payload) is not dict or (canonical and canonical_json_bytes(payload) != raw):
        raise LfWhiteningAssetProducerReplayError(f"{role} identity is invalid")
    return payload


def _verify_producer_package(package: Path) -> str:
    if not package.is_file() or package.is_symlink():
        raise LfWhiteningAssetProducerReplayError(
            "whitening producer package is unavailable"
        )
    package_digest = sha256(package.read_bytes()).hexdigest()
    if package_digest != WHITENING_ASSET_PACKAGE_SHA256:
        raise LfWhiteningAssetProducerReplayError(
            "whitening producer package digest drifted"
        )
    try:
        with ZipFile(package, "r") as archive:
            if archive.testzip() is not None:
                raise LfWhiteningAssetProducerReplayError(
                    "whitening producer package member checksum drifted"
                )
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            if (
                not names
                or names != tuple(sorted(names))
                or len({name.casefold() for name in names}) != len(names)
                or not WHITENING_ASSET_REQUIRED_CLOSURE.issubset(names)
            ):
                raise LfWhiteningAssetProducerReplayError(
                    "whitening producer package closure drifted"
                )
            for info in infos:
                member = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    member.is_absolute()
                    or any(part in {"", ".", ".."} for part in member.parts)
                    or info.is_dir()
                    or mode != 0o100644
                    or stat.S_ISLNK(mode)
                    or mode & 0o111
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                ):
                    raise LfWhiteningAssetProducerReplayError(
                        "whitening producer package member drifted"
                    )
    except (OSError, UnicodeError, ValueError) as exc:
        raise LfWhiteningAssetProducerReplayError(
            "whitening producer package is invalid"
        ) from exc
    return package_digest


def _producer_protocol_from_package(package: Path):
    required = (
        WHITENING_FIT_PROTOCOL_PATH,
        Path("configs/experiments/lf_whitening_null_fit_manifest.json"),
        Path("configs/experiments/lf_whitened_score_screening_manifest.json"),
        COMPONENT_PATH,
        RUNTIME_PATH,
    )
    with TemporaryDirectory(prefix="ceg_wm_lf_producer_") as temporary:
        root = Path(temporary)
        with ZipFile(package, "r") as archive:
            for relative in required:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(relative.as_posix()))
        protocol, fit_manifest, screening_manifest = (
            load_lf_whitened_score_screening_protocol(
                root / WHITENING_FIT_PROTOCOL_PATH, repository_root=root
            )
        )
        component_document = json.loads((root / COMPONENT_PATH).read_text("utf-8"))
        runtime_document = json.loads((root / RUNTIME_PATH).read_text("utf-8"))
    if (
        type(component_document) is not dict
        or type(component_document.get("method_adapter")) is not dict
        or type(runtime_document) is not dict
    ):
        raise LfWhiteningAssetProducerReplayError(
            "whitening producer configuration is invalid"
        )
    return (
        protocol,
        fit_manifest,
        screening_manifest,
        canonical_digest(
            {
                **component_document["method_adapter"],
                "schema_version": component_document.get("schema_version"),
            }
        ),
        canonical_digest(runtime_document),
    )


def _producer_unit_bindings(
    *, protocol, fit_manifest, screening_manifest, candidate_digest: str,
    root_key_public_digest: str,
):
    fit_key_family_digest = canonical_digest(
        {
            "manifest_digest": fit_manifest.digest(),
            "role": "key_free_clean_public_null_fit",
        }
    )
    screening_key_family_digest = canonical_digest(
        {
            "manifest_digest": screening_manifest.digest(),
            "root_key_public_digest": root_key_public_digest,
            "role": "registered_lf_whitened_screening_key_family",
        }
    )
    bindings = []
    for unit in protocol.unit_roster:
        if unit.unit_index == 0:
            key_family_digest = canonical_digest(
                {
                    "root_key_public_digest": root_key_public_digest,
                    "role": "registered_lf_clean_runtime_preflight_key_family",
                    "run_id": protocol.run_id,
                }
            )
            identity = AnalysisUnitIdentity(
                unit_id="lf_clean_public_vae_runtime_preflight",
                case_id="clean_public_vae_runtime_preflight",
                source_cluster_id=derive_source_cluster_id(
                    prompt_digest=protocol.operational_smoke_prompt_digest,
                    generation_seed=protocol.operational_smoke_generation_seed,
                    image_lineage_digest=protocol.operational_smoke_image_lineage_digest,
                    registered_key_family_digest=key_family_digest,
                ),
                prompt_digest=protocol.operational_smoke_prompt_digest,
                generation_seed=protocol.operational_smoke_generation_seed,
                image_lineage_digest=protocol.operational_smoke_image_lineage_digest,
                registered_key_family_digest=key_family_digest,
            )
        elif unit.unit_index <= 32:
            identity = derive_lf_whitening_analysis_identity(
                fit_manifest.entries[unit.unit_index - 1],
                fit_manifest,
                key_family_digest=fit_key_family_digest,
            )
        else:
            identity = derive_lf_whitening_analysis_identity(
                screening_manifest.entries[unit.unit_index - 33],
                screening_manifest,
                key_family_digest=screening_key_family_digest,
            )
        fit = 1 <= unit.unit_index <= 32
        bindings.append(
            create_frozen_development_unit_binding(
                unit,
                analysis_unit_identity=identity,
                scientific_question_id=(
                    "lf_clean_public_vae_runtime_preflight"
                    if unit.unit_index == 0
                    else "lf_clean_null_whitening_asset_fit"
                    if fit
                    else "lf_raw_whitened_key_attribution_screening"
                ),
                development_case_id=(
                    "clean_public_vae_runtime_preflight"
                    if unit.unit_index == 0
                    else "clean_public_vae_null_fit"
                    if fit
                    else "paired_clean_lf_raw_whitened_screening"
                ),
                candidate_identity=protocol.candidate_identity,
                candidate_config_digest=candidate_digest,
            )
        )
    return tuple(bindings)


def _replay_asset_from_evidence(
    evidence, *, protocol_digest: str, candidate_digest: str,
    fit_manifest_file_sha256: str,
):
    rows = []
    for expected, (record, marker) in enumerate(evidence, start=1):
        record.validate()
        if (
            record.unit_index != expected
            or record.execution_status != "success"
            or record.failure_class is not None
            or record.responsibility_id != "lf_whitening_null_fit"
            or record.protocol_digest != protocol_digest
            or marker.protocol_digest != protocol_digest
            or record.candidate_config_digest != candidate_digest
            or record.method_code_revision != WHITENING_ASSET_PRODUCER_REVISION
            or marker.revision != WHITENING_ASSET_PRODUCER_REVISION
            or record.attempt_index != 0
            or marker.attempt_index != 0
        ):
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer scientific evidence drifted"
            )
        values = record.operation_result_payload.get("clean_null_band_energy_sums")
        if type(values) not in {tuple, list}:
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer statistic is missing"
            )
        rows.append(tuple(float(value) for value in values))
    result = fit_lf_null_whitening_asset(
        rows, fit_manifest_sha256=fit_manifest_file_sha256
    )
    asset = LfNullWhiteningAsset.from_canonical_payload(
        result.canonical_payload,
        whitening_asset_digest=result.whitening_asset_digest,
    )
    asset.validate()
    final_payload = evidence[-1][0].operation_result_payload
    if (
        final_payload.get("whitening_asset_payload") != asset.canonical_payload
        or final_payload.get("whitening_asset_digest") != asset.whitening_asset_digest
        or any(
            record.operation_result_payload.get("whitening_asset_payload") is not None
            or record.operation_result_payload.get("whitening_asset_digest") is not None
            for record, _marker in evidence[:-1]
        )
    ):
        raise LfWhiteningAssetProducerReplayError(
            "whitening producer replay differs from committed asset"
        )
    return asset


def _replay_verified_whitening_asset(
    *,
    repository: Path,
    whitening_asset_persistent_root: Path,
    base_root_key: str,
    required_protocol,
):
    try:
        package = whitening_asset_persistent_root / WHITENING_ASSET_PACKAGE
        package_digest = _verify_producer_package(package)
        (
            fit_protocol,
            fit_manifest,
            screening_manifest,
            adapter_config_digest,
            runtime_config_digest,
        ) = _producer_protocol_from_package(package)
        if (
            required_protocol.whitening_asset_fit_producer_revision
            != WHITENING_ASSET_PRODUCER_REVISION
            or fit_protocol.protocol_id != required_protocol.whitening_asset_fit_identity
            or fit_protocol.run_id != required_protocol.whitening_asset_fit_run_id
            or fit_protocol.digest()
            != required_protocol.whitening_asset_fit_protocol_digest
        ):
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer authority drifted"
            )
        fit_digest = fit_protocol.digest()
        fit_registered_root = _derive_screening_registered_root(
            base_root_key,
            protocol_digest=fit_digest,
            screening_manifest_digest=screening_manifest.digest(),
            key_family_namespace=screening_manifest.key_family_namespace,
        )
        fit_public_root = identify_root_key(fit_registered_root).root_key_public_digest
        candidate_digest = canonical_digest(
            {
                "adapter_config_digest": adapter_config_digest,
                "candidate_identity": fit_protocol.candidate_identity,
                "null_fit_manifest_digest": fit_manifest.digest(),
                "runtime_config_digest": runtime_config_digest,
                "screening_manifest_digest": screening_manifest.digest(),
            }
        )
        authority_digest = canonical_digest(
            {
                "null_fit_manifest_digest": fit_manifest.digest(),
                "protocol_digest": fit_digest,
                "root_key_public_digest": fit_public_root,
                "run_id": fit_protocol.run_id,
                "screening_manifest_digest": screening_manifest.digest(),
            }
        )
        worker_identity = FrozenWorkerIdentity(
            revision=WHITENING_ASSET_PRODUCER_REVISION,
            protocol_digest=fit_digest,
            execution_intent_authority_digest=authority_digest,
            input_manifest_digest=canonical_digest(
                {
                    "null_fit_manifest_digest": fit_manifest.digest(),
                    "screening_manifest_digest": screening_manifest.digest(),
                }
            ),
            candidate_config_digest=candidate_digest,
            unit_roster_digest=fit_protocol.unit_roster_digest,
        )
        run_root = whitening_asset_persistent_root / fit_protocol.run_id
        required_directories = tuple(
            run_root / name
            for name in (
                "leases",
                "intents",
                "bundles",
                "markers",
                "receipts",
                "module_outcomes",
            )
        )
        if not run_root.is_dir() or run_root.is_symlink() or any(
            not path.is_dir() or path.is_symlink() for path in required_directories
        ):
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer persistent evidence is incomplete"
            )
        if _safe_json(
            run_root / "frozen_worker_identity.json",
            role="whitening producer worker identity",
            canonical=True,
        ) != asdict(worker_identity):
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer worker identity drifted"
            )
        server_receipts = tuple(
            sorted(run_root.glob("server_receipts/*/execution_receipt.json"))
        )
        successful = tuple(
            payload
            for path in server_receipts
            if (payload := _safe_json(path, role="whitening producer server receipt"))
            .get("exit_code")
            == 0
        )
        if len(server_receipts) != 1 or len(successful) != 1:
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer successful server receipt is not unique"
            )
        server_receipt = successful[0]
        session_id = server_receipt.get("session_id")
        session_receipts = tuple(sorted((run_root / "receipts").glob("*.json")))
        if (
            type(session_id) is not str
            or len(session_receipts) != 1
            or session_receipts[0].stem != session_id
        ):
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer session receipt identity drifted"
            )
        session_receipt = _safe_json(
            session_receipts[0],
            role="whitening producer session receipt",
            canonical=True,
        )
        committed_ids = session_receipt.get("committed_unit_ids")
        if (
            server_receipt.get("committed_revision")
            != WHITENING_ASSET_PRODUCER_REVISION
            or server_receipt.get("run_id") != fit_protocol.run_id
            or server_receipt.get("execution_package_sha256") != package_digest
            or server_receipt.get("termination_reason") != "frozen_roster_complete"
            or server_receipt.get("committed_unit_count") != 41
            or session_receipt.get("revision") != WHITENING_ASSET_PRODUCER_REVISION
            or session_receipt.get("run_id") != fit_protocol.run_id
            or session_receipt.get("package_sha256") != package_digest
            or session_receipt.get("termination_reason") != "frozen_roster_complete"
            or type(committed_ids) is not list
            or len(committed_ids) != 41
        ):
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer receipt binding drifted"
            )
        before = tuple(
            (path.relative_to(run_root).as_posix(), sha256(path.read_bytes()).hexdigest())
            for path in sorted(run_root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        )
        store = DevelopmentPersistentStore(
            whitening_asset_persistent_root,
            run_id=fit_protocol.run_id,
            worker_identity=worker_identity,
            registered_unit_bindings=_producer_unit_bindings(
                protocol=fit_protocol,
                fit_manifest=fit_manifest,
                screening_manifest=screening_manifest,
                candidate_digest=candidate_digest,
                root_key_public_digest=fit_public_root,
            ),
        )
        evidence = store.verified_terminal_scientific_evidence_for_unit_indexes(
            tuple(range(1, 33)), now_epoch_seconds=int(time.time())
        )
        asset = _replay_asset_from_evidence(
            evidence,
            protocol_digest=fit_digest,
            candidate_digest=candidate_digest,
            fit_manifest_file_sha256=fit_protocol.null_fit_manifest_file_sha256,
        )
        required_asset_digest = getattr(required_protocol, "whitening_asset_digest", None)
        if required_asset_digest is not None and (
            required_asset_digest != WHITENING_ASSET_DIGEST
            or asset.whitening_asset_digest != required_asset_digest
        ):
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer asset digest differs from current frozen authority"
            )
        after = tuple(
            (path.relative_to(run_root).as_posix(), sha256(path.read_bytes()).hexdigest())
            for path in sorted(run_root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        )
        if before != after:
            raise LfWhiteningAssetProducerReplayError(
                "whitening producer evidence was modified during replay"
            )
        return asset
    except LfWhiteningAssetProducerReplayError:
        raise
    except Exception as exc:
        raise LfWhiteningAssetProducerReplayError(
            "whitening producer evidence replay failed"
        ) from exc


def execute_lf_whitened_directional_validation_session(
    *,
    repository_root: str | Path,
    expected_revision: str,
    persistent_root: str | Path,
    whitening_asset_persistent_root: str | Path,
    cache_root: str | Path,
    run_id: str,
    session_id: str,
    execution_package_sha256: str,
    environment: Mapping[str, str],
) -> tuple[int, dict[str, object]]:
    """Run or resume one operational smoke and thirty-two scientific units."""

    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    fit_persistent = Path(whitening_asset_persistent_root).resolve()
    cache = Path(cache_root).resolve()
    root_key = environment.get("CEG_WM_ROOT_KEY")
    hf_token = environment.get("HF_TOKEN")
    if not root_key or not hf_token:
        raise LfWhitenedDirectionalEntrypointError("HF_TOKEN and CEG_WM_ROOT_KEY are required")
    if type(execution_package_sha256) is not str or len(execution_package_sha256) != 64:
        raise LfWhitenedDirectionalEntrypointError("execution package digest is invalid")
    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        repository / PROTOCOL_PATH, repository_root=repository
    )
    if run_id != protocol.run_id:
        raise LfWhitenedDirectionalEntrypointError("run identity drifted")
    backend = Sd35PipelineBackend(
        cache_root=cache,
        persistent_root=persistent,
        hf_token=hf_token,
        prompt=protocol.operational_smoke_prompt,
    )
    runtime_adapter = create_runtime_adapter(backend, repository / RUNTIME_PATH)
    runtime_session = runtime_adapter.initialize("cuda")
    adapter = CegWmExperimentAdapter(load_ceg_wm_experiment_adapter_configuration(repository / COMPONENT_PATH))
    protocol_digest = protocol.digest()
    registered_root_key = _derive_registered_experiment_root(
        root_key,
        protocol_digest=protocol_digest,
        manifest_digest=manifest.digest(),
        key_family_namespace=manifest.key_family_namespace,
    )
    public_root = identify_root_key(registered_root_key).root_key_public_digest
    base_public_root = identify_root_key(root_key).root_key_public_digest
    deny_axes = load_authority_deny_axes(protocol.prior_development_manifests, repository)
    if public_root == base_public_root or public_root in set(deny_axes.key_control_identities):
        runtime_adapter.close()
        raise LfWhitenedDirectionalEntrypointError("directional root overlaps a prior authority")
    try:
        asset = _replay_verified_whitening_asset(
            repository=repository,
            whitening_asset_persistent_root=fit_persistent,
            base_root_key=root_key,
            required_protocol=protocol,
        )
    except Exception:
        runtime_adapter.close()
        raise
    candidate_digest = canonical_digest({
        "adapter_config_digest": adapter.configuration.config_digest,
        "candidate_identity": protocol.candidate_identity,
        "component_implementation_digest": protocol.component_implementation_digest,
        "manifest_digest": manifest.digest(),
        "public_callable": protocol.public_callable,
        "runtime_config_digest": runtime_session.runtime_config_digest,
        "whitening_asset_digest": asset.whitening_asset_digest,
        "whitening_asset_fit_producer_revision": protocol.whitening_asset_fit_producer_revision,
    })
    authority_digest = canonical_digest({
        "manifest_digest": manifest.digest(),
        "protocol_digest": protocol_digest,
        "root_key_public_digest": public_root,
        "run_id": run_id,
        "whitening_asset_digest": asset.whitening_asset_digest,
    })
    runner = LfWhitenedDirectionalValidationRunner(
        protocol=protocol,
        manifest=manifest,
        adapter=adapter,
        runtime_adapter=runtime_adapter,
        whitening_asset=asset,
        method_code_revision=expected_revision,
        run_id=run_id,
        registered_root_key=registered_root_key,
        root_key_public_digest=public_root,
        protocol_digest=protocol_digest,
        execution_intent_authority_digest=authority_digest,
        candidate_config_digest=candidate_digest,
    )
    store = DevelopmentPersistentStore(
        persistent,
        run_id=run_id,
        worker_identity=FrozenWorkerIdentity(
            revision=expected_revision,
            protocol_digest=protocol_digest,
            execution_intent_authority_digest=authority_digest,
            input_manifest_digest=manifest.digest(),
            candidate_config_digest=candidate_digest,
            unit_roster_digest=protocol.unit_roster_digest,
        ),
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    started_epoch = int(time.time())
    lease = store.acquire_lease(
        session_id=session_id,
        now_epoch_seconds=started_epoch,
        lease_duration_seconds=HARD_SESSION_CAP_SECONDS - 1,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=started_epoch)
    committed_before = cursor.initial_committed_count
    termination_reason = "frozen_roster_complete"
    failure: dict[str, object] | None = None
    aggregate: dict[str, object] | None = None
    active_unit_index: int | None = None
    try:
        while cursor.next_unit_index < len(protocol.unit_roster):
            now = int(time.time())
            if now - started_epoch >= SOFT_STOP_SECONDS:
                termination_reason = "soft_stop_after_current_unit"
                break
            unit = protocol.unit_roster[cursor.next_unit_index]
            active_unit_index = unit.unit_index
            intent = store.create_session_intent(cursor, lease, now_epoch_seconds=now)
            entry = None if unit.unit_index == 0 else manifest.entries[unit.source_cluster_ordinal]
            backend.set_development_generation_prompts(
                protocol.operational_smoke_prompt if entry is None else entry.prompt
            )
            attempted_at = monotonic()
            try:
                latent = _base_latent(
                    protocol.operational_smoke_generation_seed if entry is None else entry.generation_seed,
                    height=runtime_session.image_height,
                    width=runtime_session.image_width,
                )
                if unit.unit_index == 0:
                    record = runner.execute_operational_smoke(
                        base_latent=latent,
                        attempt_index=intent.attempt_index,
                        retry_parent_intent_digest=intent.parent_attempt_intent_digest,
                        maximum_duration_seconds=unit.maximum_duration_seconds,
                        started_monotonic=attempted_at,
                    )
                else:
                    record = runner.execute_scientific_cluster(
                        cluster_ordinal=unit.source_cluster_ordinal,
                        base_latent=latent,
                        attempt_index=intent.attempt_index,
                        retry_parent_intent_digest=intent.parent_attempt_intent_digest,
                        maximum_duration_seconds=unit.maximum_duration_seconds,
                        started_monotonic=attempted_at,
                    )
            except Exception as exc:
                if unit.unit_index == 0:
                    raise
                resource = _is_resource_failure(exc)
                category = (
                    exc.category
                    if isinstance(exc, LfWhitenedDirectionalEvidenceViolation)
                    else "nonfinite_violation" if "finite" in str(exc).lower()
                    else "identity_violation" if "identity" in str(exc).lower() or "configuration" in str(exc).lower()
                    else "budget_violation" if "budget" in str(exc).lower()
                    else "integrity_violation" if "integrity" in str(exc).lower()
                    else "resource_failure" if resource
                    else "implementation_failure"
                )
                record = runner.create_failed_scientific_record(
                    cluster_ordinal=unit.source_cluster_ordinal,
                    attempt_index=intent.attempt_index,
                    retry_parent_intent_digest=intent.parent_attempt_intent_digest,
                    maximum_duration_seconds=unit.maximum_duration_seconds,
                    actual_elapsed_seconds=float(monotonic() - attempted_at),
                    failure_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                    resource_failure=resource,
                    failure_category=category,
                    failure_diagnostics=(exc.diagnostics if isinstance(exc, LfWhitenedDirectionalEvidenceViolation) else None),
                )
            marker = store.commit_session_unit(
                cursor,
                lease,
                intent,
                record=record,
                raw_secret_values=(root_key, registered_root_key, hf_token),
                now_epoch_seconds=max(now, int(time.time())),
            )
            if marker.attempt_disposition == "retryable_resource_failure":
                termination_reason = "retryable_resource_failure_after_committed_attempt"
                break
            if type(record) is DevelopmentScientificRecord and record.failure_class == "resource_failure":
                termination_reason = "terminal_resource_failure_after_committed_attempt"
                break
        if cursor.next_unit_index == len(protocol.unit_roster):
            verified = store.verified_terminal_scientific_evidence(now_epoch_seconds=int(time.time()))
            aggregate = asdict(runner.replay_directional_aggregate(verified))
    except Exception as exc:
        termination_reason = "worker_execution_failure"
        failure = {
            "failure_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "stage": "lf_whitened_directional_unit_execution",
            "unit_index": active_unit_index,
            "scientific_claims_supported": False,
        }
    finally:
        runtime_adapter.close()
    ended_epoch = int(time.time())
    session_commits = tuple(item.unit_id for item in cursor.committed_units if item.session_id == session_id)
    receipt = SessionReceipt(
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        session_id=session_id,
        run_id=run_id,
        started_at_utc=datetime.fromtimestamp(started_epoch, timezone.utc).isoformat().replace("+00:00", "Z"),
        ended_at_utc=datetime.fromtimestamp(ended_epoch, timezone.utc).isoformat().replace("+00:00", "Z"),
        gpu_model=_session_runtime_identity(role="gpu", display_value=torch.cuda.get_device_name(0)),
        cuda_identity=_session_runtime_identity(role="cuda", display_value=torch.version.cuda or "unknown"),
        environment_digest=_environment_digest(),
        revision=expected_revision,
        package_sha256=execution_package_sha256,
        walltime_seconds=float(ended_epoch - started_epoch),
        peak_vram_bytes=max(1, int(torch.cuda.max_memory_allocated(0))),
        termination_reason=termination_reason,
        soft_stop_seconds=SOFT_STOP_SECONDS,
        hard_session_cap_seconds=HARD_SESSION_CAP_SECONDS,
        gpu_mix_policy=GPU_MIX_POLICY,
        committed_unit_ids=session_commits,
        public_secret_identity_digests=(public_root,),
    )
    receipt_path = store.write_session_receipt(
        receipt,
        raw_secret_values=(root_key, registered_root_key, hf_token),
        session_cursor=cursor,
    )
    result_root = persistent / run_id / "session_results"
    result_root.mkdir(parents=True, exist_ok=True)
    archive = result_root / f"{session_id}.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as target:
        target.write(receipt_path, "session_receipt.json")
        target.writestr("committed_unit_ids.json", _canonical_bytes(list(session_commits)))
        if aggregate is not None:
            target.writestr("directional_aggregate.json", _canonical_bytes(aggregate))
        if failure is not None:
            target.writestr("diagnostic.json", _canonical_bytes(failure))
    return (3 if failure is not None else 0), {
        "artifact_kind": "lf_whitened_directional_validation_failure" if failure is not None else "lf_whitened_directional_validation_result",
        ("diagnostic_zip" if failure is not None else "result_zip"): str(archive),
        "protocol_digest": protocol_digest,
        "input_manifest_digest": manifest.digest(),
        "candidate_config_digest": candidate_digest,
        "whitening_asset_digest": asset.whitening_asset_digest,
        "whitening_asset_fit_producer_revision": protocol.whitening_asset_fit_producer_revision,
        "unit_roster_digest": protocol.unit_roster_digest,
        "source_cluster_deny_list_digest": protocol.source_cluster_deny_list_digest,
        "package_sha256": execution_package_sha256,
        "committed_unit_count": len(cursor.committed_units),
        "session_committed_unit_count": len(cursor.committed_units) - committed_before,
        "termination_reason": termination_reason,
        "directional_aggregate": aggregate,
        "formal_tau_created": False,
        "fpr_estimated": False,
        "candidate_promoted": False,
        "scientific_claims_supported": False,
    }


__all__ = [
    "LfWhitenedDirectionalEntrypointError",
    "execute_lf_whitened_directional_validation_session",
]
