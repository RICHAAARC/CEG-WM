"""Production session worker for the frozen development exploration roster."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import torch

from experiments.attacks import load_attack_registry
from experiments.methods import CegWmExperimentAdapter, load_ceg_wm_experiment_adapter_configuration
from experiments.protocol.development_exploration import (
    DEVELOPMENT_DEPENDENCY_LAYERS,
    build_development_cross_fit_plan,
    create_frozen_development_execution_intent_authority,
    development_cross_fit_source_cluster_ids,
    load_frozen_development_exploration_protocol,
)
from experiments.runners.development_exploration import DevelopmentExplorationRunner
from experiments.runners.development_inputs import (
    DevelopmentInputError,
    build_development_manifest_and_key_roster,
    build_development_manifest_and_key_roster_from_public_digest,
    load_development_prompt_roster,
)
from experiments.runners.development_persistence import (
    DIAGNOSTIC_SCHEMA_VERSION,
    GPU_MIX_POLICY,
    HARD_SESSION_CAP_SECONDS,
    SOFT_STOP_SECONDS,
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
    SessionReceipt,
    canonical_digest,
)
from runtime import (
    RuntimeAdapterError,
    RuntimeContentExecutionError,
    Sd35PipelineBackend,
    create_runtime_adapter,
)
from main import identify_root_key


PROTOCOL_PATH = Path(
    "configs/experiments/thirteen_module_mechanism_screening.json"
)
PROMPT_ROSTER_PATH = Path(
    "configs/experiments/thirteen_module_mechanism_screening_prompt_roster.json"
)
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")


class DevelopmentEntrypointError(RuntimeError):
    """The production development session could not continue safely."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _build_or_verify_package(repository: Path, persistent_root: Path, revision: str) -> Path:
    package_root = persistent_root / "development_execution_packages"
    package_root.mkdir(parents=True, exist_ok=True)
    package = package_root / f"ceg_wm_development_{revision}.zip"
    tracked = tuple(line for line in _git(repository, "ls-files").splitlines() if line)
    if not tracked:
        raise DevelopmentEntrypointError("repository has no tracked execution files")
    temporary = package.with_suffix(".building.zip")
    if not package.exists():
        if temporary.exists():
            temporary.unlink()
        with ZipFile(temporary, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in tracked:
                source = repository / relative
                if not source.is_file() or source.is_symlink():
                    raise DevelopmentEntrypointError("tracked package member is unavailable")
                info = ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source.read_bytes(), compress_type=ZIP_DEFLATED)
        try:
            try:
                target = package.open("xb")
            except FileExistsError:
                pass
            else:
                try:
                    with temporary.open("rb") as source, target:
                        shutil.copyfileobj(source, target)
                except Exception:
                    package.unlink(missing_ok=True)
                    raise
        finally:
            temporary.unlink(missing_ok=True)
    if not package.is_file():
        raise DevelopmentEntrypointError("development execution package is invalid")
    with ZipFile(package) as archive:
        if archive.testzip() is not None:
            raise DevelopmentEntrypointError("development execution package is invalid")
    return package


def _environment_digest() -> str:
    return canonical_digest(
        {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": tuple(os.sys.version_info[:3]),
            "torch": torch.__version__,
        }
    )


def _session_runtime_identity(*, role: str, display_value: str) -> str:
    """Convert runtime display metadata into a persisted stable identity."""

    normalized = re.sub(r"[^a-z0-9]+", "_", display_value.strip().lower()).strip("_")
    if role not in {"gpu", "cuda"} or not normalized:
        raise DevelopmentEntrypointError("session runtime identity is unavailable")
    return f"{role}_{normalized}"


def _candidate_digest(protocol) -> str:
    return canonical_digest(tuple((item.responsibility_id, item.candidate_config_digest) for item in protocol.module_matrix))


def _base_latent(seed: int, *, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((1, 16, height // 8, width // 8), generator=generator, dtype=torch.float32).to(device="cuda:0", dtype=torch.float16)


def replay_complete_development_module_outcomes(
    *,
    repository_root: Path,
    persistent_root: Path,
    run_id: str,
    producer_revision: str,
    now_epoch_seconds: int | None = None,
) -> dict[str, str]:
    """Create missing outcomes from a fully verified completed run, without GPU."""

    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    if type(run_id) is not str or not run_id:
        raise DevelopmentEntrypointError("outcome replay run identity is invalid")
    if (
        type(producer_revision) is not str
        or re.fullmatch(r"[0-9a-f]{40}", producer_revision) is None
    ):
        raise DevelopmentEntrypointError("outcome replay revision is invalid")
    run_root = persistent / run_id
    if (
        not run_root.is_dir()
        or run_root.is_symlink()
        or not (run_root / "frozen_worker_identity.json").is_file()
    ):
        raise DevelopmentEntrypointError(
            "outcome replay requires an existing frozen run"
        )
    protocol = load_frozen_development_exploration_protocol(repository / PROTOCOL_PATH)
    prompts = load_development_prompt_roster(repository / PROMPT_ROSTER_PATH)
    receipt_paths = tuple(sorted((persistent / run_id / "receipts").glob("*.json")))
    if not receipt_paths:
        raise DevelopmentEntrypointError("outcome replay lacks session receipts")
    public_digests: set[str] = set()
    for path in receipt_paths:
        try:
            payload = json.loads(path.read_text("utf-8"))
            values = payload["public_secret_identity_digests"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
            raise DevelopmentEntrypointError(
                "outcome replay receipt identity is unreadable"
            ) from exc
        if type(values) is not list or len(values) != 1 or type(values[0]) is not str:
            raise DevelopmentEntrypointError(
                "outcome replay receipt public identity is invalid"
            )
        public_digests.add(values[0])
    if len(public_digests) != 1:
        raise DevelopmentEntrypointError(
            "outcome replay receipt public identity drifted"
        )
    manifest, public_key_roster = (
        build_development_manifest_and_key_roster_from_public_digest(
            protocol,
            prompts,
            next(iter(public_digests)),
        )
    )
    authority = create_frozen_development_execution_intent_authority(
        protocol,
        run_id=run_id,
        seed_namespace=prompts.seed_namespace,
        input_manifest=manifest,
        public_key_roster=public_key_roster,
    )
    provisional_runner = DevelopmentExplorationRunner._for_verified_record_replay(
        intent_authority=authority,
        method_code_revision=producer_revision,
    )
    worker_identity = FrozenWorkerIdentity(
        revision=producer_revision,
        protocol_digest=authority.protocol_digest,
        execution_intent_authority_digest=authority.authority_digest,
        input_manifest_digest=authority.input_manifest_digest,
        candidate_config_digest=_candidate_digest(protocol),
        unit_roster_digest=protocol.study_budget.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        persistent,
        run_id=run_id,
        worker_identity=worker_identity,
        registered_unit_bindings=provisional_runner.create_persistence_unit_bindings(),
    )
    recovery_time = int(time.time()) if now_epoch_seconds is None else now_epoch_seconds
    cross_fit_plans = {
        responsibility_id: build_development_cross_fit_plan(
            responsibility_id=responsibility_id,
            execution_intent_authority=authority,
            expected_execution_intent_authority_digest=authority.authority_digest,
            expected_source_cluster_ids=development_cross_fit_source_cluster_ids(
                authority,
                responsibility_id=responsibility_id,
            ),
        )
        for responsibility_id in (
            "lf_detector",
            "hf_detector",
            "content_detector",
        )
    }
    runner = DevelopmentExplorationRunner._for_verified_record_replay(
        intent_authority=authority,
        method_code_revision=producer_revision,
        persistence_store=store,
    )
    expected_responsibilities = tuple(
        item.responsibility_id for item in protocol.module_matrix
    )
    outcome_root = store.run_root / "module_outcomes"
    outcome_entries = tuple(outcome_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in outcome_entries):
        raise DevelopmentEntrypointError(
            "outcome replay found a non-regular module outcome"
        )
    observed_names = {path.name for path in outcome_entries}
    expected_names = {
        f"{responsibility_id}.json"
        for responsibility_id in expected_responsibilities
    }
    if not observed_names.issubset(expected_names):
        raise DevelopmentEntrypointError(
            "outcome replay found an unknown module outcome"
        )
    missing_responsibilities = tuple(
        responsibility_id
        for responsibility_id in expected_responsibilities
        if f"{responsibility_id}.json" not in observed_names
    )
    replay_responsibilities = tuple(
        dict.fromkeys(
            (*missing_responsibilities, "conditional_recovery_decision")
        )
    )
    outcomes = runner.replay_and_persist_completed_module_outcomes(
        responsibility_ids=replay_responsibilities,
        cross_fit_plans=cross_fit_plans,
        now_epoch_seconds=recovery_time,
    )
    final_entries = tuple(outcome_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in final_entries):
        raise DevelopmentEntrypointError(
            "outcome replay produced a non-regular module outcome"
        )
    final_names = {path.name for path in final_entries}
    if set(outcomes) != set(replay_responsibilities) or final_names != expected_names:
        raise DevelopmentEntrypointError(
            "outcome replay did not rebuild every frozen responsibility"
        )
    return {
        responsibility_id: outcome.outcome_record.outcome_record_id
        for responsibility_id, outcome in outcomes.items()
    }


def execute_development_exploration_session(
    *,
    repository_root: Path,
    expected_revision: str,
    persistent_root: Path,
    cache_root: Path,
    run_id: str,
    session_id: str,
    environment: Mapping[str, str],
    maximum_wiring_clusters: int | None = None,
    stop_before_scientific_units: bool = False,
) -> tuple[int, dict[str, object]]:
    """Execute consecutive currently reachable frozen units and persist each one."""

    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    cache = Path(cache_root).resolve()
    root_key = environment.get("CEG_WM_ROOT_KEY")
    hf_token = environment.get("HF_TOKEN")
    if not root_key or not hf_token:
        raise DevelopmentEntrypointError("HF_TOKEN and CEG_WM_ROOT_KEY are required")
    if maximum_wiring_clusters is not None and (
        type(maximum_wiring_clusters) is not int
        or not 1 <= maximum_wiring_clusters <= 8
    ):
        raise DevelopmentEntrypointError("maximum wiring cluster count is invalid")
    if type(stop_before_scientific_units) is not bool:
        raise DevelopmentEntrypointError(
            "stop before scientific units flag must be boolean"
        )
    protocol = load_frozen_development_exploration_protocol(repository / PROTOCOL_PATH)
    prompts = load_development_prompt_roster(repository / PROMPT_ROSTER_PATH)
    required_prompt_count = 1 + max(
        unit.source_cluster_ordinal for unit in protocol.unit_roster
    )
    if len(prompts.entries) != required_prompt_count:
        raise DevelopmentEntrypointError(
            "prompt roster differs from the active protocol cluster requirement"
        )
    manifest, public_key_roster = build_development_manifest_and_key_roster(protocol, prompts, root_key)
    authority = create_frozen_development_execution_intent_authority(
        protocol,
        run_id=run_id,
        seed_namespace=prompts.seed_namespace,
        input_manifest=manifest,
        public_key_roster=public_key_roster,
    )
    package = _build_or_verify_package(repository, persistent, expected_revision)
    backend = Sd35PipelineBackend(
        cache_root=cache,
        persistent_root=persistent,
        hf_token=hf_token,
        prompt=prompts.entries[0].prompt,
    )
    runtime_adapter = create_runtime_adapter(backend, repository / RUNTIME_PATH)
    session = runtime_adapter.initialize("cuda")
    adapter = CegWmExperimentAdapter(load_ceg_wm_experiment_adapter_configuration(repository / COMPONENT_PATH))
    attack_registry = load_attack_registry(repository / COMPONENT_PATH)
    environment_digest = _environment_digest()
    resource_digest = canonical_digest({"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda})
    provisional_runner = DevelopmentExplorationRunner(
        intent_authority=authority,
        adapter=adapter,
        runtime_adapter=runtime_adapter,
        attack_registry=attack_registry,
        method_code_revision=expected_revision,
        environment_digest=environment_digest,
        resource_identity_digest=resource_digest,
    )
    package_sha256 = _sha256_file(package)
    worker_identity = FrozenWorkerIdentity(
        revision=expected_revision,
        protocol_digest=authority.protocol_digest,
        execution_intent_authority_digest=authority.authority_digest,
        input_manifest_digest=authority.input_manifest_digest,
        candidate_config_digest=_candidate_digest(protocol),
        unit_roster_digest=protocol.study_budget.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        persistent,
        run_id=run_id,
        worker_identity=worker_identity,
        registered_unit_bindings=provisional_runner.create_persistence_unit_bindings(),
    )
    runner = DevelopmentExplorationRunner(
        intent_authority=authority,
        adapter=adapter,
        runtime_adapter=runtime_adapter,
        attack_registry=attack_registry,
        method_code_revision=expected_revision,
        environment_digest=environment_digest,
        resource_identity_digest=resource_digest,
        persistence_store=store,
    )
    started_epoch = int(time.time())
    lease = store.acquire_lease(
        session_id=session_id,
        now_epoch_seconds=started_epoch,
        lease_duration_seconds=HARD_SESSION_CAP_SECONDS - 1,
    )
    session_cursor = store.open_session_cursor(
        lease,
        now_epoch_seconds=started_epoch,
    )
    committed_before = session_cursor.initial_committed_count
    termination_reason = "frozen_roster_complete"
    worker_failure_type: str | None = None
    active_stage = "session_initialization"
    active_responsibility: str | None = None
    active_unit_index: int | None = None
    from scripts.experiment_execution.development_exploration_worker_inputs import (
        DevelopmentDependencyInputBlocked,
        DevelopmentProductionInputBuilder,
        ROUTING_REFERENCE_SCHEDULER_READY,
        ROUTING_REFERENCE_SESSION_STOP,
    )
    try:
        input_builder = DevelopmentProductionInputBuilder(
            cache_root=cache,
            prompt_roster=prompts,
            protocol=protocol,
            authority=authority,
            registered_root_key=root_key,
            runtime_adapter=runtime_adapter,
            persistence_store=store,
            session_cursor=session_cursor,
            runner=runner,
            hf_token=hf_token,
        )
        latent_factory = lambda seed: _base_latent(
            seed,
            height=session.image_height,
            width=session.image_width,
        )
        operational_complete = True
        wiring_clusters_committed_this_session = 0
        while session_cursor.next_unit_index < len(protocol.unit_roster):
            unit = protocol.unit_roster[session_cursor.next_unit_index]
            if unit.phase not in {
                "development_environment_preflight",
                "development_full_chain_wiring",
            }:
                break
            if (
                unit.phase == "development_full_chain_wiring"
                and maximum_wiring_clusters is not None
                and wiring_clusters_committed_this_session
                >= maximum_wiring_clusters
            ):
                operational_complete = False
                termination_reason = "maximum_wiring_clusters_reached"
                break
            now = int(time.time())
            if now - started_epoch >= SOFT_STOP_SECONDS:
                operational_complete = False
                termination_reason = "soft_stop_after_operational_unit"
                break
            entry = prompts.entries[unit.source_cluster_ordinal]
            backend.set_development_generation_prompts(entry.prompt)
            active_stage = "operational_unit"
            active_responsibility = unit.responsibility_id
            active_unit_index = unit.unit_index
            attempt_started = time.monotonic()
            intent = runner.create_operational_intent(
                lease,
                session_cursor,
                now_epoch_seconds=now,
            )
            operational_inputs = input_builder.build_operational_inputs(
                latent_factory(entry.generation_seed),
                source_cluster_ordinal=unit.source_cluster_ordinal,
            )
            if unit.phase == "development_environment_preflight":
                operational_receipt = runner.execute_preflight_cluster(
                    unit.source_cluster_ordinal,
                    operational_inputs["content_embedder"],
                )
            else:
                operational_receipt = runner.execute_wiring_smoke_cluster(
                    unit.source_cluster_ordinal,
                    operational_inputs,
                )
            operational_receipt = replace(
                operational_receipt,
                elapsed_seconds=time.monotonic() - attempt_started,
            )
            runner.commit_operational_receipt(
                lease,
                session_cursor,
                intent,
                operational_receipt,
                now_epoch_seconds=max(now + 1, int(time.time())),
                raw_secret_values=(root_key, hf_token),
            )
            if unit.phase == "development_full_chain_wiring":
                wiring_clusters_committed_this_session += 1
        if stop_before_scientific_units and operational_complete:
            termination_reason = "authorized_operational_boundary_reached"
            operational_complete = False
        cross_fit_plans = (
            {
                responsibility_id: build_development_cross_fit_plan(
                    responsibility_id=responsibility_id,
                    execution_intent_authority=authority,
                    expected_execution_intent_authority_digest=authority.authority_digest,
                    expected_source_cluster_ids=development_cross_fit_source_cluster_ids(
                        authority,
                        responsibility_id=responsibility_id,
                    ),
                )
                for responsibility_id in (
                    "lf_detector",
                    "hf_detector",
                    "content_detector",
                )
            }
            if operational_complete
            else {}
        )
        verified_outcomes = {}

        def refresh_verified_outcomes(now_epoch_seconds: int) -> None:
            for dependency_layer in DEVELOPMENT_DEPENDENCY_LAYERS:
                for responsibility_id in dependency_layer:
                    if responsibility_id in verified_outcomes:
                        continue
                    indexes = tuple(
                        binding.unit_index
                        for binding in store.registered_unit_bindings
                        if binding.responsibility_id == responsibility_id
                        and binding.phase not in {
                            "development_environment_preflight",
                            "development_full_chain_wiring",
                            "development_routing_reference_fit",
                        }
                    )
                    if not indexes or max(indexes) >= session_cursor.next_unit_index:
                        continue
                    outcome = runner.build_verified_module_outcome_record(
                        responsibility_id=responsibility_id,
                        cross_fit_plans=cross_fit_plans,
                        now_epoch_seconds=now_epoch_seconds,
                    )
                    runner.persist_verified_module_outcome(outcome)
                    verified_outcomes[responsibility_id] = outcome

        while operational_complete:
            now = int(time.time())
            if now - started_epoch >= SOFT_STOP_SECONDS:
                termination_reason = "soft_stop_after_current_unit"
                break
            if session_cursor.next_unit_index >= len(protocol.unit_roster):
                break
            unit = protocol.unit_roster[session_cursor.next_unit_index]
            if unit.phase == "development_routing_reference_fit":
                active_stage = "routing_reference_fit"
                active_responsibility = "content_router"
                active_unit_index = unit.unit_index
                reference_status = input_builder.prepare_routing_reference_fit(
                    backend,
                    latent_factory,
                    lease=lease,
                    soft_stop_epoch_seconds=started_epoch + SOFT_STOP_SECONDS,
                )
                if reference_status in ROUTING_REFERENCE_SESSION_STOP:
                    termination_reason = (
                        "resource_retry_after_committed_reference"
                        if reference_status == "retryable_stop"
                        else "soft_stop_after_reference_measurement"
                    )
                    break
                if reference_status not in ROUTING_REFERENCE_SCHEDULER_READY:
                    raise RuntimeError("routing reference scheduler status is invalid")
                continue
            active_stage = "scientific_unit"
            active_responsibility = unit.responsibility_id
            active_unit_index = unit.unit_index
            prompt_entry = prompts.entries[unit.source_cluster_ordinal]
            backend.set_development_generation_prompts(prompt_entry.prompt)
            refresh_verified_outcomes(now)
            decision = runner.decide_verified_module_execution(
                responsibility_id=unit.responsibility_id,
                outcomes_by_responsibility=verified_outcomes,
                cross_fit_plans=cross_fit_plans,
                now_epoch_seconds=now,
            )
            attempt_started = time.monotonic()
            intent = runner.create_scientific_intent(
                lease,
                session_cursor,
                now_epoch_seconds=now,
            )
            if not decision.approved:
                runner.commit_claimed_terminal_failure(
                    lease,
                    session_cursor,
                    intent,
                    failure_class="dependency_blocked",
                    failure_reason=decision.decision_reason,
                    attempt_started_monotonic=attempt_started,
                    raw_secret_values=(root_key, hf_token),
                )
                continue
            try:
                unit_input = input_builder.build(
                    unit,
                    latent_factory(prompt_entry.generation_seed),
                    intent=intent,
                    now_epoch_seconds=now,
                )
            except DevelopmentDependencyInputBlocked:
                if time.monotonic() - attempt_started > unit.maximum_duration_seconds:
                    runner.commit_claimed_resource_failure(
                        lease,
                        session_cursor,
                        intent,
                        failure_reason="unit_duration_exceeded_during_input_preparation",
                        attempt_started_monotonic=attempt_started,
                        raw_secret_values=(root_key, hf_token),
                    )
                    termination_reason = "resource_retry_after_committed_unit"
                    break
                runner.commit_claimed_terminal_failure(
                    lease,
                    session_cursor,
                    intent,
                    failure_class="dependency_blocked",
                    failure_reason="verified_dependency_input_incomplete",
                    attempt_started_monotonic=attempt_started,
                    raw_secret_values=(root_key, hf_token),
                )
                continue
            except (
                MemoryError,
                OSError,
                RuntimeAdapterError,
                RuntimeContentExecutionError,
                torch.cuda.OutOfMemoryError,
            ):
                runner.commit_claimed_resource_failure(
                    lease,
                    session_cursor,
                    intent,
                    failure_reason="input_preparation_resource_exhausted",
                    attempt_started_monotonic=attempt_started,
                    raw_secret_values=(root_key, hf_token),
                )
                termination_reason = "resource_retry_after_committed_unit"
                break
            except DevelopmentInputError as exc:
                if time.monotonic() - attempt_started > unit.maximum_duration_seconds:
                    runner.commit_claimed_resource_failure(
                        lease,
                        session_cursor,
                        intent,
                        failure_reason="unit_duration_exceeded_during_input_preparation",
                        attempt_started_monotonic=attempt_started,
                        raw_secret_values=(root_key, hf_token),
                    )
                    termination_reason = "resource_retry_after_committed_unit"
                    break
                runner.commit_claimed_terminal_failure(
                    lease,
                    session_cursor,
                    intent,
                    failure_class="implementation_failure",
                    failure_reason=f"{type(exc).__module__}.{type(exc).__qualname__}",
                    attempt_started_monotonic=attempt_started,
                    raw_secret_values=(root_key, hf_token),
                )
                continue
            except Exception as exc:
                if time.monotonic() - attempt_started > unit.maximum_duration_seconds:
                    runner.commit_claimed_resource_failure(
                        lease,
                        session_cursor,
                        intent,
                        failure_reason="unit_duration_exceeded_during_input_preparation",
                        attempt_started_monotonic=attempt_started,
                        raw_secret_values=(root_key, hf_token),
                    )
                    termination_reason = "resource_retry_after_committed_unit"
                    break
                runner.commit_claimed_terminal_failure(
                    lease,
                    session_cursor,
                    intent,
                    failure_class="implementation_failure",
                    failure_reason=f"{type(exc).__module__}.{type(exc).__qualname__}",
                    attempt_started_monotonic=attempt_started,
                    raw_secret_values=(root_key, hf_token),
                )
                continue
            if time.monotonic() - attempt_started > unit.maximum_duration_seconds:
                runner.commit_claimed_resource_failure(
                    lease,
                    session_cursor,
                    intent,
                    failure_reason="unit_duration_exceeded_during_input_preparation",
                    attempt_started_monotonic=attempt_started,
                    raw_secret_values=(root_key, hf_token),
                )
                termination_reason = "resource_retry_after_committed_unit"
                break
            executed = runner.execute_and_commit_claimed_session_unit(
                lease,
                session_cursor,
                intent,
                unit_input,
                attempt_started_monotonic=attempt_started,
                raw_secret_values=(root_key, hf_token),
            )
            if (
                executed.record.failure_class == "resource_failure"
                and executed.record.execution_status == "retry"
            ):
                termination_reason = "resource_retry_after_committed_unit"
                break
        refresh_verified_outcomes(int(time.time()))
    except Exception as exc:
        worker_failure_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
        termination_reason = "worker_input_or_session_failure"
    finally:
        runtime_adapter.close()
    ended_epoch = int(time.time())
    committed_units = session_cursor.committed_units
    session_commits = tuple(
        item.unit_id
        for item in committed_units
        if item.session_id == session_id
    )
    receipt = SessionReceipt(
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        session_id=session_id,
        run_id=run_id,
        started_at_utc=datetime.fromtimestamp(started_epoch, timezone.utc).isoformat().replace("+00:00", "Z"),
        ended_at_utc=datetime.fromtimestamp(ended_epoch, timezone.utc).isoformat().replace("+00:00", "Z"),
        gpu_model=_session_runtime_identity(
            role="gpu",
            display_value=torch.cuda.get_device_name(0),
        ),
        cuda_identity=_session_runtime_identity(
            role="cuda",
            display_value=torch.version.cuda or "unknown",
        ),
        environment_digest=environment_digest,
        revision=expected_revision,
        package_sha256=package_sha256,
        walltime_seconds=float(ended_epoch - started_epoch),
        peak_vram_bytes=max(1, int(torch.cuda.max_memory_allocated(0))),
        termination_reason=termination_reason,
        soft_stop_seconds=SOFT_STOP_SECONDS,
        hard_session_cap_seconds=HARD_SESSION_CAP_SECONDS,
        gpu_mix_policy=GPU_MIX_POLICY,
        committed_unit_ids=session_commits,
        public_secret_identity_digests=(
            identify_root_key(root_key).root_key_public_digest,
        ),
    )
    receipt_path = store.write_session_receipt(
        receipt,
        raw_secret_values=(root_key, hf_token),
        session_cursor=session_cursor,
    )
    result_dir = persistent / run_id / "session_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_zip = result_dir / f"{session_id}.zip"
    with ZipFile(result_zip, "x", compression=ZIP_DEFLATED) as archive:
        archive.write(receipt_path, "session_receipt.json")
        archive.writestr("committed_unit_ids.json", _canonical_bytes(list(session_commits)))
        if worker_failure_type is not None:
            archive.writestr(
                "diagnostic.json",
                _canonical_bytes(
                    {
                        "failure_type": worker_failure_type,
                        "stage": active_stage,
                        "responsibility_id": active_responsibility,
                        "unit_index": active_unit_index,
                        "scientific_claims_supported": False,
                        "termination_reason": termination_reason,
                    }
                ),
            )
    artifact_kind = (
        "development_exploration_diagnostic"
        if worker_failure_type is not None
        else "development_exploration_result"
    )
    return (3 if worker_failure_type is not None else 0), {
        "artifact_kind": artifact_kind,
        ("diagnostic_zip" if worker_failure_type is not None else "result_zip"): str(result_zip),
        "protocol_digest": authority.protocol_digest,
        "execution_intent_authority_digest": authority.authority_digest,
        "input_manifest_digest": authority.input_manifest_digest,
        "candidate_config_digest": _candidate_digest(protocol),
        "unit_roster_digest": protocol.study_budget.unit_roster_digest,
        "package_sha256": package_sha256,
        "committed_unit_count": len(committed_units),
        "session_committed_unit_count": len(committed_units) - committed_before,
        "termination_reason": termination_reason,
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-completed-outcomes", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--persistent-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--producer-revision", required=True)
    arguments = parser.parse_args()
    if not arguments.replay_completed_outcomes:
        parser.error("only completed outcome replay is available from this CLI")
    repository = Path(arguments.repository_root).resolve()
    replay_revision = _git(repository, "rev-parse", "HEAD")
    if _git(repository, "status", "--porcelain"):
        raise DevelopmentEntrypointError(
            "outcome replay repository worktree must be clean"
        )
    outcomes = replay_complete_development_module_outcomes(
        repository_root=repository,
        persistent_root=Path(arguments.persistent_root),
        run_id=arguments.run_id,
        producer_revision=arguments.producer_revision,
    )
    print(
        json.dumps(
            {
                "producer_revision": arguments.producer_revision,
                "replay_revision": replay_revision,
                "run_id": arguments.run_id,
                "module_outcome_ids": outcomes,
                "scientific_claims_supported": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
