"""GPU worker for the frozen salient-local-LF mask/write pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from time import monotonic
import traceback
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from experiments.methods import CegWmExperimentAdapter, load_ceg_wm_experiment_adapter_configuration
from experiments.protocol.salient_local_lf_mask_write_validation import (
    OPERATIONAL_UNIT_COUNT, canonical_digest,
    load_salient_local_lf_mask_write_validation_protocol,
)
from experiments.runners.development_persistence import (
    DIAGNOSTIC_SCHEMA_VERSION, GPU_MIX_POLICY, HARD_SESSION_CAP_SECONDS,
    SOFT_STOP_SECONDS, DevelopmentPersistentStore, FrozenWorkerIdentity,
    SessionReceipt,
)
from experiments.runners.salient_local_lf_mask_write_validation import (
    SalientLocalLfMaskWriteIdentityError,
    SalientLocalLfMaskWriteIntegrityError,
    SalientLocalLfMaskWriteValidationRunner,
    aggregate_supports_scientific_claim,
)
from main import identify_root_key, key_schedule_sha256_counter
from runtime import InspyrenetSaliencyRuntime, Sd35PipelineBackend, create_runtime_adapter
from scripts.experiment_execution.development_exploration_entrypoint import (
    _base_latent, _canonical_bytes, _environment_digest, _session_runtime_identity,
)


PROTOCOL_PATH = Path("configs/experiments/salient_local_lf_mask_write_validation.json")
COMPONENT_PATH = Path("configs/experiments/internal_execution_components.json")
RUNTIME_PATH = Path("configs/runtime/runtime_sd35_flowmatch.json")
WORKER_RESULT_PREFIX = "CEG_WM_SALIENT_LOCAL_LF_WORKER_RESULT="


class SalientLocalLfMaskWriteEntrypointError(RuntimeError):
    """The worker could not preserve the frozen public boundary."""


def _registered_root(base_root: str, *, protocol_digest: str, manifest_digest: str) -> str:
    stream = key_schedule_sha256_counter(base_root, {
        "candidate_id": "content_embedding_global_hf_local_lf",
        "operator": "carrier_template", "responsibility_domain": "content_embedder",
        "model_revision": canonical_digest({
            "derivation_identity": "salient_local_lf_mask_write_validation_registered_content_key_derivation",
            "protocol_digest": protocol_digest, "manifest_digest": manifest_digest,
        }), "tensor_role": "base_gaussian",
    }, (8,))
    return "ceg-wm-salient-local-lf-mask-write:" + stream.domain_digest


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    result = []
    current: BaseException | None = error
    seen = set()
    while current is not None and id(current) not in seen and len(result) < 8:
        seen.add(id(current)); result.append(current)
        current = current.__cause__ or current.__context__
    return tuple(result)


def _is_resource_failure(error: BaseException) -> bool:
    kinds = tuple(dict.fromkeys((MemoryError, getattr(torch, "OutOfMemoryError", MemoryError),
                                getattr(torch.cuda, "OutOfMemoryError", MemoryError))))
    return any(isinstance(item, kinds) for item in _exception_chain(error))


def _classify_scientific_failure(error: BaseException) -> tuple[str, str]:
    if type(error) is SalientLocalLfMaskWriteIdentityError:
        return "identity_failure", "salient_local_lf_public_observation_identity_drift"
    if type(error) is SalientLocalLfMaskWriteIntegrityError:
        return "integrity_failure", "salient_local_lf_public_materialization_integrity_drift"
    if _is_resource_failure(error):
        return "resource_failure", "salient_local_lf_resource_failure"
    if type(error) is OSError:
        return "environment_failure", "salient_local_lf_environment_failure"
    return "implementation_failure", "salient_local_lf_implementation_failure"


def _safe_failure(error: BaseException, *, repository: Path, operation_identity: str,
                  unit_index: int | None) -> dict[str, object]:
    chain = _exception_chain(error)
    # Raw exception messages can contain paths, prompts, keys, or provider text.
    # The bounded diagnostic therefore records only the stable exact type role.
    message = f"{type(chain[0]).__module__}.{type(chain[0]).__qualname__}"
    frames = []
    for frame in traceback.extract_tb(error.__traceback__):
        try:
            relative = Path(frame.filename).resolve().relative_to(repository.resolve()).as_posix()
        except (OSError, ValueError):
            continue
        frames.append({"path": relative, "line": frame.lineno, "function": frame.name})
        if len(frames) == 8:
            break
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "failure_class": "resource_failure" if _is_resource_failure(error) else "implementation_failure",
        "failure_type_chain": [f"{type(item).__module__}.{type(item).__qualname__}" for item in chain],
        "failure_message_redacted": message,
        "operation_identity": operation_identity,
        "unit_index": unit_index,
        "package_relative_frames": frames,
        "sanitized_stdout": "", "sanitized_stderr": "",
        "completed_operation": None, "not_executed_operation": "remaining_frozen_roster",
    }


def execute_salient_local_lf_mask_write_validation_session(
    *, repository_root: str | Path, expected_revision: str,
    persistent_root: str | Path, cache_root: str | Path,
    run_id: str, session_id: str, execution_package_sha256: str,
    environment: Mapping[str, str],
) -> tuple[int, dict[str, object]]:
    repository = Path(repository_root).resolve()
    persistent = Path(persistent_root).resolve()
    cache = Path(cache_root).resolve()
    root_secret = environment.get("CEG_WM_ROOT_KEY")
    hf_token = environment.get("HF_TOKEN")
    checkpoint_text = environment.get("CEG_WM_INSPYRENET_CHECKPOINT_PATH")
    if not root_secret or not hf_token or not checkpoint_text:
        raise SalientLocalLfMaskWriteEntrypointError("required execution secret or checkpoint path is unavailable")
    if len(expected_revision) != 40 or len(execution_package_sha256) != 64:
        raise SalientLocalLfMaskWriteEntrypointError("execution authority is invalid")
    protocol = load_salient_local_lf_mask_write_validation_protocol(
        repository / PROTOCOL_PATH, repository_root=repository,
    )
    if run_id != protocol.run_id:
        raise SalientLocalLfMaskWriteEntrypointError("run identity drifted")
    backend = Sd35PipelineBackend(cache_root=cache, persistent_root=persistent,
                                  hf_token=hf_token, prompt=protocol.operational_prompt)
    runtime = create_runtime_adapter(backend, repository / RUNTIME_PATH)
    runtime_session = runtime.initialize("cuda")
    saliency = InspyrenetSaliencyRuntime(
        checkpoint_path=Path(checkpoint_text),
        checkpoint_asset_identity=str(protocol.raw["checkpoint_asset_identity"]),
        checkpoint_asset_basename=str(protocol.raw["checkpoint_asset_basename"]),
        selected_device="cuda",
    )
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(repository / COMPONENT_PATH),
        runtime_adapter=runtime,
    )
    protocol_digest = protocol.digest()
    registered_root = _registered_root(root_secret, protocol_digest=protocol_digest,
                                       manifest_digest=protocol.manifest.digest())
    public_root = identify_root_key(registered_root).root_key_public_digest
    candidate_digest = canonical_digest({
        "adapter_config_digest": adapter.configuration.config_digest,
        "candidate_identity": protocol.raw["candidate_identity"],
        "manifest_digest": protocol.manifest.digest(),
        "package_identity": execution_package_sha256,
        "runtime_config_digest": runtime_session.runtime_config_digest,
    })
    authority_digest = canonical_digest({
        "protocol_digest": protocol_digest, "manifest_digest": protocol.manifest.digest(),
        "root_key_public_digest": public_root, "run_id": run_id,
    })
    runner = SalientLocalLfMaskWriteValidationRunner(
        protocol=protocol, adapter=adapter, runtime_adapter=runtime,
        saliency_runtime=saliency, method_code_revision=expected_revision,
        registered_root_key=registered_root, protocol_digest=protocol_digest,
        execution_intent_authority_digest=authority_digest,
        candidate_config_digest=candidate_digest, package_identity=execution_package_sha256,
    )
    bindings = runner.create_persistence_unit_bindings()
    store = DevelopmentPersistentStore(
        persistent, run_id=run_id,
        worker_identity=FrozenWorkerIdentity(
            revision=expected_revision, protocol_digest=protocol_digest,
            execution_intent_authority_digest=authority_digest,
            input_manifest_digest=protocol.manifest.digest(),
            candidate_config_digest=candidate_digest,
            unit_roster_digest=protocol.unit_roster_digest,
        ), registered_unit_bindings=bindings,
    )
    started_epoch = int(time.time())
    lease = store.acquire_lease(session_id=session_id, now_epoch_seconds=started_epoch,
                                lease_duration_seconds=HARD_SESSION_CAP_SECONDS - 1)
    cursor = store.open_session_cursor(lease, now_epoch_seconds=started_epoch)
    committed_before = cursor.initial_committed_count
    failure_diagnostic = None
    aggregate = None
    scientific_claims_supported = False
    termination_reason = "frozen_roster_incomplete"
    active_unit: int | None = None
    try:
        while cursor.next_unit_index < len(protocol.unit_roster):
            active_unit = cursor.next_unit_index
            if int(time.time()) - started_epoch >= SOFT_STOP_SECONDS:
                termination_reason = "soft_stop_after_current_unit"
                break
            unit = protocol.unit_roster[active_unit]
            intent = store.create_session_intent(cursor, lease, now_epoch_seconds=int(time.time()))
            started = monotonic()
            try:
                if active_unit == 0:
                    record = runner.execute_checkpoint_runtime_preflight(attempt_index=intent.attempt_index)
                else:
                    prompt = protocol.operational_prompt if active_unit == 1 else protocol.manifest.entries[active_unit - OPERATIONAL_UNIT_COUNT].prompt
                    seed = protocol.operational_generation_seed if active_unit == 1 else protocol.manifest.entries[active_unit - OPERATIONAL_UNIT_COUNT].generation_seed
                    backend.set_development_generation_prompts(prompt)
                    latent = _base_latent(seed, height=runtime_session.image_height, width=runtime_session.image_width)
                    if active_unit == 1:
                        record = runner.execute_public_runtime_preflight(base_latent=latent, attempt_index=intent.attempt_index)
                    else:
                        record = runner.execute_scientific_unit(unit_index=active_unit, base_latent=latent,
                                                                attempt_index=intent.attempt_index)
            except Exception as exc:
                if active_unit < OPERATIONAL_UNIT_COUNT:
                    raise
                failure_class, failure_reason = _classify_scientific_failure(exc)
                record = runner.create_failed_scientific_record(
                    unit_index=active_unit, attempt_index=intent.attempt_index,
                    elapsed=float(monotonic() - started), failure_class=failure_class,
                    failure_reason=failure_reason,
                )
            store.commit_session_unit(cursor, lease, intent, record=record,
                                      raw_secret_values=(root_secret, registered_root, hf_token, checkpoint_text),
                                      now_epoch_seconds=max(int(time.time()), started_epoch + 1))
        if cursor.next_unit_index == len(protocol.unit_roster):
            evidence = store.verified_terminal_scientific_evidence_for_unit_indexes(
                tuple(range(OPERATIONAL_UNIT_COUNT, len(protocol.unit_roster))),
                now_epoch_seconds=int(time.time()),
            )
            aggregate_value = runner.replay_aggregate(evidence)
            aggregate = asdict(aggregate_value)
            scientific_claims_supported = aggregate_supports_scientific_claim(aggregate_value)
            termination_reason = "frozen_roster_complete"
    except Exception as exc:
        failure_diagnostic = _safe_failure(
            exc, repository=repository,
            operation_identity="salient_local_lf_mask_write_validation_execution",
            unit_index=active_unit,
        )
        termination_reason = "operational_preflight_failed" if active_unit in {None, 0, 1} else "scientific_execution_failed"
    finally:
        runtime.close()
    ended_epoch = int(time.time())
    session_commits = tuple(item.unit_id for item in cursor.committed_units if item.session_id == session_id)
    receipt = SessionReceipt(
        schema_version=DIAGNOSTIC_SCHEMA_VERSION, session_id=session_id, run_id=run_id,
        started_at_utc=datetime.fromtimestamp(started_epoch, timezone.utc).isoformat().replace("+00:00", "Z"),
        ended_at_utc=datetime.fromtimestamp(ended_epoch, timezone.utc).isoformat().replace("+00:00", "Z"),
        gpu_model=_session_runtime_identity(role="gpu", display_value=torch.cuda.get_device_name(0)),
        cuda_identity=_session_runtime_identity(role="cuda", display_value=torch.version.cuda or "unknown"),
        environment_digest=_environment_digest(), revision=expected_revision,
        package_sha256=execution_package_sha256, walltime_seconds=float(ended_epoch - started_epoch),
        peak_vram_bytes=max(1, int(torch.cuda.max_memory_allocated(0))),
        termination_reason=termination_reason, soft_stop_seconds=SOFT_STOP_SECONDS,
        hard_session_cap_seconds=HARD_SESSION_CAP_SECONDS, gpu_mix_policy=GPU_MIX_POLICY,
        committed_unit_ids=session_commits, public_secret_identity_digests=(public_root,),
    )
    receipt_path = store.write_session_receipt(receipt, raw_secret_values=(root_secret, registered_root, hf_token, checkpoint_text), session_cursor=cursor)
    result_root = persistent / run_id / "session_results"
    result_root.mkdir(parents=True, exist_ok=True)
    archive = result_root / f"{session_id}.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as target:
        target.write(receipt_path, "session_receipt.json")
        target.writestr("committed_unit_ids.json", _canonical_bytes(list(session_commits)))
        if aggregate is not None:
            target.writestr("salient_local_lf_mask_write_aggregate.json", _canonical_bytes(aggregate))
        if failure_diagnostic is not None:
            target.writestr("diagnostic.json", _canonical_bytes(failure_diagnostic))
    is_failure = failure_diagnostic is not None
    result = {
        "artifact_kind": "salient_local_lf_mask_write_validation_failure" if is_failure else "salient_local_lf_mask_write_validation_result",
        "diagnostic_zip" if is_failure else "result_zip": str(archive),
        "protocol_digest": protocol_digest, "input_manifest_digest": protocol.manifest.digest(),
        "candidate_config_digest": candidate_digest, "unit_roster_digest": protocol.unit_roster_digest,
        "package_sha256": execution_package_sha256, "committed_unit_count": len(cursor.committed_units),
        "session_committed_unit_count": len(cursor.committed_units) - committed_before,
        "termination_reason": termination_reason,
        "salient_local_lf_mask_write_aggregate": aggregate,
        "formal_tau_created": False, "fpr_estimated": False, "candidate_promoted": False,
        "scientific_claims_supported": bool(
            aggregate is not None and scientific_claims_supported
        ),
    }
    return (3 if is_failure else 0), result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("repository-root", "expected-revision", "persistent-root", "cache-root", "run-id", "session-id", "execution-package-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    code, result = execute_salient_local_lf_mask_write_validation_session(
        repository_root=args.repository_root, expected_revision=args.expected_revision,
        persistent_root=args.persistent_root, cache_root=args.cache_root,
        run_id=args.run_id, session_id=args.session_id,
        execution_package_sha256=args.execution_package_sha256, environment=os.environ,
    )
    print(WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["execute_salient_local_lf_mask_write_validation_session", "SalientLocalLfMaskWriteEntrypointError"]
