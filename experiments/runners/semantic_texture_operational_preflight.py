"""Two-unit, zero-science semantic-texture operational preflight."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from time import monotonic
from typing import Callable

from experiments.methods import CegWmExperimentAdapter


PREFLIGHT_SCHEMA_VERSION = 2
PREFLIGHT_PROFILE_ID = "semantic_texture_operational_preflight"
WRITE_UNIT_ID = "semantic_texture_write_operational"
BLIND_DETECTION_UNIT_ID = "semantic_texture_blind_detection_operational"
UNIT_ROSTER = (WRITE_UNIT_ID, BLIND_DETECTION_UNIT_ID)
ALLOWED_BLOCKED_CLASSES = frozenset(
    {
        "environment_blocked",
        "resource_blocked",
        "implementation_blocked",
        "identity_blocked",
        "integrity_blocked",
    }
)
ALLOWED_PRE_EXECUTION_STAGES = frozenset(
    {
        "required_environment",
        "runtime_backend_construction",
        "runtime_configuration",
        "runtime_initialization",
        "semantic_runtime_initialization",
        "experiment_adapter_initialization",
        "latent_preparation",
        "runner_admission",
    }
)
ASSET_AUTHORITY_STATUS = "identity_blocked"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class SemanticTextureOperationalPreflightError(RuntimeError):
    """The operational preflight could not preserve its fixed boundary."""


class SemanticTextureOperationalBlockedError(RuntimeError):
    """A classified operational failure without scientific interpretation."""

    def __init__(
        self,
        blocked_class: str,
        message: str,
    ) -> None:
        if blocked_class not in ALLOWED_BLOCKED_CLASSES:
            raise SemanticTextureOperationalPreflightError(
                "blocked classification is not registered"
            )
        super().__init__(message)
        self.blocked_class = blocked_class


@dataclass(frozen=True)
class SemanticTextureOperationalConfiguration:
    """Validated package-local operational configuration."""

    schema_version: int
    profile_id: str
    unit_roster: tuple[str, str]
    asset_authority_status: str
    generation_prompt: str
    generation_negative_prompt: str
    generation_seed: int
    model_id: str = field(compare=False)
    model_revision: str = field(compare=False)
    minimum_cuda_vram_bytes: int
    minimum_free_ephemeral_bytes: int
    requirements_lock_sha256: str = field(compare=False)
    seeded_latent_protocol: str
    inspyrenet_source_repository: str = field(compare=False)
    inspyrenet_checkpoint_filename: str = field(compare=False)
    configuration_digest: str


@dataclass(frozen=True)
class OperationalUnitOutcome:
    """One bounded, non-scientific unit outcome."""

    unit_id: str
    started: bool
    status: str
    blocked_class: str | None
    sanitized_error_category: str | None
    sanitized_error_message: str | None
    sanitized_trace_tail: tuple[str, ...]
    elapsed_seconds: float
    public_result_identity: str | None
    witness_identity: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "blocked_class": self.blocked_class,
            "elapsed_seconds": self.elapsed_seconds,
            "public_result_identity": self.public_result_identity,
            "sanitized_error_category": self.sanitized_error_category,
            "sanitized_error_message": self.sanitized_error_message,
            "sanitized_trace_tail": list(self.sanitized_trace_tail),
            "started": self.started,
            "status": self.status,
            "unit_id": self.unit_id,
            "witness_identity": self.witness_identity,
        }


@dataclass(frozen=True)
class SemanticTextureOperationalResult:
    """Package-local result; it is not a governed scientific record."""

    schema_version: int
    profile_id: str
    observed_repository_revision: str
    run_id: str
    configuration_digest: str
    asset_authority_status: str
    model_id: str
    model_revision: str
    pre_execution_stage: str | None
    unit_outcomes: tuple[OperationalUnitOutcome, OperationalUnitOutcome]
    aggregate: None
    blocked_class: str
    status: str
    scientific_unit_count: int
    science_started: bool
    formal_tau_created: bool
    candidate_promoted: bool
    scientific_claims_supported: bool
    result_identity: str

    def as_dict(self) -> dict[str, object]:
        return {
            "aggregate": self.aggregate,
            "asset_authority_status": self.asset_authority_status,
            "blocked_class": self.blocked_class,
            "candidate_promoted": self.candidate_promoted,
            "configuration_digest": self.configuration_digest,
            "formal_tau_created": self.formal_tau_created,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "observed_repository_revision": self.observed_repository_revision,
            "pre_execution_stage": self.pre_execution_stage,
            "profile_id": self.profile_id,
            "result_identity": self.result_identity,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "science_started": self.science_started,
            "scientific_claims_supported": self.scientific_claims_supported,
            "scientific_unit_count": self.scientific_unit_count,
            "status": self.status,
            "unit_outcomes": [item.as_dict() for item in self.unit_outcomes],
        }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _blocked_class(error: BaseException) -> str:
    if isinstance(error, SemanticTextureOperationalBlockedError):
        return error.blocked_class
    if isinstance(error, (MemoryError, OSError)):
        return "resource_blocked"
    return "implementation_blocked"


def load_semantic_texture_operational_configuration(
    path: str | Path,
) -> SemanticTextureOperationalConfiguration:
    """Load the fixed two-unit, asset-blocked configuration."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticTextureOperationalPreflightError(
            "operational preflight configuration is unreadable"
        ) from exc
    required = {
        "asset_authority_status",
        "generation_negative_prompt",
        "generation_prompt",
        "generation_seed",
        "inspyrenet_checkpoint_filename",
        "inspyrenet_source_repository",
        "minimum_cuda_vram_bytes",
        "minimum_free_ephemeral_bytes",
        "model_id",
        "model_revision",
        "profile_id",
        "requirements_lock_sha256",
        "schema_version",
        "seeded_latent_protocol",
        "unit_roster",
        "zero_science_boundary",
    }
    if type(raw) is not dict or set(raw) != required:
        raise SemanticTextureOperationalPreflightError(
            "operational preflight configuration fields drifted"
        )
    zero_science = raw["zero_science_boundary"]
    if (
        raw["schema_version"] != PREFLIGHT_SCHEMA_VERSION
        or raw["profile_id"] != PREFLIGHT_PROFILE_ID
        or raw["unit_roster"] != list(UNIT_ROSTER)
        or raw["asset_authority_status"] != ASSET_AUTHORITY_STATUS
        or raw["generation_prompt"] != "a red cube"
        or raw["generation_negative_prompt"] != ""
        or raw["generation_seed"] != 2026081701
        or raw["minimum_cuda_vram_bytes"] != 23622320128
        or raw["minimum_free_ephemeral_bytes"] != 34359738368
        or type(raw["requirements_lock_sha256"]) is not str
        or _DIGEST.fullmatch(raw["requirements_lock_sha256"]) is None
        or raw["seeded_latent_protocol"]
        != "cpu_float32_generator_shape_1x16xheight_div8xwidth_div8_then_available_cuda_float16_once"
        or any(
            type(raw[field]) is not str or not raw[field]
            for field in (
                "model_id",
                "model_revision",
                "inspyrenet_source_repository",
                "inspyrenet_checkpoint_filename",
            )
        )
        or zero_science
        != {
            "aggregate": None,
            "candidate_promoted": False,
            "formal_tau_created": False,
            "science_started": False,
            "scientific_claims_supported": False,
            "scientific_unit_count": 0,
        }
    ):
        raise SemanticTextureOperationalPreflightError(
            "operational preflight configuration identity drifted"
        )
    configuration_identity = {
        field_name: field_value
        for field_name, field_value in raw.items()
        if field_name
        not in {
            "generation_negative_prompt",
            "generation_prompt",
            "generation_seed",
            "inspyrenet_source_repository",
            "inspyrenet_checkpoint_filename",
            "model_id",
            "model_revision",
            "requirements_lock_sha256",
        }
    }
    return SemanticTextureOperationalConfiguration(
        schema_version=raw["schema_version"],
        profile_id=raw["profile_id"],
        unit_roster=tuple(raw["unit_roster"]),
        asset_authority_status=raw["asset_authority_status"],
        generation_prompt=raw["generation_prompt"],
        generation_negative_prompt=raw["generation_negative_prompt"],
        generation_seed=raw["generation_seed"],
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        minimum_cuda_vram_bytes=raw["minimum_cuda_vram_bytes"],
        minimum_free_ephemeral_bytes=raw["minimum_free_ephemeral_bytes"],
        requirements_lock_sha256=raw["requirements_lock_sha256"],
        seeded_latent_protocol=raw["seeded_latent_protocol"],
        inspyrenet_source_repository=raw["inspyrenet_source_repository"],
        inspyrenet_checkpoint_filename=raw["inspyrenet_checkpoint_filename"],
        configuration_digest=_digest(configuration_identity),
    )


def _public_identity(observation: object) -> tuple[str, str | None]:
    result_identity = getattr(observation, "result_identity", None)
    if (
        type(result_identity) is not str
        or _DIGEST.fullmatch(result_identity) is None
    ):
        raise SemanticTextureOperationalBlockedError(
            "integrity_blocked",
            "public adapter result identity is unavailable",
        )
    result = getattr(observation, "result", None)
    witness = getattr(result, "witness", None)
    witness_identity = getattr(witness, "witness_identity", None)
    if (
        type(witness_identity) is not str
        or _DIGEST.fullmatch(witness_identity) is None
    ):
        raise SemanticTextureOperationalBlockedError(
            "integrity_blocked",
            "semantic-texture write witness identity is unavailable",
        )
    return result_identity, witness_identity


def _live_blind_detection_via_public_adapter(
    adapter: CegWmExperimentAdapter,
    *,
    detection_image_rgb8: object,
    detection_key: str,
    semantic_runtime: object,
    whitening_asset: object,
    hf_null: object,
    lf_null: object,
) -> object:
    """Reserved public call edge; Phase A has no authority to reach it."""

    return adapter.detect_semantic_texture_candidate(
        detection_image_rgb8,
        detection_key,
        semantic_runtime,
        whitening_asset,
        hf_null=hf_null,
        lf_null=lf_null,
    )


def _unit_failure(
    unit_id: str,
    *,
    started: bool,
    elapsed_seconds: float,
    error: BaseException,
) -> OperationalUnitOutcome:
    blocked_class = _blocked_class(error)
    return OperationalUnitOutcome(
        unit_id=unit_id,
        started=started,
        status="blocked",
        blocked_class=blocked_class,
        sanitized_error_category=blocked_class,
        sanitized_error_message=None,
        sanitized_trace_tail=(),
        elapsed_seconds=max(0.0, elapsed_seconds),
        public_result_identity=None,
        witness_identity=None,
    )


def _operational_result(
    configuration: SemanticTextureOperationalConfiguration,
    *,
    observed_repository_revision: str,
    run_id: str,
    pre_execution_stage: str | None,
    write_outcome: OperationalUnitOutcome,
    detector_outcome: OperationalUnitOutcome,
) -> SemanticTextureOperationalResult:
    if _REVISION.fullmatch(observed_repository_revision) is None:
        raise SemanticTextureOperationalPreflightError(
            "source revision must be exact"
        )
    if not run_id or len(run_id) > 96 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise SemanticTextureOperationalPreflightError("run identity is invalid")
    blocked_class = (
        write_outcome.blocked_class
        if write_outcome.status == "blocked"
        else detector_outcome.blocked_class
    )
    if blocked_class not in ALLOWED_BLOCKED_CLASSES:
        raise SemanticTextureOperationalPreflightError(
            "result blocked classification drifted"
        )
    pre_execution_failure = (
        write_outcome.started is False and detector_outcome.started is False
    )
    if pre_execution_failure:
        if pre_execution_stage not in ALLOWED_PRE_EXECUTION_STAGES:
            raise SemanticTextureOperationalPreflightError(
                "pre-execution stage is not registered"
            )
    elif pre_execution_stage is not None:
        raise SemanticTextureOperationalPreflightError(
            "started operational result retains a pre-execution stage"
        )
    unsigned = {
        "aggregate": None,
        "asset_authority_status": configuration.asset_authority_status,
        "blocked_class": blocked_class,
        "candidate_promoted": False,
        "configuration_digest": configuration.configuration_digest,
        "formal_tau_created": False,
        "pre_execution_stage": pre_execution_stage,
        "profile_id": configuration.profile_id,
        "run_id": run_id,
        "schema_version": configuration.schema_version,
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "status": "blocked",
        "unit_outcomes": [
            write_outcome.as_dict(),
            detector_outcome.as_dict(),
        ],
    }
    result = SemanticTextureOperationalResult(
        schema_version=configuration.schema_version,
        profile_id=configuration.profile_id,
        observed_repository_revision=observed_repository_revision,
        run_id=run_id,
        configuration_digest=configuration.configuration_digest,
        asset_authority_status=configuration.asset_authority_status,
        model_id=configuration.model_id,
        model_revision=configuration.model_revision,
        pre_execution_stage=pre_execution_stage,
        unit_outcomes=(write_outcome, detector_outcome),
        aggregate=None,
        blocked_class=blocked_class,
        status="blocked",
        scientific_unit_count=0,
        science_started=False,
        formal_tau_created=False,
        candidate_promoted=False,
        scientific_claims_supported=False,
        result_identity=_digest(unsigned),
    )
    if not math.isfinite(sum(item.elapsed_seconds for item in result.unit_outcomes)):
        raise SemanticTextureOperationalPreflightError(
            "unit timing is non-finite"
        )
    return result


def create_semantic_texture_operational_pre_execution_failure(
    configuration: SemanticTextureOperationalConfiguration,
    *,
    observed_repository_revision: str,
    run_id: str,
    blocked_class: str,
    pre_execution_stage: str,
) -> SemanticTextureOperationalResult:
    """Create the fixed two-unstarted outcome before trusted execution begins."""

    error = SemanticTextureOperationalBlockedError(
        blocked_class,
        "trusted semantic-texture execution did not start",
    )
    return _operational_result(
        configuration,
        observed_repository_revision=observed_repository_revision,
        run_id=run_id,
        pre_execution_stage=pre_execution_stage,
        write_outcome=_unit_failure(
            WRITE_UNIT_ID,
            started=False,
            elapsed_seconds=0.0,
            error=error,
        ),
        detector_outcome=_unit_failure(
            BLIND_DETECTION_UNIT_ID,
            started=False,
            elapsed_seconds=0.0,
            error=error,
        ),
    )


def execute_semantic_texture_operational_preflight(
    adapter: CegWmExperimentAdapter,
    configuration: SemanticTextureOperationalConfiguration,
    *,
    observed_repository_revision: str,
    run_id: str,
    base_latent: object,
    detection_key: str,
    semantic_runtime: object,
    monotonic_clock: Callable[[], float] = monotonic,
) -> SemanticTextureOperationalResult:
    """Run the live write unit and stop at the missing detector asset authority."""

    if _REVISION.fullmatch(observed_repository_revision) is None:
        raise SemanticTextureOperationalPreflightError(
            "source revision must be exact"
        )
    if not run_id or len(run_id) > 96 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise SemanticTextureOperationalPreflightError("run identity is invalid")
    if configuration.asset_authority_status != ASSET_AUTHORITY_STATUS:
        raise SemanticTextureOperationalPreflightError(
            "asset authority status drifted"
        )

    write_started = monotonic_clock()
    try:
        write_observation = (
            adapter.execute_semantic_texture_content_write_and_vae(
                base_latent,
                detection_key,
                semantic_runtime,
            )
        )
        public_result_identity, witness_identity = _public_identity(
            write_observation
        )
        write_outcome = OperationalUnitOutcome(
            unit_id=WRITE_UNIT_ID,
            started=True,
            status="passed",
            blocked_class=None,
            sanitized_error_category=None,
            sanitized_error_message=None,
            sanitized_trace_tail=(),
            elapsed_seconds=max(0.0, monotonic_clock() - write_started),
            public_result_identity=public_result_identity,
            witness_identity=witness_identity,
        )
    except Exception as exc:
        write_outcome = _unit_failure(
            WRITE_UNIT_ID,
            started=True,
            elapsed_seconds=monotonic_clock() - write_started,
            error=exc,
        )
        inherited = SemanticTextureOperationalBlockedError(
            write_outcome.blocked_class or "implementation_blocked",
            "blind detection was not started because the write unit blocked",
        )
        detector_outcome = _unit_failure(
            BLIND_DETECTION_UNIT_ID,
            started=False,
            elapsed_seconds=0.0,
            error=inherited,
        )
    else:
        detector_started = monotonic_clock()
        detector_outcome = _unit_failure(
            BLIND_DETECTION_UNIT_ID,
            started=True,
            elapsed_seconds=monotonic_clock() - detector_started,
            error=SemanticTextureOperationalBlockedError(
                "identity_blocked",
                "dedicated semantic-texture detector assets are not authorized",
            ),
        )

    return _operational_result(
        configuration,
        observed_repository_revision=observed_repository_revision,
        run_id=run_id,
        pre_execution_stage=None,
        write_outcome=write_outcome,
        detector_outcome=detector_outcome,
    )


__all__ = [
    "ALLOWED_BLOCKED_CLASSES",
    "ALLOWED_PRE_EXECUTION_STAGES",
    "ASSET_AUTHORITY_STATUS",
    "BLIND_DETECTION_UNIT_ID",
    "OperationalUnitOutcome",
    "PREFLIGHT_PROFILE_ID",
    "PREFLIGHT_SCHEMA_VERSION",
    "SemanticTextureOperationalBlockedError",
    "SemanticTextureOperationalConfiguration",
    "SemanticTextureOperationalPreflightError",
    "SemanticTextureOperationalResult",
    "UNIT_ROSTER",
    "WRITE_UNIT_ID",
    "create_semantic_texture_operational_pre_execution_failure",
    "execute_semantic_texture_operational_preflight",
    "load_semantic_texture_operational_configuration",
]
