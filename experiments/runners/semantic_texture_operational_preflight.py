"""Two-unit, zero-science semantic-texture operational preflight."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from time import monotonic
from typing import Callable

from experiments.methods import CegWmExperimentAdapter


PREFLIGHT_SCHEMA_VERSION = 1
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
    model_id: str
    model_revision: str
    inspyrenet_source_revision: str
    inspyrenet_checkpoint_revision: str
    inspyrenet_checkpoint_sha256: str
    inspyrenet_checkpoint_size_bytes: int
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
    source_revision: str
    run_id: str
    configuration_digest: str
    package_identity: str
    asset_authority_status: str
    model_id: str
    model_revision: str
    inspyrenet_source_revision: str
    inspyrenet_checkpoint_revision: str
    inspyrenet_checkpoint_sha256: str
    inspyrenet_checkpoint_size_bytes: int
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
            "inspyrenet_checkpoint_revision": self.inspyrenet_checkpoint_revision,
            "inspyrenet_checkpoint_sha256": self.inspyrenet_checkpoint_sha256,
            "inspyrenet_checkpoint_size_bytes": self.inspyrenet_checkpoint_size_bytes,
            "inspyrenet_source_revision": self.inspyrenet_source_revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "package_identity": self.package_identity,
            "profile_id": self.profile_id,
            "result_identity": self.result_identity,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "science_started": self.science_started,
            "scientific_claims_supported": self.scientific_claims_supported,
            "scientific_unit_count": self.scientific_unit_count,
            "source_revision": self.source_revision,
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
        "inspyrenet_checkpoint_revision",
        "inspyrenet_checkpoint_sha256",
        "inspyrenet_checkpoint_size_bytes",
        "inspyrenet_source_revision",
        "model_id",
        "model_revision",
        "profile_id",
        "schema_version",
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
        or type(raw["model_id"]) is not str
        or not raw["model_id"]
        or _REVISION.fullmatch(raw["model_revision"]) is None
        or _REVISION.fullmatch(raw["inspyrenet_source_revision"]) is None
        or _REVISION.fullmatch(raw["inspyrenet_checkpoint_revision"]) is None
        or _DIGEST.fullmatch(raw["inspyrenet_checkpoint_sha256"]) is None
        or type(raw["inspyrenet_checkpoint_size_bytes"]) is not int
        or raw["inspyrenet_checkpoint_size_bytes"] <= 0
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
    return SemanticTextureOperationalConfiguration(
        schema_version=raw["schema_version"],
        profile_id=raw["profile_id"],
        unit_roster=tuple(raw["unit_roster"]),
        asset_authority_status=raw["asset_authority_status"],
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        inspyrenet_source_revision=raw["inspyrenet_source_revision"],
        inspyrenet_checkpoint_revision=raw["inspyrenet_checkpoint_revision"],
        inspyrenet_checkpoint_sha256=raw["inspyrenet_checkpoint_sha256"],
        inspyrenet_checkpoint_size_bytes=raw[
            "inspyrenet_checkpoint_size_bytes"
        ],
        configuration_digest=_digest(raw),
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


def execute_semantic_texture_operational_preflight(
    adapter: CegWmExperimentAdapter,
    configuration: SemanticTextureOperationalConfiguration,
    *,
    source_revision: str,
    run_id: str,
    package_identity: str,
    base_latent: object,
    detection_key: str,
    semantic_runtime: object,
    monotonic_clock: Callable[[], float] = monotonic,
) -> SemanticTextureOperationalResult:
    """Run the live write unit and stop at the missing detector asset authority."""

    if _REVISION.fullmatch(source_revision) is None:
        raise SemanticTextureOperationalPreflightError(
            "source revision must be exact"
        )
    if not run_id or len(run_id) > 96 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise SemanticTextureOperationalPreflightError("run identity is invalid")
    if _DIGEST.fullmatch(package_identity) is None:
        raise SemanticTextureOperationalPreflightError(
            "package identity must be SHA-256"
        )
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

    blocked_class = detector_outcome.blocked_class
    if write_outcome.status == "blocked":
        blocked_class = write_outcome.blocked_class
    if blocked_class not in ALLOWED_BLOCKED_CLASSES:
        raise SemanticTextureOperationalPreflightError(
            "result blocked classification drifted"
        )
    unsigned = {
        "aggregate": None,
        "asset_authority_status": configuration.asset_authority_status,
        "blocked_class": blocked_class,
        "candidate_promoted": False,
        "configuration_digest": configuration.configuration_digest,
        "formal_tau_created": False,
        "inspyrenet_checkpoint_revision": (
            configuration.inspyrenet_checkpoint_revision
        ),
        "inspyrenet_checkpoint_sha256": (
            configuration.inspyrenet_checkpoint_sha256
        ),
        "inspyrenet_checkpoint_size_bytes": (
            configuration.inspyrenet_checkpoint_size_bytes
        ),
        "inspyrenet_source_revision": configuration.inspyrenet_source_revision,
        "model_id": configuration.model_id,
        "model_revision": configuration.model_revision,
        "package_identity": package_identity,
        "profile_id": configuration.profile_id,
        "run_id": run_id,
        "schema_version": configuration.schema_version,
        "science_started": False,
        "scientific_claims_supported": False,
        "scientific_unit_count": 0,
        "source_revision": source_revision,
        "status": "blocked",
        "unit_outcomes": [
            write_outcome.as_dict(),
            detector_outcome.as_dict(),
        ],
    }
    result_identity = _digest(unsigned)
    result = SemanticTextureOperationalResult(
        schema_version=configuration.schema_version,
        profile_id=configuration.profile_id,
        source_revision=source_revision,
        run_id=run_id,
        configuration_digest=configuration.configuration_digest,
        package_identity=package_identity,
        asset_authority_status=configuration.asset_authority_status,
        model_id=configuration.model_id,
        model_revision=configuration.model_revision,
        inspyrenet_source_revision=configuration.inspyrenet_source_revision,
        inspyrenet_checkpoint_revision=(
            configuration.inspyrenet_checkpoint_revision
        ),
        inspyrenet_checkpoint_sha256=(
            configuration.inspyrenet_checkpoint_sha256
        ),
        inspyrenet_checkpoint_size_bytes=(
            configuration.inspyrenet_checkpoint_size_bytes
        ),
        unit_outcomes=(write_outcome, detector_outcome),
        aggregate=None,
        blocked_class=blocked_class,
        status="blocked",
        scientific_unit_count=0,
        science_started=False,
        formal_tau_created=False,
        candidate_promoted=False,
        scientific_claims_supported=False,
        result_identity=result_identity,
    )
    if not math.isfinite(sum(item.elapsed_seconds for item in result.unit_outcomes)):
        raise SemanticTextureOperationalPreflightError(
            "unit timing is non-finite"
        )
    return result


__all__ = [
    "ALLOWED_BLOCKED_CLASSES",
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
    "execute_semantic_texture_operational_preflight",
    "load_semantic_texture_operational_configuration",
]
