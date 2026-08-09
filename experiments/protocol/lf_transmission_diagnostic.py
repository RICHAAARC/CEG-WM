"""Frozen development protocol for locating LF carrier signal loss."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping

from experiments.protocol.development_exploration import DevelopmentStudyUnit
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)


PROTOCOL_ID = "ceg_wm_lf_transmission_diagnostic"
PROTOCOL_VERSION = "1.0.0"
RUN_ID = "ceg_wm_lf_carrier_to_detector_transmission_diagnostic"
SCIENTIFIC_CLUSTER_COUNT = 8
OPERATIONAL_UNIT_COUNT = 0
SCIENTIFIC_UNIT_COUNT = 8
MAXIMUM_TOTAL_UNITS = 8
MAXIMUM_ATTEMPTS_PER_UNIT = 2
MAXIMUM_DURATION_SECONDS = 2700
SIGNAL_POSITIONS = (
    "callback_pre_write",
    "actual_dtype_post_write",
    "scheduler_suffix_final",
    "rgb_vae_reencoded",
)
CLAIM_BOUNDARY = (
    "development_lf_transport_direction_only_no_threshold_no_promotion_no_claim"
)
STOP_RULE = (
    "registered_minus_primary_null_positive_at_least_seven_of_eight_and_"
    "registered_minus_wrong_key_positive_at_least_seven_of_eight_with_no_"
    "budget_integrity_or_nonfinite_failure_then_allow_request_for_lf_"
    "directional_validation"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class LfTransmissionProtocolError(ValueError):
    """The checked-in LF transmission diagnostic protocol is inconsistent."""


def canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class LfTransmissionManifestEntry:
    cluster_ordinal: int
    cluster_identity: str
    prompt: str
    prompt_digest: str
    generation_seed: int
    image_lineage_identity: str
    image_lineage_digest: str
    split: str
    role_id: str

    def validate(self, *, image_lineage_namespace: str) -> None:
        if type(self.cluster_ordinal) is not int or not 0 <= self.cluster_ordinal < 8:
            raise LfTransmissionProtocolError("cluster ordinal is invalid")
        if (
            type(self.cluster_identity) is not str
            or not self.cluster_identity
            or type(self.prompt) is not str
            or not self.prompt
            or self.split != "development"
            or self.role_id != "lf_transmission_diagnostic"
        ):
            raise LfTransmissionProtocolError("manifest entry identity is invalid")
        if self.prompt_digest != sha256(self.prompt.encode("utf-8")).hexdigest():
            raise LfTransmissionProtocolError("prompt digest drifted")
        if type(self.generation_seed) is not int or self.generation_seed < 0:
            raise LfTransmissionProtocolError("generation seed is invalid")
        if (
            type(self.image_lineage_identity) is not str
            or not self.image_lineage_identity
            or self.image_lineage_digest
            != canonical_digest(
                {
                    "cluster_identity": self.cluster_identity,
                    "image_lineage_namespace": image_lineage_namespace,
                    "image_lineage_identity": self.image_lineage_identity,
                    "generation_seed": self.generation_seed,
                }
            )
        ):
            raise LfTransmissionProtocolError("image lineage digest drifted")


@dataclass(frozen=True, slots=True)
class LfTransmissionManifest:
    schema_version: str
    manifest_id: str
    seed_namespace: str
    source_cluster_namespace: str
    image_lineage_namespace: str
    registered_key_derivation_identity: str
    registered_key_domain_identity: str
    registered_key_family_namespace: str
    entries: tuple[LfTransmissionManifestEntry, ...]

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if self.schema_version != "ceg_wm_lf_transmission_manifest_v1":
            raise LfTransmissionProtocolError("manifest schema drifted")
        if (
            self.manifest_id != "lf_transmission_diagnostic_eight_cluster_manifest"
            or self.seed_namespace != "lf_transmission_diagnostic_20260809"
            or self.source_cluster_namespace
            != "lf_transmission_source_clusters_20260809"
            or self.image_lineage_namespace
            != "lf_transmission_paired_rgb_lineages_20260809"
            or self.registered_key_derivation_identity
            != "lf_transmission_registered_key_subdomain_derivation"
            or self.registered_key_domain_identity
            != "lf_carrier_registered_detection_key_domain"
            or self.registered_key_family_namespace
            != "lf_transmission_registered_key_family_20260809"
        ):
            raise LfTransmissionProtocolError("manifest identity drifted")
        if len(self.entries) != SCIENTIFIC_CLUSTER_COUNT:
            raise LfTransmissionProtocolError("manifest cluster count drifted")
        for entry in self.entries:
            entry.validate(
                image_lineage_namespace=self.image_lineage_namespace
            )
        if tuple(item.cluster_ordinal for item in self.entries) != tuple(range(8)):
            raise LfTransmissionProtocolError("manifest order drifted")
        if len({item.cluster_identity for item in self.entries}) != len(self.entries):
            raise LfTransmissionProtocolError("manifest clusters collide")
        if len({item.generation_seed for item in self.entries}) != len(self.entries):
            raise LfTransmissionProtocolError("manifest seeds collide")


@dataclass(frozen=True, slots=True)
class LfTransmissionDiagnosticProtocol:
    schema_version: str
    protocol_id: str
    protocol_version: str
    run_id: str
    split: str
    role_id: str
    candidate_identity: str
    manifest_path: str
    manifest_file_sha256: str
    signal_positions: tuple[str, ...]
    operational_unit_count: int
    scientific_cluster_count: int
    maximum_total_units: int
    maximum_attempts_per_unit: int
    maximum_duration_seconds_per_unit: int
    stop_rule: str
    claim_boundary: str
    unit_roster_digest: str

    @property
    def unit_roster(self) -> tuple[DevelopmentStudyUnit, ...]:
        return tuple(
            DevelopmentStudyUnit(
                unit_index=ordinal,
                phase="development_scientific_breadth",
                responsibility_id="lf_detector",
                source_cluster_ordinal=ordinal,
                content_branch_id="lf_only",
                geometry_case_id="not_applicable",
                maximum_record_attempts=self.maximum_attempts_per_unit,
                maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
            )
            for ordinal in range(self.scientific_cluster_count)
        )

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if (
            self.schema_version != "ceg_wm_lf_transmission_diagnostic_protocol_v1"
            or self.protocol_id != PROTOCOL_ID
            or self.protocol_version != PROTOCOL_VERSION
            or self.run_id != RUN_ID
            or self.split != "development"
            or self.role_id != "lf_transmission_diagnostic"
            or self.candidate_identity != "lf_low_pass"
        ):
            raise LfTransmissionProtocolError("protocol identity drifted")
        if _DIGEST.fullmatch(self.manifest_file_sha256) is None:
            raise LfTransmissionProtocolError("manifest file digest is invalid")
        if self.signal_positions != SIGNAL_POSITIONS:
            raise LfTransmissionProtocolError("signal positions drifted")
        if (
            self.operational_unit_count != OPERATIONAL_UNIT_COUNT
            or self.scientific_cluster_count != SCIENTIFIC_CLUSTER_COUNT
            or self.maximum_total_units != MAXIMUM_TOTAL_UNITS
            or len(self.unit_roster) != MAXIMUM_TOTAL_UNITS
            or self.maximum_attempts_per_unit != MAXIMUM_ATTEMPTS_PER_UNIT
            or self.maximum_duration_seconds_per_unit != MAXIMUM_DURATION_SECONDS
            or self.stop_rule != STOP_RULE
            or self.claim_boundary != CLAIM_BOUNDARY
        ):
            raise LfTransmissionProtocolError("protocol budget or rule drifted")
        if self.unit_roster_digest != canonical_digest(
            tuple(asdict(unit) for unit in self.unit_roster)
        ):
            raise LfTransmissionProtocolError("unit roster digest drifted")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LfTransmissionProtocolError("checked-in JSON is unreadable") from exc
    if type(value) is not dict:
        raise LfTransmissionProtocolError("checked-in JSON must be a mapping")
    return value


def load_lf_transmission_manifest(path: str | Path) -> LfTransmissionManifest:
    raw = _load_json(Path(path))
    try:
        manifest = LfTransmissionManifest(
            schema_version=raw["schema_version"],
            manifest_id=raw["manifest_id"],
            seed_namespace=raw["seed_namespace"],
            source_cluster_namespace=raw["source_cluster_namespace"],
            image_lineage_namespace=raw["image_lineage_namespace"],
            registered_key_derivation_identity=(
                raw["registered_key_derivation_identity"]
            ),
            registered_key_domain_identity=raw["registered_key_domain_identity"],
            registered_key_family_namespace=raw["registered_key_family_namespace"],
            entries=tuple(
                LfTransmissionManifestEntry(**item) for item in raw["entries"]
            ),
        )
    except (KeyError, TypeError) as exc:
        raise LfTransmissionProtocolError("manifest schema is invalid") from exc
    manifest.validate()
    return manifest


def derive_lf_transmission_analysis_identity(
    entry: LfTransmissionManifestEntry,
    manifest: LfTransmissionManifest,
    *,
    root_key_public_digest: str,
) -> AnalysisUnitIdentity:
    """Derive the frozen LF source-cluster and registered key-family identity."""

    manifest.validate()
    entry.validate(image_lineage_namespace=manifest.image_lineage_namespace)
    if _DIGEST.fullmatch(root_key_public_digest) is None:
        raise LfTransmissionProtocolError("root key public digest is invalid")
    key_family = canonical_digest(
        {
            "registered_key_derivation_identity": (
                manifest.registered_key_derivation_identity
            ),
            "registered_key_domain_identity": manifest.registered_key_domain_identity,
            "registered_key_family_namespace": (
                manifest.registered_key_family_namespace
            ),
            "root_key_public_digest": root_key_public_digest,
            "seed_namespace": manifest.seed_namespace,
        }
    )
    derived_cluster = derive_source_cluster_id(
        prompt_digest=entry.prompt_digest,
        generation_seed=entry.generation_seed,
        image_lineage_digest=entry.image_lineage_digest,
        registered_key_family_digest=key_family,
    )
    return AnalysisUnitIdentity(
        unit_id=f"lf_transmission_cluster_{entry.cluster_ordinal:02d}",
        case_id="paired_clean_lf_transport_observation",
        source_cluster_id=derived_cluster,
        prompt_digest=entry.prompt_digest,
        generation_seed=entry.generation_seed,
        image_lineage_digest=entry.image_lineage_digest,
        registered_key_family_digest=key_family,
    )


def load_lf_transmission_protocol(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[LfTransmissionDiagnosticProtocol, LfTransmissionManifest]:
    raw = _load_json(Path(path))
    try:
        protocol = LfTransmissionDiagnosticProtocol(
            **{
                **raw,
                "signal_positions": tuple(raw["signal_positions"]),
            }
        )
    except (KeyError, TypeError) as exc:
        raise LfTransmissionProtocolError("protocol schema is invalid") from exc
    protocol.validate()
    manifest_path = Path(repository_root) / protocol.manifest_path
    if sha256(manifest_path.read_bytes()).hexdigest() != protocol.manifest_file_sha256:
        raise LfTransmissionProtocolError("manifest file digest drifted")
    manifest = load_lf_transmission_manifest(manifest_path)
    return protocol, manifest


__all__ = [
    "CLAIM_BOUNDARY",
    "LfTransmissionDiagnosticProtocol",
    "LfTransmissionManifest",
    "LfTransmissionManifestEntry",
    "LfTransmissionProtocolError",
    "SIGNAL_POSITIONS",
    "STOP_RULE",
    "RUN_ID",
    "canonical_digest",
    "derive_lf_transmission_analysis_identity",
    "load_lf_transmission_manifest",
    "load_lf_transmission_protocol",
]
