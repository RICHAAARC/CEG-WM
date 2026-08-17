"""Frozen development protocol for locating HF carrier signal loss."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping

from experiments.protocol.development_support import DevelopmentStudyUnit


PROTOCOL_ID = "ceg_wm_hf_transmission_diagnostic"
PROTOCOL_VERSION = "1.0.0"
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
    "development_hf_transport_direction_only_no_threshold_no_promotion_no_claim"
)
STOP_RULE = (
    "registered_minus_primary_null_positive_at_least_seven_of_eight_and_"
    "registered_minus_wrong_key_positive_at_least_seven_of_eight_with_no_"
    "budget_integrity_or_nonfinite_failure"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class HfTransmissionProtocolError(ValueError):
    """The checked-in HF transmission diagnostic protocol is inconsistent."""


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
class HfTransmissionManifestEntry:
    cluster_ordinal: int
    cluster_identity: str
    prompt: str
    prompt_digest: str
    generation_seed: int
    image_lineage_identity: str
    image_lineage_digest: str
    split: str
    role_id: str

    def validate(self) -> None:
        if type(self.cluster_ordinal) is not int or not 0 <= self.cluster_ordinal < 8:
            raise HfTransmissionProtocolError("cluster ordinal is invalid")
        if (
            type(self.cluster_identity) is not str
            or not self.cluster_identity
            or type(self.prompt) is not str
            or not self.prompt
            or self.split != "development"
            or self.role_id != "hf_transmission_diagnostic"
        ):
            raise HfTransmissionProtocolError("manifest entry identity is invalid")
        if self.prompt_digest != sha256(self.prompt.encode("utf-8")).hexdigest():
            raise HfTransmissionProtocolError("prompt digest drifted")
        if type(self.generation_seed) is not int or self.generation_seed < 0:
            raise HfTransmissionProtocolError("generation seed is invalid")
        if (
            type(self.image_lineage_identity) is not str
            or not self.image_lineage_identity
            or self.image_lineage_digest
            != canonical_digest(
                {
                    "cluster_identity": self.cluster_identity,
                    "image_lineage_identity": self.image_lineage_identity,
                    "generation_seed": self.generation_seed,
                }
            )
        ):
            raise HfTransmissionProtocolError("image lineage digest drifted")


@dataclass(frozen=True, slots=True)
class HfTransmissionManifest:
    schema_version: str
    manifest_id: str
    seed_namespace: str
    entries: tuple[HfTransmissionManifestEntry, ...]

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if self.schema_version != "ceg_wm_hf_transmission_manifest_v1":
            raise HfTransmissionProtocolError("manifest schema drifted")
        if (
            self.manifest_id != "hf_transmission_diagnostic_eight_cluster_manifest"
            or self.seed_namespace != "hf_transmission_diagnostic_20260808"
        ):
            raise HfTransmissionProtocolError("manifest identity drifted")
        if len(self.entries) != SCIENTIFIC_CLUSTER_COUNT:
            raise HfTransmissionProtocolError("manifest cluster count drifted")
        for entry in self.entries:
            entry.validate()
        if tuple(item.cluster_ordinal for item in self.entries) != tuple(range(8)):
            raise HfTransmissionProtocolError("manifest order drifted")
        if len({item.cluster_identity for item in self.entries}) != len(self.entries):
            raise HfTransmissionProtocolError("manifest clusters collide")
        if len({item.generation_seed for item in self.entries}) != len(self.entries):
            raise HfTransmissionProtocolError("manifest seeds collide")


@dataclass(frozen=True, slots=True)
class HfTransmissionDiagnosticProtocol:
    schema_version: str
    protocol_id: str
    protocol_version: str
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
                responsibility_id="hf_detector",
                source_cluster_ordinal=ordinal,
                content_branch_id="hf_only",
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
            self.schema_version != "ceg_wm_hf_transmission_diagnostic_protocol_v1"
            or self.protocol_id != PROTOCOL_ID
            or self.protocol_version != PROTOCOL_VERSION
            or self.split != "development"
            or self.role_id != "hf_transmission_diagnostic"
            or self.candidate_identity != "hf_sparse_tail"
        ):
            raise HfTransmissionProtocolError("protocol identity drifted")
        if _DIGEST.fullmatch(self.manifest_file_sha256) is None:
            raise HfTransmissionProtocolError("manifest file digest is invalid")
        if self.signal_positions != SIGNAL_POSITIONS:
            raise HfTransmissionProtocolError("signal positions drifted")
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
            raise HfTransmissionProtocolError("protocol budget or rule drifted")
        if self.unit_roster_digest != canonical_digest(
            tuple(asdict(unit) for unit in self.unit_roster)
        ):
            raise HfTransmissionProtocolError("unit roster digest drifted")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HfTransmissionProtocolError("checked-in JSON is unreadable") from exc
    if type(value) is not dict:
        raise HfTransmissionProtocolError("checked-in JSON must be a mapping")
    return value


def load_hf_transmission_manifest(path: str | Path) -> HfTransmissionManifest:
    raw = _load_json(Path(path))
    try:
        manifest = HfTransmissionManifest(
            schema_version=raw["schema_version"],
            manifest_id=raw["manifest_id"],
            seed_namespace=raw["seed_namespace"],
            entries=tuple(
                HfTransmissionManifestEntry(**item) for item in raw["entries"]
            ),
        )
    except (KeyError, TypeError) as exc:
        raise HfTransmissionProtocolError("manifest schema is invalid") from exc
    manifest.validate()
    return manifest


def load_hf_transmission_protocol(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[HfTransmissionDiagnosticProtocol, HfTransmissionManifest]:
    raw = _load_json(Path(path))
    try:
        protocol = HfTransmissionDiagnosticProtocol(
            **{
                **raw,
                "signal_positions": tuple(raw["signal_positions"]),
            }
        )
    except (KeyError, TypeError) as exc:
        raise HfTransmissionProtocolError("protocol schema is invalid") from exc
    protocol.validate()
    manifest_path = Path(repository_root) / protocol.manifest_path
    if sha256(manifest_path.read_bytes()).hexdigest() != protocol.manifest_file_sha256:
        raise HfTransmissionProtocolError("manifest file digest drifted")
    manifest = load_hf_transmission_manifest(manifest_path)
    return protocol, manifest


__all__ = [
    "CLAIM_BOUNDARY",
    "HfTransmissionDiagnosticProtocol",
    "HfTransmissionManifest",
    "HfTransmissionManifestEntry",
    "HfTransmissionProtocolError",
    "SIGNAL_POSITIONS",
    "STOP_RULE",
    "canonical_digest",
    "load_hf_transmission_manifest",
    "load_hf_transmission_protocol",
]
