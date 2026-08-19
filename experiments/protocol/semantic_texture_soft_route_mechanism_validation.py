"""Pure soft-route mechanism-validation protocol contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite, nextafter
from pathlib import Path
import re
from typing import Mapping, Sequence

from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)


PROTOCOL_ID = "semantic_texture_soft_route_mechanism_validation"
MANIFEST_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
SELECTION_ROLE = "semantic_texture_soft_route_candidate_selection"
CONFIRMATION_ROLE = "semantic_texture_soft_route_untouched_confirmation"
SOURCE_ROSTER_PATH = "configs/experiments/hf_only_reference_prompt_roster.json"
SOURCE_ROSTER_ROWS_DIGEST = "9488cc26be1723bf40e1e7336c97a2b9c5307c1d4a8ac97f184855a832f12e69"
KEY_FAMILY_NAMESPACE_DIGEST = "c94f68e1aaf69b710630d3a3401262c3a7af7afdeeffe6d0f9ea3eb63e1777b1"
SELECTION_ENTRIES_DIGEST = "5e270361246f7a591fa6423c5e3be360c17cf074a131ef6e1a61a684c74d8fb4"
SELECTION_MANIFEST_DIGEST = "55d0dff884155c087b6adb9895a621243669d8d89715f9dae7a54b0613576bec"
CONFIRMATION_ENTRIES_DIGEST = "0ebcec073a815d0f5f90cbb569d93b385ea85f5a641c7e335e27f10d1e694cad"
CONFIRMATION_MANIFEST_DIGEST = "bd7f1f76a5eb39fc3b8fd225c806e4f189dda8d4adee4758ccb7ac6f3f7f16f1"
CLUSTER_COUNT = 32
ALPHA_SELECTION_FLOAT64_HEX = "0x1.999999999999ap-4"
ASSET_BUNDLE_DIGEST = "f9dd6df410cb4f7895376c65c5f6d3e764f6cfddabd0d64d525fdaaefd93de3d"
ASSET_BUNDLE_SHA256 = "126f73150584d5c5a1e5b5e2dbffa9bb0379a9375c202ab49a87b56f99c41ea7"
ASSET_BUNDLE_RELATIVE_PATH = f"semantic_texture_soft_detector_assets/{ASSET_BUNDLE_DIGEST}/semantic_texture_soft_detector_asset_bundle.json"
SELECTION_ARTIFACT_RELATIVE_PATH = "semantic_texture_soft_route_mechanism_validation/candidate_selection/semantic_texture_soft_route_selection_artifact.json"
ARMS = (
    "clean_unwatermarked",
    "hf_only",
    "lf_only",
    "semantic_texture_soft_routed",
    "semantic_texture_route_disabled",
)
ATTACKS = ("identity", "crop_0_75")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class SoftRouteMechanismProtocolError(ValueError):
    """A soft-route mechanism validation literal protocol artifact is malformed or unauthorized."""


def canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_digest(value: object, role: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise SoftRouteMechanismProtocolError(f"{role} digest is invalid")
    return value


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismManifestEntry:
    source_row: int
    prompt_text: str
    prompt_digest: str
    generation_seed: int
    manifest_cluster_label: str
    source_cluster_id: str
    registered_key_family_digest: str
    image_lineage_identity: str
    image_lineage_digest: str

    def validate(self, *, ordinal: int, role_id: str) -> None:
        if type(self.source_row) is not int or self.source_row < 65:
            raise SoftRouteMechanismProtocolError("source row is invalid")
        if type(self.prompt_text) is not str or not self.prompt_text:
            raise SoftRouteMechanismProtocolError("prompt text is invalid")
        if self.prompt_digest != sha256(self.prompt_text.encode("utf-8")).hexdigest():
            raise SoftRouteMechanismProtocolError("prompt digest drifted")
        if type(self.generation_seed) is not int:
            raise SoftRouteMechanismProtocolError("generation seed is invalid")
        expected_lineage = canonical_digest(
            {
                "generation_seed": self.generation_seed,
                "identity_role": "soft_route_mechanism_image_lineage",
                "prompt_digest": self.prompt_digest,
                "registered_key_family_digest": self.registered_key_family_digest,
                "role_id": role_id,
            }
        )
        expected_lineage_digest = canonical_digest(
            {
                "generation_seed": self.generation_seed,
                "image_lineage_identity": expected_lineage,
                "image_lineage_namespace": PROTOCOL_ID,
            }
        )
        if (
            self.image_lineage_identity != expected_lineage
            or self.image_lineage_digest != expected_lineage_digest
            or self.registered_key_family_digest != KEY_FAMILY_NAMESPACE_DIGEST
        ):
            raise SoftRouteMechanismProtocolError("entry identity drifted")
        expected_cluster = derive_source_cluster_id(
            prompt_digest=self.prompt_digest,
            generation_seed=self.generation_seed,
            image_lineage_digest=self.image_lineage_digest,
            registered_key_family_digest=self.registered_key_family_digest,
        )
        if (
            self.source_cluster_id != expected_cluster
            or self.manifest_cluster_label != expected_cluster
        ):
            raise SoftRouteMechanismProtocolError("source cluster identity drifted")
        identity = AnalysisUnitIdentity(
            unit_id=expected_cluster,
            case_id="semantic_texture_soft_route_mechanism_validation",
            source_cluster_id=self.source_cluster_id,
            prompt_digest=self.prompt_digest,
            generation_seed=self.generation_seed,
            image_lineage_digest=self.image_lineage_digest,
            registered_key_family_digest=self.registered_key_family_digest,
        )
        if identity.validate():
            raise SoftRouteMechanismProtocolError("attack identity drifted")


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismManifest:
    schema_version: int
    protocol_id: str
    role_id: str
    source_roster_path: str
    source_roster_rows_digest: str
    key_family_namespace_digest: str
    skipped_raw_rows: tuple[Mapping[str, object], ...]
    entries: tuple[SoftRouteMechanismManifestEntry, ...]

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())

    def validate(self, *, expected_role: str) -> None:
        if (
            self.schema_version != MANIFEST_SCHEMA_VERSION
            or self.protocol_id != PROTOCOL_ID
            or self.role_id != expected_role
            or self.source_roster_path != SOURCE_ROSTER_PATH
            or self.source_roster_rows_digest != SOURCE_ROSTER_ROWS_DIGEST
            or self.key_family_namespace_digest != KEY_FAMILY_NAMESPACE_DIGEST
            or len(self.entries) != CLUSTER_COUNT
        ):
            raise SoftRouteMechanismProtocolError("manifest authority drifted")
        if self.skipped_raw_rows != (
            {
                "raw_row": 82,
                "reason": "run_b_target_prompt_text_and_digest_overlap",
                "prompt_digest": "9f967626aad3681265acae6ac6a6a2d349237425eb02f3b3328e403006e1b905",
            },
        ):
            raise SoftRouteMechanismProtocolError("skip ledger drifted")
        expected_seed = 202608190200 if expected_role == SELECTION_ROLE else 202608190300
        expected_rows = (
            tuple(range(65, 82)) + tuple(range(83, 98))
            if expected_role == SELECTION_ROLE
            else tuple(range(98, 130))
        )
        for ordinal, (entry, source_row) in enumerate(
            zip(self.entries, expected_rows, strict=True)
        ):
            if type(entry) is not SoftRouteMechanismManifestEntry:
                raise SoftRouteMechanismProtocolError("entry type drifted")
            entry.validate(ordinal=ordinal, role_id=expected_role)
            if entry.source_row != source_row or entry.generation_seed != expected_seed + ordinal:
                raise SoftRouteMechanismProtocolError("literal roster drifted")
        expected_entries_digest = (
            SELECTION_ENTRIES_DIGEST
            if expected_role == SELECTION_ROLE
            else CONFIRMATION_ENTRIES_DIGEST
        )
        if canonical_digest([asdict(entry) for entry in self.entries]) != expected_entries_digest:
            raise SoftRouteMechanismProtocolError("literal entry digest drifted")
        expected_manifest_digest = (
            SELECTION_MANIFEST_DIGEST
            if expected_role == SELECTION_ROLE
            else CONFIRMATION_MANIFEST_DIGEST
        )
        if self.digest() != expected_manifest_digest:
            raise SoftRouteMechanismProtocolError("literal manifest digest drifted")
        axes = tuple(
            tuple(getattr(entry, field) for entry in self.entries)
            for field in (
                "source_row", "prompt_text", "prompt_digest", "generation_seed",
                "source_cluster_id", "image_lineage_identity", "image_lineage_digest",
            )
        )
        if any(len(set(axis)) != CLUSTER_COUNT for axis in axes):
            raise SoftRouteMechanismProtocolError("manifest identity collides")


def load_manifest(path: str | Path, *, expected_role: str) -> SoftRouteMechanismManifest:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if type(raw) is not dict or set(raw) != {
            "schema_version", "protocol_id", "role_id", "source_roster_path",
            "source_roster_rows_digest", "key_family_namespace_digest", "skipped_raw_rows", "entries",
        }:
            raise TypeError
        manifest = SoftRouteMechanismManifest(
            **{
                **raw,
                "skipped_raw_rows": tuple(raw["skipped_raw_rows"]),
                "entries": tuple(SoftRouteMechanismManifestEntry(**entry) for entry in raw["entries"]),
            }
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SoftRouteMechanismProtocolError("manifest is unreadable") from exc
    manifest.validate(expected_role=expected_role)
    try:
        roster = json.loads((Path(path).resolve().parents[2] / SOURCE_ROSTER_PATH).read_text(encoding="utf-8"))
        rows = roster["rows"]
        if roster["rows_digest"] != SOURCE_ROSTER_ROWS_DIGEST:
            raise ValueError
        by_row = {entry["source_row"]: entry for entry in rows}
        for entry in manifest.entries:
            source = by_row[entry.source_row]
            if source["prompt_text"] != entry.prompt_text or source["prompt_digest"] != entry.prompt_digest:
                raise ValueError
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SoftRouteMechanismProtocolError("source roster binding drifted") from exc
    return manifest


def load_soft_route_mechanism_configuration(path: str | Path) -> Mapping[str, object]:
    """Load the frozen soft-route mechanism validation identities without introducing a parallel schema."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            type(raw) is not dict
            or raw["schema_version"] != 1
            or raw["protocol_id"] != PROTOCOL_ID
            or raw["candidate_selection_entries_digest"] != SELECTION_ENTRIES_DIGEST
            or raw["candidate_selection_manifest_digest"] != SELECTION_MANIFEST_DIGEST
            or raw["untouched_confirmation_entries_digest"] != CONFIRMATION_ENTRIES_DIGEST
            or raw["untouched_confirmation_manifest_digest"] != CONFIRMATION_MANIFEST_DIGEST
            or raw["asset_bundle_relative_path"] != ASSET_BUNDLE_RELATIVE_PATH
            or raw["asset_bundle_digest"] != ASSET_BUNDLE_DIGEST
            or raw["asset_bundle_sha256"] != ASSET_BUNDLE_SHA256
            or raw["selection_artifact_relative_path"] != SELECTION_ARTIFACT_RELATIVE_PATH
            or raw["alpha_selection_float64_hex"] != ALPHA_SELECTION_FLOAT64_HEX
            or raw["source_cluster_count"] != CLUSTER_COUNT
            or raw["maximum_record_attempts"] != 1
            or tuple(raw["arms"]) != ARMS
            or raw["attacks"] != [
                {"attack_id": "identity"},
                {"attack_id": "crop", "crop_fraction": 0.75},
            ]
            or (raw["combined_relative_l2_numerator"], raw["combined_relative_l2_denominator"]) != (3, 250)
            or raw["diagnostic_only"] is not True
        ):
            raise ValueError
        return raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SoftRouteMechanismProtocolError("soft-route mechanism validation configuration authority drifted") from exc


def validate_split_disjointness(selection: SoftRouteMechanismManifest, confirmation: SoftRouteMechanismManifest) -> None:
    selection.validate(expected_role=SELECTION_ROLE)
    confirmation.validate(expected_role=CONFIRMATION_ROLE)
    for field in (
        "source_row", "prompt_text", "prompt_digest", "generation_seed",
        "source_cluster_id", "image_lineage_identity", "image_lineage_digest",
    ):
        if set(getattr(entry, field) for entry in selection.entries) & set(
            getattr(entry, field) for entry in confirmation.entries
        ):
            raise SoftRouteMechanismProtocolError("split identity overlap")


def load_selection_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
) -> Mapping[str, object]:
    """Load only an exact, selection-bound provisional authority record."""

    try:
        artifact_blob = Path(path).read_bytes()
        if (
            _DIGEST.fullmatch(expected_sha256) is None
            or sha256(artifact_blob).hexdigest() != expected_sha256
        ):
            raise ValueError
        raw = json.loads(artifact_blob)
        required = {
            "protocol_id", "selection_manifest_digest", "provisional_calibration",
            "candidate_selection_passed", "diagnostic_only", "science_started",
            "scientific_unit_count", "candidate_promoted", "formal_tau_created",
            "formal_fpr_created",
        }
        calibration = raw["provisional_calibration"]
        if (
            type(raw) is not dict or set(raw) != required
            or raw["protocol_id"] != PROTOCOL_ID
            or raw["selection_manifest_digest"] != SELECTION_MANIFEST_DIGEST
            or raw["candidate_selection_passed"] is not True
            or raw["diagnostic_only"] is not True
            or raw["science_started"] is not False
            or raw["scientific_unit_count"] != 0
            or raw["candidate_promoted"] is not False
            or raw["formal_tau_created"] is not False
            or raw["formal_fpr_created"] is not False
            or type(calibration) is not dict
            or set(calibration) != {"selection_manifest_digest", "hf_detector_identity", "lf_detector_identity", "hf_null_identity", "lf_null_identity", "tau_hf_provisional", "tau_lf_provisional", "tau_max_provisional", "hf_records", "lf_records", "retired"}
            or calibration["selection_manifest_digest"] != SELECTION_MANIFEST_DIGEST
            or calibration["retired"] is not False
            or len(calibration["hf_records"]) != CLUSTER_COUNT
            or len(calibration["lf_records"]) != CLUSTER_COUNT
        ):
            raise ValueError
        return raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SoftRouteMechanismProtocolError("selection artifact authority drifted") from exc


def provisional_tau(values: Sequence[float]) -> float:
    """Tie-safe provisional operating point: fourth largest plus one ULP."""
    if len(values) != CLUSTER_COUNT or any(not isfinite(value) for value in values):
        raise SoftRouteMechanismProtocolError("provisional null values are invalid")
    ordered = sorted(float(value) for value in values)
    return nextafter(ordered[-4], float("inf"))


__all__ = [
    "ALPHA_SELECTION_FLOAT64_HEX", "ARMS", "ATTACKS", "CLUSTER_COUNT", "CONFIRMATION_ENTRIES_DIGEST",
    "CONFIRMATION_MANIFEST_DIGEST", "CONFIRMATION_ROLE", "SoftRouteMechanismManifest",
    "SoftRouteMechanismManifestEntry", "KEY_FAMILY_NAMESPACE_DIGEST", "PROTOCOL_ID",
    "SELECTION_ENTRIES_DIGEST", "SELECTION_MANIFEST_DIGEST", "SELECTION_ROLE",
    "SoftRouteMechanismProtocolError", "canonical_digest", "load_soft_route_mechanism_configuration", "load_manifest",
    "load_selection_artifact", "provisional_tau", "validate_split_disjointness",
]
