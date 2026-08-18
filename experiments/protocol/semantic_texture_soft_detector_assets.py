"""Frozen Phase-B primary-null inputs and public detector asset bundle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Mapping

MANIFEST_SCHEMA_VERSION = "ceg_wm_semantic_texture_soft_detector_assets_manifest_v1"
ASSET_BUNDLE_SCHEMA_VERSION = 1
ASSET_NAMESPACE = "semantic_texture_soft_detector_assets_primary_null_v1"
WHITENING_FIT_ROLE = "semantic_texture_soft_lf_whitening_fit"
BRANCH_NULL_ROLE = "semantic_texture_soft_branch_primary_null"
WHITENING_FIT_COUNT = 32
BRANCH_NULL_COUNT = 32
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class SemanticTextureSoftDetectorAssetProtocolError(ValueError):
    """A Phase-B manifest or asset bundle is not exact and reusable."""


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
        raise SemanticTextureSoftDetectorAssetProtocolError(
            f"{role} must be a lowercase SHA-256 digest"
        )
    return value


@dataclass(frozen=True, slots=True)
class SemanticTextureSoftDetectorManifestEntry:
    source_row: int
    prompt_text: str
    prompt_digest: str
    generation_seed: int
    source_cluster_id: str
    image_lineage_identity: str
    image_lineage_digest: str

    def validate(self, *, ordinal: int, role_id: str, namespace: str) -> None:
        if type(self.source_row) is not int or self.source_row < 1:
            raise SemanticTextureSoftDetectorAssetProtocolError("source row is invalid")
        if type(self.prompt_text) is not str or not self.prompt_text:
            raise SemanticTextureSoftDetectorAssetProtocolError("prompt text is invalid")
        if self.prompt_digest != sha256(self.prompt_text.encode("utf-8")).hexdigest():
            raise SemanticTextureSoftDetectorAssetProtocolError("prompt digest drifted")
        if type(self.generation_seed) is not int or self.generation_seed < 0:
            raise SemanticTextureSoftDetectorAssetProtocolError("generation seed is invalid")
        expected_cluster = f"{namespace}:{role_id}:cluster:{ordinal:02d}"
        expected_lineage = f"{namespace}:{role_id}:image:{ordinal:02d}"
        if (
            self.source_cluster_id != expected_cluster
            or self.image_lineage_identity != expected_lineage
        ):
            raise SemanticTextureSoftDetectorAssetProtocolError("manifest identity drifted")
        expected_lineage_digest = canonical_digest(
            {
                "cluster_identity": expected_cluster,
                "generation_seed": self.generation_seed,
                "image_lineage_identity": expected_lineage,
                "image_lineage_namespace": namespace,
            }
        )
        if self.image_lineage_digest != expected_lineage_digest:
            raise SemanticTextureSoftDetectorAssetProtocolError("image lineage drifted")


@dataclass(frozen=True, slots=True)
class SemanticTextureSoftDetectorManifest:
    schema_version: str
    manifest_id: str
    role_id: str
    seed_namespace: str
    source_cluster_namespace: str
    image_lineage_namespace: str
    key_family_namespace: str
    key_family_namespace_digest: str
    source_roster_path: str
    source_roster_rows_digest: str
    entries: tuple[SemanticTextureSoftDetectorManifestEntry, ...]

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())

    def validate(self, *, expected_role: str, count: int) -> None:
        if (
            self.schema_version != MANIFEST_SCHEMA_VERSION
            or self.role_id != expected_role
            or self.manifest_id != f"{expected_role}_v1"
            or self.seed_namespace != ASSET_NAMESPACE
            or self.source_cluster_namespace != ASSET_NAMESPACE
            or self.image_lineage_namespace != ASSET_NAMESPACE
            or self.key_family_namespace != ASSET_NAMESPACE
            or self.key_family_namespace_digest
            != sha256(ASSET_NAMESPACE.encode("utf-8")).hexdigest()
            or self.source_roster_path
            != "configs/experiments/hf_only_reference_prompt_roster.json"
        ):
            raise SemanticTextureSoftDetectorAssetProtocolError("manifest authority drifted")
        _require_digest(self.source_roster_rows_digest, "source roster rows")
        if len(self.entries) != count:
            raise SemanticTextureSoftDetectorAssetProtocolError("manifest count drifted")
        for ordinal, entry in enumerate(self.entries, start=1):
            if type(entry) is not SemanticTextureSoftDetectorManifestEntry:
                raise SemanticTextureSoftDetectorAssetProtocolError("manifest entry type drifted")
            entry.validate(ordinal=ordinal, role_id=expected_role, namespace=ASSET_NAMESPACE)
            expected_source_row = (
                ordinal
                if expected_role == WHITENING_FIT_ROLE
                else WHITENING_FIT_COUNT + ordinal
            )
            if entry.source_row != expected_source_row:
                raise SemanticTextureSoftDetectorAssetProtocolError(
                    "manifest source-row authority drifted"
                )
        axes = (
            tuple(entry.source_row for entry in self.entries),
            tuple(entry.prompt_text for entry in self.entries),
            tuple(entry.prompt_digest for entry in self.entries),
            tuple(entry.generation_seed for entry in self.entries),
            tuple(entry.source_cluster_id for entry in self.entries),
            tuple(entry.image_lineage_identity for entry in self.entries),
            tuple(entry.image_lineage_digest for entry in self.entries),
        )
        if any(len(set(axis)) != count for axis in axes):
            raise SemanticTextureSoftDetectorAssetProtocolError("manifest axis collides")


def load_manifest(path: str | Path, *, expected_role: str, count: int) -> SemanticTextureSoftDetectorManifest:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = SemanticTextureSoftDetectorManifest(
            **{**raw, "entries": tuple(SemanticTextureSoftDetectorManifestEntry(**entry) for entry in raw["entries"])}
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SemanticTextureSoftDetectorAssetProtocolError("manifest is unreadable") from exc
    manifest.validate(expected_role=expected_role, count=count)
    try:
        roster_path = Path(path).resolve().parents[2] / manifest.source_roster_path
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        rows = roster["rows"]
        source_rows_digest = roster["rows_digest"]
    except (
        IndexError,
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise SemanticTextureSoftDetectorAssetProtocolError(
            "source prompt roster is unreadable"
        ) from exc
    if (
        type(rows) is not list
        or manifest.source_roster_rows_digest != source_rows_digest
        or any(
            entry.source_row > len(rows)
            or rows[entry.source_row - 1].get("source_row") != entry.source_row
            or rows[entry.source_row - 1].get("prompt_text") != entry.prompt_text
            or rows[entry.source_row - 1].get("prompt_digest") != entry.prompt_digest
            for entry in manifest.entries
        )
    ):
        raise SemanticTextureSoftDetectorAssetProtocolError(
            "source prompt roster authority drifted"
        )
    return manifest


def validate_partition_disjointness(
    whitening_fit: SemanticTextureSoftDetectorManifest,
    branch_null: SemanticTextureSoftDetectorManifest,
    *,
    target_prompt: str,
    target_seed: int,
) -> None:
    whitening_fit.validate(expected_role=WHITENING_FIT_ROLE, count=WHITENING_FIT_COUNT)
    branch_null.validate(expected_role=BRANCH_NULL_ROLE, count=BRANCH_NULL_COUNT)
    for field in (
        "source_row",
        "prompt_text",
        "prompt_digest",
        "generation_seed",
        "source_cluster_id",
        "image_lineage_identity",
        "image_lineage_digest",
    ):
        left = {getattr(entry, field) for entry in whitening_fit.entries}
        right = {getattr(entry, field) for entry in branch_null.entries}
        if left & right:
            raise SemanticTextureSoftDetectorAssetProtocolError("primary-null partitions overlap")
    if (
        target_prompt in {entry.prompt_text for entry in (*whitening_fit.entries, *branch_null.entries)}
        or target_seed in {entry.generation_seed for entry in (*whitening_fit.entries, *branch_null.entries)}
    ):
        raise SemanticTextureSoftDetectorAssetProtocolError("target overlaps primary-null partitions")


@dataclass(frozen=True, slots=True)
class SemanticTextureBranchNullRecordPayload:
    """A pure canonical primary-null record; no detector object ownership."""

    score_float64_hex: str
    source_cluster_id: str
    sample_id: str

    def validate(self) -> None:
        if (
            type(self.score_float64_hex) is not str
            or type(self.source_cluster_id) is not str
            or not self.source_cluster_id
            or type(self.sample_id) is not str
            or not self.sample_id
        ):
            raise SemanticTextureSoftDetectorAssetProtocolError("branch record is invalid")
        try:
            value = float.fromhex(self.score_float64_hex)
        except ValueError as exc:
            raise SemanticTextureSoftDetectorAssetProtocolError("branch score is invalid") from exc
        if not isfinite(value) or value.hex() != self.score_float64_hex:
            raise SemanticTextureSoftDetectorAssetProtocolError("branch score is non-canonical")

    def as_dict(self) -> dict[str, str]:
        return {
            "score_float64_hex": self.score_float64_hex,
            "source_cluster_id": self.source_cluster_id,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True, slots=True)
class SemanticTextureBranchNullPayload:
    """Pure transport representation for one diagnostic empirical CDF."""

    branch: str
    detector_identity: str
    partition_identity: str
    records: tuple[SemanticTextureBranchNullRecordPayload, ...]

    @classmethod
    def from_mapping(cls, payload: object) -> "SemanticTextureBranchNullPayload":
        if type(payload) is not dict or set(payload) != {
            "branch", "detector_identity", "partition_identity", "records"
        } or type(payload["records"]) is not list:
            raise SemanticTextureSoftDetectorAssetProtocolError("branch calibration payload drifted")
        try:
            value = cls(
                branch=payload["branch"],
                detector_identity=payload["detector_identity"],
                partition_identity=payload["partition_identity"],
                records=tuple(
                    SemanticTextureBranchNullRecordPayload(**item)
                    for item in payload["records"]
                    if type(item) is dict
                    and set(item) == {"score_float64_hex", "source_cluster_id", "sample_id"}
                ),
            )
        except TypeError as exc:
            raise SemanticTextureSoftDetectorAssetProtocolError("branch calibration payload is invalid") from exc
        if len(value.records) != len(payload["records"]):
            raise SemanticTextureSoftDetectorAssetProtocolError("branch record fields drifted")
        value.validate()
        return value

    def validate(self) -> None:
        if self.branch not in {"hf", "lf"}:
            raise SemanticTextureSoftDetectorAssetProtocolError("branch calibration role is invalid")
        _require_digest(self.detector_identity, "branch detector identity")
        _require_digest(self.partition_identity, "branch partition identity")
        if len(self.records) != BRANCH_NULL_COUNT:
            raise SemanticTextureSoftDetectorAssetProtocolError("branch record count drifted")
        if any(type(record) is not SemanticTextureBranchNullRecordPayload for record in self.records):
            raise SemanticTextureSoftDetectorAssetProtocolError("branch record type drifted")
        for record in self.records:
            record.validate()
        identities = tuple((record.source_cluster_id, record.sample_id) for record in self.records)
        if len(set(identities)) != len(identities):
            raise SemanticTextureSoftDetectorAssetProtocolError("branch record identity drifted")
        ordered = tuple(
            sorted(
                self.records,
                key=lambda record: (
                    float.fromhex(record.score_float64_hex),
                    record.source_cluster_id,
                    record.sample_id,
                ),
            )
        )
        if self.records != ordered:
            raise SemanticTextureSoftDetectorAssetProtocolError("branch record order drifted")

    def as_dict(self) -> dict[str, object]:
        return {
            "branch": self.branch,
            "detector_identity": self.detector_identity,
            "partition_identity": self.partition_identity,
            "records": [record.as_dict() for record in self.records],
        }


_WHITENING_PAYLOAD_FIELDS = {
    "artifact_role",
    "band_identity",
    "candidate_id",
    "detrend_identity",
    "fit_manifest_sha256",
    "fit_source_cluster_count",
    "latent_shape",
    "lf_carrier_config_digest",
    "observation_protocol",
    "regularization_ratio",
    "route_candidate_id",
    "transform_identity",
    "weights_binary32_be_hex",
}


def _validate_whitening_payload(payload: object, *, manifest_digest: str, carrier_digest: str) -> None:
    if type(payload) is not dict or set(payload) != _WHITENING_PAYLOAD_FIELDS:
        raise SemanticTextureSoftDetectorAssetProtocolError("whitening asset payload drifted")
    if (
        payload["artifact_role"] != "lf_semantic_texture_soft_clean_null_whitening_operator"
        or payload["candidate_id"] != "lf_semantic_texture_soft_whitened_matched_score"
        or payload["route_candidate_id"] != "routing_semantic_texture_soft"
        or payload["fit_source_cluster_count"] != WHITENING_FIT_COUNT
        or payload["fit_manifest_sha256"] != manifest_digest
        or payload["lf_carrier_config_digest"] != carrier_digest
    ):
        raise SemanticTextureSoftDetectorAssetProtocolError("whitening bundle binding drifted")
    weights = payload["weights_binary32_be_hex"]
    if type(weights) is not list or len(weights) != 96 or any(
        type(word) is not str or re.fullmatch(r"[0-9a-f]{8}", word) is None
        for word in weights
    ):
        raise SemanticTextureSoftDetectorAssetProtocolError("whitening asset weights drifted")


@dataclass(frozen=True, slots=True)
class SemanticTextureSoftDetectorAssetBundle:
    schema_version: int
    bundle_digest: str
    whitening_manifest_digest: str
    branch_null_manifest_digest: str
    lf_carrier_config_digest: str
    whitening_asset_payload: dict[str, object]
    whitening_asset_digest: str
    hf_null_payload: SemanticTextureBranchNullPayload
    lf_null_payload: SemanticTextureBranchNullPayload

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "whitening_manifest_digest": self.whitening_manifest_digest,
            "branch_null_manifest_digest": self.branch_null_manifest_digest,
            "lf_carrier_config_digest": self.lf_carrier_config_digest,
            "whitening_asset_payload": self.whitening_asset_payload,
            "whitening_asset_digest": self.whitening_asset_digest,
            "hf_null_payload": self.hf_null_payload.as_dict(),
            "lf_null_payload": self.lf_null_payload.as_dict(),
        }

    @classmethod
    def from_mapping(cls, payload: object) -> "SemanticTextureSoftDetectorAssetBundle":
        """Load only the exact JSON-compatible Phase-B bundle envelope."""

        expected = {
            "schema_version",
            "bundle_digest",
            "whitening_manifest_digest",
            "branch_null_manifest_digest",
            "lf_carrier_config_digest",
            "whitening_asset_payload",
            "whitening_asset_digest",
            "hf_null_payload",
            "lf_null_payload",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise SemanticTextureSoftDetectorAssetProtocolError("asset bundle fields drifted")
        try:
            bundle = cls(
                schema_version=payload["schema_version"],
                bundle_digest=payload["bundle_digest"],
                whitening_manifest_digest=payload["whitening_manifest_digest"],
                branch_null_manifest_digest=payload["branch_null_manifest_digest"],
                lf_carrier_config_digest=payload["lf_carrier_config_digest"],
                whitening_asset_payload=payload["whitening_asset_payload"],
                whitening_asset_digest=payload["whitening_asset_digest"],
                hf_null_payload=SemanticTextureBranchNullPayload.from_mapping(
                    payload["hf_null_payload"]
                ),
                lf_null_payload=SemanticTextureBranchNullPayload.from_mapping(
                    payload["lf_null_payload"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise SemanticTextureSoftDetectorAssetProtocolError(
                "asset bundle mapping is invalid"
            ) from exc
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if self.schema_version != ASSET_BUNDLE_SCHEMA_VERSION or self.bundle_digest != canonical_digest(self.canonical_payload()):
            raise SemanticTextureSoftDetectorAssetProtocolError("asset bundle digest drifted")
        for item, role in ((self.whitening_manifest_digest, "whitening manifest"), (self.branch_null_manifest_digest, "branch null manifest"), (self.lf_carrier_config_digest, "LF carrier configuration"), (self.whitening_asset_digest, "whitening asset")):
            _require_digest(item, role)
        _validate_whitening_payload(
            self.whitening_asset_payload,
            manifest_digest=self.whitening_manifest_digest,
            carrier_digest=self.lf_carrier_config_digest,
        )
        if (
            type(self.hf_null_payload) is not SemanticTextureBranchNullPayload
            or type(self.lf_null_payload) is not SemanticTextureBranchNullPayload
        ):
            raise SemanticTextureSoftDetectorAssetProtocolError("branch bundle payload type drifted")
        self.hf_null_payload.validate()
        self.lf_null_payload.validate()
        if (
            self.hf_null_payload.branch != "hf"
            or self.lf_null_payload.branch != "lf"
            or self.hf_null_payload.partition_identity != self.branch_null_manifest_digest
            or self.lf_null_payload.partition_identity != self.branch_null_manifest_digest
        ):
            raise SemanticTextureSoftDetectorAssetProtocolError("branch bundle identity drifted")


def create_asset_bundle(*, whitening_manifest_digest: str, branch_null_manifest_digest: str, lf_carrier_config_digest: str, whitening_asset_payload: dict[str, object], whitening_asset_digest: str, hf_null_payload: SemanticTextureBranchNullPayload, lf_null_payload: SemanticTextureBranchNullPayload) -> SemanticTextureSoftDetectorAssetBundle:
    payload = {
        "schema_version": ASSET_BUNDLE_SCHEMA_VERSION,
        "whitening_manifest_digest": whitening_manifest_digest,
        "branch_null_manifest_digest": branch_null_manifest_digest,
        "lf_carrier_config_digest": lf_carrier_config_digest,
        "whitening_asset_payload": whitening_asset_payload,
        "whitening_asset_digest": whitening_asset_digest,
        "hf_null_payload": hf_null_payload.as_dict(),
        "lf_null_payload": lf_null_payload.as_dict(),
    }
    bundle = SemanticTextureSoftDetectorAssetBundle(
        schema_version=ASSET_BUNDLE_SCHEMA_VERSION,
        bundle_digest=canonical_digest(payload),
        whitening_manifest_digest=whitening_manifest_digest,
        branch_null_manifest_digest=branch_null_manifest_digest,
        lf_carrier_config_digest=lf_carrier_config_digest,
        whitening_asset_payload=whitening_asset_payload,
        whitening_asset_digest=whitening_asset_digest,
        hf_null_payload=hf_null_payload,
        lf_null_payload=lf_null_payload,
    )
    bundle.validate()
    return bundle


__all__ = [
    "ASSET_BUNDLE_SCHEMA_VERSION", "ASSET_NAMESPACE", "BRANCH_NULL_COUNT", "BRANCH_NULL_ROLE", "SemanticTextureBranchNullPayload", "SemanticTextureBranchNullRecordPayload", "SemanticTextureSoftDetectorAssetBundle", "SemanticTextureSoftDetectorAssetProtocolError", "SemanticTextureSoftDetectorManifest", "SemanticTextureSoftDetectorManifestEntry", "WHITENING_FIT_COUNT", "WHITENING_FIT_ROLE", "canonical_digest", "create_asset_bundle", "load_manifest", "validate_partition_disjointness",
]
