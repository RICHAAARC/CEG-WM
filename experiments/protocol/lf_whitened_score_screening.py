"""Frozen development protocol for LF clean-null fit and score screening."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

from experiments.protocol.development_exploration import DevelopmentStudyUnit
from experiments.protocol.internal_splits import AnalysisUnitIdentity, derive_source_cluster_id


PROTOCOL_ID = "ceg_wm_lf_whitened_score_screening"
PROTOCOL_VERSION = "1.0.0"
RUN_ID = "ceg_wm_lf_whitening_asset_fit_and_score_screening"
NULL_FIT_CLUSTER_COUNT = 32
SCREENING_CLUSTER_COUNT = 8
SCIENTIFIC_UNIT_COUNT = 40
OPERATIONAL_UNIT_COUNT = 1
MAXIMUM_TOTAL_UNITS = 41
MAXIMUM_ATTEMPTS_PER_UNIT = 2
MAXIMUM_DURATION_SECONDS = 2700
MARGIN_FLOOR = float.fromhex("0x1.0000000000000p-10")
CLAIM_BOUNDARY = "development_lf_whitened_score_screening_no_threshold_no_promotion"
STOP_RULE = (
    "complete_all_eight_screening_clusters_then_require_seven_registered_null_"
    "and_seven_registered_max_wrong_margins_above_the_binary32_numerical_"
    "tolerance_and_preregistered_minimum_meaningful_normalized_margin_floor_and_"
    "six_positive_raw_to_whitened_improvements_with_positive_mean_and_no_"
    "identity_budget_integrity_or_nonfinite_failure"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class LfWhitenedScreeningProtocolError(ValueError):
    """The checked-in LF whitening protocol is inconsistent."""


def canonical_digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LfWhiteningManifestEntry:
    cluster_ordinal: int
    cluster_identity: str
    prompt: str
    prompt_digest: str
    generation_seed: int
    image_lineage_identity: str
    image_lineage_digest: str
    split: str
    role_id: str

    def validate(self, *, expected_role: str, image_lineage_namespace: str, count: int) -> None:
        if type(self.cluster_ordinal) is not int or not 0 <= self.cluster_ordinal < count:
            raise LfWhitenedScreeningProtocolError("cluster ordinal is invalid")
        if any(type(value) is not str or not value for value in (self.cluster_identity, self.prompt, self.image_lineage_identity)):
            raise LfWhitenedScreeningProtocolError("manifest identity is invalid")
        if self.split != "development" or self.role_id != expected_role:
            raise LfWhitenedScreeningProtocolError("manifest role drifted")
        if self.prompt_digest != sha256(self.prompt.encode("utf-8")).hexdigest():
            raise LfWhitenedScreeningProtocolError("prompt digest drifted")
        if type(self.generation_seed) is not int or self.generation_seed < 0:
            raise LfWhitenedScreeningProtocolError("generation seed is invalid")
        if self.image_lineage_digest != canonical_digest({
            "cluster_identity": self.cluster_identity,
            "generation_seed": self.generation_seed,
            "image_lineage_identity": self.image_lineage_identity,
            "image_lineage_namespace": image_lineage_namespace,
        }):
            raise LfWhitenedScreeningProtocolError("image lineage digest drifted")


@dataclass(frozen=True, slots=True)
class LfWhiteningManifest:
    schema_version: str
    manifest_id: str
    role_id: str
    seed_namespace: str
    source_cluster_namespace: str
    image_lineage_namespace: str
    key_family_namespace: str
    entries: tuple[LfWhiteningManifestEntry, ...]

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self, *, expected_role: str, count: int) -> None:
        expected_schema = f"ceg_wm_{expected_role}_manifest_v1"
        if self.schema_version != expected_schema or self.role_id != expected_role:
            raise LfWhitenedScreeningProtocolError("manifest authority drifted")
        if any(type(value) is not str or not value for value in (self.manifest_id, self.seed_namespace, self.source_cluster_namespace, self.image_lineage_namespace, self.key_family_namespace)):
            raise LfWhitenedScreeningProtocolError("manifest namespace is invalid")
        if len(self.entries) != count or tuple(item.cluster_ordinal for item in self.entries) != tuple(range(count)):
            raise LfWhitenedScreeningProtocolError("manifest coverage drifted")
        for item in self.entries:
            item.validate(expected_role=expected_role, image_lineage_namespace=self.image_lineage_namespace, count=count)
        for values in (
            tuple(item.cluster_identity for item in self.entries),
            tuple(item.prompt_digest for item in self.entries),
            tuple(item.generation_seed for item in self.entries),
            tuple(item.image_lineage_digest for item in self.entries),
        ):
            if len(set(values)) != count:
                raise LfWhitenedScreeningProtocolError("manifest axis collides")


@dataclass(frozen=True, slots=True)
class LfWhitenedScoreScreeningProtocol:
    schema_version: str
    protocol_id: str
    protocol_version: str
    run_id: str
    split: str
    candidate_identity: str
    null_fit_manifest_path: str
    null_fit_manifest_file_sha256: str
    screening_manifest_path: str
    screening_manifest_file_sha256: str
    operational_smoke_prompt: str
    operational_smoke_prompt_digest: str
    operational_smoke_generation_seed: int
    operational_smoke_image_lineage_digest: str
    null_fit_cluster_count: int
    screening_cluster_count: int
    operational_unit_count: int
    maximum_total_units: int
    maximum_attempts_per_unit: int
    maximum_duration_seconds_per_unit: int
    margin_floor: float
    stop_rule: str
    claim_boundary: str
    unit_roster_digest: str

    @property
    def unit_roster(self) -> tuple[DevelopmentStudyUnit, ...]:
        operational = (
            DevelopmentStudyUnit(
                unit_index=0,
                phase="development_environment_preflight",
                responsibility_id="lf_clean_public_vae_runtime_preflight",
                source_cluster_ordinal=0,
                content_branch_id="clean_control",
                geometry_case_id="not_applicable",
                maximum_record_attempts=self.maximum_attempts_per_unit,
                maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
            ),
        )
        fit = tuple(DevelopmentStudyUnit(
            unit_index=self.operational_unit_count + index,
            phase="development_scientific_breadth",
            responsibility_id="lf_whitening_null_fit",
            source_cluster_ordinal=index,
            content_branch_id="clean_control",
            geometry_case_id="not_applicable",
            maximum_record_attempts=self.maximum_attempts_per_unit,
            maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
        ) for index in range(self.null_fit_cluster_count))
        screening = tuple(DevelopmentStudyUnit(
            unit_index=(
                self.operational_unit_count
                + self.null_fit_cluster_count
                + index
            ),
            phase="development_scientific_breadth",
            responsibility_id="lf_whitened_score_screening",
            source_cluster_ordinal=index,
            content_branch_id="lf_only",
            geometry_case_id="not_applicable",
            maximum_record_attempts=self.maximum_attempts_per_unit,
            maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
        ) for index in range(self.screening_cluster_count))
        return operational + fit + screening

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if (
            self.schema_version != "ceg_wm_lf_whitened_score_screening_protocol_v1"
            or self.protocol_id != PROTOCOL_ID
            or self.protocol_version != PROTOCOL_VERSION
            or self.run_id != RUN_ID
            or self.split != "development"
            or self.candidate_identity != "lf_null_whitened_matched_score"
        ):
            raise LfWhitenedScreeningProtocolError("protocol identity drifted")
        if _DIGEST.fullmatch(self.null_fit_manifest_file_sha256) is None or _DIGEST.fullmatch(self.screening_manifest_file_sha256) is None:
            raise LfWhitenedScreeningProtocolError("manifest file digest is invalid")
        if (
            type(self.operational_smoke_prompt) is not str
            or not self.operational_smoke_prompt
            or self.operational_smoke_prompt_digest
            != sha256(self.operational_smoke_prompt.encode("utf-8")).hexdigest()
            or type(self.operational_smoke_generation_seed) is not int
            or self.operational_smoke_generation_seed < 0
            or _DIGEST.fullmatch(self.operational_smoke_image_lineage_digest)
            is None
        ):
            raise LfWhitenedScreeningProtocolError(
                "operational smoke identity is invalid"
            )
        if (
            self.null_fit_cluster_count != NULL_FIT_CLUSTER_COUNT
            or self.screening_cluster_count != SCREENING_CLUSTER_COUNT
            or self.operational_unit_count != OPERATIONAL_UNIT_COUNT
            or self.maximum_total_units != MAXIMUM_TOTAL_UNITS
            or len(self.unit_roster) != MAXIMUM_TOTAL_UNITS
            or self.maximum_attempts_per_unit != MAXIMUM_ATTEMPTS_PER_UNIT
            or self.maximum_duration_seconds_per_unit != MAXIMUM_DURATION_SECONDS
            or self.margin_floor != MARGIN_FLOOR
            or self.stop_rule != STOP_RULE
            or self.claim_boundary != CLAIM_BOUNDARY
        ):
            raise LfWhitenedScreeningProtocolError("protocol budget or rule drifted")
        if self.unit_roster_digest != canonical_digest(tuple(asdict(unit) for unit in self.unit_roster)):
            raise LfWhitenedScreeningProtocolError("unit roster digest drifted")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LfWhitenedScreeningProtocolError("checked-in JSON is unreadable") from exc
    if type(value) is not dict:
        raise LfWhitenedScreeningProtocolError("checked-in JSON must be a mapping")
    return value


def load_lf_whitening_manifest(path: str | Path, *, expected_role: str, count: int) -> LfWhiteningManifest:
    raw = _load_json(Path(path))
    try:
        manifest = LfWhiteningManifest(
            schema_version=raw["schema_version"], manifest_id=raw["manifest_id"], role_id=raw["role_id"],
            seed_namespace=raw["seed_namespace"], source_cluster_namespace=raw["source_cluster_namespace"],
            image_lineage_namespace=raw["image_lineage_namespace"], key_family_namespace=raw["key_family_namespace"],
            entries=tuple(LfWhiteningManifestEntry(**item) for item in raw["entries"]),
        )
    except (KeyError, TypeError) as exc:
        raise LfWhitenedScreeningProtocolError("manifest schema is invalid") from exc
    manifest.validate(expected_role=expected_role, count=count)
    return manifest


def load_lf_whitened_score_screening_protocol(path: str | Path, *, repository_root: str | Path) -> tuple[LfWhitenedScoreScreeningProtocol, LfWhiteningManifest, LfWhiteningManifest]:
    raw = _load_json(Path(path))
    try:
        protocol = LfWhitenedScoreScreeningProtocol(**raw)
    except TypeError as exc:
        raise LfWhitenedScreeningProtocolError("protocol schema is invalid") from exc
    protocol.validate()
    root = Path(repository_root)
    fit_path = root / protocol.null_fit_manifest_path
    screening_path = root / protocol.screening_manifest_path
    if sha256(fit_path.read_bytes()).hexdigest() != protocol.null_fit_manifest_file_sha256 or sha256(screening_path.read_bytes()).hexdigest() != protocol.screening_manifest_file_sha256:
        raise LfWhitenedScreeningProtocolError("manifest file digest drifted")
    fit = load_lf_whitening_manifest(fit_path, expected_role="lf_whitening_null_fit", count=NULL_FIT_CLUSTER_COUNT)
    screening = load_lf_whitening_manifest(screening_path, expected_role="lf_whitened_score_screening", count=SCREENING_CLUSTER_COUNT)
    axes = (
        ({item.prompt_digest for item in fit.entries}, {item.prompt_digest for item in screening.entries}),
        ({item.generation_seed for item in fit.entries}, {item.generation_seed for item in screening.entries}),
        ({item.image_lineage_digest for item in fit.entries}, {item.image_lineage_digest for item in screening.entries}),
    )
    if any(left & right for left, right in axes) or fit.key_family_namespace == screening.key_family_namespace:
        raise LfWhitenedScreeningProtocolError("fit and screening manifests overlap")
    if (
        protocol.operational_smoke_prompt_digest
        in {item.prompt_digest for item in (*fit.entries, *screening.entries)}
        or protocol.operational_smoke_generation_seed
        in {item.generation_seed for item in (*fit.entries, *screening.entries)}
        or protocol.operational_smoke_image_lineage_digest
        in {
            item.image_lineage_digest
            for item in (*fit.entries, *screening.entries)
        }
    ):
        raise LfWhitenedScreeningProtocolError(
            "operational smoke overlaps scientific manifests"
        )
    return protocol, fit, screening


def derive_lf_whitening_analysis_identity(entry: LfWhiteningManifestEntry, manifest: LfWhiteningManifest, *, key_family_digest: str) -> AnalysisUnitIdentity:
    if _DIGEST.fullmatch(key_family_digest) is None:
        raise LfWhitenedScreeningProtocolError("key family digest is invalid")
    source_cluster = derive_source_cluster_id(
        prompt_digest=entry.prompt_digest,
        generation_seed=entry.generation_seed,
        image_lineage_digest=entry.image_lineage_digest,
        registered_key_family_digest=key_family_digest,
    )
    return AnalysisUnitIdentity(
        unit_id=f"{manifest.role_id}_cluster_{entry.cluster_ordinal:02d}",
        case_id=("clean_public_vae_null_fit" if manifest.role_id == "lf_whitening_null_fit" else "paired_clean_lf_raw_whitened_screening"),
        source_cluster_id=source_cluster,
        prompt_digest=entry.prompt_digest,
        generation_seed=entry.generation_seed,
        image_lineage_digest=entry.image_lineage_digest,
        registered_key_family_digest=key_family_digest,
    )


__all__ = [
    "CLAIM_BOUNDARY", "MARGIN_FLOOR", "NULL_FIT_CLUSTER_COUNT", "RUN_ID", "SCREENING_CLUSTER_COUNT", "STOP_RULE",
    "LfWhitenedScoreScreeningProtocol", "LfWhitenedScreeningProtocolError", "LfWhiteningManifest", "LfWhiteningManifestEntry",
    "canonical_digest", "derive_lf_whitening_analysis_identity", "load_lf_whitened_score_screening_protocol", "load_lf_whitening_manifest",
]
