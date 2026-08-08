"""Frozen development protocol for HF-only detector direction validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping

from experiments.protocol.development_exploration import DevelopmentStudyUnit


PROTOCOL_ID = "ceg_wm_hf_only_detector_directional_validation"
PROTOCOL_VERSION = "1.0.0"
OPERATIONAL_UNIT_COUNT = 2
SCIENTIFIC_CLUSTER_COUNT = 32
INITIAL_GPU_GATE_SCIENTIFIC_UNIT_COUNT = 8
MAXIMUM_TOTAL_UNITS = 34
MAXIMUM_ATTEMPTS_PER_UNIT = 2
MAXIMUM_DURATION_SECONDS = 2700
WRONG_KEY_ROSTER_SIZE = 4
PRACTICAL_MARGIN_FLOOR = 0.001
CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250
MINIMUM_DIRECTIONAL_SUCCESS_COUNT = 28
CONFIDENCE_LEVEL = 0.95
MARGIN_QUANTILE_PROBABILITY = 0.25
MARGIN_QUANTILE_METHOD = "nearest_rank"
CLAIM_BOUNDARY = (
    "development_hf_detector_direction_only_no_threshold_no_fpr_no_promotion_no_claim"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class HfDetectorDirectionalProtocolError(ValueError):
    """The checked-in HF detector direction protocol is inconsistent."""


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
class PriorDevelopmentManifestBinding:
    path: str
    file_sha256: str

    def validate(self) -> None:
        if type(self.path) is not str or not self.path.startswith(
            "configs/experiments/"
        ):
            raise HfDetectorDirectionalProtocolError(
                "prior development manifest path is invalid"
            )
        if _DIGEST.fullmatch(self.file_sha256) is None:
            raise HfDetectorDirectionalProtocolError(
                "prior development manifest digest is invalid"
            )


@dataclass(frozen=True, slots=True)
class HfDetectorDirectionalManifestEntry:
    entry_role: str
    cluster_ordinal: int
    cluster_identity: str
    prompt: str
    generation_seed: int

    @property
    def prompt_digest(self) -> str:
        return sha256(self.prompt.encode("utf-8")).hexdigest()

    @property
    def image_lineage_identity(self) -> str:
        return "paired_clean_hf_final_rgb_lineage"

    @property
    def image_lineage_digest(self) -> str:
        return canonical_digest(
            {
                "cluster_identity": self.cluster_identity,
                "entry_role": self.entry_role,
                "generation_seed": self.generation_seed,
                "image_lineage_identity": self.image_lineage_identity,
            }
        )

    def validate(self) -> None:
        limit = (
            OPERATIONAL_UNIT_COUNT
            if self.entry_role == "operational_smoke"
            else SCIENTIFIC_CLUSTER_COUNT
            if self.entry_role == "scientific_directional_validation"
            else -1
        )
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < limit
            or type(self.cluster_identity) is not str
            or not self.cluster_identity
            or type(self.prompt) is not str
            or not self.prompt
            or type(self.generation_seed) is not int
            or self.generation_seed < 0
        ):
            raise HfDetectorDirectionalProtocolError(
                "directional manifest entry is invalid"
            )


@dataclass(frozen=True, slots=True)
class HfDetectorDirectionalManifest:
    schema_version: str
    manifest_id: str
    split: str
    seed_namespace: str
    entries: tuple[HfDetectorDirectionalManifestEntry, ...]

    @property
    def operational_entries(self) -> tuple[HfDetectorDirectionalManifestEntry, ...]:
        return tuple(item for item in self.entries if item.entry_role == "operational_smoke")

    @property
    def scientific_entries(self) -> tuple[HfDetectorDirectionalManifestEntry, ...]:
        return tuple(
            item
            for item in self.entries
            if item.entry_role == "scientific_directional_validation"
        )

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if (
            self.schema_version != "ceg_wm_hf_detector_directional_manifest_v1"
            or self.manifest_id
            != "hf_only_detector_directional_validation_manifest"
            or self.split != "development"
            or self.seed_namespace
            != "hf_only_detector_directional_validation_20260808"
        ):
            raise HfDetectorDirectionalProtocolError(
                "directional manifest identity drifted"
            )
        if len(self.entries) != MAXIMUM_TOTAL_UNITS:
            raise HfDetectorDirectionalProtocolError(
                "directional manifest entry count drifted"
            )
        for entry in self.entries:
            entry.validate()
        if tuple(item.cluster_ordinal for item in self.operational_entries) != tuple(
            range(OPERATIONAL_UNIT_COUNT)
        ) or tuple(item.cluster_ordinal for item in self.scientific_entries) != tuple(
            range(SCIENTIFIC_CLUSTER_COUNT)
        ):
            raise HfDetectorDirectionalProtocolError(
                "directional manifest role order drifted"
            )
        if len({item.cluster_identity for item in self.entries}) != len(self.entries):
            raise HfDetectorDirectionalProtocolError(
                "directional manifest cluster identities collide"
            )
        if len({item.generation_seed for item in self.entries}) != len(self.entries):
            raise HfDetectorDirectionalProtocolError(
                "directional manifest generation seeds collide"
            )
        if len({item.prompt_digest for item in self.entries}) != len(self.entries):
            raise HfDetectorDirectionalProtocolError(
                "directional manifest prompts collide"
            )


@dataclass(frozen=True, slots=True)
class HfOnlyDetectorDirectionalProtocol:
    schema_version: str
    protocol_id: str
    protocol_version: str
    split: str
    role_id: str
    candidate_identity: str
    detector_operation_identity: str
    manifest_path: str
    manifest_file_sha256: str
    prior_development_manifests: tuple[PriorDevelopmentManifestBinding, ...]
    source_cluster_deny_list_digest: str
    operational_unit_count: int
    scientific_cluster_count: int
    initial_gpu_gate_scientific_unit_count: int
    maximum_total_units: int
    maximum_attempts_per_unit: int
    maximum_duration_seconds_per_unit: int
    wrong_key_roster_size: int
    practical_margin_floor: float
    content_relative_l2_numerator: int
    content_relative_l2_denominator: int
    minimum_registered_minus_null_success_count: int
    minimum_registered_minus_wrong_success_count: int
    confidence_level: float
    confidence_lower_bound_requirement: str
    margin_quantile_probability: float
    margin_quantile_method: str
    claim_boundary: str
    unit_roster_digest: str

    @property
    def unit_roster(self) -> tuple[DevelopmentStudyUnit, ...]:
        operational = tuple(
            DevelopmentStudyUnit(
                unit_index=index,
                phase="development_environment_preflight",
                responsibility_id="development_environment_preflight",
                source_cluster_ordinal=index,
                content_branch_id="hf_only",
                geometry_case_id="not_applicable",
                maximum_record_attempts=self.maximum_attempts_per_unit,
                maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
            )
            for index in range(self.operational_unit_count)
        )
        scientific = tuple(
            DevelopmentStudyUnit(
                unit_index=self.operational_unit_count + ordinal,
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
        return operational + scientific

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if (
            self.schema_version
            != "ceg_wm_hf_only_detector_directional_protocol_v1"
            or self.protocol_id != PROTOCOL_ID
            or self.protocol_version != PROTOCOL_VERSION
            or self.split != "development"
            or self.role_id != "hf_only_detector_directional_validation"
            or self.candidate_identity != "hf_sparse_tail"
            or self.detector_operation_identity != "main.hf_detector"
        ):
            raise HfDetectorDirectionalProtocolError(
                "directional protocol identity drifted"
            )
        if _DIGEST.fullmatch(self.manifest_file_sha256) is None or _DIGEST.fullmatch(
            self.source_cluster_deny_list_digest
        ) is None:
            raise HfDetectorDirectionalProtocolError(
                "directional protocol digest is invalid"
            )
        if not self.prior_development_manifests:
            raise HfDetectorDirectionalProtocolError(
                "directional deny-list bindings are missing"
            )
        for binding in self.prior_development_manifests:
            binding.validate()
        if (
            self.operational_unit_count != OPERATIONAL_UNIT_COUNT
            or self.scientific_cluster_count != SCIENTIFIC_CLUSTER_COUNT
            or self.initial_gpu_gate_scientific_unit_count
            != INITIAL_GPU_GATE_SCIENTIFIC_UNIT_COUNT
            or self.maximum_total_units != MAXIMUM_TOTAL_UNITS
            or len(self.unit_roster) != MAXIMUM_TOTAL_UNITS
            or self.maximum_attempts_per_unit != MAXIMUM_ATTEMPTS_PER_UNIT
            or self.maximum_duration_seconds_per_unit != MAXIMUM_DURATION_SECONDS
            or self.wrong_key_roster_size != WRONG_KEY_ROSTER_SIZE
            or self.practical_margin_floor != PRACTICAL_MARGIN_FLOOR
            or self.content_relative_l2_numerator
            != CONTENT_RELATIVE_L2_NUMERATOR
            or self.content_relative_l2_denominator
            != CONTENT_RELATIVE_L2_DENOMINATOR
            or self.minimum_registered_minus_null_success_count
            != MINIMUM_DIRECTIONAL_SUCCESS_COUNT
            or self.minimum_registered_minus_wrong_success_count
            != MINIMUM_DIRECTIONAL_SUCCESS_COUNT
            or self.confidence_level != CONFIDENCE_LEVEL
            or self.confidence_lower_bound_requirement
            != "strictly_greater_than_one_half"
            or self.margin_quantile_probability != MARGIN_QUANTILE_PROBABILITY
            or self.margin_quantile_method != MARGIN_QUANTILE_METHOD
            or self.claim_boundary != CLAIM_BOUNDARY
        ):
            raise HfDetectorDirectionalProtocolError(
                "directional protocol budget or gate drifted"
            )
        if self.unit_roster_digest != canonical_digest(
            tuple(asdict(unit) for unit in self.unit_roster)
        ):
            raise HfDetectorDirectionalProtocolError(
                "directional unit roster digest drifted"
            )


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HfDetectorDirectionalProtocolError(
            "checked-in directional JSON is unreadable"
        ) from exc
    if type(value) is not dict:
        raise HfDetectorDirectionalProtocolError(
            "checked-in directional JSON must be a mapping"
        )
    return value


def _prior_source_bindings(
    bindings: tuple[PriorDevelopmentManifestBinding, ...],
    repository_root: Path,
) -> tuple[tuple[str, int], ...]:
    values: set[tuple[str, int]] = set()
    for binding in bindings:
        path = repository_root / binding.path
        if sha256(path.read_bytes()).hexdigest() != binding.file_sha256:
            raise HfDetectorDirectionalProtocolError(
                "prior development manifest file digest drifted"
            )
        raw = _load_json(path)
        entries = raw.get("entries")
        if type(entries) is not list or not entries:
            raise HfDetectorDirectionalProtocolError(
                "prior development manifest entries are invalid"
            )
        for entry in entries:
            if type(entry) is not dict:
                raise HfDetectorDirectionalProtocolError(
                    "prior development manifest entry is invalid"
                )
            prompt = entry.get("prompt")
            seed = entry.get("generation_seed")
            if type(prompt) is not str or not prompt or type(seed) is not int:
                raise HfDetectorDirectionalProtocolError(
                    "prior development source identity is invalid"
                )
            values.add((sha256(prompt.encode("utf-8")).hexdigest(), seed))
    return tuple(sorted(values))


def load_hf_detector_directional_manifest(
    path: str | Path,
) -> HfDetectorDirectionalManifest:
    raw = _load_json(Path(path))
    try:
        manifest = HfDetectorDirectionalManifest(
            schema_version=raw["schema_version"],
            manifest_id=raw["manifest_id"],
            split=raw["split"],
            seed_namespace=raw["seed_namespace"],
            entries=tuple(
                HfDetectorDirectionalManifestEntry(**item)
                for item in raw["entries"]
            ),
        )
    except (KeyError, TypeError) as exc:
        raise HfDetectorDirectionalProtocolError(
            "directional manifest schema is invalid"
        ) from exc
    manifest.validate()
    return manifest


def load_hf_only_detector_directional_protocol(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[HfOnlyDetectorDirectionalProtocol, HfDetectorDirectionalManifest]:
    raw = _load_json(Path(path))
    try:
        protocol = HfOnlyDetectorDirectionalProtocol(
            **{
                **raw,
                "prior_development_manifests": tuple(
                    PriorDevelopmentManifestBinding(**item)
                    for item in raw["prior_development_manifests"]
                ),
            }
        )
    except (KeyError, TypeError) as exc:
        raise HfDetectorDirectionalProtocolError(
            "directional protocol schema is invalid"
        ) from exc
    protocol.validate()
    repository = Path(repository_root)
    manifest_path = repository / protocol.manifest_path
    if sha256(manifest_path.read_bytes()).hexdigest() != protocol.manifest_file_sha256:
        raise HfDetectorDirectionalProtocolError(
            "directional manifest file digest drifted"
        )
    manifest = load_hf_detector_directional_manifest(manifest_path)
    prior_bindings = _prior_source_bindings(
        protocol.prior_development_manifests, repository
    )
    if protocol.source_cluster_deny_list_digest != canonical_digest(
        {
            "manifest_bindings": tuple(
                asdict(item) for item in protocol.prior_development_manifests
            ),
            "prior_prompt_seed_bindings": prior_bindings,
        }
    ):
        raise HfDetectorDirectionalProtocolError(
            "directional source deny-list digest drifted"
        )
    if any(
        (entry.prompt_digest, entry.generation_seed) in set(prior_bindings)
        for entry in manifest.entries
    ):
        raise HfDetectorDirectionalProtocolError(
            "directional source cluster overlaps prior development data"
        )
    return protocol, manifest


__all__ = [
    "CLAIM_BOUNDARY",
    "HfDetectorDirectionalManifest",
    "HfDetectorDirectionalManifestEntry",
    "HfDetectorDirectionalProtocolError",
    "HfOnlyDetectorDirectionalProtocol",
    "canonical_digest",
    "load_hf_detector_directional_manifest",
    "load_hf_only_detector_directional_protocol",
]
