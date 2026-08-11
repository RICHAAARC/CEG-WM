"""Frozen development protocol for real Q/K synchronization-write diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re

from experiments.protocol.development_exploration import DevelopmentStudyUnit
from experiments.protocol.hf_only_detector_directional_validation import (
    AuthorityDenyAxes,
    PriorDevelopmentManifestBinding,
    load_authority_deny_axes,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    derive_source_cluster_id,
)


PROTOCOL_ID = "ceg_wm_qk_synchronization_write_diagnosis"
PROTOCOL_VERSION = "1.0.0"
RUN_ID = "ceg_wm_qk_differentiable_vae_resource_qualification"
OPERATIONAL_UNIT_COUNT = 1
AUTHORIZED_OPERATIONAL_UNIT_COUNT = 1
AUTHORIZED_SCIENTIFIC_UNIT_COUNT = 0
AUTHORIZED_TOTAL_UNIT_COUNT = 1
AUTHORIZED_MAXIMUM_ATTEMPTS_PER_UNIT = 1
RATIO_PROBE_CLUSTER_COUNT = 4
RATIO_PROBE_UNIT_COUNT = 12
TRANSFORM_PROBE_CLUSTER_COUNT = 4
TRANSFORM_PROBE_UNIT_COUNT = 16
SCIENTIFIC_UNIT_COUNT = RATIO_PROBE_UNIT_COUNT + TRANSFORM_PROBE_UNIT_COUNT
MAXIMUM_TOTAL_UNITS = OPERATIONAL_UNIT_COUNT + SCIENTIFIC_UNIT_COUNT
MAXIMUM_ATTEMPTS_PER_UNIT = 2
MAXIMUM_DURATION_SECONDS = 2700
WRONG_KEY_INDEXES = (0, 1, 2, 3)
GEOMETRY_RATIO_ROSTER = (
    ("geometry_content_ratio_one_sixteenth", 1.0 / 16.0),
    ("geometry_content_ratio_one_eighth", 1.0 / 8.0),
    ("geometry_content_ratio_one_quarter", 1.0 / 4.0),
)
TRANSFORM_PROBE_ROSTER = (
    ("identity", 1.0, 1.0, 0.0),
    ("crop", 0.75, 1.0, 0.0),
    ("scale", 1.0, 0.7071067811865476, 0.0),
    ("rotation", 1.0, 1.0, 16.0),
)
LINE_SEARCH_FACTORS = tuple(1.0 / (2**index) for index in range(8))
CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250
CONTENT_PROJECTION_RELATIVE_LIMIT = 1.0e-4
CALLBACK_INDEX = 18
QK_OBSERVATION_SCHEDULE_INDEX = 7
FUTURE_SPLIT_EXCLUSION_ROLES = (
    "candidate_selection",
    "calibration",
    "evaluation",
)
PASSING_MODULE_OUTCOME = "mechanism_signal_observed"
PASSING_CANDIDATE_RECOMMENDATION = "candidate_worth_further_selection"
CLAIM_BOUNDARY = (
    "development_qk_synchronization_write_diagnosis_only_no_ratio_selection_"
    "no_estimator_no_threshold_no_fpr_no_promotion_no_calibration"
)
RATIO_ELIGIBILITY_RULE = (
    "after_all_twelve_ratio_probe_units_are_terminal_choose_the_first_ratio_in_"
    "ascending_frozen_order_with_four_of_four_write_accepted_positive_actual_"
    "registered_gain_positive_keyed_gain_margin_and_zero_identity_budget_"
    "integrity_or_nonfinite_violation"
)
TRANSFORM_DEPENDENCY_RULE = (
    "execute_the_sixteen_preregistered_transform_probe_units_only_after_one_"
    "ratio_is_eligible_otherwise_commit_dependency_blocked_excluded_records"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class QkSynchronizationWriteProtocolError(ValueError):
    """The checked-in Q/K synchronization-write protocol is inconsistent."""


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
class QkSynchronizationManifestEntry:
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
        if (
            type(self.cluster_ordinal) is not int
            or not 0 <= self.cluster_ordinal < RATIO_PROBE_CLUSTER_COUNT
            or any(
                type(value) is not str or not value
                for value in (
                    self.cluster_identity,
                    self.prompt,
                    self.image_lineage_identity,
                )
            )
            or self.split != "development"
            or self.role_id != "qk_synchronization_write_diagnosis"
            or type(self.generation_seed) is not int
            or self.generation_seed < 0
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis manifest entry is invalid"
            )
        if self.prompt_digest != sha256(self.prompt.encode("utf-8")).hexdigest():
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis prompt digest drifted"
            )
        expected_lineage = canonical_digest(
            {
                "cluster_identity": self.cluster_identity,
                "generation_seed": self.generation_seed,
                "image_lineage_identity": self.image_lineage_identity,
                "image_lineage_namespace": image_lineage_namespace,
            }
        )
        if self.image_lineage_digest != expected_lineage:
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis image lineage digest drifted"
            )


@dataclass(frozen=True, slots=True)
class QkSynchronizationManifest:
    schema_version: str
    manifest_id: str
    role_id: str
    seed_namespace: str
    source_cluster_namespace: str
    image_lineage_namespace: str
    content_key_family_namespace: str
    geometry_key_family_namespace: str
    entries: tuple[QkSynchronizationManifestEntry, ...]

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if (
            self.schema_version
            != "ceg_wm_qk_synchronization_write_diagnosis_manifest"
            or self.manifest_id
            != "qk_synchronization_write_diagnosis_source_clusters"
            or self.role_id != "qk_synchronization_write_diagnosis"
            or self.seed_namespace
            != "qk_synchronization_write_diagnosis_seed_namespace"
            or self.source_cluster_namespace
            != "qk_synchronization_write_diagnosis_source_cluster_namespace"
            or self.image_lineage_namespace
            != "qk_synchronization_write_diagnosis_public_rgb8_lineage_namespace"
            or self.content_key_family_namespace
            != "qk_synchronization_write_diagnosis_hf_content_key_family"
            or self.geometry_key_family_namespace
            != "qk_synchronization_write_diagnosis_geometry_key_family"
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis manifest authority drifted"
            )
        if (
            len(self.entries) != RATIO_PROBE_CLUSTER_COUNT
            or tuple(item.cluster_ordinal for item in self.entries)
            != tuple(range(RATIO_PROBE_CLUSTER_COUNT))
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis manifest coverage drifted"
            )
        for item in self.entries:
            item.validate(image_lineage_namespace=self.image_lineage_namespace)
        for values in (
            tuple(item.cluster_identity for item in self.entries),
            tuple(item.prompt_digest for item in self.entries),
            tuple(item.generation_seed for item in self.entries),
            tuple(item.image_lineage_digest for item in self.entries),
        ):
            if len(set(values)) != RATIO_PROBE_CLUSTER_COUNT:
                raise QkSynchronizationWriteProtocolError(
                    "Q/K diagnosis manifest axis collides"
                )


@dataclass(frozen=True, slots=True)
class GeometryRatioProbeSpecification:
    ratio_identity: str
    ratio: float


@dataclass(frozen=True, slots=True)
class TransformedRelationProbeSpecification:
    transform_identity: str
    crop_fraction: float
    scale_factor: float
    rotation_degrees: float


@dataclass(frozen=True, slots=True)
class QkSynchronizationWriteProtocol:
    schema_version: str
    protocol_id: str
    protocol_version: str
    run_id: str
    split: str
    role_id: str
    candidate_identity: str
    content_branch_id: str
    routing_mode: str
    public_method_callable: str
    public_runtime_chain: str
    manifest_path: str
    manifest_file_sha256: str
    prior_development_manifests: tuple[PriorDevelopmentManifestBinding, ...]
    source_cluster_deny_list_digest: str
    future_split_exclusion_roles: tuple[str, ...]
    content_registered_key_derivation_identity: str
    geometry_registered_key_derivation_identity: str
    wrong_key_control_identity: str
    candidate_specification_sha256: str
    method_review_reference: str
    method_reviewed_revision: str
    runtime_configuration_path: str
    runtime_configuration_file_sha256: str
    internal_execution_components_path: str
    internal_execution_components_file_sha256: str
    operational_smoke_prompt: str
    operational_smoke_prompt_digest: str
    operational_smoke_generation_seed: int
    operational_smoke_image_lineage_identity: str
    operational_smoke_image_lineage_digest: str
    operational_unit_count: int
    authorized_operational_unit_count: int
    authorized_scientific_unit_count: int
    authorized_total_unit_count: int
    authorized_maximum_attempts_per_unit: int
    authorized_unit_roster_digest: str
    ratio_probe_cluster_count: int
    ratio_probe_unit_count: int
    transform_probe_cluster_count: int
    transform_probe_unit_count: int
    scientific_unit_count: int
    maximum_total_units: int
    maximum_attempts_per_unit: int
    maximum_duration_seconds_per_unit: int
    wrong_key_indexes: tuple[int, ...]
    geometry_ratio_roster: tuple[GeometryRatioProbeSpecification, ...]
    transform_probe_roster: tuple[TransformedRelationProbeSpecification, ...]
    line_search_factors: tuple[float, ...]
    content_relative_l2_numerator: int
    content_relative_l2_denominator: int
    content_projection_relative_limit: float
    callback_index: int
    qk_observation_schedule_index: int
    qk_observation_conditioning_identity: str
    qk_observation_layer_names: tuple[str, ...]
    ratio_eligibility_rule: str
    transform_dependency_rule: str
    passing_module_outcome: str
    passing_candidate_recommendation: str
    claim_boundary: str
    unit_roster_digest: str

    @property
    def unit_roster(self) -> tuple[DevelopmentStudyUnit, ...]:
        operational = (
            DevelopmentStudyUnit(
                unit_index=0,
                phase="development_environment_preflight",
                responsibility_id="qk_synchronization_write_runtime_preflight",
                source_cluster_ordinal=0,
                content_branch_id="hf_only",
                geometry_case_id="identity",
                maximum_record_attempts=self.maximum_attempts_per_unit,
                maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
            ),
        )
        ratio_units = tuple(
            DevelopmentStudyUnit(
                unit_index=(
                    self.operational_unit_count
                    + ratio_index * self.ratio_probe_cluster_count
                    + cluster_ordinal
                ),
                phase="development_scientific_breadth",
                responsibility_id="geometry_write_ratio_probe",
                source_cluster_ordinal=cluster_ordinal,
                content_branch_id="hf_only",
                geometry_case_id=specification.ratio_identity,
                maximum_record_attempts=self.maximum_attempts_per_unit,
                maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
            )
            for ratio_index, specification in enumerate(self.geometry_ratio_roster)
            for cluster_ordinal in range(self.ratio_probe_cluster_count)
        )
        transform_units = tuple(
            DevelopmentStudyUnit(
                unit_index=(
                    self.operational_unit_count
                    + self.ratio_probe_unit_count
                    + transform_index * self.transform_probe_cluster_count
                    + cluster_ordinal
                ),
                phase="development_scientific_breadth",
                responsibility_id="transformed_relation_probe",
                source_cluster_ordinal=cluster_ordinal,
                content_branch_id="hf_only",
                geometry_case_id=specification.transform_identity,
                maximum_record_attempts=self.maximum_attempts_per_unit,
                maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
            )
            for transform_index, specification in enumerate(
                self.transform_probe_roster
            )
            for cluster_ordinal in range(self.transform_probe_cluster_count)
        )
        return operational + ratio_units + transform_units

    @property
    def authorized_unit_roster(self) -> tuple[DevelopmentStudyUnit, ...]:
        return (
            DevelopmentStudyUnit(
                unit_index=0,
                phase="development_environment_preflight",
                responsibility_id="qk_synchronization_write_runtime_preflight",
                source_cluster_ordinal=0,
                content_branch_id="hf_only",
                geometry_case_id="identity",
                maximum_record_attempts=(
                    self.authorized_maximum_attempts_per_unit
                ),
                maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
            ),
        )

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if (
            self.schema_version
            != "ceg_wm_qk_synchronization_write_diagnosis_protocol"
            or self.protocol_id != PROTOCOL_ID
            or self.protocol_version != PROTOCOL_VERSION
            or self.run_id != RUN_ID
            or self.split != "development"
            or self.role_id != "qk_synchronization_write_diagnosis"
            or self.candidate_identity != "qk_relation_similarity"
            or self.content_branch_id != "hf_only"
            or self.routing_mode != "routing_disabled"
            or self.public_method_callable
            != "experiments.methods.CegWmExperimentAdapter.execute_qk_synchronization_write"
            or self.public_runtime_chain
            != "runtime.public_suffix_replay_to_main_qk_relation"
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis protocol identity drifted"
            )
        for digest in (
            self.manifest_file_sha256,
            self.source_cluster_deny_list_digest,
            self.candidate_specification_sha256,
            self.runtime_configuration_file_sha256,
            self.internal_execution_components_file_sha256,
        ):
            if _DIGEST.fullmatch(digest) is None:
                raise QkSynchronizationWriteProtocolError(
                    "Q/K diagnosis digest is invalid"
                )
        if (
            _REVISION.fullmatch(self.method_reviewed_revision) is None
            or type(self.method_review_reference) is not str
            or not self.method_review_reference
            or self.future_split_exclusion_roles
            != FUTURE_SPLIT_EXCLUSION_ROLES
            or not self.prior_development_manifests
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis review or split authority is invalid"
            )
        for binding in self.prior_development_manifests:
            binding.validate()
        if (
            self.content_registered_key_derivation_identity
            != "qk_diagnosis_hf_content_registered_key_derivation"
            or self.geometry_registered_key_derivation_identity
            != "qk_diagnosis_geometry_registered_key_derivation"
            or self.wrong_key_control_identity
            != "qk_diagnosis_four_geometry_wrong_key_controls"
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis key authority drifted"
            )
        if (
            self.operational_unit_count != OPERATIONAL_UNIT_COUNT
            or self.authorized_operational_unit_count
            != AUTHORIZED_OPERATIONAL_UNIT_COUNT
            or self.authorized_scientific_unit_count
            != AUTHORIZED_SCIENTIFIC_UNIT_COUNT
            or self.authorized_total_unit_count != AUTHORIZED_TOTAL_UNIT_COUNT
            or self.authorized_maximum_attempts_per_unit
            != AUTHORIZED_MAXIMUM_ATTEMPTS_PER_UNIT
            or len(self.authorized_unit_roster) != AUTHORIZED_TOTAL_UNIT_COUNT
            or self.ratio_probe_cluster_count != RATIO_PROBE_CLUSTER_COUNT
            or self.ratio_probe_unit_count != RATIO_PROBE_UNIT_COUNT
            or self.transform_probe_cluster_count
            != TRANSFORM_PROBE_CLUSTER_COUNT
            or self.transform_probe_unit_count != TRANSFORM_PROBE_UNIT_COUNT
            or self.scientific_unit_count != SCIENTIFIC_UNIT_COUNT
            or self.maximum_total_units != MAXIMUM_TOTAL_UNITS
            or len(self.unit_roster) != MAXIMUM_TOTAL_UNITS
            or self.maximum_attempts_per_unit != MAXIMUM_ATTEMPTS_PER_UNIT
            or self.maximum_duration_seconds_per_unit
            != MAXIMUM_DURATION_SECONDS
            or self.wrong_key_indexes != WRONG_KEY_INDEXES
            or tuple(
                (item.ratio_identity, item.ratio)
                for item in self.geometry_ratio_roster
            )
            != GEOMETRY_RATIO_ROSTER
            or tuple(
                (
                    item.transform_identity,
                    item.crop_fraction,
                    item.scale_factor,
                    item.rotation_degrees,
                )
                for item in self.transform_probe_roster
            )
            != TRANSFORM_PROBE_ROSTER
            or self.line_search_factors != LINE_SEARCH_FACTORS
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis roster or budget drifted"
            )
        if (
            self.content_relative_l2_numerator
            != CONTENT_RELATIVE_L2_NUMERATOR
            or self.content_relative_l2_denominator
            != CONTENT_RELATIVE_L2_DENOMINATOR
            or self.content_projection_relative_limit
            != CONTENT_PROJECTION_RELATIVE_LIMIT
            or self.callback_index != CALLBACK_INDEX
            or self.qk_observation_schedule_index
            != QK_OBSERVATION_SCHEDULE_INDEX
            or self.qk_observation_conditioning_identity
            != "sd3_empty_text_triplet_without_cfg"
            or self.qk_observation_layer_names
            != (
                "transformer_blocks.0.attn",
                "transformer_blocks.23.attn",
            )
            or self.ratio_eligibility_rule != RATIO_ELIGIBILITY_RULE
            or self.transform_dependency_rule != TRANSFORM_DEPENDENCY_RULE
            or self.passing_module_outcome != PASSING_MODULE_OUTCOME
            or self.passing_candidate_recommendation
            != PASSING_CANDIDATE_RECOMMENDATION
            or self.claim_boundary != CLAIM_BOUNDARY
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis scientific rule drifted"
            )
        if (
            type(self.operational_smoke_prompt) is not str
            or not self.operational_smoke_prompt
            or self.operational_smoke_prompt_digest
            != sha256(self.operational_smoke_prompt.encode("utf-8")).hexdigest()
            or type(self.operational_smoke_generation_seed) is not int
            or self.operational_smoke_generation_seed < 0
            or not self.operational_smoke_image_lineage_identity
            or _DIGEST.fullmatch(self.operational_smoke_image_lineage_digest)
            is None
            or self.operational_smoke_image_lineage_digest
            != canonical_digest(
                {
                    "generation_seed": self.operational_smoke_generation_seed,
                    "image_lineage_identity": (
                        self.operational_smoke_image_lineage_identity
                    ),
                    "prompt_digest": self.operational_smoke_prompt_digest,
                }
            )
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis operational identity is invalid"
            )
        for configured_path in (
            self.manifest_path,
            self.runtime_configuration_path,
            self.internal_execution_components_path,
        ):
            path = PurePosixPath(configured_path)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise QkSynchronizationWriteProtocolError(
                    "Q/K diagnosis configured path is invalid"
                )
        if self.unit_roster_digest != canonical_digest(
            tuple(asdict(unit) for unit in self.unit_roster)
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis unit roster digest drifted"
            )
        if self.authorized_unit_roster_digest != canonical_digest(
            tuple(asdict(unit) for unit in self.authorized_unit_roster)
        ):
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis authorized unit roster digest drifted"
            )


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QkSynchronizationWriteProtocolError(
            "checked-in Q/K diagnosis JSON is unreadable"
        ) from exc
    if type(value) is not dict:
        raise QkSynchronizationWriteProtocolError(
            "checked-in Q/K diagnosis JSON must be a mapping"
        )
    return value


def load_qk_synchronization_manifest(
    path: str | Path,
) -> QkSynchronizationManifest:
    raw = _load_json(Path(path))
    try:
        manifest = QkSynchronizationManifest(
            **{
                **raw,
                "entries": tuple(
                    QkSynchronizationManifestEntry(**item)
                    for item in raw["entries"]
                ),
            }
        )
    except (KeyError, TypeError) as exc:
        raise QkSynchronizationWriteProtocolError(
            "Q/K diagnosis manifest schema is invalid"
        ) from exc
    manifest.validate()
    return manifest


def load_qk_synchronization_write_protocol(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[QkSynchronizationWriteProtocol, QkSynchronizationManifest]:
    raw = _load_json(Path(path))
    try:
        protocol = QkSynchronizationWriteProtocol(
            **{
                **raw,
                "prior_development_manifests": tuple(
                    PriorDevelopmentManifestBinding(**item)
                    for item in raw["prior_development_manifests"]
                ),
                "future_split_exclusion_roles": tuple(
                    raw["future_split_exclusion_roles"]
                ),
                "wrong_key_indexes": tuple(raw["wrong_key_indexes"]),
                "geometry_ratio_roster": tuple(
                    GeometryRatioProbeSpecification(**item)
                    for item in raw["geometry_ratio_roster"]
                ),
                "transform_probe_roster": tuple(
                    TransformedRelationProbeSpecification(**item)
                    for item in raw["transform_probe_roster"]
                ),
                "line_search_factors": tuple(raw["line_search_factors"]),
                "qk_observation_layer_names": tuple(
                    raw["qk_observation_layer_names"]
                ),
            }
        )
    except (KeyError, TypeError) as exc:
        raise QkSynchronizationWriteProtocolError(
            "Q/K diagnosis protocol schema is invalid"
        ) from exc
    protocol.validate()
    root = Path(repository_root)
    manifest_path = root / protocol.manifest_path
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise QkSynchronizationWriteProtocolError(
            "Q/K diagnosis manifest is missing"
        ) from exc
    if sha256(manifest_bytes).hexdigest() != protocol.manifest_file_sha256:
        raise QkSynchronizationWriteProtocolError(
            "Q/K diagnosis manifest file digest drifted"
        )
    manifest = load_qk_synchronization_manifest(manifest_path)
    for configured_path, expected_digest in (
        (
            protocol.runtime_configuration_path,
            protocol.runtime_configuration_file_sha256,
        ),
        (
            protocol.internal_execution_components_path,
            protocol.internal_execution_components_file_sha256,
        ),
    ):
        try:
            payload = (root / configured_path).read_bytes()
        except OSError as exc:
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis execution authority is missing"
            ) from exc
        if sha256(payload).hexdigest() != expected_digest:
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis execution authority digest drifted"
            )
    deny_axes = load_authority_deny_axes(
        protocol.prior_development_manifests,
        root,
    )
    if protocol.source_cluster_deny_list_digest != canonical_digest(
        {
            "manifest_bindings": tuple(
                asdict(item) for item in protocol.prior_development_manifests
            ),
            "authority_deny_axes": deny_axes.digest_value(),
        }
    ):
        raise QkSynchronizationWriteProtocolError(
            "Q/K diagnosis deny-list digest drifted"
        )
    new_axes = AuthorityDenyAxes(
        prompt_digests=tuple(sorted(item.prompt_digest for item in manifest.entries)),
        source_cluster_identities=tuple(
            sorted(
                (
                    manifest.source_cluster_namespace,
                    *(item.cluster_identity for item in manifest.entries),
                )
            )
        ),
        seed_namespaces=(manifest.seed_namespace,),
        generation_seeds=tuple(
            sorted(item.generation_seed for item in manifest.entries)
        ),
        image_lineage_identities=tuple(
            sorted(
                (
                    manifest.image_lineage_namespace,
                    *(item.image_lineage_identity for item in manifest.entries),
                    *(item.image_lineage_digest for item in manifest.entries),
                )
            )
        ),
        key_control_identities=tuple(
            sorted(
                (
                    manifest.content_key_family_namespace,
                    manifest.geometry_key_family_namespace,
                    protocol.content_registered_key_derivation_identity,
                    protocol.geometry_registered_key_derivation_identity,
                    protocol.wrong_key_control_identity,
                )
            )
        ),
    )
    intersections = {
        name: sorted(set(getattr(deny_axes, name)) & set(getattr(new_axes, name)))
        for name in asdict(deny_axes)
    }
    operational_values = {
        "prompt_digests": {protocol.operational_smoke_prompt_digest},
        "generation_seeds": {protocol.operational_smoke_generation_seed},
        "image_lineage_identities": {
            protocol.operational_smoke_image_lineage_identity,
            protocol.operational_smoke_image_lineage_digest,
        },
    }
    if any(intersections.values()) or any(
        values & set(getattr(deny_axes, name))
        for name, values in operational_values.items()
    ):
        raise QkSynchronizationWriteProtocolError(
            "Q/K diagnosis overlaps a prior authority axis"
        )
    if (
        protocol.operational_smoke_prompt_digest
        in {item.prompt_digest for item in manifest.entries}
        or protocol.operational_smoke_generation_seed
        in {item.generation_seed for item in manifest.entries}
        or protocol.operational_smoke_image_lineage_digest
        in {item.image_lineage_digest for item in manifest.entries}
    ):
        raise QkSynchronizationWriteProtocolError(
            "Q/K diagnosis operational smoke overlaps scientific manifest"
        )
    return protocol, manifest


def derive_qk_synchronization_analysis_identity(
    entry: QkSynchronizationManifestEntry,
    unit: DevelopmentStudyUnit,
    *,
    content_key_family_digest: str,
    geometry_key_family_digest: str,
) -> AnalysisUnitIdentity:
    for value in (content_key_family_digest, geometry_key_family_digest):
        if _DIGEST.fullmatch(value) is None:
            raise QkSynchronizationWriteProtocolError(
                "Q/K diagnosis key family digest is invalid"
            )
    combined_key_family_digest = canonical_digest(
        {
            "content_key_family_digest": content_key_family_digest,
            "geometry_key_family_digest": geometry_key_family_digest,
        }
    )
    return AnalysisUnitIdentity(
        unit_id=f"qk_synchronization_write_unit_{unit.unit_index:02d}",
        case_id=unit.geometry_case_id,
        source_cluster_id=derive_source_cluster_id(
            prompt_digest=entry.prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=entry.image_lineage_digest,
            registered_key_family_digest=combined_key_family_digest,
        ),
        prompt_digest=entry.prompt_digest,
        generation_seed=entry.generation_seed,
        image_lineage_digest=entry.image_lineage_digest,
        registered_key_family_digest=combined_key_family_digest,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "AUTHORIZED_SCIENTIFIC_UNIT_COUNT",
    "AUTHORIZED_TOTAL_UNIT_COUNT",
    "GEOMETRY_RATIO_ROSTER",
    "MAXIMUM_TOTAL_UNITS",
    "QkSynchronizationManifest",
    "QkSynchronizationManifestEntry",
    "QkSynchronizationWriteProtocol",
    "QkSynchronizationWriteProtocolError",
    "RATIO_PROBE_UNIT_COUNT",
    "RUN_ID",
    "SCIENTIFIC_UNIT_COUNT",
    "TRANSFORM_PROBE_UNIT_COUNT",
    "WRONG_KEY_INDEXES",
    "canonical_digest",
    "derive_qk_synchronization_analysis_identity",
    "load_qk_synchronization_manifest",
    "load_qk_synchronization_write_protocol",
]
