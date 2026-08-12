"""Frozen development protocol for the content-routing directional diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

from experiments.protocol.development_exploration import DevelopmentStudyUnit
from experiments.protocol.hf_only_detector_directional_validation import (
    AuthorityDenyAxes,
    PriorDevelopmentManifestBinding,
    load_authority_deny_axes,
)


PROTOCOL_ID = "ceg_wm_content_routing_directional_diagnosis"
PROTOCOL_VERSION = "1.0.0"
RUN_ID = "ceg_wm_content_routing_backend_binding_correction_diagnosis"
OPERATIONAL_UNIT_COUNT = 2
REFERENCE_FIT_CLUSTER_COUNT = 32
DIRECTIONAL_PROBE_CLUSTER_COUNT = 8
MAXIMUM_TOTAL_UNITS = 42
MAXIMUM_ATTEMPTS_PER_UNIT = 1
MAXIMUM_DURATION_SECONDS = 2700
CROSS_FIT_FOLD_COUNT = 4
REFERENCE_FIT_COUNT_PER_PROBE = 24
WRONG_KEY_ROSTER_SIZE = 4
CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250
MIXING_COEFFICIENT = 0.50
INCREMENTAL_INDICATOR_MEAN_REQUIREMENT = 0.5
ROUTING_COVERAGE_REQUIREMENT = 0.0
OPERATIONAL_PHASE = "development_environment_preflight"
OPERATIONAL_RESPONSIBILITY_ID = "development_environment_preflight"
OPERATIONAL_CONTENT_BRANCH_ID = "development_environment_preflight"
NOT_APPLICABLE_GEOMETRY_CASE_ID = "geometry_case_not_applicable"
OPERATIONAL_ROLE = "environment_runtime_throughput_preflight"
OPERATIONAL_CASE_IDS = (
    "environment_identity_preflight",
    "runtime_identity_preflight",
    "throughput_preflight",
)
OPERATIONAL_RESULT_RESPONSIBILITY_ID = "content_embedder"
REFERENCE_FIT_ROLE = "content_routing_reference_fit"
DIRECTIONAL_PROBE_ROLE = "content_routing_directional_probe"
REFERENCE_QUANTILE_RULE = "strictly_positive_exact_nearest_rank_p95"
REFERENCE_STATISTIC_IDENTITIES = (
    "texture_gradient_reference",
    "latent_response_reference",
    "local_sensitivity_reference",
)
ROUTING_SEMANTIC_OBSERVATION_IDENTITY = "clip_patch_prompt_similarity_without_fitted_reference"
ROUTED_ROUTE_IDENTITY = "routing_stqr"
UNIFORM_ROUTE_IDENTITY = "routing_uniform_control"
CONTENT_EMBEDDING_RESPONSIBILITY_ID = "content_embedder"
CONTENT_EMBEDDING_BRANCH_IDENTITY = "lf_hf_routed_combination"
PUBLIC_CONTENT_OPERATION = "FormalHfContentDetectionOperation"
PUBLIC_SCORE_IDENTITY = "hf_only_public_content_operation"
PUBLIC_SCORE_SEMANTICS = "content_score_equals_hf_result_hf_score"
PUBLIC_SCORE_REQUIRED_NULL_RESULT_FIELDS = (
    "lf_score",
    "lf_result",
    "combined_score",
    "diagnostic_combination",
    "diagnostic_identity",
)
LF_BRANCH_RESPONSIBILITY_IDS = ("lf_carrier", "content_embedder")
LF_DETECTOR_USAGE = "prohibited"
FUTURE_SPLIT_EXCLUSION_ROLES = (
    "routing_directional_validation",
    "candidate_selection",
    "content_threshold_fit",
    "rescue_window_fit",
    "geometry_reliability_fit",
    "end_to_end_calibration_check",
    "evaluation",
)
PASSING_OUTCOME = "routing_directional_signal_observed"
NEGATIVE_OUTCOME = "routing_directional_signal_not_observed"
PASSING_REQUEST = "allow_request_for_fixed_half_routing_directional_validation"
CLAIM_BOUNDARY = (
    "development_content_routing_directional_diagnosis_fixed_half_mixing_only_"
    "no_alpha_generalization_no_mechanism_promotion_no_threshold_no_fpr_no_"
    "combination_conclusion"
)
STOP_RULE = (
    "complete_two_operational_preflights_then_thirty_two_reference_fit_"
    "clusters_then_all_eight_paired_directional_probes_without_adaptive_"
    "sampling_or_parameter_change"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ContentRoutingDirectionalProtocolError(ValueError):
    """The checked-in routing diagnosis authority is inconsistent."""


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
class ContentRoutingManifestEntry:
    cluster_ordinal: int
    cluster_identity: str
    prompt: str
    generation_seed: int
    fold_index: int
    image_lineage_identity: str

    @property
    def prompt_digest(self) -> str:
        return sha256(self.prompt.encode("utf-8")).hexdigest()

    def image_lineage_digest(self, *, role_id: str) -> str:
        return canonical_digest(
            {
                "cluster_identity": self.cluster_identity,
                "generation_seed": self.generation_seed,
                "image_lineage_identity": self.image_lineage_identity,
                "prompt_digest": self.prompt_digest,
                "role_id": role_id,
            }
        )

    def validate(self, *, ordinal: int, role_id: str) -> None:
        if (
            type(self.cluster_ordinal) is not int
            or self.cluster_ordinal != ordinal
            or type(self.cluster_identity) is not str
            or not self.cluster_identity
            or type(self.prompt) is not str
            or not self.prompt
            or type(self.generation_seed) is not int
            or self.generation_seed < 0
            or type(self.fold_index) is not int
            or self.fold_index != ordinal % CROSS_FIT_FOLD_COUNT
            or type(self.image_lineage_identity) is not str
            or self.image_lineage_identity
            != f"{role_id}_paired_rendered_rgb8_observation"
        ):
            raise ContentRoutingDirectionalProtocolError(
                "routing manifest entry drifted"
            )


@dataclass(frozen=True, slots=True)
class ContentRoutingManifest:
    schema_version: str
    manifest_id: str
    role_id: str
    split: str
    seed_namespace: str
    source_cluster_namespace: str
    image_lineage_namespace: str
    key_family_namespace: str
    entries: tuple[ContentRoutingManifestEntry, ...]

    def validate(self, *, expected_role: str, expected_count: int) -> None:
        if (
            self.schema_version != "1.0.0"
            or self.manifest_id != f"{expected_role}_cluster_manifest"
            or self.role_id != expected_role
            or self.split != "development"
            or self.seed_namespace != f"{expected_role}_seed_namespace"
            or self.source_cluster_namespace
            != f"{expected_role}_source_cluster_namespace"
            or self.image_lineage_namespace
            != f"{expected_role}_rendered_rgb8_lineage_namespace"
            or self.key_family_namespace
            != f"{expected_role}_registered_key_family_namespace"
            or len(self.entries) != expected_count
        ):
            raise ContentRoutingDirectionalProtocolError(
                "routing manifest identity drifted"
            )
        for ordinal, entry in enumerate(self.entries):
            if type(entry) is not ContentRoutingManifestEntry:
                raise ContentRoutingDirectionalProtocolError(
                    "routing manifest entry type drifted"
                )
            entry.validate(ordinal=ordinal, role_id=expected_role)
        axes = (
            tuple(item.cluster_identity for item in self.entries),
            tuple(item.prompt_digest for item in self.entries),
            tuple(item.generation_seed for item in self.entries),
            tuple(item.image_lineage_digest(role_id=self.role_id) for item in self.entries),
        )
        if any(len(set(values)) != expected_count for values in axes):
            raise ContentRoutingDirectionalProtocolError(
                "routing manifest axes are not unique"
            )


@dataclass(frozen=True, slots=True)
class ContentRoutingOperationalUnit:
    unit_index: int
    source_cluster_ordinal: int
    phase: str
    responsibility_id: str
    content_branch_id: str
    geometry_case_id: str
    operational_role: str
    case_ids: tuple[str, ...]
    responsibility_result_digest_keys: tuple[str, ...]
    counts_as_scientific_coverage: bool
    scientific_claims_supported: bool

    def validate(self, *, ordinal: int) -> None:
        if (
            self.unit_index != ordinal
            or self.source_cluster_ordinal != ordinal
            or self.phase != OPERATIONAL_PHASE
            or self.responsibility_id != OPERATIONAL_RESPONSIBILITY_ID
            or self.content_branch_id != OPERATIONAL_CONTENT_BRANCH_ID
            or self.geometry_case_id != NOT_APPLICABLE_GEOMETRY_CASE_ID
            or self.operational_role != OPERATIONAL_ROLE
            or self.case_ids != OPERATIONAL_CASE_IDS
            or self.responsibility_result_digest_keys
            != (OPERATIONAL_RESULT_RESPONSIBILITY_ID,)
            or self.counts_as_scientific_coverage is not False
            or self.scientific_claims_supported is not False
        ):
            raise ContentRoutingDirectionalProtocolError(
                "routing operational unit identity drifted"
            )


@dataclass(frozen=True, slots=True)
class ContentRoutingDirectionalProtocol:
    schema_version: str
    protocol_id: str
    protocol_version: str
    run_id: str
    split: str
    reference_fit_manifest_path: str
    reference_fit_manifest_file_sha256: str
    directional_probe_manifest_path: str
    directional_probe_manifest_file_sha256: str
    prior_authority_bindings: tuple[PriorDevelopmentManifestBinding, ...]
    source_cluster_deny_list_digest: str
    future_split_exclusion_roles: tuple[str, ...]
    routing_candidate_identity: str
    uniform_control_identity: str
    content_embedding_responsibility_id: str
    content_embedding_branch_identity: str
    public_content_operation: str
    public_score_identity: str
    public_score_semantics: str
    public_score_required_null_result_fields: tuple[str, ...]
    lf_branch_responsibility_ids: tuple[str, ...]
    lf_detector_usage: str
    mixing_coefficient: float
    content_relative_l2_numerator: int
    content_relative_l2_denominator: int
    cross_fit_fold_count: int
    reference_fit_count_per_probe: int
    reference_quantile_rule: str
    reference_statistic_identities: tuple[str, ...]
    semantic_observation_identity: str
    wrong_key_roster_size: int
    operational_unit_count: int
    reference_fit_cluster_count: int
    directional_probe_cluster_count: int
    maximum_total_units: int
    maximum_attempts_per_unit: int
    maximum_duration_seconds_per_unit: int
    operational_units: tuple[ContentRoutingOperationalUnit, ...]
    incremental_indicator_mean_requirement: float
    routing_coverage_requirement: float
    passing_outcome: str
    negative_outcome: str
    passing_request: str
    stop_rule: str
    claim_boundary: str
    unit_roster_digest: str

    @property
    def unit_roster(self) -> tuple[DevelopmentStudyUnit, ...]:
        operational = tuple(
            DevelopmentStudyUnit(
                unit_index=authority.unit_index,
                phase=authority.phase,
                responsibility_id=authority.responsibility_id,
                source_cluster_ordinal=authority.source_cluster_ordinal,
                content_branch_id=authority.content_branch_id,
                geometry_case_id=authority.geometry_case_id,
                maximum_record_attempts=MAXIMUM_ATTEMPTS_PER_UNIT,
                maximum_duration_seconds=MAXIMUM_DURATION_SECONDS,
            )
            for authority in self.operational_units
        )
        reference = tuple(
            DevelopmentStudyUnit(
                unit_index=OPERATIONAL_UNIT_COUNT + ordinal,
                phase="development_routing_reference_fit",
                responsibility_id="content_router",
                source_cluster_ordinal=ordinal,
                content_branch_id="routing_observation_reference_fit",
                geometry_case_id=NOT_APPLICABLE_GEOMETRY_CASE_ID,
                maximum_record_attempts=MAXIMUM_ATTEMPTS_PER_UNIT,
                maximum_duration_seconds=MAXIMUM_DURATION_SECONDS,
            )
            for ordinal in range(REFERENCE_FIT_CLUSTER_COUNT)
        )
        probes = tuple(
            DevelopmentStudyUnit(
                unit_index=(
                    OPERATIONAL_UNIT_COUNT
                    + REFERENCE_FIT_CLUSTER_COUNT
                    + ordinal
                ),
                phase="development_content_routing_directional_probe",
                responsibility_id="content_router",
                source_cluster_ordinal=ordinal,
                content_branch_id="paired_routed_uniform_content_embedding",
                geometry_case_id=NOT_APPLICABLE_GEOMETRY_CASE_ID,
                maximum_record_attempts=MAXIMUM_ATTEMPTS_PER_UNIT,
                maximum_duration_seconds=MAXIMUM_DURATION_SECONDS,
            )
            for ordinal in range(DIRECTIONAL_PROBE_CLUSTER_COUNT)
        )
        return (*operational, *reference, *probes)

    def validate(self) -> None:
        expected = {
            "schema_version": "1.0.0",
            "protocol_id": PROTOCOL_ID,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": RUN_ID,
            "split": "development",
            "future_split_exclusion_roles": FUTURE_SPLIT_EXCLUSION_ROLES,
            "routing_candidate_identity": ROUTED_ROUTE_IDENTITY,
            "uniform_control_identity": UNIFORM_ROUTE_IDENTITY,
            "content_embedding_responsibility_id": CONTENT_EMBEDDING_RESPONSIBILITY_ID,
            "content_embedding_branch_identity": CONTENT_EMBEDDING_BRANCH_IDENTITY,
            "public_content_operation": PUBLIC_CONTENT_OPERATION,
            "public_score_identity": PUBLIC_SCORE_IDENTITY,
            "public_score_semantics": PUBLIC_SCORE_SEMANTICS,
            "public_score_required_null_result_fields": PUBLIC_SCORE_REQUIRED_NULL_RESULT_FIELDS,
            "lf_branch_responsibility_ids": LF_BRANCH_RESPONSIBILITY_IDS,
            "lf_detector_usage": LF_DETECTOR_USAGE,
            "mixing_coefficient": MIXING_COEFFICIENT,
            "content_relative_l2_numerator": CONTENT_RELATIVE_L2_NUMERATOR,
            "content_relative_l2_denominator": CONTENT_RELATIVE_L2_DENOMINATOR,
            "cross_fit_fold_count": CROSS_FIT_FOLD_COUNT,
            "reference_fit_count_per_probe": REFERENCE_FIT_COUNT_PER_PROBE,
            "reference_quantile_rule": REFERENCE_QUANTILE_RULE,
            "reference_statistic_identities": REFERENCE_STATISTIC_IDENTITIES,
            "semantic_observation_identity": ROUTING_SEMANTIC_OBSERVATION_IDENTITY,
            "wrong_key_roster_size": WRONG_KEY_ROSTER_SIZE,
            "operational_unit_count": OPERATIONAL_UNIT_COUNT,
            "reference_fit_cluster_count": REFERENCE_FIT_CLUSTER_COUNT,
            "directional_probe_cluster_count": DIRECTIONAL_PROBE_CLUSTER_COUNT,
            "maximum_total_units": MAXIMUM_TOTAL_UNITS,
            "maximum_attempts_per_unit": MAXIMUM_ATTEMPTS_PER_UNIT,
            "maximum_duration_seconds_per_unit": MAXIMUM_DURATION_SECONDS,
            "incremental_indicator_mean_requirement": INCREMENTAL_INDICATOR_MEAN_REQUIREMENT,
            "routing_coverage_requirement": ROUTING_COVERAGE_REQUIREMENT,
            "passing_outcome": PASSING_OUTCOME,
            "negative_outcome": NEGATIVE_OUTCOME,
            "passing_request": PASSING_REQUEST,
            "stop_rule": STOP_RULE,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise ContentRoutingDirectionalProtocolError(
                    f"routing protocol {field_name} drifted"
                )
        if len(self.operational_units) != OPERATIONAL_UNIT_COUNT:
            raise ContentRoutingDirectionalProtocolError(
                "routing operational unit count drifted"
            )
        for ordinal, unit in enumerate(self.operational_units):
            if type(unit) is not ContentRoutingOperationalUnit:
                raise ContentRoutingDirectionalProtocolError(
                    "routing operational unit exact type required"
                )
            unit.validate(ordinal=ordinal)
        if (
            not self.prior_authority_bindings
            or any(
                type(item) is not PriorDevelopmentManifestBinding
                for item in self.prior_authority_bindings
            )
        ):
            raise ContentRoutingDirectionalProtocolError(
                "routing prior authority bindings are missing"
            )
        for item in self.prior_authority_bindings:
            item.validate()
        digest_values = (
            self.reference_fit_manifest_file_sha256,
            self.directional_probe_manifest_file_sha256,
            self.source_cluster_deny_list_digest,
            self.unit_roster_digest,
        )
        if any(_DIGEST.fullmatch(value) is None for value in digest_values):
            raise ContentRoutingDirectionalProtocolError(
                "routing protocol digest is invalid"
            )
        if self.unit_roster_digest != canonical_digest(
            tuple(asdict(item) for item in self.unit_roster)
        ):
            raise ContentRoutingDirectionalProtocolError(
                "routing unit roster digest drifted"
            )

    def digest(self) -> str:
        self.validate()
        return canonical_digest(asdict(self))


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContentRoutingDirectionalProtocolError(
            "routing authority JSON is unreadable"
        ) from exc
    if type(value) is not dict:
        raise ContentRoutingDirectionalProtocolError(
            "routing authority JSON must be a mapping"
        )
    return value


def load_content_routing_manifest(
    path: str | Path,
    *,
    expected_role: str,
    expected_count: int,
) -> ContentRoutingManifest:
    raw = _load_json(Path(path))
    expected_keys = {
        "schema_version",
        "manifest_id",
        "role_id",
        "split",
        "seed_namespace",
        "source_cluster_namespace",
        "image_lineage_namespace",
        "key_family_namespace",
        "entries",
    }
    if set(raw) != expected_keys or type(raw["entries"]) is not list:
        raise ContentRoutingDirectionalProtocolError(
            "routing manifest schema drifted"
        )
    try:
        manifest = ContentRoutingManifest(
            **{
                **raw,
                "entries": tuple(
                    ContentRoutingManifestEntry(**item)
                    for item in raw["entries"]
                ),
            }
        )
    except (TypeError, KeyError) as exc:
        raise ContentRoutingDirectionalProtocolError(
            "routing manifest entry schema drifted"
        ) from exc
    manifest.validate(expected_role=expected_role, expected_count=expected_count)
    return manifest


def _manifest_axes(manifest: ContentRoutingManifest) -> AuthorityDenyAxes:
    return AuthorityDenyAxes(
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
                    *(
                        item.image_lineage_digest(role_id=manifest.role_id)
                        for item in manifest.entries
                    ),
                )
            )
        ),
        key_control_identities=(manifest.key_family_namespace,),
    )


def _axis_intersections(
    left: AuthorityDenyAxes,
    right: AuthorityDenyAxes,
) -> dict[str, tuple[object, ...]]:
    return {
        field_name: tuple(
            sorted(set(getattr(left, field_name)) & set(getattr(right, field_name)))
        )
        for field_name in asdict(left)
    }


def load_content_routing_directional_protocol(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[
    ContentRoutingDirectionalProtocol,
    ContentRoutingManifest,
    ContentRoutingManifest,
]:
    raw = _load_json(Path(path))
    if type(raw.get("prior_authority_bindings")) is not list:
        raise ContentRoutingDirectionalProtocolError(
            "routing prior authority schema drifted"
        )
    try:
        protocol = ContentRoutingDirectionalProtocol(
            **{
                **raw,
                "prior_authority_bindings": tuple(
                    PriorDevelopmentManifestBinding(**item)
                    for item in raw["prior_authority_bindings"]
                ),
                "future_split_exclusion_roles": tuple(
                    raw["future_split_exclusion_roles"]
                ),
                "reference_statistic_identities": tuple(
                    raw["reference_statistic_identities"]
                ),
                "public_score_required_null_result_fields": tuple(
                    raw["public_score_required_null_result_fields"]
                ),
                "lf_branch_responsibility_ids": tuple(
                    raw["lf_branch_responsibility_ids"]
                ),
                "operational_units": tuple(
                    ContentRoutingOperationalUnit(
                        **{
                            **item,
                            "case_ids": tuple(item["case_ids"]),
                            "responsibility_result_digest_keys": tuple(
                                item["responsibility_result_digest_keys"]
                            ),
                        }
                    )
                    for item in raw["operational_units"]
                ),
            }
        )
    except (TypeError, KeyError) as exc:
        raise ContentRoutingDirectionalProtocolError(
            "routing protocol schema drifted"
        ) from exc
    protocol.validate()
    root = Path(repository_root)
    reference_path = root / protocol.reference_fit_manifest_path
    probe_path = root / protocol.directional_probe_manifest_path
    if (
        sha256(reference_path.read_bytes()).hexdigest()
        != protocol.reference_fit_manifest_file_sha256
        or sha256(probe_path.read_bytes()).hexdigest()
        != protocol.directional_probe_manifest_file_sha256
    ):
        raise ContentRoutingDirectionalProtocolError(
            "routing manifest file digest drifted"
        )
    reference = load_content_routing_manifest(
        reference_path,
        expected_role=REFERENCE_FIT_ROLE,
        expected_count=REFERENCE_FIT_CLUSTER_COUNT,
    )
    probes = load_content_routing_manifest(
        probe_path,
        expected_role=DIRECTIONAL_PROBE_ROLE,
        expected_count=DIRECTIONAL_PROBE_CLUSTER_COUNT,
    )
    prior_axes = load_authority_deny_axes(protocol.prior_authority_bindings, root)
    if protocol.source_cluster_deny_list_digest != canonical_digest(
        {
            "authority_deny_axes": prior_axes.digest_value(),
            "manifest_bindings": tuple(
                asdict(item) for item in protocol.prior_authority_bindings
            ),
        }
    ):
        raise ContentRoutingDirectionalProtocolError(
            "routing source deny-list digest drifted"
        )
    reference_axes = _manifest_axes(reference)
    probe_axes = _manifest_axes(probes)
    if (
        any(_axis_intersections(prior_axes, reference_axes).values())
        or any(_axis_intersections(prior_axes, probe_axes).values())
        or any(_axis_intersections(reference_axes, probe_axes).values())
    ):
        raise ContentRoutingDirectionalProtocolError(
            "routing reference or probe authority overlaps a denied axis"
        )
    return protocol, reference, probes


def reference_entries_for_probe(
    probe: ContentRoutingManifestEntry,
    reference_manifest: ContentRoutingManifest,
) -> tuple[ContentRoutingManifestEntry, ...]:
    if type(probe) is not ContentRoutingManifestEntry:
        raise TypeError("routing probe entry exact type required")
    reference_manifest.validate(
        expected_role=REFERENCE_FIT_ROLE,
        expected_count=REFERENCE_FIT_CLUSTER_COUNT,
    )
    selected = tuple(
        item
        for item in reference_manifest.entries
        if item.fold_index != probe.fold_index
    )
    if len(selected) != REFERENCE_FIT_COUNT_PER_PROBE:
        raise ContentRoutingDirectionalProtocolError(
            "routing cross-fit reference count drifted"
        )
    return selected


__all__ = [
    "CLAIM_BOUNDARY",
    "CONTENT_RELATIVE_L2_DENOMINATOR",
    "CONTENT_RELATIVE_L2_NUMERATOR",
    "DIRECTIONAL_PROBE_CLUSTER_COUNT",
    "ContentRoutingDirectionalProtocol",
    "ContentRoutingDirectionalProtocolError",
    "ContentRoutingManifest",
    "ContentRoutingManifestEntry",
    "ContentRoutingOperationalUnit",
    "load_content_routing_directional_protocol",
    "load_content_routing_manifest",
    "reference_entries_for_probe",
]
