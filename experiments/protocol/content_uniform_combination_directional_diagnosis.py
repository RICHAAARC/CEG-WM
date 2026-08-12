"""Frozen development protocol for disabled-routing content combination diagnosis."""

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


PROTOCOL_ID = "ceg_wm_content_uniform_combination_directional_diagnosis"
PROTOCOL_VERSION = "1.0.0"
RUN_ID = "ceg_wm_content_uniform_combination_whitening_asset_replay_correction_diagnosis"
OPERATIONAL_UNIT_COUNT = 1
REFERENCE_FIT_CLUSTER_COUNT = 32
DIRECTIONAL_PROBE_CLUSTER_COUNT = 8
MAXIMUM_TOTAL_UNITS = 41
MAXIMUM_ATTEMPTS_PER_UNIT = 1
MAXIMUM_DURATION_SECONDS = 2700
CROSS_FIT_FOLD_COUNT = 4
REFERENCE_FIT_COUNT_PER_PROBE = 24
WRONG_KEY_ROSTER_SIZE = 4
CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250
MIXING_COEFFICIENTS = (0.25, 0.50, 0.75)
COMBINATION_FUNCTIONS = (
    "hf_only_standardized_score",
    "weighted_hf_lf_standardized_score",
    "maximum_hf_lf_standardized_score",
)
COMBINATION_WEIGHTS = (0.25, 0.50, 0.75)
UNIFORM_ROUTE_IDENTITY = "routing_uniform_control"
LF_DETECTOR_IDENTITY = "lf_null_whitened_matched_score"
REFERENCE_FIT_ROLE = "content_uniform_combination_reference_fit"
DIRECTIONAL_PROBE_ROLE = "content_uniform_combination_directional_probe"
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
ATTRIBUTION_MARGIN_FLOOR = 0.0009765625
ATTRIBUTION_SUCCESS_COUNT_REQUIREMENT = 7
DIRECTIONAL_IMPROVEMENT_COUNT_REQUIREMENT = 4
IDENTITY_SUCCESS_COUNT_MAXIMUM_LOSS = 1
PASSING_OUTCOME = "mechanism_signal_observed"
NEGATIVE_OUTCOME = "mechanism_signal_not_observed"
PASSING_RECOMMENDATION = "candidate_worth_further_selection"
NEGATIVE_RECOMMENDATION = "candidate_not_recommended_for_selection"
PASSING_REQUEST = "allow_request_for_content_combination_candidate_selection"
FUTURE_SPLIT_EXCLUSION_ROLES = (
    "content_combination_candidate_selection",
    "content_threshold_fit",
    "rescue_window_fit",
    "geometry_reliability_fit",
    "end_to_end_calibration_check",
    "evaluation",
)
STOP_RULE = (
    "complete_one_operational_preflight_then_thirty_two_reference_fit_clusters_"
    "then_all_eight_six_image_probes_without_adaptive_sampling_parameter_change_"
    "or_result_conditioned_candidate_choice"
)
CLAIM_BOUNDARY = (
    "development_disabled_routing_content_combination_directional_diagnosis_only_"
    "no_candidate_selection_no_promotion_no_threshold_no_fpr_no_calibration_"
    "no_evaluation_no_joint_claim"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ContentUniformCombinationDirectionalProtocolError(ValueError):
    """The checked-in combination diagnosis authority is inconsistent."""


def canonical_digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
        sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ContentUniformCombinationManifestEntry:
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
        return canonical_digest({
            "cluster_identity": self.cluster_identity,
            "generation_seed": self.generation_seed,
            "image_lineage_identity": self.image_lineage_identity,
            "prompt_digest": self.prompt_digest,
            "role_id": role_id,
        })

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
            or self.image_lineage_identity != f"{role_id}_paired_rendered_rgb8_observation"
        ):
            raise ContentUniformCombinationDirectionalProtocolError(
                "combination manifest entry drifted"
            )


@dataclass(frozen=True, slots=True)
class ContentUniformCombinationManifest:
    schema_version: str
    manifest_id: str
    role_id: str
    split: str
    seed_namespace: str
    source_cluster_namespace: str
    image_lineage_namespace: str
    key_family_namespace: str
    entries: tuple[ContentUniformCombinationManifestEntry, ...]

    def validate(self, *, expected_role: str, expected_count: int) -> None:
        if (
            self.schema_version != "1.0.0"
            or self.manifest_id != f"{expected_role}_cluster_manifest"
            or self.role_id != expected_role
            or self.split != "development"
            or self.seed_namespace != f"{expected_role}_seed_namespace"
            or self.source_cluster_namespace != f"{expected_role}_source_cluster_namespace"
            or self.image_lineage_namespace != f"{expected_role}_rendered_rgb8_lineage_namespace"
            or self.key_family_namespace != f"{expected_role}_registered_key_family_namespace"
            or len(self.entries) != expected_count
        ):
            raise ContentUniformCombinationDirectionalProtocolError(
                "combination manifest identity drifted"
            )
        for ordinal, entry in enumerate(self.entries):
            if type(entry) is not ContentUniformCombinationManifestEntry:
                raise ContentUniformCombinationDirectionalProtocolError(
                    "combination manifest entry exact type required"
                )
            entry.validate(ordinal=ordinal, role_id=expected_role)
        axes = (
            tuple(item.cluster_identity for item in self.entries),
            tuple(item.prompt_digest for item in self.entries),
            tuple(item.generation_seed for item in self.entries),
            tuple(item.image_lineage_digest(role_id=self.role_id) for item in self.entries),
        )
        if any(len(set(values)) != expected_count for values in axes):
            raise ContentUniformCombinationDirectionalProtocolError(
                "combination manifest axes are not unique"
            )


@dataclass(frozen=True, slots=True)
class ContentUniformCombinationOperationalUnit:
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

    def validate(self) -> None:
        if (
            self.unit_index != 0
            or self.source_cluster_ordinal != 0
            or self.phase != OPERATIONAL_PHASE
            or self.responsibility_id != OPERATIONAL_RESPONSIBILITY_ID
            or self.content_branch_id != OPERATIONAL_CONTENT_BRANCH_ID
            or self.geometry_case_id != NOT_APPLICABLE_GEOMETRY_CASE_ID
            or self.operational_role != OPERATIONAL_ROLE
            or self.case_ids != OPERATIONAL_CASE_IDS
            or self.responsibility_result_digest_keys != (OPERATIONAL_RESULT_RESPONSIBILITY_ID,)
            or self.counts_as_scientific_coverage is not False
            or self.scientific_claims_supported is not False
        ):
            raise ContentUniformCombinationDirectionalProtocolError(
                "combination operational unit drifted"
            )


@dataclass(frozen=True, slots=True)
class ContentUniformCombinationDirectionalProtocol:
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
    uniform_route_identity: str
    lf_detector_identity: str
    whitening_asset_fit_identity: str
    whitening_asset_fit_producer_revision: str
    whitening_asset_fit_protocol_digest: str
    whitening_asset_fit_run_id: str
    whitening_asset_digest: str
    mixing_coefficients: tuple[float, ...]
    combination_functions: tuple[str, ...]
    combination_weights: tuple[float, ...]
    wrong_key_roster_size: int
    content_relative_l2_numerator: int
    content_relative_l2_denominator: int
    cross_fit_fold_count: int
    reference_fit_count_per_probe: int
    operational_unit_count: int
    reference_fit_cluster_count: int
    directional_probe_cluster_count: int
    maximum_total_units: int
    maximum_attempts_per_unit: int
    maximum_duration_seconds_per_unit: int
    operational_units: tuple[ContentUniformCombinationOperationalUnit, ...]
    attribution_margin_floor: float
    attribution_success_count_requirement: int
    directional_improvement_count_requirement: int
    identity_success_count_maximum_loss: int
    passing_outcome: str
    negative_outcome: str
    passing_recommendation: str
    negative_recommendation: str
    passing_request: str
    stop_rule: str
    claim_boundary: str
    unit_roster_digest: str

    @property
    def unit_roster(self) -> tuple[DevelopmentStudyUnit, ...]:
        operational = (DevelopmentStudyUnit(
            unit_index=0,
            phase=OPERATIONAL_PHASE,
            responsibility_id=OPERATIONAL_RESPONSIBILITY_ID,
            source_cluster_ordinal=0,
            content_branch_id=OPERATIONAL_CONTENT_BRANCH_ID,
            geometry_case_id=NOT_APPLICABLE_GEOMETRY_CASE_ID,
            maximum_record_attempts=MAXIMUM_ATTEMPTS_PER_UNIT,
            maximum_duration_seconds=MAXIMUM_DURATION_SECONDS,
        ),)
        reference = tuple(DevelopmentStudyUnit(
            unit_index=OPERATIONAL_UNIT_COUNT + ordinal,
            phase="development_content_combination_reference_fit",
            responsibility_id="content_detector",
            source_cluster_ordinal=ordinal,
            content_branch_id="paired_clean_branch_null_reference",
            geometry_case_id=NOT_APPLICABLE_GEOMETRY_CASE_ID,
            maximum_record_attempts=MAXIMUM_ATTEMPTS_PER_UNIT,
            maximum_duration_seconds=MAXIMUM_DURATION_SECONDS,
        ) for ordinal in range(REFERENCE_FIT_CLUSTER_COUNT))
        probes = tuple(DevelopmentStudyUnit(
            unit_index=OPERATIONAL_UNIT_COUNT + REFERENCE_FIT_CLUSTER_COUNT + ordinal,
            phase="development_content_uniform_combination_directional_probe",
            responsibility_id="content_detector",
            source_cluster_ordinal=ordinal,
            content_branch_id="six_image_uniform_combination_probe",
            geometry_case_id=NOT_APPLICABLE_GEOMETRY_CASE_ID,
            maximum_record_attempts=MAXIMUM_ATTEMPTS_PER_UNIT,
            maximum_duration_seconds=MAXIMUM_DURATION_SECONDS,
        ) for ordinal in range(DIRECTIONAL_PROBE_CLUSTER_COUNT))
        return (*operational, *reference, *probes)

    def validate(self) -> None:
        expected = {
            "schema_version": "1.0.0",
            "protocol_id": PROTOCOL_ID,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": RUN_ID,
            "split": "development",
            "future_split_exclusion_roles": FUTURE_SPLIT_EXCLUSION_ROLES,
            "uniform_route_identity": UNIFORM_ROUTE_IDENTITY,
            "lf_detector_identity": LF_DETECTOR_IDENTITY,
            "whitening_asset_fit_identity": "ceg_wm_lf_whitened_score_screening",
            "mixing_coefficients": MIXING_COEFFICIENTS,
            "combination_functions": COMBINATION_FUNCTIONS,
            "combination_weights": COMBINATION_WEIGHTS,
            "wrong_key_roster_size": WRONG_KEY_ROSTER_SIZE,
            "content_relative_l2_numerator": CONTENT_RELATIVE_L2_NUMERATOR,
            "content_relative_l2_denominator": CONTENT_RELATIVE_L2_DENOMINATOR,
            "cross_fit_fold_count": CROSS_FIT_FOLD_COUNT,
            "reference_fit_count_per_probe": REFERENCE_FIT_COUNT_PER_PROBE,
            "operational_unit_count": OPERATIONAL_UNIT_COUNT,
            "reference_fit_cluster_count": REFERENCE_FIT_CLUSTER_COUNT,
            "directional_probe_cluster_count": DIRECTIONAL_PROBE_CLUSTER_COUNT,
            "maximum_total_units": MAXIMUM_TOTAL_UNITS,
            "maximum_attempts_per_unit": MAXIMUM_ATTEMPTS_PER_UNIT,
            "maximum_duration_seconds_per_unit": MAXIMUM_DURATION_SECONDS,
            "attribution_margin_floor": ATTRIBUTION_MARGIN_FLOOR,
            "attribution_success_count_requirement": ATTRIBUTION_SUCCESS_COUNT_REQUIREMENT,
            "directional_improvement_count_requirement": DIRECTIONAL_IMPROVEMENT_COUNT_REQUIREMENT,
            "identity_success_count_maximum_loss": IDENTITY_SUCCESS_COUNT_MAXIMUM_LOSS,
            "passing_outcome": PASSING_OUTCOME,
            "negative_outcome": NEGATIVE_OUTCOME,
            "passing_recommendation": PASSING_RECOMMENDATION,
            "negative_recommendation": NEGATIVE_RECOMMENDATION,
            "passing_request": PASSING_REQUEST,
            "stop_rule": STOP_RULE,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise ContentUniformCombinationDirectionalProtocolError(
                    f"combination protocol {field_name} drifted"
                )
        if len(self.operational_units) != 1:
            raise ContentUniformCombinationDirectionalProtocolError(
                "combination operational count drifted"
            )
        self.operational_units[0].validate()
        if not self.prior_authority_bindings:
            raise ContentUniformCombinationDirectionalProtocolError(
                "combination prior authorities are missing"
            )
        for binding in self.prior_authority_bindings:
            binding.validate()
        for value in (
            self.reference_fit_manifest_file_sha256,
            self.directional_probe_manifest_file_sha256,
            self.source_cluster_deny_list_digest,
            self.whitening_asset_digest,
            self.whitening_asset_fit_protocol_digest,
            self.unit_roster_digest,
        ):
            if _DIGEST.fullmatch(value) is None:
                raise ContentUniformCombinationDirectionalProtocolError(
                    "combination digest is invalid"
                )
        if re.fullmatch(r"[0-9a-f]{40}", self.whitening_asset_fit_producer_revision) is None:
            raise ContentUniformCombinationDirectionalProtocolError(
                "whitening producer revision is invalid"
            )
        if self.whitening_asset_fit_run_id != "ceg_wm_lf_whitening_asset_fit_and_score_screening":
            raise ContentUniformCombinationDirectionalProtocolError(
                "whitening producer run drifted"
            )
        if self.unit_roster_digest != canonical_digest(tuple(asdict(item) for item in self.unit_roster)):
            raise ContentUniformCombinationDirectionalProtocolError(
                "combination unit roster digest drifted"
            )

    def digest(self) -> str:
        self.validate()
        return canonical_digest(asdict(self))


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContentUniformCombinationDirectionalProtocolError(
            "combination authority JSON is unreadable"
        ) from exc
    if type(value) is not dict:
        raise ContentUniformCombinationDirectionalProtocolError(
            "combination authority JSON must be a mapping"
        )
    return value


def load_content_uniform_combination_manifest(
    path: str | Path, *, expected_role: str, expected_count: int
) -> ContentUniformCombinationManifest:
    raw = _load_json(Path(path))
    if set(raw) != {"schema_version", "manifest_id", "role_id", "split", "seed_namespace",
        "source_cluster_namespace", "image_lineage_namespace", "key_family_namespace", "entries"} or type(raw["entries"]) is not list:
        raise ContentUniformCombinationDirectionalProtocolError(
            "combination manifest schema drifted"
        )
    try:
        manifest = ContentUniformCombinationManifest(**{
            **raw,
            "entries": tuple(ContentUniformCombinationManifestEntry(**item) for item in raw["entries"]),
        })
    except (TypeError, KeyError) as exc:
        raise ContentUniformCombinationDirectionalProtocolError(
            "combination manifest entry schema drifted"
        ) from exc
    manifest.validate(expected_role=expected_role, expected_count=expected_count)
    return manifest


def _manifest_axes(manifest: ContentUniformCombinationManifest) -> AuthorityDenyAxes:
    return AuthorityDenyAxes(
        prompt_digests=tuple(sorted(item.prompt_digest for item in manifest.entries)),
        source_cluster_identities=tuple(sorted((manifest.source_cluster_namespace,
            *(item.cluster_identity for item in manifest.entries)))),
        seed_namespaces=(manifest.seed_namespace,),
        generation_seeds=tuple(sorted(item.generation_seed for item in manifest.entries)),
        image_lineage_identities=tuple(sorted((manifest.image_lineage_namespace,
            *(item.image_lineage_identity for item in manifest.entries),
            *(item.image_lineage_digest(role_id=manifest.role_id) for item in manifest.entries)))),
        key_control_identities=(manifest.key_family_namespace,),
    )


def _intersects(left: AuthorityDenyAxes, right: AuthorityDenyAxes) -> bool:
    return any(set(getattr(left, name)) & set(getattr(right, name)) for name in asdict(left))


def load_content_uniform_combination_directional_protocol(
    path: str | Path, *, repository_root: str | Path
) -> tuple[ContentUniformCombinationDirectionalProtocol, ContentUniformCombinationManifest, ContentUniformCombinationManifest]:
    raw = _load_json(Path(path))
    try:
        protocol = ContentUniformCombinationDirectionalProtocol(**{
            **raw,
            "prior_authority_bindings": tuple(PriorDevelopmentManifestBinding(**item) for item in raw["prior_authority_bindings"]),
            "future_split_exclusion_roles": tuple(raw["future_split_exclusion_roles"]),
            "mixing_coefficients": tuple(raw["mixing_coefficients"]),
            "combination_functions": tuple(raw["combination_functions"]),
            "combination_weights": tuple(raw["combination_weights"]),
            "operational_units": tuple(ContentUniformCombinationOperationalUnit(**{
                **item,
                "case_ids": tuple(item["case_ids"]),
                "responsibility_result_digest_keys": tuple(item["responsibility_result_digest_keys"]),
            }) for item in raw["operational_units"]),
        })
    except (TypeError, KeyError) as exc:
        raise ContentUniformCombinationDirectionalProtocolError(
            "combination protocol schema drifted"
        ) from exc
    protocol.validate()
    root = Path(repository_root)
    reference_path = root / protocol.reference_fit_manifest_path
    probe_path = root / protocol.directional_probe_manifest_path
    if sha256(reference_path.read_bytes()).hexdigest() != protocol.reference_fit_manifest_file_sha256 or sha256(probe_path.read_bytes()).hexdigest() != protocol.directional_probe_manifest_file_sha256:
        raise ContentUniformCombinationDirectionalProtocolError(
            "combination manifest file digest drifted"
        )
    reference = load_content_uniform_combination_manifest(reference_path,
        expected_role=REFERENCE_FIT_ROLE, expected_count=REFERENCE_FIT_CLUSTER_COUNT)
    probes = load_content_uniform_combination_manifest(probe_path,
        expected_role=DIRECTIONAL_PROBE_ROLE, expected_count=DIRECTIONAL_PROBE_CLUSTER_COUNT)
    prior_axes = load_authority_deny_axes(protocol.prior_authority_bindings, root)
    expected_deny = canonical_digest({
        "authority_deny_axes": prior_axes.digest_value(),
        "manifest_bindings": tuple(asdict(item) for item in protocol.prior_authority_bindings),
    })
    if protocol.source_cluster_deny_list_digest != expected_deny:
        raise ContentUniformCombinationDirectionalProtocolError(
            "combination source deny-list digest drifted"
        )
    reference_axes = _manifest_axes(reference)
    probe_axes = _manifest_axes(probes)
    if _intersects(prior_axes, reference_axes) or _intersects(prior_axes, probe_axes) or _intersects(reference_axes, probe_axes):
        raise ContentUniformCombinationDirectionalProtocolError(
            "combination reference or probe overlaps a denied axis"
        )
    return protocol, reference, probes


def reference_entries_for_probe(
    probe: ContentUniformCombinationManifestEntry,
    reference_manifest: ContentUniformCombinationManifest,
) -> tuple[ContentUniformCombinationManifestEntry, ...]:
    if type(probe) is not ContentUniformCombinationManifestEntry:
        raise TypeError("combination probe entry exact type required")
    reference_manifest.validate(expected_role=REFERENCE_FIT_ROLE,
        expected_count=REFERENCE_FIT_CLUSTER_COUNT)
    selected = tuple(item for item in reference_manifest.entries if item.fold_index != probe.fold_index)
    if len(selected) != REFERENCE_FIT_COUNT_PER_PROBE:
        raise ContentUniformCombinationDirectionalProtocolError(
            "combination cross-fit reference count drifted"
        )
    return selected


__all__ = [
    "CLAIM_BOUNDARY", "COMBINATION_FUNCTIONS", "COMBINATION_WEIGHTS",
    "CONTENT_RELATIVE_L2_DENOMINATOR", "CONTENT_RELATIVE_L2_NUMERATOR",
    "DIRECTIONAL_PROBE_CLUSTER_COUNT", "MIXING_COEFFICIENTS",
    "ContentUniformCombinationDirectionalProtocol",
    "ContentUniformCombinationDirectionalProtocolError",
    "ContentUniformCombinationManifest", "ContentUniformCombinationManifestEntry",
    "ContentUniformCombinationOperationalUnit",
    "load_content_uniform_combination_directional_protocol",
    "load_content_uniform_combination_manifest", "reference_entries_for_probe",
]
