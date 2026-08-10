"""Frozen development protocol for LF whitened directional validation."""

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
from experiments.protocol.lf_whitened_score_screening import (
    LfWhiteningManifest,
    LfWhiteningManifestEntry,
    load_lf_whitening_manifest,
)


PROTOCOL_ID = "ceg_wm_lf_whitened_directional_validation"
PROTOCOL_VERSION = "1.0.0"
RUN_ID = "ceg_wm_lf_whitened_directional_validation_prepared_feature_execution"
SCIENTIFIC_CLUSTER_COUNT = 32
OPERATIONAL_UNIT_COUNT = 1
MAXIMUM_TOTAL_UNITS = 33
MAXIMUM_ATTEMPTS_PER_UNIT = 2
MAXIMUM_DURATION_SECONDS = 2700
WRONG_KEY_ROSTER_SIZE = 4
PRACTICAL_MARGIN_FLOOR = float.fromhex("0x1.0000000000000p-10")
CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250
MINIMUM_DIRECTIONAL_SUCCESS_COUNT = 28
CONFIDENCE_LEVEL = 0.95
FUTURE_SPLIT_EXCLUSION_ROLES = (
    "candidate_selection",
    "calibration",
    "evaluation",
)
PASSING_MODULE_OUTCOME = "mechanism_signal_observed"
PASSING_CANDIDATE_RECOMMENDATION = "candidate_worth_further_selection"
CLAIM_BOUNDARY = (
    "development_lf_whitened_directional_validation_only_no_threshold_no_fpr_"
    "no_promotion_no_calibration"
)
STOP_RULE = (
    "complete_all_thirty_two_scientific_clusters_then_require_twenty_eight_"
    "registered_primary_null_and_twenty_eight_registered_max_wrong_margins_"
    "strictly_above_the_frozen_floor_with_both_one_sided_exact_ninety_five_"
    "percent_lower_bounds_strictly_above_one_half_and_no_identity_budget_"
    "integrity_or_nonfinite_violation"
)
LF_DIRECTIONAL_COMPONENT_IDS = (
    "key_schedule",
    "lf_carrier",
    "content_embedder",
    "lf_detector",
)
_EXPECTED_COMPONENT_SOURCE_PATHS = (
    "main/shared/key_schedule.py",
    "main/content_chain/lf_carrier.py",
    "main/content_chain/embedder.py",
    "main/content_chain/lf_detector.py",
    "main/content_chain/lf_whitening.py",
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class LfWhitenedDirectionalProtocolError(ValueError):
    """The checked-in LF whitened directional protocol is inconsistent."""


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
class ComponentSourceBindingAuthority:
    component_id: str
    source_role: str
    implementation_path: str
    implementation_symbol: str
    source_sha256: str

    def validate(self) -> None:
        path = PurePosixPath(self.implementation_path)
        if (
            self.component_id not in LF_DIRECTIONAL_COMPONENT_IDS
            or self.source_role
            not in {"component_implementation", "candidate_public_asset_contract"}
            or path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "main"
            or path.name == "__init__.py"
            or path.suffix != ".py"
            or type(self.implementation_symbol) is not str
            or not self.implementation_symbol
            or _DIGEST.fullmatch(self.source_sha256) is None
        ):
            raise LfWhitenedDirectionalProtocolError(
                "component source binding is invalid"
            )


@dataclass(frozen=True, slots=True)
class LfWhitenedDirectionalProtocol:
    schema_version: str
    protocol_id: str
    protocol_version: str
    run_id: str
    split: str
    role_id: str
    candidate_identity: str
    public_callable: str
    manifest_path: str
    manifest_file_sha256: str
    prior_development_manifests: tuple[PriorDevelopmentManifestBinding, ...]
    source_cluster_deny_list_digest: str
    future_split_exclusion_roles: tuple[str, ...]
    registered_key_derivation_identity: str
    wrong_key_control_identity: str
    candidate_specification_sha256: str
    method_review_reference: str
    method_reviewed_revision: str
    whitening_asset_fit_producer_revision: str
    whitening_asset_fit_identity: str
    whitening_asset_fit_run_id: str
    whitening_asset_fit_protocol_digest: str
    whitening_null_fit_manifest_file_sha256: str
    ordered_component_ids: tuple[str, ...]
    component_source_bindings: tuple[ComponentSourceBindingAuthority, ...]
    component_implementation_digest: str
    operational_smoke_prompt: str
    operational_smoke_prompt_digest: str
    operational_smoke_generation_seed: int
    operational_smoke_image_lineage_identity: str
    operational_smoke_image_lineage_digest: str
    operational_unit_count: int
    scientific_cluster_count: int
    maximum_total_units: int
    maximum_attempts_per_unit: int
    maximum_duration_seconds_per_unit: int
    wrong_key_roster_size: int
    practical_margin_floor: float
    content_relative_l2_numerator: int
    content_relative_l2_denominator: int
    minimum_registered_minus_null_success_count: int
    minimum_registered_minus_max_wrong_success_count: int
    confidence_level: float
    confidence_lower_bound_requirement: str
    stop_rule: str
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
                responsibility_id="lf_whitened_detector_runtime_preflight",
                source_cluster_ordinal=0,
                content_branch_id="lf_only",
                geometry_case_id="not_applicable",
                maximum_record_attempts=self.maximum_attempts_per_unit,
                maximum_duration_seconds=self.maximum_duration_seconds_per_unit,
            ),
        )
        scientific = tuple(
            DevelopmentStudyUnit(
                unit_index=self.operational_unit_count + ordinal,
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
        return operational + scientific

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if (
            self.schema_version
            != "ceg_wm_lf_whitened_directional_validation_protocol_v1"
            or self.protocol_id != PROTOCOL_ID
            or self.protocol_version != PROTOCOL_VERSION
            or self.run_id != RUN_ID
            or self.split != "development"
            or self.role_id != "lf_whitened_directional_validation"
            or self.candidate_identity != "lf_null_whitened_matched_score"
            or self.public_callable != "main.lf_null_whitened_matched_detector"
        ):
            raise LfWhitenedDirectionalProtocolError("protocol identity drifted")
        for digest in (
            self.manifest_file_sha256,
            self.source_cluster_deny_list_digest,
            self.candidate_specification_sha256,
            self.whitening_asset_fit_protocol_digest,
            self.whitening_null_fit_manifest_file_sha256,
            self.component_implementation_digest,
        ):
            if _DIGEST.fullmatch(digest) is None:
                raise LfWhitenedDirectionalProtocolError(
                    "protocol digest is invalid"
                )
        if (
            type(self.method_review_reference) is not str
            or not self.method_review_reference
            or _REVISION.fullmatch(self.method_reviewed_revision) is None
            or _REVISION.fullmatch(self.whitening_asset_fit_producer_revision) is None
            or self.whitening_asset_fit_identity
            != "ceg_wm_lf_whitened_score_screening"
            or self.whitening_asset_fit_run_id
            != "ceg_wm_lf_whitening_asset_fit_and_score_screening"
        ):
            raise LfWhitenedDirectionalProtocolError(
                "method or whitening fit authority is invalid"
            )
        if not self.prior_development_manifests:
            raise LfWhitenedDirectionalProtocolError(
                "prior development authority bindings are missing"
            )
        for binding in self.prior_development_manifests:
            binding.validate()
        if self.future_split_exclusion_roles != FUTURE_SPLIT_EXCLUSION_ROLES:
            raise LfWhitenedDirectionalProtocolError(
                "future split exclusion roles drifted"
            )
        if (
            self.registered_key_derivation_identity
            != "lf_whitened_directional_registered_key_derivation"
            or self.wrong_key_control_identity
            != "lf_whitened_directional_four_key_max_control"
        ):
            raise LfWhitenedDirectionalProtocolError("key control identity drifted")
        if self.ordered_component_ids != LF_DIRECTIONAL_COMPONENT_IDS:
            raise LfWhitenedDirectionalProtocolError(
                "component order or membership drifted"
            )
        if tuple(
            binding.implementation_path for binding in self.component_source_bindings
        ) != _EXPECTED_COMPONENT_SOURCE_PATHS:
            raise LfWhitenedDirectionalProtocolError(
                "component source closure drifted"
            )
        for binding in self.component_source_bindings:
            binding.validate()
        closure_payload = {
            "ordered_component_ids": list(self.ordered_component_ids),
            "source_bindings": [
                asdict(binding) for binding in self.component_source_bindings
            ],
        }
        if self.component_implementation_digest != canonical_digest(closure_payload):
            raise LfWhitenedDirectionalProtocolError(
                "component implementation digest drifted"
            )
        if (
            type(self.operational_smoke_prompt) is not str
            or not self.operational_smoke_prompt
            or self.operational_smoke_prompt_digest
            != sha256(self.operational_smoke_prompt.encode("utf-8")).hexdigest()
            or type(self.operational_smoke_generation_seed) is not int
            or self.operational_smoke_generation_seed < 0
            or type(self.operational_smoke_image_lineage_identity) is not str
            or not self.operational_smoke_image_lineage_identity
            or self.operational_smoke_image_lineage_digest
            != canonical_digest(
                {
                    "generation_seed": self.operational_smoke_generation_seed,
                    "image_lineage_identity": (
                        self.operational_smoke_image_lineage_identity
                    ),
                    "responsibility_id": (
                        "lf_whitened_detector_runtime_preflight"
                    ),
                }
            )
        ):
            raise LfWhitenedDirectionalProtocolError(
                "operational smoke identity is invalid"
            )
        if (
            self.operational_unit_count != OPERATIONAL_UNIT_COUNT
            or self.scientific_cluster_count != SCIENTIFIC_CLUSTER_COUNT
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
            or self.minimum_registered_minus_max_wrong_success_count
            != MINIMUM_DIRECTIONAL_SUCCESS_COUNT
            or self.confidence_level != CONFIDENCE_LEVEL
            or self.confidence_lower_bound_requirement
            != "strictly_greater_than_one_half"
            or self.stop_rule != STOP_RULE
            or self.passing_module_outcome != PASSING_MODULE_OUTCOME
            or self.passing_candidate_recommendation
            != PASSING_CANDIDATE_RECOMMENDATION
            or self.claim_boundary != CLAIM_BOUNDARY
        ):
            raise LfWhitenedDirectionalProtocolError(
                "protocol budget or scientific gate drifted"
            )
        if self.unit_roster_digest != canonical_digest(
            tuple(asdict(unit) for unit in self.unit_roster)
        ):
            raise LfWhitenedDirectionalProtocolError(
                "unit roster digest drifted"
            )


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LfWhitenedDirectionalProtocolError(
            "checked-in directional JSON is unreadable"
        ) from exc
    if type(value) is not dict:
        raise LfWhitenedDirectionalProtocolError(
            "checked-in directional JSON must be a mapping"
        )
    return value


def load_lf_whitened_directional_validation_protocol(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[LfWhitenedDirectionalProtocol, LfWhiteningManifest]:
    raw = _load_json(Path(path))
    try:
        protocol = LfWhitenedDirectionalProtocol(
            **{
                **raw,
                "prior_development_manifests": tuple(
                    PriorDevelopmentManifestBinding(**item)
                    for item in raw["prior_development_manifests"]
                ),
                "future_split_exclusion_roles": tuple(
                    raw["future_split_exclusion_roles"]
                ),
                "ordered_component_ids": tuple(raw["ordered_component_ids"]),
                "component_source_bindings": tuple(
                    ComponentSourceBindingAuthority(**item)
                    for item in raw["component_source_bindings"]
                ),
            }
        )
    except (KeyError, TypeError) as exc:
        raise LfWhitenedDirectionalProtocolError(
            "directional protocol schema is invalid"
        ) from exc
    protocol.validate()
    root = Path(repository_root)
    manifest_path = root / protocol.manifest_path
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise LfWhitenedDirectionalProtocolError(
            "directional manifest is missing or unreadable"
        ) from exc
    if sha256(manifest_bytes).hexdigest() != protocol.manifest_file_sha256:
        raise LfWhitenedDirectionalProtocolError(
            "directional manifest file digest drifted"
        )
    manifest = load_lf_whitening_manifest(
        manifest_path,
        expected_role="lf_whitened_directional_validation",
        count=SCIENTIFIC_CLUSTER_COUNT,
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
        raise LfWhitenedDirectionalProtocolError(
            "source cluster deny-list digest drifted"
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
        key_control_identities=(
            manifest.key_family_namespace,
            protocol.registered_key_derivation_identity,
            protocol.wrong_key_control_identity,
        ),
    )
    operational_prompt_digest = protocol.operational_smoke_prompt_digest
    operational_values = {
        "prompt_digests": {operational_prompt_digest},
        "generation_seeds": {protocol.operational_smoke_generation_seed},
        "image_lineage_identities": {
            protocol.operational_smoke_image_lineage_identity,
            protocol.operational_smoke_image_lineage_digest,
        },
    }
    intersections = {
        name: sorted(set(getattr(deny_axes, name)) & set(getattr(new_axes, name)))
        for name in asdict(deny_axes)
    }
    if any(intersections.values()) or any(
        values & set(getattr(deny_axes, name))
        for name, values in operational_values.items()
    ):
        raise LfWhitenedDirectionalProtocolError(
            "directional manifest overlaps a prior authority axis"
        )
    if (
        operational_prompt_digest
        in {item.prompt_digest for item in manifest.entries}
        or protocol.operational_smoke_generation_seed
        in {item.generation_seed for item in manifest.entries}
        or protocol.operational_smoke_image_lineage_digest
        in {item.image_lineage_digest for item in manifest.entries}
    ):
        raise LfWhitenedDirectionalProtocolError(
            "operational smoke overlaps scientific manifest"
        )
    return protocol, manifest


def derive_lf_whitened_directional_analysis_identity(
    entry: LfWhiteningManifestEntry,
    manifest: LfWhiteningManifest,
    *,
    key_family_digest: str,
) -> AnalysisUnitIdentity:
    if _DIGEST.fullmatch(key_family_digest) is None:
        raise LfWhitenedDirectionalProtocolError("key family digest is invalid")
    return AnalysisUnitIdentity(
        unit_id=f"lf_whitened_directional_cluster_{entry.cluster_ordinal:02d}",
        case_id="paired_clean_lf_whitened_directional_validation",
        source_cluster_id=derive_source_cluster_id(
            prompt_digest=entry.prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=entry.image_lineage_digest,
            registered_key_family_digest=key_family_digest,
        ),
        prompt_digest=entry.prompt_digest,
        generation_seed=entry.generation_seed,
        image_lineage_digest=entry.image_lineage_digest,
        registered_key_family_digest=key_family_digest,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "FUTURE_SPLIT_EXCLUSION_ROLES",
    "LF_DIRECTIONAL_COMPONENT_IDS",
    "PRACTICAL_MARGIN_FLOOR",
    "RUN_ID",
    "SCIENTIFIC_CLUSTER_COUNT",
    "ComponentSourceBindingAuthority",
    "LfWhitenedDirectionalProtocol",
    "LfWhitenedDirectionalProtocolError",
    "canonical_digest",
    "derive_lf_whitened_directional_analysis_identity",
    "load_lf_whitened_directional_validation_protocol",
]
