"""hf_only_reference_validation reference specification and deterministic frozen split materialization."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
    INTERNAL_VALIDATION_PROTOCOL_ID,
    INTERNAL_VALIDATION_PROTOCOL_VERSION,
    SplitAssignment,
    derive_source_cluster_id,
)


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
HF_ONLY_REFERENCE_SCHEMA_VERSION = "ceg_wm_hf_only_reference_run_v1"
HF_ONLY_REFERENCE_PROMPT_ROSTER_SCHEMA_VERSION = "ceg_wm_hf_only_reference_prompt_roster_v1"
HF_ONLY_REFERENCE_COMPACT_MANIFEST_SCHEMA_VERSION = "ceg_wm_hf_only_reference_compact_manifest_v1"
HF_ONLY_REFERENCE_GATE_ID = "hf_reference_candidate_frozen"
HF_ONLY_REFERENCE_RESULT_GATE_ID = "hf_detector_reference_gate_passed"
HF_ONLY_REFERENCE_DETECTOR_MODE = "hf_only"
HF_ONLY_REFERENCE_SPLITS = ("content_threshold_fit", "untouched_confirmation")
HF_ONLY_REFERENCE_DATASET_ID = "nateraw/parti-prompts"
HF_ONLY_REFERENCE_DATASET_REVISION = "944b156abfdad7627c3221b5ec4f6a6fb060a197"
HF_ONLY_REFERENCE_DATASET_FILE = "PartiPrompts.tsv"
HF_ONLY_REFERENCE_DATASET_SHA256 = (
    "fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb"
)
HF_ONLY_REFERENCE_DATASET_LICENSE = "Apache-2.0"
HF_ONLY_REFERENCE_PROMPT_COUNT = 1632
HF_ONLY_REFERENCE_CATEGORY_COUNT = 12
HF_ONLY_REFERENCE_PROMPTS_PER_SPLIT = 816
HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT = 4096
HF_ONLY_REFERENCE_MANIFEST_IDENTITIES = {
    "content_threshold_fit": (
        "ceg_wm_hf_only_reference_content_threshold_fit_manifest_v1",
        "1",
        "1150c89cc11db5d6bbe9cb17f032c0a5934f9189db0feb4e5de3998e5302719b",
    ),
    "untouched_confirmation": (
        "ceg_wm_hf_only_reference_untouched_confirmation_manifest_v1",
        "1",
        "35a23a03927fd1a48a54926c33ba9f32f63f58a3d4707dab17d0cec6ce8528cf",
    ),
}
HF_ONLY_REFERENCE_PRIMARY_NULL_TARGET_FPR = 0.001
HF_ONLY_REFERENCE_CONFIDENCE_LEVEL = 0.95
HF_ONLY_REFERENCE_THRESHOLD_TAIL_FAILURE_PROBABILITY = (
    1.0 - HF_ONLY_REFERENCE_PRIMARY_NULL_TARGET_FPR
) ** HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT
HF_ONLY_REFERENCE_ZERO_FAILURE_CP_UPPER_95 = 1.0 - (
    1.0 - HF_ONLY_REFERENCE_CONFIDENCE_LEVEL
) ** (1.0 / HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT)
HF_ONLY_REFERENCE_THRESHOLD_RULE = "float64_nextafter_above_fit_primary_null_maximum"
HF_ONLY_REFERENCE_THRESHOLD_COMPARISON = "score_greater_than_or_equal_to_tau"
HF_ONLY_REFERENCE_CLAIM_BOUNDARY = (
    "protocol_budget_and_cpu_constraints_only_no_gpu_run_no_scientific_result"
)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    raw: dict[str, Any],
    expected: Iterable[str],
    *,
    context: str,
) -> None:
    expected_set = set(expected)
    actual_set = set(raw)
    if actual_set != expected_set:
        missing = ",".join(sorted(expected_set - actual_set))
        extra = ",".join(sorted(actual_set - expected_set))
        raise ValueError(f"{context}_keys_invalid:missing={missing}:extra={extra}")


def _require_nonempty_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name}_missing_or_invalid")
    return value


def _require_digest(value: object, field_name: str) -> str:
    if type(value) is not str or not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name}_invalid")
    return value


def _require_revision(value: object, field_name: str) -> str:
    if type(value) is not str or not _REVISION_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name}_invalid")
    return value


@dataclass(frozen=True)
class FrozenPromptRosterEntry:
    source_row: int
    prompt_text: str
    prompt_digest: str
    category: str
    challenge: str

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        if (
            type(self.source_row) is not int
            or isinstance(self.source_row, bool)
            or self.source_row < 1
        ):
            violations.append("source_row_invalid")
        if not _DIGEST_PATTERN.fullmatch(self.prompt_digest):
            violations.append("prompt_digest_invalid")
        elif hashlib.sha256(self.prompt_text.encode("utf-8")).hexdigest() != self.prompt_digest:
            violations.append("prompt_digest_text_mismatch")
        if not self.prompt_text:
            violations.append("prompt_text_missing")
        if not self.category.strip():
            violations.append("category_missing")
        if not self.challenge.strip():
            violations.append("challenge_missing")
        return tuple(violations)


@dataclass(frozen=True)
class FrozenPromptRoster:
    schema_version: str
    dataset_id: str
    dataset_revision: str
    dataset_file: str
    dataset_file_sha256: str
    license: str
    prompt_text_normalization: str
    expected_prompt_count: int
    expected_duplicate_prompt_count: int
    expected_category_count: int
    rows_digest: str
    rows: tuple[FrozenPromptRosterEntry, ...]

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        expected_scalars = (
            ("schema_version", HF_ONLY_REFERENCE_PROMPT_ROSTER_SCHEMA_VERSION),
            ("dataset_id", HF_ONLY_REFERENCE_DATASET_ID),
            ("dataset_revision", HF_ONLY_REFERENCE_DATASET_REVISION),
            ("dataset_file", HF_ONLY_REFERENCE_DATASET_FILE),
            ("dataset_file_sha256", HF_ONLY_REFERENCE_DATASET_SHA256),
            ("license", HF_ONLY_REFERENCE_DATASET_LICENSE),
            ("prompt_text_normalization", "exact_utf8_decoded_tsv_prompt_field"),
            ("expected_prompt_count", HF_ONLY_REFERENCE_PROMPT_COUNT),
            ("expected_duplicate_prompt_count", 0),
            ("expected_category_count", HF_ONLY_REFERENCE_CATEGORY_COUNT),
        )
        for name, expected in expected_scalars:
            if getattr(self, name) != expected:
                violations.append(f"{name}_frozen_value_mismatch")
        if len(self.rows) != HF_ONLY_REFERENCE_PROMPT_COUNT:
            violations.append("prompt_count_mismatch")
        source_rows = tuple(row.source_row for row in self.rows)
        if source_rows != tuple(range(1, HF_ONLY_REFERENCE_PROMPT_COUNT + 1)):
            violations.append("source_row_order_or_coverage_mismatch")
        prompt_digests = tuple(row.prompt_digest for row in self.rows)
        if len(set(prompt_digests)) != len(prompt_digests):
            violations.append("prompt_digest_duplicate")
        if len({row.category for row in self.rows}) != HF_ONLY_REFERENCE_CATEGORY_COUNT:
            violations.append("category_count_mismatch")
        for row in self.rows:
            violations.extend(row.validate())
        if self.rows_digest != _canonical_digest(
            [asdict(row) for row in self.rows]
        ):
            violations.append("rows_digest_mismatch")
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True)
class CompactHfOnlyReferenceSplitManifest:
    schema_version: str
    protocol_id: str
    protocol_version: str
    manifest_id: str
    manifest_revision: str
    split: str
    roster_rows_digest: str
    source_row_selector: str
    expected_prompt_count: int
    seed_namespace: str
    seed_derivation: str
    base_replicates_per_prompt: int
    extra_replicates_for_first_selected_prompts: int
    image_lineage_derivation: str
    registered_key_family_digest: str
    expected_assignment_count: int
    expected_materialized_manifest_digest: str

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        expected_selector = {
            "content_threshold_fit": (
                "stratified_alternating_v1_fit_gets_first_26_odd_strata_extras"
            ),
            "untouched_confirmation": (
                "stratified_alternating_v1_confirmation_gets_last_26_odd_strata_extras"
            ),
        }
        expected_seed_namespace = {
            "content_threshold_fit": "ceg_wm_hf_only_threshold_fit_seed_v1",
            "untouched_confirmation": "ceg_wm_hf_only_reference_confirmation_seed_v1",
        }
        expected_scalars = (
            ("schema_version", HF_ONLY_REFERENCE_COMPACT_MANIFEST_SCHEMA_VERSION),
            ("protocol_id", INTERNAL_VALIDATION_PROTOCOL_ID),
            ("protocol_version", INTERNAL_VALIDATION_PROTOCOL_VERSION),
            ("expected_prompt_count", HF_ONLY_REFERENCE_PROMPTS_PER_SPLIT),
            ("seed_derivation", "sha256_canonical_json_low_63_bits"),
            ("base_replicates_per_prompt", 5),
            ("extra_replicates_for_first_selected_prompts", 16),
            (
                "image_lineage_derivation",
                "sha256_canonical_json_dataset_row_category_challenge_seed_replication",
            ),
            ("expected_assignment_count", HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT),
        )
        for name, expected in expected_scalars:
            if getattr(self, name) != expected:
                violations.append(f"{name}_frozen_value_mismatch")
        if self.split not in HF_ONLY_REFERENCE_SPLITS:
            violations.append("split_invalid")
        else:
            if self.source_row_selector != expected_selector[self.split]:
                violations.append("source_row_selector_invalid")
            if self.seed_namespace != expected_seed_namespace[self.split]:
                violations.append("seed_namespace_invalid")
            manifest_id, revision, materialized_digest = (
                HF_ONLY_REFERENCE_MANIFEST_IDENTITIES[self.split]
            )
            if self.manifest_id != manifest_id:
                violations.append("manifest_id_frozen_value_mismatch")
            if self.manifest_revision != revision:
                violations.append("manifest_revision_frozen_value_mismatch")
            if self.expected_materialized_manifest_digest != materialized_digest:
                violations.append(
                    "expected_materialized_manifest_digest_frozen_value_mismatch"
                )
        for name in (
            "roster_rows_digest",
            "registered_key_family_digest",
            "expected_materialized_manifest_digest",
        ):
            if not _DIGEST_PATTERN.fullmatch(getattr(self, name)):
                violations.append(f"{name}_invalid")
        return tuple(dict.fromkeys(violations))


@dataclass(frozen=True)
class HfReferenceCandidateFreeze:
    gate_id: str
    detector_mode: str
    candidate_binding_digest: str
    freeze_semantics: str


@dataclass(frozen=True)
class HfOnlyReferenceSpecification:
    raw: dict[str, Any]

    def digest(self) -> str:
        return _canonical_digest(self.raw)

    def validate(self) -> tuple[str, ...]:
        violations: list[str] = []
        try:
            _validate_hf_only_reference_specification_raw(self.raw)
        except ValueError as error:
            violations.append(str(error))
        return tuple(violations)

    def freeze_reference_candidate(self) -> HfReferenceCandidateFreeze:
        violations = self.validate()
        if violations:
            raise ValueError(", ".join(violations))
        binding = self.raw["candidate_binding"]
        return HfReferenceCandidateFreeze(
            gate_id=HF_ONLY_REFERENCE_GATE_ID,
            detector_mode=HF_ONLY_REFERENCE_DETECTOR_MODE,
            candidate_binding_digest=binding["candidate_binding_digest"],
            freeze_semantics="identity_and_runtime_qualification_fact_only_not_result_gate",
        )


@dataclass(frozen=True)
class HfOnlyReferenceBundle:
    specification: HfOnlyReferenceSpecification
    roster: FrozenPromptRoster
    compact_manifests: tuple[CompactHfOnlyReferenceSplitManifest, ...]
    materialized_manifests: tuple[FrozenSplitManifest, ...]


def _validate_hf_only_reference_specification_raw(raw: dict[str, Any]) -> None:
    _require_exact_keys(
        raw,
        (
            "schema_version",
            "protocol_id",
            "protocol_version",
            "phase",
            "detector_mode",
            "prerequisite_gate_id",
            "result_gate_id",
            "result_gate_prerequisite_forbidden",
            "claim_boundary",
            "dataset",
            "candidate_binding",
            "method_execution",
            "key_controls",
            "split_manifests",
            "run_phases",
            "execution_budget",
            "metric_plan",
            "statistics",
            "forbidden_scope",
        ),
        context="hf_only_reference_specification",
    )
    expected_scalars = (
        ("schema_version", HF_ONLY_REFERENCE_SCHEMA_VERSION),
        ("protocol_id", INTERNAL_VALIDATION_PROTOCOL_ID),
        ("protocol_version", INTERNAL_VALIDATION_PROTOCOL_VERSION),
        ("phase", "hf_only_reference_protocol_and_budget_freeze_only"),
        ("detector_mode", HF_ONLY_REFERENCE_DETECTOR_MODE),
        ("prerequisite_gate_id", HF_ONLY_REFERENCE_GATE_ID),
        ("result_gate_id", HF_ONLY_REFERENCE_RESULT_GATE_ID),
        ("result_gate_prerequisite_forbidden", True),
        ("claim_boundary", HF_ONLY_REFERENCE_CLAIM_BOUNDARY),
    )
    for name, expected in expected_scalars:
        if raw[name] != expected:
            raise ValueError(f"{name}_frozen_value_mismatch")
    _validate_dataset(raw["dataset"])
    _validate_candidate_binding(raw["candidate_binding"])
    _validate_method_execution(raw["method_execution"])
    _validate_key_controls(raw["key_controls"])
    _validate_split_manifests(raw["split_manifests"])
    _validate_run_phases(raw["run_phases"])
    _validate_execution_budget(raw["execution_budget"])
    _validate_metric_plan(raw["metric_plan"])
    _validate_statistics(raw["statistics"])
    forbidden_scope = raw["forbidden_scope"]
    if forbidden_scope != [
        "lf",
        "routing",
        "geometry",
        "combined_detector",
        "post_hf_only_reference_validation",
        "held_out_evaluation",
        "external_baseline",
        "formal_evaluation",
        "gpu_execution",
        "scientific_result_claim",
    ]:
        raise ValueError("forbidden_scope_invalid")


def _validate_dataset(raw: object) -> None:
    if type(raw) is not dict:
        raise ValueError("dataset_invalid")
    _require_exact_keys(
        raw,
        (
            "dataset_id",
            "revision",
            "file",
            "file_sha256",
            "license",
            "prompt_count",
            "duplicate_prompt_count",
            "category_count",
            "roster_path",
            "roster_file_sha256",
            "roster_rows_digest",
            "runtime_network_access",
            "dataset_snapshot_path",
        ),
        context="dataset",
    )
    expected = {
        "dataset_id": HF_ONLY_REFERENCE_DATASET_ID,
        "revision": HF_ONLY_REFERENCE_DATASET_REVISION,
        "file": HF_ONLY_REFERENCE_DATASET_FILE,
        "file_sha256": HF_ONLY_REFERENCE_DATASET_SHA256,
        "license": HF_ONLY_REFERENCE_DATASET_LICENSE,
        "prompt_count": HF_ONLY_REFERENCE_PROMPT_COUNT,
        "duplicate_prompt_count": 0,
        "category_count": HF_ONLY_REFERENCE_CATEGORY_COUNT,
        "roster_path": "configs/experiments/hf_only_reference_prompt_roster.json",
        "dataset_snapshot_path": (
            "configs/experiments/assets/"
            "parti_prompts_dataset_snapshot.txt"
        ),
        "runtime_network_access": "forbidden_use_frozen_roster_only",
    }
    for name, value in expected.items():
        if raw[name] != value:
            raise ValueError(f"dataset_{name}_frozen_value_mismatch")
    _require_digest(raw["roster_file_sha256"], "roster_file_sha256")
    _require_digest(raw["roster_rows_digest"], "roster_rows_digest")


def _validate_candidate_binding(raw: object) -> None:
    if type(raw) is not dict:
        raise ValueError("candidate_binding_invalid")
    _require_exact_keys(
        raw,
        (
            "method_candidate_id",
            "key_schedule_candidate_id",
            "runtime_candidate_id",
            "candidate_specification_path",
            "candidate_specification_sha256",
            "method_reviewed_revision",
            "protocol_authorization_parent_revision",
            "method_source_files",
            "method_source_bundle_digest",
            "formal_method_adapter_config_path",
            "formal_method_adapter_config_digest",
            "hf_formal_component_identities",
            "runtime_config_path",
            "runtime_config_sha256",
            "runtime_qualification",
            "required_execution_package_bindings",
            "candidate_binding_digest",
        ),
        context="candidate_binding",
    )
    expected = {
        "method_candidate_id": "hf_sparse_tail",
        "key_schedule_candidate_id": "key_schedule_sha256_counter",
        "runtime_candidate_id": "runtime_sd35_flowmatch",
        "candidate_specification_path": "docs/design/candidate_specifications.md",
        "candidate_specification_sha256": (
            "fc241dc0e85089e1d71635765527b44a9cd453ac7b221ac2e9dc177088108337"
        ),
        "method_reviewed_revision": "ee512b31917fdf31d76e7237d3bba2b9c8ec4c64",
        "protocol_authorization_parent_revision": (
            "8a557c076b79788405fe16f99e4acb6f43e89681"
        ),
        "runtime_config_path": "configs/runtime/runtime_sd35_flowmatch.json",
        "runtime_config_sha256": (
            "a097e826551dfa74601defef406d027cd6428946894f85012d2c65268b0bff1f"
        ),
    }
    for name, value in expected.items():
        if raw[name] != value:
            raise ValueError(f"candidate_binding_{name}_mismatch")
    source_files = raw["method_source_files"]
    if type(source_files) is not list or not source_files:
        raise ValueError("method_source_files_invalid")
    expected_paths = (
        "main/__init__.py",
        "main/shared/__init__.py",
        "main/shared/key_schedule.py",
        "main/shared/rgb8.py",
        "main/content_chain/__init__.py",
        "main/content_chain/routing.py",
        "main/content_chain/lf_carrier.py",
        "main/content_chain/hf_carrier.py",
        "main/content_chain/embedder.py",
        "main/content_chain/lf_detector.py",
        "main/content_chain/hf_detector.py",
        "main/content_chain/detector.py",
        "experiments/methods/ceg_wm.py",
        "runtime/__init__.py",
        "runtime/adapter.py",
        "runtime/backend.py",
        "runtime/configuration.py",
        "runtime/content_write.py",
        "runtime/qk_observation.py",
        "runtime/sd35_backend.py",
    )
    if tuple(entry.get("path") for entry in source_files) != expected_paths:
        raise ValueError("method_source_file_paths_invalid")
    for entry in source_files:
        if type(entry) is not dict or set(entry) != {"path", "sha256"}:
            raise ValueError("method_source_file_entry_invalid")
        _require_digest(entry["sha256"], "method_source_file_sha256")
    expected_bundle = _canonical_digest(source_files)
    if raw["method_source_bundle_digest"] != expected_bundle:
        raise ValueError("method_source_bundle_digest_mismatch")
    if (
        raw["formal_method_adapter_config_path"]
        != "configs/experiments/internal_execution_components.json"
        or raw["formal_method_adapter_config_digest"]
        != "e539b1237e654574242976304f266cb0362f950394358f63f9e3ab96393061b7"
    ):
        raise ValueError("formal_method_adapter_binding_mismatch")
    if raw["hf_formal_component_identities"] != [
        {
            "responsibility": "hf_carrier",
            "public_callable": "main.hf_carrier",
            "result_identity_field": "carrier_config_digest",
        },
        {
            "responsibility": "content_embedder",
            "public_callable": "main.content_embedder",
            "result_identity_field": "embedding_result_identity",
        },
        {
            "responsibility": "hf_detector",
            "public_callable": "main.hf_detector",
            "result_identity_field": "detector_identity",
        },
        {
            "responsibility": "content_detector",
            "public_callable": "main.content_detector",
            "result_identity_field": "content_config_digest",
        },
        {
            "responsibility": "runtime_hf_content_materialization",
            "public_callable": "runtime.Sd35RuntimeAdapter.execute_content_write_and_vae",
            "result_identity_field": "runtime_config_digest_and_materialization_integrity",
        },
    ]:
        raise ValueError("hf_formal_component_identities_invalid")
    if raw["required_execution_package_bindings"] != {
        "required_fields": [
            "exact_method_code_revision",
            "adapter_config_digest",
            "hf_template_digest",
            "hf_detector_config_digest",
            "formal_content_detector_config_digest",
            "candidate_config_digest",
            "execution_config_digest",
        ],
        "derivation_authority": (
            "formal_execution_package_from_frozen_manifest_and_registered_key"
        ),
        "absence_semantics": "hf_only_threshold_fit_gpu_execution_preflight_fail_closed",
        "hf_only_reference_protocol_status": "not_yet_materialized_no_result_claim",
    }:
        raise ValueError("required_execution_package_bindings_invalid")
    qualification = raw["runtime_qualification"]
    if type(qualification) is not dict:
        raise ValueError("runtime_qualification_invalid")
    _require_exact_keys(
        qualification,
        (
            "candidate_revision",
            "execution_package_sha256",
            "profile",
            "status",
            "run_id",
            "result_zip_sha256",
            "evidence_semantics",
        ),
        context="runtime_qualification",
    )
    expected_qualification = {
        "candidate_revision": "8b2344756c4c247906ff0d4eab68e46a773e13f5",
        "execution_package_sha256": (
            "8290abeed79931eb7208ac9ca280f1ea401f4725abfead35f12617a0ef54dd38"
        ),
        "profile": "qualification",
        "status": "passed",
        "run_id": "20260729T110628Z",
        "result_zip_sha256": (
            "d9b7d91d41cc963098c077268445ad80e9994c809227ca2f68615a37ac93ac37"
        ),
        "evidence_semantics": (
            "runtime_executability_and_identity_only_not_fixed_fpr_or_scientific_effect"
        ),
    }
    if qualification != expected_qualification:
        raise ValueError("runtime_qualification_frozen_identity_mismatch")
    binding_payload = {
        key: value for key, value in raw.items() if key != "candidate_binding_digest"
    }
    if raw["candidate_binding_digest"] != _canonical_digest(binding_payload):
        raise ValueError("candidate_binding_digest_mismatch")


def _validate_method_execution(raw: object) -> None:
    if type(raw) is not dict:
        raise ValueError("method_execution_invalid")
    expected = {
        "write_identity": "hf_sparse_tail_registered_key_write",
        "detect_identity": "hf_sparse_tail_direct_registered_or_control_key_score",
        "latent_dtype": "float16",
        "template_dtype": "float32",
        "score_dtype": "float32",
        "image_height": 512,
        "image_width": 512,
        "inference_steps": 20,
        "guidance_scale": 4.5,
        "callback_index": 18,
        "content_relative_l2_nominal": "3/250",
        "content_relative_l2_limit": "3/250",
        "vae_encode_protocol": "posterior_mode_sub_shift_factor_mul_scaling_factor",
        "fixed_fpr_target": HF_ONLY_REFERENCE_PRIMARY_NULL_TARGET_FPR,
    }
    _require_exact_keys(raw, expected, context="method_execution")
    if raw != expected:
        raise ValueError("method_execution_frozen_value_mismatch")


def _validate_key_controls(raw: object) -> None:
    if type(raw) is not dict:
        raise ValueError("key_controls_invalid")
    _require_exact_keys(
        raw,
        (
            "registered_key_family_id",
            "registered_key_family_digest",
            "registered_key_secret_persistence",
            "wrong_key_roster_id",
            "wrong_key_derivation",
            "wrong_key_count_per_source_cluster",
            "public_noise_identity",
            "public_noise_secret_independence",
            "primary_null_identity",
            "wrong_key_null_identity",
            "nulls_must_remain_separate",
        ),
        context="key_controls",
    )
    expected_except_digest = {
        "registered_key_family_id": "ceg_wm_hf_only_reference_registered_family_v1",
        "registered_key_secret_persistence": "forbidden",
        "wrong_key_roster_id": "ceg_wm_hf_only_reference_wrong_key_roster_v1",
        "wrong_key_derivation": "key_schedule_sha256_counter_wrong_key_domain",
        "wrong_key_count_per_source_cluster": 1,
        "public_noise_identity": "key_schedule_sha256_counter_public_noise",
        "public_noise_secret_independence": "required",
        "primary_null_identity": "unwatermarked_image_with_registered_detection_key",
        "wrong_key_null_identity": "watermarked_image_with_preregistered_wrong_detection_key",
        "nulls_must_remain_separate": True,
    }
    for name, expected in expected_except_digest.items():
        if raw[name] != expected:
            raise ValueError(f"key_controls_{name}_mismatch")
    expected_family_digest = _canonical_digest(
        {
            "key_schedule_candidate_id": "key_schedule_sha256_counter",
            "registered_key_family_id": raw["registered_key_family_id"],
            "wrong_key_roster_id": raw["wrong_key_roster_id"],
        }
    )
    if raw["registered_key_family_digest"] != expected_family_digest:
        raise ValueError("registered_key_family_digest_mismatch")


def _validate_split_manifests(raw: object) -> None:
    if type(raw) is not dict or tuple(raw) != HF_ONLY_REFERENCE_SPLITS:
        raise ValueError("split_manifests_identity_or_order_invalid")
    expected = {
        "content_threshold_fit": {
            "path": "configs/experiments/hf_only_content_threshold_fit_manifest.json",
            "file_sha256": (
                "554d1f488dadc433dba57a2b4e9435a8f55eeecef77929c76161076753b2886a"
            ),
            "materialized_manifest_digest": (
                "1150c89cc11db5d6bbe9cb17f032c0a5934f9189db0feb4e5de3998e5302719b"
            ),
        },
        "untouched_confirmation": {
            "path": (
                "configs/experiments/hf_only_untouched_confirmation_manifest.json"
            ),
            "file_sha256": (
                "43d7d6d81f827348655f6bd0caa4b808f401cf20b1153923d14c5cc86270e6cd"
            ),
            "materialized_manifest_digest": (
                "35a23a03927fd1a48a54926c33ba9f32f63f58a3d4707dab17d0cec6ce8528cf"
            ),
        },
    }
    if raw != expected:
        raise ValueError("split_manifest_bindings_invalid")


def _validate_run_phases(raw: object) -> None:
    if type(raw) is not dict or tuple(raw) != (
        "threshold_fit",
        "untouched_confirmation",
    ):
        raise ValueError("run_phases_identity_or_order_invalid")
    if raw["threshold_fit"] != {
        "run_phase_id": "hf_only_threshold_fit_v1",
        "accessible_split": "content_threshold_fit",
        "package_scope": "threshold_fit_only",
        "prerequisite_gates": [HF_ONLY_REFERENCE_GATE_ID],
        "forbidden_split_access": ["untouched_confirmation"],
        "same_run_confirmation": "forbidden",
        "output_artifact_schema": "ceg_wm_hf_only_tau_frozen_v1",
        "output_gate_id": "hf_only_tau_frozen",
        "output_required_fields": [
            "protocol_id",
            "protocol_version",
            "protocol_digest",
            "fit_manifest_digest",
            "candidate_binding_digest",
            "threshold_rule",
            "tau_float64_hex",
            "fit_record_collection_sha256",
            "artifact_sha256",
            "independent_review_revision",
            "independent_review_decision",
        ],
        "artifact_semantics": "frozen_tau_fact_only_not_hf_scientific_result",
    }:
        raise ValueError("threshold_fit_run_phase_invalid")
    if raw["untouched_confirmation"] != {
        "run_phase_id": "hf_only_reference_untouched_confirmation_v1",
        "accessible_split": "untouched_confirmation",
        "package_scope": "confirmation_only",
        "prerequisite_gates": [
            "candidate_selection_frozen",
            "hf_only_tau_frozen",
        ],
        "package_authorization_status": (
            "blocked_until_independently_reviewed_tau_artifact_bound"
        ),
        "required_tau_binding_fields": [
            "tau_float64_hex",
            "artifact_sha256",
            "independent_review_revision",
            "independent_review_decision",
        ],
        "required_review_decision": "APPROVE",
        "tau_refit": "forbidden",
        "same_run_as_threshold_fit": "forbidden",
    }:
        raise ValueError("untouched_confirmation_run_phase_invalid")


def _validate_execution_budget(raw: object) -> None:
    if type(raw) is not dict:
        raise ValueError("execution_budget_invalid")
    _require_exact_keys(
        raw,
        (
            "threshold_fit",
            "untouched_confirmation",
            "failure_and_denominator_policy",
            "shard_and_resource_boundary",
        ),
        context="execution_budget",
    )
    if raw["threshold_fit"] != {
        "source_clusters": 4096,
        "clean_image_generations": 4096,
        "registered_watermarked_image_generations": 0,
        "hf_write_materializations": 0,
        "registered_key_detection_calls": 4096,
        "wrong_key_detection_calls": 0,
        "paired_quality_evaluations": 0,
        "public_noise_identity_checks": 4096,
        "total_detection_calls": 4096,
    }:
        raise ValueError("threshold_fit_execution_budget_invalid")
    if raw["untouched_confirmation"] != {
        "source_clusters": 4096,
        "clean_image_generations": 4096,
        "registered_watermarked_image_generations": 4096,
        "hf_write_materializations": 4096,
        "registered_key_detection_calls": 8192,
        "wrong_key_detection_calls": 4096,
        "paired_quality_evaluations": 4096,
        "public_noise_identity_checks": 4096,
        "total_detection_calls": 12288,
    }:
        raise ValueError("untouched_confirmation_execution_budget_invalid")
    if raw["failure_and_denominator_policy"] != {
        "planned_source_clusters_remain_denominator": True,
        "failed_excluded_and_retry_records_retained": True,
        "missing_required_outcome_fails_gate": True,
        "early_stopping": "forbidden",
        "maximum_record_attempts_per_unit": 3,
        "post_result_budget_expansion": "forbidden_without_new_protocol_revision",
    }:
        raise ValueError("failure_and_denominator_policy_invalid")
    if raw["shard_and_resource_boundary"] != {
        "shard_count_per_phase": 16,
        "source_clusters_per_shard": 256,
        "shard_assignment": "ascending_materialized_assignment_index",
        "cross_phase_shard_reuse": "forbidden",
        "gpu_execution_authorized_by_hf_only_reference_protocol": False,
        "resource_class_and_walltime": (
            "must_be_frozen_in_separately_authorized_execution_package"
        ),
        "resource_expansion": "requires_new_protocol_revision_and_authorization",
    }:
        raise ValueError("shard_and_resource_boundary_invalid")


def _validate_metric_plan(raw: object) -> None:
    expected = {
        "metric_schema_version": "ceg_wm_hf_only_reference_metric_plan_v1",
        "implementation_status": "formula_identities_frozen_metrics_not_yet_implemented",
        "held_out_evaluation_access": "forbidden",
        "metric_split_bindings": [
            {
                "metric_id": "hf_only_reference_tau_fit",
                "allowed_splits": ["content_threshold_fit"],
            },
            {
                "metric_id": "hf_only_reference_primary_null_fixed_fpr",
                "allowed_splits": ["untouched_confirmation"],
            },
            {
                "metric_id": "hf_only_reference_registered_tpr",
                "allowed_splits": ["untouched_confirmation"],
            },
            {
                "metric_id": "hf_only_reference_wrong_key_false_accept",
                "allowed_splits": ["untouched_confirmation"],
            },
            {
                "metric_id": "hf_only_reference_paired_key_attribution",
                "allowed_splits": ["untouched_confirmation"],
            },
            {
                "metric_id": "hf_only_reference_paired_final_image_quality",
                "allowed_splits": ["untouched_confirmation"],
            },
            {
                "metric_id": "hf_only_reference_actual_dtype_integrity",
                "allowed_splits": ["untouched_confirmation"],
            },
        ],
        "formula_identities": {
            "threshold_fit": {
                "input": "4096_finite_float64_primary_null_hf_scores",
                "tau": "float64_nextafter(max(scores),positive_infinity)",
                "decision": "score_greater_than_or_equal_to_tau",
                "required_fit_false_positives": 0,
            },
            "binomial_intervals": {
                "family": "exact_clopper_pearson",
                "confidence_level": 0.95,
                "primary_null_direction": "one_sided_upper",
                "registered_tpr_direction": "one_sided_lower",
                "wrong_key_direction": "one_sided_upper",
                "paired_attribution_direction": "one_sided_lower",
                "primary_and_wrong_key_pooling": "forbidden",
            },
            "paired_attribution": {
                "pairing_unit": "same_source_cluster_and_registered_watermarked_image",
                "success": "registered_hf_score_strictly_greater_than_wrong_key_hf_score",
                "tie": "failure",
                "aggregate": "success_count_over_all_4096_planned_pairs",
            },
            "paired_quality": {
                "reference": "final_clean_rgb8_image",
                "candidate": "same_source_cluster_final_registered_watermarked_rgb8_image",
                "normalization": "uint8_channel_value_divided_by_255",
                "metrics": [
                    {
                        "metric_id": "paired_rgb8_relative_l2",
                        "formula": "l2(candidate-reference)/l2(reference)",
                        "numeric_domain": "[0,positive_infinity)",
                        "unit": "dimensionless",
                        "direction": "lower_is_less_distortion",
                    },
                    {
                        "metric_id": "paired_rgb8_mse",
                        "formula": "mean((candidate-reference)^2)_over_all_hwc_values",
                        "numeric_domain": "[0,1]",
                        "unit": "squared_normalized_rgb8_intensity",
                        "direction": "lower_is_less_distortion",
                    },
                ],
                "aggregate": (
                    "arithmetic_mean_and_two_sided_95_percent_student_t_interval_"
                    "mean_plus_or_minus_t_0_975_n_minus_1_times_sample_sd_ddof_1_over_sqrt_n"
                ),
                "interval_clipping": "forbidden",
                "missing_or_non_finite": "scientific_failure_and_gate_failure",
                "required_pair_count": 4096,
                "pass_cutoff": None,
            },
        },
    }
    if type(raw) is not dict or raw != expected:
        raise ValueError("metric_plan_invalid")


def _validate_statistics(raw: object) -> None:
    if type(raw) is not dict:
        raise ValueError("statistics_invalid")
    _require_exact_keys(
        raw,
        (
            "threshold_fit",
            "untouched_confirmation",
            "tpr",
            "wrong_key",
            "paired_attribution",
            "paired_quality",
            "actual_dtype_integrity",
            "result_gate_semantics",
        ),
        context="statistics",
    )
    threshold_fit = raw["threshold_fit"]
    expected_threshold = {
        "split": "content_threshold_fit",
        "primary_null_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "positive_count": 0,
        "fit_quantity": "hf_only_tau",
        "fit_rule": HF_ONLY_REFERENCE_THRESHOLD_RULE,
        "decision_comparison": HF_ONLY_REFERENCE_THRESHOLD_COMPARISON,
        "tail_target": HF_ONLY_REFERENCE_PRIMARY_NULL_TARGET_FPR,
        "tail_failure_probability_formula": "(1-tail_target)^n",
        "tail_failure_probability_upper_limit": 0.05,
        "confirmation_data_access": "forbidden",
    }
    if threshold_fit != expected_threshold:
        raise ValueError("threshold_fit_plan_invalid")
    if HF_ONLY_REFERENCE_THRESHOLD_TAIL_FAILURE_PROBABILITY >= 0.05:
        raise ValueError("threshold_fit_tail_budget_insufficient")
    confirmation = raw["untouched_confirmation"]
    expected_confirmation = {
        "split": "untouched_confirmation",
        "frozen_tau_source": "content_threshold_fit_only",
        "tau_refit": "forbidden",
        "primary_null_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "registered_positive_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "wrong_key_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "paired_quality_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "primary_null_zero_false_positive_required": True,
        "primary_null_one_sided_confidence_level": HF_ONLY_REFERENCE_CONFIDENCE_LEVEL,
        "primary_null_cp_upper_target": HF_ONLY_REFERENCE_PRIMARY_NULL_TARGET_FPR,
    }
    if confirmation != expected_confirmation:
        raise ValueError("untouched_confirmation_plan_invalid")
    if HF_ONLY_REFERENCE_ZERO_FAILURE_CP_UPPER_95 > HF_ONLY_REFERENCE_PRIMARY_NULL_TARGET_FPR:
        raise ValueError("confirmation_primary_null_budget_insufficient")
    tpr = raw["tpr"]
    if tpr != {
        "sample_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "interval": "one_sided_95_percent_clopper_pearson_lower",
        "minimum_lower_confidence_bound": 0.5,
        "scientific_utility_rationale": (
            "at_fpr_0_001_a_reference_detected_no_more_often_than_not_is_not_useful"
        ),
        "power_precision_rationale": (
            "n_4096_places_a_true_rate_0_55_more_than_six_worst_case_standard_errors_above_0_50"
        ),
        "semantics": "preregistered_minimum_effect_gate_not_tuned_from_results",
    }:
        raise ValueError("tpr_plan_invalid")
    wrong_key = raw["wrong_key"]
    if wrong_key != {
        "sample_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "interval": "one_sided_95_percent_clopper_pearson_upper",
        "reported_separately_from_primary_null": True,
        "zero_wrong_key_positives_required": True,
        "maximum_upper_confidence_bound": HF_ONLY_REFERENCE_PRIMARY_NULL_TARGET_FPR,
        "scientific_utility_rationale": (
            "wrong_key_false_acceptance_must_meet_the_same_0_001_ceiling_but_is_not_primary_fpr"
        ),
        "semantics": "independent_attribution_null_gate_never_pooled_with_primary_null",
    }:
        raise ValueError("wrong_key_plan_invalid")
    paired_attribution = raw["paired_attribution"]
    if paired_attribution != {
        "sample_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "success_event": "registered_score_strictly_greater_than_paired_wrong_key_score",
        "ties": "count_as_failures",
        "interval": "one_sided_95_percent_clopper_pearson_lower",
        "minimum_lower_confidence_bound": 0.5,
        "scientific_utility_rationale": (
            "registered_key_scores_must_beat_their_paired_wrong_key_controls_above_chance"
        ),
        "power_precision_rationale": (
            "n_4096_places_a_true_rate_0_55_more_than_six_worst_case_standard_errors_above_0_50"
        ),
    }:
        raise ValueError("paired_attribution_plan_invalid")
    paired_quality = raw["paired_quality"]
    if paired_quality != {
        "sample_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "interval": "two_sided_95_percent_paired_mean_interval",
        "standardized_standard_error_scale": 0.015625,
        "pass_threshold": None,
        "semantics": "paired_quality_characterization_no_preregistered_limit",
    }:
        raise ValueError("paired_quality_plan_invalid")
    if raw["actual_dtype_integrity"] != {
        "registered_positive_count": HF_ONLY_REFERENCE_SOURCE_CLUSTERS_PER_SPLIT,
        "required_for_every_registered_positive": True,
        "required_runtime_dtype": "float16",
        "measurement_dtype": "float32",
        "content_relative_l2_limit": "3/250",
        "required_checks": [
            "materialization_integrity_passed",
            "actual_dtype_matches_runtime",
            "actual_relative_l2_finite",
            "actual_relative_l2_at_most_hard_limit",
        ],
        "failure_semantics": "scientific_failure_and_hf_only_reference_negative_closure",
    }:
        raise ValueError("actual_dtype_integrity_plan_invalid")
    result_gate = raw["result_gate_semantics"]
    if result_gate != {
        "high_specificity_gate": (
            "zero_primary_null_false_positives_and_one_sided_95_percent_cp_upper_at_most_0_001"
        ),
        "identity_gate": (
            "all_frozen_candidate_runtime_dtype_key_and_manifest_identities_match"
        ),
        "tpr_gate": "one_sided_95_percent_lower_bound_at_least_0_50",
        "wrong_key_gate": (
            "zero_positive_and_separate_one_sided_95_percent_upper_bound_at_most_0_001"
        ),
        "paired_attribution_gate": (
            "registered_greater_than_wrong_key_one_sided_95_percent_lower_bound_above_0_50"
        ),
        "actual_dtype_integrity_gate": "all_registered_positive_records_pass",
        "paired_quality": "reported_as_characterization_without_pass_threshold",
        "negative_closure": (
            "failure_of_any_gate_closes_hf_reference_research_question_negative_and_forbids_unapproved_follow_on_validation"
        ),
        "result_gate_id": HF_ONLY_REFERENCE_RESULT_GATE_ID,
        "result_gate_is_prerequisite": False,
    }:
        raise ValueError("result_gate_semantics_invalid")


def load_frozen_prompt_roster(path: str | Path) -> FrozenPromptRoster:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    _require_exact_keys(
        raw,
        (
            "schema_version",
            "dataset_id",
            "dataset_revision",
            "dataset_file",
            "dataset_file_sha256",
            "license",
            "prompt_text_normalization",
            "expected_prompt_count",
            "expected_duplicate_prompt_count",
            "expected_category_count",
            "rows_digest",
            "rows",
        ),
        context="prompt_roster",
    )
    rows: list[FrozenPromptRosterEntry] = []
    for raw_row in raw["rows"]:
        if type(raw_row) is not dict:
            raise ValueError("prompt_roster_row_invalid")
        _require_exact_keys(
            raw_row,
            ("source_row", "prompt_text", "prompt_digest", "category", "challenge"),
            context="prompt_roster_row",
        )
        rows.append(FrozenPromptRosterEntry(**raw_row))
    raw["rows"] = tuple(rows)
    roster = FrozenPromptRoster(**raw)
    violations = roster.validate()
    if violations:
        raise ValueError(", ".join(violations))
    return roster


def load_compact_hf_only_reference_split_manifest(path: str | Path) -> CompactHfOnlyReferenceSplitManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    _require_exact_keys(
        raw,
        tuple(CompactHfOnlyReferenceSplitManifest.__dataclass_fields__),
        context="compact_manifest",
    )
    specification = CompactHfOnlyReferenceSplitManifest(**raw)
    violations = specification.validate()
    if violations:
        raise ValueError(", ".join(violations))
    return specification


def load_hf_only_reference_specification(
    path: str | Path,
) -> HfOnlyReferenceSpecification:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    specification = HfOnlyReferenceSpecification(raw=raw)
    violations = specification.validate()
    if violations:
        raise ValueError(", ".join(violations))
    return specification


def _selected_roster_rows(
    roster: FrozenPromptRoster,
    compact: CompactHfOnlyReferenceSplitManifest,
) -> tuple[FrozenPromptRosterEntry, ...]:
    strata: dict[tuple[str, str], list[FrozenPromptRosterEntry]] = {}
    for row in roster.rows:
        strata.setdefault((row.category, row.challenge), []).append(row)
    odd_strata = tuple(
        identity
        for identity in sorted(strata)
        if len(strata[identity]) % 2
    )
    fit_extra_strata = frozenset(odd_strata[: len(odd_strata) // 2])
    selected_rows: list[FrozenPromptRosterEntry] = []
    for identity in sorted(strata):
        rows = sorted(strata[identity], key=lambda row: row.source_row)
        fit_gets_extra = identity in fit_extra_strata
        for index, row in enumerate(rows):
            assigned_to_fit = index % 2 == (0 if fit_gets_extra else 1)
            if (compact.split == "content_threshold_fit") == assigned_to_fit:
                selected_rows.append(row)
    selected = tuple(sorted(selected_rows, key=lambda row: row.source_row))
    if len(selected) != compact.expected_prompt_count:
        raise ValueError("selected_prompt_count_mismatch")
    return selected


def _generation_seed(
    compact: CompactHfOnlyReferenceSplitManifest,
    row: FrozenPromptRosterEntry,
    replication_index: int,
) -> int:
    digest = _canonical_digest(
        {
            "replication_index": replication_index,
            "seed_namespace": compact.seed_namespace,
            "source_row": row.source_row,
        }
    )
    return int(digest[:16], 16) & ((1 << 63) - 1)


def _image_lineage_digest(
    compact: CompactHfOnlyReferenceSplitManifest,
    row: FrozenPromptRosterEntry,
    *,
    generation_seed: int,
    replication_index: int,
) -> str:
    return _canonical_digest(
        {
            "category": row.category,
            "challenge": row.challenge,
            "dataset_file_sha256": HF_ONLY_REFERENCE_DATASET_SHA256,
            "dataset_revision": HF_ONLY_REFERENCE_DATASET_REVISION,
            "generation_seed": generation_seed,
            "replication_index": replication_index,
            "source_row": row.source_row,
        }
    )


def materialize_hf_only_reference_split_manifest(
    compact: CompactHfOnlyReferenceSplitManifest,
    roster: FrozenPromptRoster,
) -> FrozenSplitManifest:
    compact_violations = compact.validate()
    roster_violations = roster.validate()
    if compact_violations or roster_violations:
        raise ValueError(", ".join((*compact_violations, *roster_violations)))
    if compact.roster_rows_digest != roster.rows_digest:
        raise ValueError("compact_manifest_roster_digest_mismatch")
    assignments: list[SplitAssignment] = []
    selected = _selected_roster_rows(roster, compact)
    for selected_index, row in enumerate(selected):
        replicate_count = compact.base_replicates_per_prompt + (
            1
            if selected_index
            < compact.extra_replicates_for_first_selected_prompts
            else 0
        )
        for replication_index in range(replicate_count):
            generation_seed = _generation_seed(compact, row, replication_index)
            lineage_digest = _image_lineage_digest(
                compact,
                row,
                generation_seed=generation_seed,
                replication_index=replication_index,
            )
            cluster_id = derive_source_cluster_id(
                prompt_digest=row.prompt_digest,
                generation_seed=generation_seed,
                image_lineage_digest=lineage_digest,
                registered_key_family_digest=compact.registered_key_family_digest,
            )
            identity = AnalysisUnitIdentity(
                unit_id=(
                    f"hf_only_reference_{compact.split}_row_{row.source_row:04d}_"
                    f"rep_{replication_index:02d}"
                ),
                case_id=f"hf_only_reference_{compact.split}",
                source_cluster_id=cluster_id,
                prompt_digest=row.prompt_digest,
                generation_seed=generation_seed,
                image_lineage_digest=lineage_digest,
                registered_key_family_digest=compact.registered_key_family_digest,
            )
            assignments.append(SplitAssignment(identity=identity, split=compact.split))
    manifest = FrozenSplitManifest(
        protocol_id=compact.protocol_id,
        protocol_version=compact.protocol_version,
        manifest_id=compact.manifest_id,
        manifest_revision=compact.manifest_revision,
        assignments=tuple(assignments),
    )
    violations = manifest.validate(require_all_splits=False)
    if violations:
        raise ValueError(", ".join(violations))
    if len(manifest.assignments) != compact.expected_assignment_count:
        raise ValueError("materialized_assignment_count_mismatch")
    if manifest.digest() != compact.expected_materialized_manifest_digest:
        raise ValueError("materialized_manifest_digest_mismatch")
    return manifest


def validate_hf_only_reference_manifest_pair(
    threshold_fit: FrozenSplitManifest,
    untouched_confirmation: FrozenSplitManifest,
    roster: FrozenPromptRoster,
) -> tuple[str, ...]:
    violations: list[str] = []
    if {assignment.split for assignment in threshold_fit.assignments} != {
        "content_threshold_fit"
    }:
        violations.append("threshold_fit_split_identity_invalid")
    if {assignment.split for assignment in untouched_confirmation.assignments} != {
        "untouched_confirmation"
    }:
        violations.append("untouched_confirmation_split_identity_invalid")
    threshold_clusters = {
        assignment.identity.source_cluster_id
        for assignment in threshold_fit.assignments
    }
    confirmation_clusters = {
        assignment.identity.source_cluster_id
        for assignment in untouched_confirmation.assignments
    }
    if threshold_clusters & confirmation_clusters:
        violations.append("source_cluster_split_leakage")
    threshold_prompts = {
        assignment.identity.prompt_digest for assignment in threshold_fit.assignments
    }
    confirmation_prompts = {
        assignment.identity.prompt_digest
        for assignment in untouched_confirmation.assignments
    }
    if threshold_prompts & confirmation_prompts:
        violations.append("prompt_split_leakage")
    if len(threshold_prompts | confirmation_prompts) != HF_ONLY_REFERENCE_PROMPT_COUNT:
        violations.append("prompt_roster_union_incomplete")
    if len(threshold_clusters) != len(threshold_fit.assignments):
        violations.append("threshold_fit_source_cluster_duplicate")
    if len(confirmation_clusters) != len(untouched_confirmation.assignments):
        violations.append("untouched_confirmation_source_cluster_duplicate")
    if len(threshold_prompts) != HF_ONLY_REFERENCE_PROMPTS_PER_SPLIT:
        violations.append("threshold_fit_prompt_count_mismatch")
    if len(confirmation_prompts) != HF_ONLY_REFERENCE_PROMPTS_PER_SPLIT:
        violations.append("untouched_confirmation_prompt_count_mismatch")
    roster_by_digest = {row.prompt_digest: row for row in roster.rows}
    if set(roster_by_digest) != threshold_prompts | confirmation_prompts:
        violations.append("materialized_prompt_roster_identity_mismatch")
    else:
        threshold_counts: dict[tuple[str, str], int] = {}
        confirmation_counts: dict[tuple[str, str], int] = {}
        for digest in threshold_prompts:
            row = roster_by_digest[digest]
            identity = (row.category, row.challenge)
            threshold_counts[identity] = threshold_counts.get(identity, 0) + 1
        for digest in confirmation_prompts:
            row = roster_by_digest[digest]
            identity = (row.category, row.challenge)
            confirmation_counts[identity] = (
                confirmation_counts.get(identity, 0) + 1
            )
        for identity in {
            (row.category, row.challenge) for row in roster.rows
        }:
            if abs(
                threshold_counts.get(identity, 0)
                - confirmation_counts.get(identity, 0)
            ) > 1:
                violations.append("category_challenge_stratum_imbalance")
                break
    return tuple(violations)


def validate_bound_authority_files(
    specification: HfOnlyReferenceSpecification,
    repository_root: str | Path,
) -> tuple[str, ...]:
    """Build-time check only; runtime uses the already frozen public digests."""
    violations = list(specification.validate())
    if violations:
        return tuple(violations)
    root = Path(repository_root)
    binding = specification.raw["candidate_binding"]
    candidate_specification_path = root / binding["candidate_specification_path"]
    if (
        not candidate_specification_path.is_file()
        or _file_sha256(candidate_specification_path)
        != binding["candidate_specification_sha256"]
    ):
        violations.append("candidate_specification_authority_mismatch")
    for entry in binding["method_source_files"]:
        path = root / entry["path"]
        if not path.is_file() or _file_sha256(path) != entry["sha256"]:
            violations.append(f"bound_authority_file_mismatch:{entry['path']}")
    runtime_path = root / binding["runtime_config_path"]
    if (
        not runtime_path.is_file()
        or _file_sha256(runtime_path) != binding["runtime_config_sha256"]
    ):
        violations.append("bound_authority_file_mismatch:runtime_config")
    adapter_path = root / binding["formal_method_adapter_config_path"]
    if not adapter_path.is_file():
        violations.append("bound_authority_file_mismatch:formal_method_adapter")
    else:
        try:
            with adapter_path.open("r", encoding="utf-8") as handle:
                execution_components = json.load(handle)
            adapter_payload = dict(execution_components["method_adapter"])
            adapter_payload["schema_version"] = execution_components["schema_version"]
        except (KeyError, TypeError, json.JSONDecodeError):
            violations.append("bound_authority_file_mismatch:formal_method_adapter")
        else:
            if _canonical_digest(adapter_payload) != binding[
                "formal_method_adapter_config_digest"
            ]:
                violations.append(
                    "bound_authority_file_mismatch:formal_method_adapter"
                )
    dataset = specification.raw["dataset"]
    for path_key, digest_key in (
        ("roster_path", "roster_file_sha256"),
        ("dataset_snapshot_path", "file_sha256"),
    ):
        artifact = root / dataset[path_key]
        if not artifact.is_file() or _file_sha256(artifact) != dataset[digest_key]:
            violations.append(f"bound_authority_file_mismatch:{path_key}")
    return tuple(violations)


def load_hf_only_reference_bundle(
    repository_root: str | Path,
) -> HfOnlyReferenceBundle:
    root = Path(repository_root)
    specification = load_hf_only_reference_specification(
        root / "configs/experiments/hf_only_reference_validation.json"
    )
    authority_violations = validate_bound_authority_files(specification, root)
    if authority_violations:
        raise ValueError(", ".join(authority_violations))
    dataset = specification.raw["dataset"]
    roster = load_frozen_prompt_roster(root / dataset["roster_path"])
    if roster.rows_digest != dataset["roster_rows_digest"]:
        raise ValueError("bundle_roster_rows_digest_mismatch")
    snapshot_path = root / dataset["dataset_snapshot_path"]
    with snapshot_path.open("r", encoding="utf-8", newline="") as handle:
        snapshot_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(snapshot_rows) != len(roster.rows):
        raise ValueError("bundle_dataset_snapshot_row_count_mismatch")
    for roster_row, snapshot_row in zip(roster.rows, snapshot_rows, strict=True):
        if (
            roster_row.prompt_text != snapshot_row.get("Prompt")
            or roster_row.category != snapshot_row.get("Category")
            or roster_row.challenge != snapshot_row.get("Challenge")
            or hashlib.sha256(
                snapshot_row.get("Prompt", "").encode("utf-8")
            ).hexdigest()
            != roster_row.prompt_digest
        ):
            raise ValueError("bundle_dataset_snapshot_roster_mismatch")
    key_family_digest = specification.raw["key_controls"][
        "registered_key_family_digest"
    ]
    compact_manifests: list[CompactHfOnlyReferenceSplitManifest] = []
    materialized_manifests: list[FrozenSplitManifest] = []
    for split in HF_ONLY_REFERENCE_SPLITS:
        binding = specification.raw["split_manifests"][split]
        compact_path = root / binding["path"]
        if _file_sha256(compact_path) != binding["file_sha256"]:
            raise ValueError(f"bundle_compact_manifest_file_sha256_mismatch:{split}")
        compact = load_compact_hf_only_reference_split_manifest(compact_path)
        if (
            compact.split != split
            or compact.roster_rows_digest != roster.rows_digest
            or compact.registered_key_family_digest != key_family_digest
        ):
            raise ValueError(f"bundle_compact_manifest_binding_mismatch:{split}")
        materialized = materialize_hf_only_reference_split_manifest(compact, roster)
        if materialized.digest() != binding["materialized_manifest_digest"]:
            raise ValueError(f"bundle_materialized_manifest_digest_mismatch:{split}")
        compact_manifests.append(compact)
        materialized_manifests.append(materialized)
    pair_violations = validate_hf_only_reference_manifest_pair(
        materialized_manifests[0],
        materialized_manifests[1],
        roster,
    )
    if pair_violations:
        raise ValueError(", ".join(pair_violations))
    specification.freeze_reference_candidate()
    return HfOnlyReferenceBundle(
        specification=specification,
        roster=roster,
        compact_manifests=tuple(compact_manifests),
        materialized_manifests=tuple(materialized_manifests),
    )
