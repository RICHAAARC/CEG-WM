"""Frozen Stage-A contrastive LF branch-attribution protocol contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite, nextafter, sqrt
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

from experiments.protocol.internal_splits import derive_source_cluster_id


PROTOCOL_ID = "contrastive_lf_branch_attribution"
SCHEMA_VERSION = 1
NULL_FIT_ROLE = "contrastive_lf_null_fit"
SELECTION_ROLE = "contrastive_lf_candidate_selection"
CONFIRMATION_ROLE = "contrastive_lf_untouched_confirmation"
ROLES = (NULL_FIT_ROLE, SELECTION_ROLE, CONFIRMATION_ROLE)
MULTISCALE_CANDIDATE_ID = "lf_multiscale_lowpass_contrastive"
SINGLE_SCALE_CANDIDATE_ID = "lf_five_by_five_lowpass_contrastive"
CANDIDATE_IDS = (MULTISCALE_CANDIDATE_ID, SINGLE_SCALE_CANDIDATE_ID)
CANDIDATE_ROLE_LABELS = (
    "multiscale_primary_candidate",
    "single_scale_fallback_candidate",
)
HF_CANDIDATE_ID = "hf_sparse_tail"
KEY_FAMILY_NAMESPACE_DIGEST = (
    "c94f68e1aaf69b710630d3a3401262c3a7af7afdeeffe6d0f9ea3eb63e1777b1"
)
SOURCE_SNAPSHOT_PATH = (
    "configs/experiments/assets/parti_prompts_dataset_snapshot.txt"
)
SOURCE_SNAPSHOT_SHA256 = (
    "fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb"
)
PROMPT_ROSTER_PATH = (
    "configs/experiments/contrastive_lf_branch_attribution_prompt_roster.json"
)
PROMPT_ROSTER_DIGEST = (
    "92bdc0daa75425878345b4747f8d63233cd64ce49a7abcaa9db427cdc8de3548"
)
SOURCE_ROSTER_ROWS_DIGEST = (
    "7d55ce5897f0fe33c0ff3c815d2ea45618a60666ed086cd8a915ac037d371792"
)
CONFIG_DIGEST = (
    "389a7e24656c07b59b7a6f222c67f8058295efe976129b90e05f020625351ea4"
)
CONFIG_PATH = "configs/experiments/contrastive_lf_branch_attribution.json"
MANIFEST_PATHS = {
    NULL_FIT_ROLE: "configs/experiments/contrastive_lf_null_fit_manifest.json",
    SELECTION_ROLE: (
        "configs/experiments/contrastive_lf_candidate_selection_manifest.json"
    ),
    CONFIRMATION_ROLE: (
        "configs/experiments/contrastive_lf_untouched_confirmation_manifest.json"
    ),
}
ENTRIES_DIGESTS = {
    NULL_FIT_ROLE: (
        "69993f38aa7bf2cdb945421f0e640bd4933d2035e5598f40b75b6a2911eb55d7"
    ),
    SELECTION_ROLE: (
        "199b17c5c76157178b602f0abfe0d356ca35f1ca624542b3f1978f4fb4e064a3"
    ),
    CONFIRMATION_ROLE: (
        "26334fd5cbd15172e5d07970723288a6cb2c72be24c23dd27c4b0903424cd1a9"
    ),
}
MANIFEST_DIGESTS = {
    NULL_FIT_ROLE: (
        "d77ca1688f8b231e7b1526fd6b7bcad1b49dbdc897c9d7335064bb68a7f3607c"
    ),
    SELECTION_ROLE: (
        "137e56f2df31766ca1038e1d53a6e1d2667adb4d3d8b92982e222f868dcbd50b"
    ),
    CONFIRMATION_ROLE: (
        "13b9d9e7972483be363a3bdaa54597c1691191095c5022077e613d3c0e91a171"
    ),
}
SOURCE_ROWS_BY_ROLE = {
    NULL_FIT_ROLE: tuple(range(132, 164)),
    SELECTION_ROLE: tuple(range(164, 196)),
    CONFIRMATION_ROLE: tuple(range(196, 228)),
}
SEED_BASE_BY_ROLE = {
    NULL_FIT_ROLE: 202608210000,
    SELECTION_ROLE: 202608210100,
    CONFIRMATION_ROLE: 202608210200,
}
CLUSTER_COUNT = 32
MAXIMUM_RECORD_ATTEMPTS = 1
NULL_FIT_ARMS = ("clean_unwatermarked",)
SELECTION_ARMS = (
    "clean_unwatermarked",
    "hf_only",
    "multiscale_low_frequency_only",
    "single_scale_low_frequency_only",
)
CONFIRMATION_ARMS = (
    "clean_unwatermarked",
    "hf_only",
    "selected_low_frequency_only",
)
ATTACKS = (
    "identity",
    "jpeg_quality_70",
    "gaussian_blur_sigma_1",
    "gaussian_noise_sigma_0_01",
)
EXTERNAL_WRONG_KEY_INDEXES = tuple(range(8))
INTERNAL_DECOY_INDEXES = tuple(range(8))
EXTERNAL_WRONG_KEY_ROSTER_IDENTITY = (
    "contrastive_lf_external_wrong_key_roster"
)
EXTERNAL_WRONG_KEY_ROSTER_DIGEST = (
    "d89725b2cec922e584d62247990b468d09354b58b64217db58c404adaaf3d23f"
)
INTERNAL_DECOY_ROSTER_IDENTITY = "contrastive_lf_internal_decoy_roster"
INTERNAL_DECOY_ROSTER_DIGEST = (
    "9b04ce25938fc49a820b8c631a7e8a5ae487a1e7f6738b49e5dc0ef8cb5be2c2"
)
GATE_ORDER = (
    "identity_integrity_denominator",
    "budget_finite_replay",
    "hf_anchor",
    "candidate_attribution_null_wrong",
    "blur_complement",
    "quality",
)
RESULT_CLASSIFICATIONS = (
    "success",
    "scientific_failure",
    "insufficient_evidence",
    "operational_failure",
)
EXECUTION_STATUSES = ("completed", "failed", "unstarted")
OPERATIONAL_FAILURE_CLASSES = frozenset(
    {"runtime_failure", "dependency_failure", "codec_failure", "resource_failure"}
)
RECORD_KINDS = (
    "clean_base_observation",
    "null_statistic",
    "base_generation",
    "attacked_observation",
    "detector",
    "budget",
    "quality",
)
QUALITY_EPSILON_FLOAT64_HEX = "0x1.0203040506070p-16"
PILLOW_VERSION = "12.3.0"
JPEG_GOLDEN_DIGESTS = {
    "jpeg_candidate_attack_encoded": (
        "9a202effd37e2b693f70fad3e9e01bc41d68df1fab4138de349a39963b49c80b"
    ),
    "jpeg_candidate_attack_decoded": (
        "e4c5fc8268dce4b00f6b36bdcf542968bdba495a153cc273e9ece7279dd7029b"
    ),
    "jpeg_stability_probe_encoded": (
        "e8c89254351499471fe086704f3ecc6e2fb76f7cccd33e5a8ecc2fc33fc54c36"
    ),
    "jpeg_stability_probe_decoded": (
        "0d0e02529511b7cae4de3479b9645569c3985750655217abfb6c7d6326b362c7"
    ),
}
BLIND_DETECTOR_INPUTS = (
    "current_rgb8",
    "detection_key",
    "public_frozen_assets",
)
FORBIDDEN_DETECTOR_INPUTS = (
    "reference_image",
    "prompt",
    "embed_record",
    "private_latent",
    "embed_route",
    "qk_cache",
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_FAILURE_TEXT = re.compile(
    r"(?i)(traceback|secret|token|root[_ -]?key|/(?:home|content)/|[a-z]:\\)"
)


class ContrastiveLfProtocolError(ValueError):
    """The checked-in contrastive LF attribution protocol is invalid."""


def canonical_digest(value: object) -> str:
    """Return the protocol-wide stable JSON SHA-256."""

    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_digest(value: object, field_name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ContrastiveLfProtocolError(f"{field_name}_invalid")
    return value


def _require_relative_path(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise ContrastiveLfProtocolError(f"{field_name}_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ContrastiveLfProtocolError(f"{field_name}_invalid")
    return value


@dataclass(frozen=True, slots=True)
class ContrastiveLfManifestEntry:
    """One immutable prompt/seed/lineage/source-cluster assignment."""

    source_row: int
    prompt_text: str
    prompt_digest: str
    generation_seed: int
    source_cluster_id: str
    registered_key_family_digest: str
    image_lineage_identity: str
    image_lineage_digest: str

    def validate(self, *, role_id: str, ordinal: int) -> None:
        if role_id not in ROLES or ordinal not in range(CLUSTER_COUNT):
            raise ContrastiveLfProtocolError("manifest_entry_position_invalid")
        if (
            type(self.source_row) is not int
            or self.source_row != SOURCE_ROWS_BY_ROLE[role_id][ordinal]
            or type(self.prompt_text) is not str
            or not self.prompt_text
            or self.prompt_digest
            != sha256(self.prompt_text.encode("utf-8")).hexdigest()
            or type(self.generation_seed) is not int
            or self.generation_seed != SEED_BASE_BY_ROLE[role_id] + ordinal
            or self.registered_key_family_digest
            != KEY_FAMILY_NAMESPACE_DIGEST
        ):
            raise ContrastiveLfProtocolError("manifest_entry_authority_drifted")
        expected_lineage_identity = canonical_digest(
            {
                "generation_seed": self.generation_seed,
                "identity_role": (
                    "contrastive_lf_branch_attribution_image_lineage"
                ),
                "prompt_digest": self.prompt_digest,
                "protocol_id": PROTOCOL_ID,
                "registered_key_family_digest": (
                    self.registered_key_family_digest
                ),
                "role_id": role_id,
            }
        )
        expected_lineage_digest = canonical_digest(
            {
                "generation_seed": self.generation_seed,
                "image_lineage_identity": expected_lineage_identity,
                "image_lineage_namespace": PROTOCOL_ID,
                "role_id": role_id,
            }
        )
        expected_cluster = derive_source_cluster_id(
            prompt_digest=self.prompt_digest,
            generation_seed=self.generation_seed,
            image_lineage_digest=expected_lineage_digest,
            registered_key_family_digest=self.registered_key_family_digest,
        )
        if (
            self.image_lineage_identity != expected_lineage_identity
            or self.image_lineage_digest != expected_lineage_digest
            or self.source_cluster_id != expected_cluster
        ):
            raise ContrastiveLfProtocolError("manifest_entry_identity_drifted")


@dataclass(frozen=True, slots=True)
class ContrastiveLfManifest:
    """One literal 32-cluster Stage-A role manifest."""

    schema_version: int
    protocol_id: str
    role_id: str
    source_roster_path: str
    source_roster_rows_digest: str
    candidate_family_id: str
    candidate_ids: tuple[str, ...]
    registered_key_family_namespace_digest: str
    entries_digest: str
    entries: tuple[ContrastiveLfManifestEntry, ...]
    manifest_digest: str

    def payload_without_manifest_digest(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("manifest_digest")
        return payload

    def validate(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or self.protocol_id != PROTOCOL_ID
            or self.role_id not in ROLES
            or self.source_roster_path != PROMPT_ROSTER_PATH
            or self.source_roster_rows_digest != SOURCE_ROSTER_ROWS_DIGEST
            or self.candidate_family_id != PROTOCOL_ID
            or self.candidate_ids != CANDIDATE_IDS
            or self.registered_key_family_namespace_digest
            != KEY_FAMILY_NAMESPACE_DIGEST
            or len(self.entries) != CLUSTER_COUNT
        ):
            raise ContrastiveLfProtocolError("manifest_authority_drifted")
        for ordinal, entry in enumerate(self.entries):
            if type(entry) is not ContrastiveLfManifestEntry:
                raise ContrastiveLfProtocolError("manifest_entry_type_invalid")
            entry.validate(role_id=self.role_id, ordinal=ordinal)
        if self.entries_digest != canonical_digest(
            [asdict(entry) for entry in self.entries]
        ) or self.entries_digest != ENTRIES_DIGESTS[self.role_id]:
            raise ContrastiveLfProtocolError("manifest_entries_digest_drifted")
        if self.manifest_digest != canonical_digest(
            self.payload_without_manifest_digest()
        ) or self.manifest_digest != MANIFEST_DIGESTS[self.role_id]:
            raise ContrastiveLfProtocolError("manifest_digest_drifted")
        for field_name in (
            "source_row",
            "prompt_digest",
            "generation_seed",
            "source_cluster_id",
            "image_lineage_identity",
            "image_lineage_digest",
        ):
            values = [getattr(entry, field_name) for entry in self.entries]
            if len(set(values)) != CLUSTER_COUNT:
                raise ContrastiveLfProtocolError("manifest_axis_collides")


def _load_json_mapping(path: str | Path, error_role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContrastiveLfProtocolError(f"{error_role}_unreadable") from exc
    if type(value) is not dict:
        raise ContrastiveLfProtocolError(f"{error_role}_invalid")
    return value


def load_prompt_roster(path: str | Path) -> Mapping[str, object]:
    """Authenticate the public snapshot selection and frozen prior-use scan."""

    raw = _load_json_mapping(path, "prompt_roster")
    expected_keys = {
        "schema_version",
        "roster_id",
        "source_snapshot_path",
        "source_snapshot_sha256",
        "selection_rule",
        "exclusion_rule",
        "exclusion_boundary_revision",
        "exclusion_source_bindings",
        "excluded_prompt_digest_count",
        "excluded_prompt_digests_digest",
        "rows_digest",
        "rows",
        "prompt_roster_digest",
    }
    if set(raw) != expected_keys:
        raise ContrastiveLfProtocolError("prompt_roster_schema_drifted")
    payload = dict(raw)
    payload.pop("prompt_roster_digest")
    if (
        raw["schema_version"] != SCHEMA_VERSION
        or raw["roster_id"]
        != "contrastive_lf_branch_attribution_prompt_roster"
        or raw["source_snapshot_path"] != SOURCE_SNAPSHOT_PATH
        or raw["source_snapshot_sha256"] != SOURCE_SNAPSHOT_SHA256
        or raw["selection_rule"]
        != (
            "ascending_source_row_first_ninety_six_nonempty_unique_prompt_"
            "digests_after_registered_usage_exclusion"
        )
        or raw["exclusion_rule"]
        != (
            "all_checked_in_experiment_json_at_boundary_revision_except_"
            "source_universe_rows"
        )
        or raw["exclusion_boundary_revision"]
        != "55eed59023ab0a06870c402047cc0eefd79a846f"
        or raw["excluded_prompt_digest_count"] != 324
        or raw["excluded_prompt_digests_digest"]
        != "f30699bb8d6a029d04df7a647355872bd58ea6c4b427e5c21d088fb2c04f4147"
        or raw["rows_digest"] != SOURCE_ROSTER_ROWS_DIGEST
        or raw["prompt_roster_digest"] != PROMPT_ROSTER_DIGEST
        or canonical_digest(payload) != PROMPT_ROSTER_DIGEST
    ):
        raise ContrastiveLfProtocolError("prompt_roster_authority_drifted")
    bindings = raw["exclusion_source_bindings"]
    rows = raw["rows"]
    if type(bindings) is not list or len(bindings) != 30:
        raise ContrastiveLfProtocolError("prompt_exclusion_bindings_drifted")
    repository_root = Path(path).resolve().parents[2]
    seen_paths: set[str] = set()
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "relative_path",
            "file_sha256",
        }:
            raise ContrastiveLfProtocolError("prompt_exclusion_binding_invalid")
        relative = _require_relative_path(
            binding["relative_path"], "exclusion_source_path"
        )
        expected_sha = _require_digest(
            binding["file_sha256"], "exclusion_source_file"
        )
        if relative in seen_paths:
            raise ContrastiveLfProtocolError("prompt_exclusion_path_duplicate")
        seen_paths.add(relative)
        try:
            observed_sha = sha256(
                (repository_root / relative).read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise ContrastiveLfProtocolError(
                "prompt_exclusion_source_unreadable"
            ) from exc
        if observed_sha != expected_sha:
            raise ContrastiveLfProtocolError("prompt_exclusion_source_drifted")
    if type(rows) is not list or len(rows) != 96:
        raise ContrastiveLfProtocolError("prompt_roster_count_drifted")
    if canonical_digest(rows) != SOURCE_ROSTER_ROWS_DIGEST:
        raise ContrastiveLfProtocolError("prompt_roster_rows_digest_drifted")
    required_row_keys = {
        "source_row",
        "prompt_text",
        "prompt_digest",
        "category",
        "challenge",
        "role_id",
        "generation_seed",
    }
    for ordinal, row in enumerate(rows):
        role_id = ROLES[ordinal // CLUSTER_COUNT]
        role_ordinal = ordinal % CLUSTER_COUNT
        if (
            type(row) is not dict
            or set(row) != required_row_keys
            or row["source_row"] != 132 + ordinal
            or row["role_id"] != role_id
            or row["generation_seed"]
            != SEED_BASE_BY_ROLE[role_id] + role_ordinal
            or type(row["prompt_text"]) is not str
            or not row["prompt_text"]
            or row["prompt_digest"]
            != sha256(row["prompt_text"].encode("utf-8")).hexdigest()
            or type(row["category"]) is not str
            or not row["category"]
            or type(row["challenge"]) is not str
            or not row["challenge"]
        ):
            raise ContrastiveLfProtocolError("prompt_roster_row_drifted")
    if len({row["prompt_digest"] for row in rows}) != 96:
        raise ContrastiveLfProtocolError("prompt_roster_digest_collides")
    return raw


def load_manifest(path: str | Path, *, expected_role: str) -> ContrastiveLfManifest:
    """Load one exact literal manifest and bind it to the prompt roster."""

    raw = _load_json_mapping(path, "manifest")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "role_id",
        "source_roster_path",
        "source_roster_rows_digest",
        "candidate_family_id",
        "candidate_ids",
        "registered_key_family_namespace_digest",
        "entries_digest",
        "entries",
        "manifest_digest",
    }
    if set(raw) != expected_keys:
        raise ContrastiveLfProtocolError("manifest_schema_drifted")
    try:
        manifest = ContrastiveLfManifest(
            schema_version=raw["schema_version"],
            protocol_id=raw["protocol_id"],
            role_id=raw["role_id"],
            source_roster_path=raw["source_roster_path"],
            source_roster_rows_digest=raw["source_roster_rows_digest"],
            candidate_family_id=raw["candidate_family_id"],
            candidate_ids=tuple(raw["candidate_ids"]),
            registered_key_family_namespace_digest=(
                raw["registered_key_family_namespace_digest"]
            ),
            entries_digest=raw["entries_digest"],
            entries=tuple(
                ContrastiveLfManifestEntry(**entry)
                for entry in raw["entries"]
            ),
            manifest_digest=raw["manifest_digest"],
        )
    except (KeyError, TypeError) as exc:
        raise ContrastiveLfProtocolError("manifest_payload_invalid") from exc
    if expected_role not in ROLES or manifest.role_id != expected_role:
        raise ContrastiveLfProtocolError("manifest_role_drifted")
    manifest.validate()
    roster_path = Path(path).resolve().parents[2] / PROMPT_ROSTER_PATH
    roster = load_prompt_roster(roster_path)
    roster_rows = {
        (row["role_id"], row["source_row"]): row for row in roster["rows"]
    }
    for entry in manifest.entries:
        row = roster_rows[(expected_role, entry.source_row)]
        if (
            row["prompt_text"] != entry.prompt_text
            or row["prompt_digest"] != entry.prompt_digest
            or row["generation_seed"] != entry.generation_seed
        ):
            raise ContrastiveLfProtocolError("manifest_roster_binding_drifted")
    return manifest


def validate_split_disjointness(
    manifests: Sequence[ContrastiveLfManifest],
) -> None:
    """Require all three split axes to be pairwise disjoint."""

    if len(manifests) != len(ROLES) or tuple(
        manifest.role_id for manifest in manifests
    ) != ROLES:
        raise ContrastiveLfProtocolError("split_manifest_order_drifted")
    for manifest in manifests:
        manifest.validate()
    for field_name in (
        "source_row",
        "prompt_digest",
        "generation_seed",
        "source_cluster_id",
        "image_lineage_identity",
        "image_lineage_digest",
    ):
        seen: set[object] = set()
        for manifest in manifests:
            values = {getattr(entry, field_name) for entry in manifest.entries}
            if seen & values:
                raise ContrastiveLfProtocolError("split_identity_overlap")
            seen.update(values)


def load_configuration(path: str | Path) -> Mapping[str, object]:
    """Load the exact Stage-A configuration without accepting extra keys."""

    raw = _load_json_mapping(path, "configuration")
    observed_digest = raw.get("config_digest")
    payload = dict(raw)
    payload.pop("config_digest", None)
    if (
        observed_digest != CONFIG_DIGEST
        or canonical_digest(payload) != CONFIG_DIGEST
        or raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("protocol_id") != PROTOCOL_ID
        or tuple(raw.get("roles", ())) != ROLES
        or tuple(item["candidate_id"] for item in raw.get("candidates", ()))
        != CANDIDATE_IDS
        or tuple(
            item["candidate_role_label"] for item in raw.get("candidates", ())
        )
        != CANDIDATE_ROLE_LABELS
        or raw.get("source_cluster_count") != CLUSTER_COUNT
        or raw.get("maximum_record_attempts") != MAXIMUM_RECORD_ATTEMPTS
        or tuple(item["attack_id"] for item in raw.get("attacks", ()))
        != ATTACKS
        or tuple(raw.get("external_wrong_key_indexes", ()))
        != EXTERNAL_WRONG_KEY_INDEXES
        or tuple(raw.get("internal_decoy_indexes", ()))
        != INTERNAL_DECOY_INDEXES
        or raw.get("combined_relative_l2_numerator") != 3
        or raw.get("combined_relative_l2_denominator") != 250
        or raw.get("actual_branch_decomposition_claimed") is not False
        or raw.get("pillow_version") != PILLOW_VERSION
        or raw.get("quality_epsilon_float64_hex")
        != QUALITY_EPSILON_FLOAT64_HEX
        or raw.get("provisional_threshold_rule")
        != "nextafter(fourth_largest_z,+inf)"
        or tuple(raw.get("gate_order", ())) != GATE_ORDER
        or tuple(raw.get("result_classifications", ()))
        != RESULT_CLASSIFICATIONS
    ):
        raise ContrastiveLfProtocolError("configuration_authority_drifted")
    return raw


@dataclass(frozen=True, slots=True)
class FixedDenominators:
    """Expected persisted slot counts for one role."""

    clean_base_observation_count: int
    raw_null_statistic_count: int
    base_generation_count: int
    attacked_observation_slot_count: int
    detector_record_count: int
    budget_record_count: int
    quality_record_count: int

    @property
    def persisted_record_count(self) -> int:
        return (
            self.clean_base_observation_count
            + self.raw_null_statistic_count
            + self.base_generation_count
            + self.attacked_observation_slot_count
            + self.detector_record_count
            + self.budget_record_count
            + self.quality_record_count
        )


DENOMINATORS_BY_ROLE = {
    NULL_FIT_ROLE: FixedDenominators(32, 96, 0, 0, 0, 0, 0),
    SELECTION_ROLE: FixedDenominators(0, 0, 128, 512, 3840, 96, 384),
    CONFIRMATION_ROLE: FixedDenominators(0, 0, 96, 384, 2560, 64, 256),
}


@dataclass(frozen=True, slots=True)
class ContrastiveLfRecordTemplate:
    """One preallocated immutable protocol slot."""

    schema_version: int
    protocol_id: str
    role_id: str
    record_id: str
    slot_ordinal: int
    record_kind: str
    source_row: int
    source_cluster_id: str
    prompt_text: str
    prompt_digest: str
    generation_seed: int
    image_lineage_identity: str
    image_lineage_digest: str
    key_family_namespace_digest: str
    prompt_roster_digest: str
    source_roster_rows_digest: str
    sample_manifest_digest: str
    manifest_entries_digest: str
    candidate_id: str | None
    arm_id: str
    attack_id: str
    key_role: str | None
    wrong_key_index: int | None
    control_identity: str
    internal_decoy_score_count: int
    internal_decoy_roster_identity: str
    internal_decoy_roster_digest: str
    external_wrong_key_roster_identity: str
    external_wrong_key_roster_digest: str

    def payload_without_record_id(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("record_id")
        return payload

    def validate(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or self.protocol_id != PROTOCOL_ID
            or self.role_id not in ROLES
            or type(self.slot_ordinal) is not int
            or self.slot_ordinal < 0
            or self.record_kind not in RECORD_KINDS
            or self.key_family_namespace_digest
            != KEY_FAMILY_NAMESPACE_DIGEST
            or self.prompt_roster_digest != PROMPT_ROSTER_DIGEST
            or self.source_roster_rows_digest != SOURCE_ROSTER_ROWS_DIGEST
            or self.sample_manifest_digest != MANIFEST_DIGESTS[self.role_id]
            or self.manifest_entries_digest != ENTRIES_DIGESTS[self.role_id]
            or self.attack_id not in ATTACKS
            or self.internal_decoy_score_count not in {0, 8}
            or self.internal_decoy_roster_identity
            != INTERNAL_DECOY_ROSTER_IDENTITY
            or self.internal_decoy_roster_digest
            != INTERNAL_DECOY_ROSTER_DIGEST
            or self.external_wrong_key_roster_identity
            != EXTERNAL_WRONG_KEY_ROSTER_IDENTITY
            or self.external_wrong_key_roster_digest
            != EXTERNAL_WRONG_KEY_ROSTER_DIGEST
            or self.record_id != canonical_digest(self.payload_without_record_id())
        ):
            raise ContrastiveLfProtocolError("record_template_invalid")
        for name in (
            "record_id",
            "source_cluster_id",
            "prompt_digest",
            "image_lineage_identity",
            "image_lineage_digest",
        ):
            _require_digest(getattr(self, name), name)
        if self.source_row not in SOURCE_ROWS_BY_ROLE[self.role_id]:
            raise ContrastiveLfProtocolError("record_template_source_row_invalid")
        source_ordinal = SOURCE_ROWS_BY_ROLE[self.role_id].index(self.source_row)
        ContrastiveLfManifestEntry(
            source_row=self.source_row,
            prompt_text=self.prompt_text,
            prompt_digest=self.prompt_digest,
            generation_seed=self.generation_seed,
            source_cluster_id=self.source_cluster_id,
            registered_key_family_digest=self.key_family_namespace_digest,
            image_lineage_identity=self.image_lineage_identity,
            image_lineage_digest=self.image_lineage_digest,
        ).validate(role_id=self.role_id, ordinal=source_ordinal)
        if self.key_role not in {None, "registered", "wrong"}:
            raise ContrastiveLfProtocolError("record_template_key_role_invalid")
        if self.key_role == "wrong":
            if self.wrong_key_index not in EXTERNAL_WRONG_KEY_INDEXES:
                raise ContrastiveLfProtocolError(
                    "record_template_wrong_key_invalid"
                )
        elif self.wrong_key_index is not None:
            raise ContrastiveLfProtocolError("record_template_wrong_key_invalid")
        if self.internal_decoy_score_count == 8 and self.candidate_id not in (
            CANDIDATE_IDS
        ):
            raise ContrastiveLfProtocolError("record_template_decoy_invalid")


def _template(
    *,
    entry: ContrastiveLfManifestEntry,
    role_id: str,
    slot_ordinal: int,
    record_kind: str,
    candidate_id: str | None,
    arm_id: str,
    attack_id: str,
    key_role: str | None = None,
    wrong_key_index: int | None = None,
    control_identity: str,
) -> ContrastiveLfRecordTemplate:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "role_id": role_id,
        "slot_ordinal": slot_ordinal,
        "record_kind": record_kind,
        "source_row": entry.source_row,
        "source_cluster_id": entry.source_cluster_id,
        "prompt_text": entry.prompt_text,
        "prompt_digest": entry.prompt_digest,
        "generation_seed": entry.generation_seed,
        "image_lineage_identity": entry.image_lineage_identity,
        "image_lineage_digest": entry.image_lineage_digest,
        "key_family_namespace_digest": KEY_FAMILY_NAMESPACE_DIGEST,
        "prompt_roster_digest": PROMPT_ROSTER_DIGEST,
        "source_roster_rows_digest": SOURCE_ROSTER_ROWS_DIGEST,
        "sample_manifest_digest": MANIFEST_DIGESTS[role_id],
        "manifest_entries_digest": ENTRIES_DIGESTS[role_id],
        "candidate_id": candidate_id,
        "arm_id": arm_id,
        "attack_id": attack_id,
        "key_role": key_role,
        "wrong_key_index": wrong_key_index,
        "control_identity": control_identity,
        "internal_decoy_score_count": (
            8 if candidate_id in CANDIDATE_IDS and record_kind in {
                "null_statistic",
                "detector",
            } else 0
        ),
        "internal_decoy_roster_identity": INTERNAL_DECOY_ROSTER_IDENTITY,
        "internal_decoy_roster_digest": INTERNAL_DECOY_ROSTER_DIGEST,
        "external_wrong_key_roster_identity": (
            EXTERNAL_WRONG_KEY_ROSTER_IDENTITY
        ),
        "external_wrong_key_roster_digest": EXTERNAL_WRONG_KEY_ROSTER_DIGEST,
    }
    template = ContrastiveLfRecordTemplate(
        record_id=canonical_digest(payload), **payload
    )
    template.validate()
    return template


def build_record_templates(
    manifest: ContrastiveLfManifest,
    *,
    selected_candidate_id: str | None = None,
) -> tuple[ContrastiveLfRecordTemplate, ...]:
    """Preallocate the complete fixed-denominator record order for one role."""

    manifest.validate()
    if manifest.role_id == CONFIRMATION_ROLE:
        if selected_candidate_id not in CANDIDATE_IDS:
            raise ContrastiveLfProtocolError(
                "confirmation_selected_candidate_missing"
            )
    elif selected_candidate_id is not None:
        raise ContrastiveLfProtocolError("unexpected_selected_candidate")
    specs: list[dict[str, object]] = []
    if manifest.role_id == NULL_FIT_ROLE:
        for entry in manifest.entries:
            specs.append(
                {
                    "entry": entry,
                    "record_kind": "clean_base_observation",
                    "candidate_id": None,
                    "arm_id": "clean_unwatermarked",
                    "attack_id": "identity",
                    "control_identity": "clean_base_observation",
                }
            )
            for candidate_id in (
                HF_CANDIDATE_ID,
                MULTISCALE_CANDIDATE_ID,
                SINGLE_SCALE_CANDIDATE_ID,
            ):
                specs.append(
                    {
                        "entry": entry,
                        "record_kind": "null_statistic",
                        "candidate_id": candidate_id,
                        "arm_id": "clean_unwatermarked",
                        "attack_id": "identity",
                        "key_role": "registered",
                        "control_identity": "clean_primary_null",
                    }
                )
    else:
        if manifest.role_id == SELECTION_ROLE:
            arms = SELECTION_ARMS
            watermarked = (
                (HF_CANDIDATE_ID, "hf_only"),
                (MULTISCALE_CANDIDATE_ID, "multiscale_low_frequency_only"),
                (SINGLE_SCALE_CANDIDATE_ID, "single_scale_low_frequency_only"),
            )
        else:
            arms = CONFIRMATION_ARMS
            watermarked = (
                (HF_CANDIDATE_ID, "hf_only"),
                (selected_candidate_id, "selected_low_frequency_only"),
            )
        for entry in manifest.entries:
            for arm_id in arms:
                specs.append(
                    {
                        "entry": entry,
                        "record_kind": "base_generation",
                        "candidate_id": (
                            None
                            if arm_id == "clean_unwatermarked"
                            else next(
                                candidate
                                for candidate, arm in watermarked
                                if arm == arm_id
                            )
                        ),
                        "arm_id": arm_id,
                        "attack_id": "identity",
                        "control_identity": "base_generation",
                    }
                )
            for arm_id in arms:
                for attack_id in ATTACKS:
                    candidate_id = None
                    if arm_id != "clean_unwatermarked":
                        candidate_id = next(
                            candidate
                            for candidate, arm in watermarked
                            if arm == arm_id
                        )
                    specs.append(
                        {
                            "entry": entry,
                            "record_kind": "attacked_observation",
                            "candidate_id": candidate_id,
                            "arm_id": arm_id,
                            "attack_id": attack_id,
                            "control_identity": "attacked_observation",
                        }
                    )
            for candidate_id, arm_id in watermarked:
                for attack_id in ATTACKS:
                    specs.append(
                        {
                            "entry": entry,
                            "record_kind": "detector",
                            "candidate_id": candidate_id,
                            "arm_id": arm_id,
                            "attack_id": attack_id,
                            "key_role": "registered",
                            "control_identity": "registered_attribution",
                        }
                    )
                    for wrong_key_index in EXTERNAL_WRONG_KEY_INDEXES:
                        specs.append(
                            {
                                "entry": entry,
                                "record_kind": "detector",
                                "candidate_id": candidate_id,
                                "arm_id": arm_id,
                                "attack_id": attack_id,
                                "key_role": "wrong",
                                "wrong_key_index": wrong_key_index,
                                "control_identity": "external_wrong_key",
                            }
                        )
                    specs.append(
                        {
                            "entry": entry,
                            "record_kind": "detector",
                            "candidate_id": candidate_id,
                            "arm_id": "clean_unwatermarked",
                            "attack_id": attack_id,
                            "key_role": "registered",
                            "control_identity": "paired_primary_null",
                        }
                    )
            for candidate_id, arm_id in watermarked:
                specs.append(
                    {
                        "entry": entry,
                        "record_kind": "budget",
                        "candidate_id": candidate_id,
                        "arm_id": arm_id,
                        "attack_id": "identity",
                        "control_identity": "actual_binary32_budget",
                    }
                )
                for attack_id in ATTACKS:
                    specs.append(
                        {
                            "entry": entry,
                            "record_kind": "quality",
                            "candidate_id": candidate_id,
                            "arm_id": arm_id,
                            "attack_id": attack_id,
                            "control_identity": "paired_rgb8_quality",
                        }
                    )
    records = tuple(
        _template(slot_ordinal=ordinal, role_id=manifest.role_id, **spec)
        for ordinal, spec in enumerate(specs)
    )
    expected = DENOMINATORS_BY_ROLE[manifest.role_id]
    observed_counts = {
        kind: sum(record.record_kind == kind for record in records)
        for kind in RECORD_KINDS
    }
    if (
        observed_counts["clean_base_observation"]
        != expected.clean_base_observation_count
        or observed_counts["null_statistic"]
        != expected.raw_null_statistic_count
        or observed_counts["base_generation"]
        != expected.base_generation_count
        or observed_counts["attacked_observation"]
        != expected.attacked_observation_slot_count
        or observed_counts["detector"] != expected.detector_record_count
        or observed_counts["budget"] != expected.budget_record_count
        or observed_counts["quality"] != expected.quality_record_count
        or len(records) != expected.persisted_record_count
        or len({record.record_id for record in records}) != len(records)
    ):
        raise ContrastiveLfProtocolError("record_denominator_drifted")
    return records


@dataclass(frozen=True, slots=True)
class ContrastiveLfRecord:
    """Flattenable governed record for one preallocated slot."""

    template: ContrastiveLfRecordTemplate
    attempt_index: int
    execution_status: str
    method_config_digest: str
    implementation_revision: str
    model_identity: str
    runtime_identity: str
    codec_identity: str
    raw_score: float | None
    internal_decoy_scores: tuple[float, ...]
    registered_score: float | None
    wrong_key_score: float | None
    primary_null_score: float | None
    population_mean: float | None
    population_variance: float | None
    population_sigma: float | None
    null_asset_digest: str | None
    provisional_threshold_digest: str | None
    z_score: float | None
    key_margin: float | None
    budget_status: str | None
    materialization_replay_identity: str | None
    replay_digest: str | None
    nonfinite_detected: bool
    paired_rgb8_mse: float | None
    failure_class: str | None
    failure_reason: str | None

    def canonical_payload(self) -> dict[str, object]:
        return {**asdict(self.template), **{
            field_name: value
            for field_name, value in asdict(self).items()
            if field_name != "template"
        }}

    def validate(self) -> None:
        self.template.validate()
        if (
            self.attempt_index != 0
            or self.execution_status not in EXECUTION_STATUSES
            or self.method_config_digest != CONFIG_DIGEST
            or _REVISION.fullmatch(self.implementation_revision) is None
            or type(self.model_identity) is not str
            or not self.model_identity
            or type(self.runtime_identity) is not str
            or not self.runtime_identity
            or self.codec_identity != "pillow_rgb8_jpeg_exact_capability"
            or type(self.nonfinite_detected) is not bool
        ):
            raise ContrastiveLfProtocolError("record_authority_invalid")
        for value in (
            self.raw_score,
            self.registered_score,
            self.wrong_key_score,
            self.primary_null_score,
            self.population_mean,
            self.population_variance,
            self.population_sigma,
            self.z_score,
            self.key_margin,
            self.paired_rgb8_mse,
        ):
            if value is not None and not isfinite(value):
                raise ContrastiveLfProtocolError("record_nonfinite")
        if any(not isfinite(value) for value in self.internal_decoy_scores):
            raise ContrastiveLfProtocolError("record_decoy_nonfinite")
        expected_decoys = self.template.internal_decoy_score_count
        if self.execution_status == "completed" and len(
            self.internal_decoy_scores
        ) != expected_decoys:
            raise ContrastiveLfProtocolError("record_decoy_count_drifted")
        for field_name in (
            "null_asset_digest",
            "provisional_threshold_digest",
            "replay_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_digest(value, field_name)
        if self.execution_status == "completed":
            if self.failure_class is not None or self.failure_reason is not None:
                raise ContrastiveLfProtocolError("completed_record_has_failure")
            if self.nonfinite_detected:
                raise ContrastiveLfProtocolError("completed_record_nonfinite")
            if self.template.record_kind in {"null_statistic", "detector"}:
                if self.raw_score is None:
                    raise ContrastiveLfProtocolError(
                        "completed_score_record_missing_raw_score"
                    )
                if self.template.control_identity == "registered_attribution":
                    if self.registered_score != self.raw_score:
                        raise ContrastiveLfProtocolError(
                            "registered_score_binding_drifted"
                        )
                elif self.template.control_identity == "external_wrong_key":
                    if self.wrong_key_score != self.raw_score:
                        raise ContrastiveLfProtocolError(
                            "wrong_key_score_binding_drifted"
                        )
                elif self.template.control_identity == "paired_primary_null":
                    if self.primary_null_score != self.raw_score:
                        raise ContrastiveLfProtocolError(
                            "primary_null_score_binding_drifted"
                        )
            if self.template.record_kind == "budget" and (
                self.budget_status != "accepted"
                or self.materialization_replay_identity is None
                or self.replay_digest is None
            ):
                raise ContrastiveLfProtocolError(
                    "completed_budget_record_invalid"
                )
            if self.template.record_kind == "quality" and (
                self.paired_rgb8_mse is None or self.paired_rgb8_mse < 0.0
            ):
                raise ContrastiveLfProtocolError(
                    "completed_quality_record_invalid"
                )
        elif self.execution_status == "failed":
            validate_bounded_failure(self.failure_class, self.failure_reason)
            if _record_science_evidence_present(self):
                raise ContrastiveLfProtocolError("failed_record_has_evidence")
        elif (
            _record_science_evidence_present(self)
            or self.failure_class is not None
            or self.failure_reason is not None
            or self.nonfinite_detected
        ):
            raise ContrastiveLfProtocolError("unstarted_record_has_evidence")


def _record_science_evidence_present(record: ContrastiveLfRecord) -> bool:
    return bool(record.internal_decoy_scores) or any(
        value is not None
        for value in (
            record.raw_score,
            record.registered_score,
            record.wrong_key_score,
            record.primary_null_score,
            record.population_mean,
            record.population_variance,
            record.population_sigma,
            record.null_asset_digest,
            record.provisional_threshold_digest,
            record.z_score,
            record.key_margin,
            record.budget_status,
            record.materialization_replay_identity,
            record.replay_digest,
            record.paired_rgb8_mse,
        )
    )


def validate_bounded_failure(
    failure_class: str | None, failure_reason: str | None
) -> None:
    if (
        failure_class
        not in {
            "identity_failure",
            "integrity_failure",
            "runtime_failure",
            "dependency_failure",
            "codec_failure",
            "resource_failure",
            "operation_failure",
        }
        or type(failure_reason) is not str
        or not failure_reason
        or len(failure_reason.encode("utf-8")) > 256
        or _FORBIDDEN_FAILURE_TEXT.search(failure_reason) is not None
    ):
        raise ContrastiveLfProtocolError("bounded_failure_invalid")


def validate_failure_tail(records: Sequence[ContrastiveLfRecord]) -> None:
    """Allow one failed slot followed only by the preallocated unstarted tail."""

    if not records:
        raise ContrastiveLfProtocolError("record_collection_empty")
    first_failure: int | None = None
    for ordinal, record in enumerate(records):
        record.validate()
        if record.template.slot_ordinal != ordinal:
            raise ContrastiveLfProtocolError("record_slot_order_drifted")
        if record.execution_status == "failed":
            if first_failure is not None:
                raise ContrastiveLfProtocolError("multiple_failed_slots")
            first_failure = ordinal
        elif first_failure is None and record.execution_status != "completed":
            raise ContrastiveLfProtocolError("premature_unstarted_slot")
        elif first_failure is not None and record.execution_status != "unstarted":
            raise ContrastiveLfProtocolError("failure_tail_not_unstarted")


@dataclass(frozen=True, slots=True)
class DenominatorReport:
    record_kind: str
    expected_record_count: int
    completed_record_count: int
    failed_record_count: int
    unstarted_record_count: int

    def validate(self) -> None:
        if (
            self.record_kind not in RECORD_KINDS
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.expected_record_count,
                    self.completed_record_count,
                    self.failed_record_count,
                    self.unstarted_record_count,
                )
            )
            or self.completed_record_count
            + self.failed_record_count
            + self.unstarted_record_count
            != self.expected_record_count
            or self.failed_record_count > 1
        ):
            raise ContrastiveLfProtocolError("denominator_report_invalid")


_VALIDATED_COLLECTION_CAPABILITY = object()


def _derive_record_collection(
    records: tuple[ContrastiveLfRecord, ...],
    *,
    role_id: str,
    selected_candidate_id: str | None,
) -> tuple[
    tuple[DenominatorReport, ...], str, ContrastiveLfRecord | None
]:
    if role_id not in {SELECTION_ROLE, CONFIRMATION_ROLE}:
        raise ContrastiveLfProtocolError("record_collection_role_invalid")
    if (
        role_id == SELECTION_ROLE
        and selected_candidate_id is not None
    ) or (
        role_id == CONFIRMATION_ROLE
        and selected_candidate_id not in CANDIDATE_IDS
    ):
        raise ContrastiveLfProtocolError(
            "record_collection_selected_candidate_invalid"
        )
    repository_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(
        repository_root / MANIFEST_PATHS[role_id], expected_role=role_id
    )
    expected_templates = build_record_templates(
        manifest, selected_candidate_id=selected_candidate_id
    )
    if len(records) != len(expected_templates):
        raise ContrastiveLfProtocolError("record_collection_length_drifted")
    for record, expected_template in zip(
        records, expected_templates, strict=True
    ):
        if (
            type(record) is not ContrastiveLfRecord
            or record.template != expected_template
        ):
            raise ContrastiveLfProtocolError(
                "record_collection_template_drifted"
            )
    validate_failure_tail(records)
    expected = DENOMINATORS_BY_ROLE[role_id]
    expected_by_kind = {
        "base_generation": expected.base_generation_count,
        "attacked_observation": expected.attacked_observation_slot_count,
        "detector": expected.detector_record_count,
        "budget": expected.budget_record_count,
        "quality": expected.quality_record_count,
    }
    reports = tuple(
        DenominatorReport(
            record_kind=record_kind,
            expected_record_count=expected_count,
            completed_record_count=sum(
                record.template.record_kind == record_kind
                and record.execution_status == "completed"
                for record in records
            ),
            failed_record_count=sum(
                record.template.record_kind == record_kind
                and record.execution_status == "failed"
                for record in records
            ),
            unstarted_record_count=sum(
                record.template.record_kind == record_kind
                and record.execution_status == "unstarted"
                for record in records
            ),
        )
        for record_kind, expected_count in expected_by_kind.items()
    )
    for report in reports:
        report.validate()
    failed_records = tuple(
        record for record in records if record.execution_status == "failed"
    )
    if len(failed_records) > 1 or sum(
        report.failed_record_count for report in reports
    ) > 1:
        raise ContrastiveLfProtocolError("record_collection_failure_count_invalid")
    return (
        reports,
        canonical_digest([record.canonical_payload() for record in records]),
        failed_records[0] if failed_records else None,
    )


@dataclass(frozen=True, slots=True, init=False)
class ValidatedContrastiveLfRecordCollection:
    """Exact record collection authority produced only by validation."""

    role_id: str
    selected_candidate_id: str | None
    records: tuple[ContrastiveLfRecord, ...]
    denominator_reports: tuple[DenominatorReport, ...]
    record_collection_digest: str
    failed_record: ContrastiveLfRecord | None
    _capability: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        role_id: str,
        selected_candidate_id: str | None,
        records: tuple[ContrastiveLfRecord, ...],
        denominator_reports: tuple[DenominatorReport, ...],
        record_collection_digest: str,
        failed_record: ContrastiveLfRecord | None,
        _capability: object,
    ) -> None:
        if _capability is not _VALIDATED_COLLECTION_CAPABILITY:
            raise ContrastiveLfProtocolError(
                "record_collection_capability_invalid"
            )
        object.__setattr__(self, "role_id", role_id)
        object.__setattr__(self, "selected_candidate_id", selected_candidate_id)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "denominator_reports", denominator_reports)
        object.__setattr__(self, "record_collection_digest", record_collection_digest)
        object.__setattr__(self, "failed_record", failed_record)
        object.__setattr__(self, "_capability", _capability)

    def validate(self) -> None:
        if self._capability is not _VALIDATED_COLLECTION_CAPABILITY:
            raise ContrastiveLfProtocolError(
                "record_collection_capability_invalid"
            )
        reports, digest, failed_record = _derive_record_collection(
            self.records,
            role_id=self.role_id,
            selected_candidate_id=self.selected_candidate_id,
        )
        if (
            reports != self.denominator_reports
            or digest != self.record_collection_digest
            or failed_record != self.failed_record
        ):
            raise ContrastiveLfProtocolError(
                "record_collection_capability_drifted"
            )

    @property
    def denominator_complete(self) -> bool:
        return all(
            report.completed_record_count == report.expected_record_count
            for report in self.denominator_reports
        )

    @property
    def operational_failure_observed(self) -> bool:
        return (
            self.failed_record is not None
            and self.failed_record.failure_class in OPERATIONAL_FAILURE_CLASSES
        )


def validate_record_collection(
    records: Sequence[ContrastiveLfRecord],
    *,
    role_id: str,
    selected_candidate_id: str | None = None,
) -> ValidatedContrastiveLfRecordCollection:
    """Validate exact templates, statuses, denominators, and collection digest."""

    record_tuple = tuple(records)
    reports, digest, failed_record = _derive_record_collection(
        record_tuple,
        role_id=role_id,
        selected_candidate_id=selected_candidate_id,
    )
    return ValidatedContrastiveLfRecordCollection(
        role_id=role_id,
        selected_candidate_id=selected_candidate_id,
        records=record_tuple,
        denominator_reports=reports,
        record_collection_digest=digest,
        failed_record=failed_record,
        _capability=_VALIDATED_COLLECTION_CAPABILITY,
    )


@dataclass(frozen=True, slots=True)
class GateReport:
    gate_id: str
    gate_status: str

    def validate(self) -> None:
        if self.gate_id not in GATE_ORDER or self.gate_status not in {
            "passed",
            "failed",
            "not_evaluable",
        }:
            raise ContrastiveLfProtocolError("gate_report_invalid")


@dataclass(frozen=True, slots=True)
class ContrastiveLfProtocolResult:
    schema_version: int
    protocol_id: str
    role_id: str
    sample_manifest_digest: str
    manifest_entries_digest: str
    record_collection_digest: str
    denominator_reports: tuple[DenominatorReport, ...]
    gate_reports: tuple[GateReport, ...]
    first_failed_gate: str | None
    result_classification: str
    candidate_selection_passed: bool
    confirmation_passed: bool
    selected_candidate_id: str | None
    candidate_promoted: bool
    formal_tau_created: bool
    formal_fpr_created: bool
    full_ceg_wm_eligible: bool

    def validate(
        self,
        validated_collection: ValidatedContrastiveLfRecordCollection,
    ) -> None:
        if type(validated_collection) is not ValidatedContrastiveLfRecordCollection:
            raise ContrastiveLfProtocolError(
                "protocol_result_record_collection_invalid"
            )
        validated_collection.validate()
        if (
            self.schema_version != SCHEMA_VERSION
            or self.protocol_id != PROTOCOL_ID
            or self.role_id not in {SELECTION_ROLE, CONFIRMATION_ROLE}
            or self.sample_manifest_digest != MANIFEST_DIGESTS[self.role_id]
            or self.manifest_entries_digest != ENTRIES_DIGESTS[self.role_id]
            or _DIGEST.fullmatch(self.record_collection_digest) is None
            or self.role_id != validated_collection.role_id
            or self.record_collection_digest
            != validated_collection.record_collection_digest
            or self.denominator_reports
            != validated_collection.denominator_reports
            or self.result_classification not in RESULT_CLASSIFICATIONS
            or type(self.candidate_selection_passed) is not bool
            or type(self.confirmation_passed) is not bool
            or self.selected_candidate_id not in {*CANDIDATE_IDS, None}
            or self.candidate_promoted is not False
            or self.formal_tau_created is not False
            or self.formal_fpr_created is not False
            or self.full_ceg_wm_eligible is not False
        ):
            raise ContrastiveLfProtocolError("protocol_result_authority_invalid")
        expected = DENOMINATORS_BY_ROLE[self.role_id]
        expected_by_kind = {
            "base_generation": expected.base_generation_count,
            "attacked_observation": expected.attacked_observation_slot_count,
            "detector": expected.detector_record_count,
            "budget": expected.budget_record_count,
            "quality": expected.quality_record_count,
        }
        if tuple(report.record_kind for report in self.denominator_reports) != tuple(
            expected_by_kind
        ):
            raise ContrastiveLfProtocolError("denominator_report_order_drifted")
        for report in self.denominator_reports:
            report.validate()
            if report.expected_record_count != expected_by_kind[report.record_kind]:
                raise ContrastiveLfProtocolError("denominator_expected_drifted")
        if tuple(report.gate_id for report in self.gate_reports) != GATE_ORDER:
            raise ContrastiveLfProtocolError("gate_report_order_drifted")
        for report in self.gate_reports:
            report.validate()
        expected_first = next(
            (
                report.gate_id
                for report in self.gate_reports
                if report.gate_status == "failed"
            ),
            None,
        )
        if self.first_failed_gate != expected_first:
            raise ContrastiveLfProtocolError("first_failed_gate_drifted")
        complete = all(
            report.completed_record_count == report.expected_record_count
            for report in self.denominator_reports
        )
        gates_pass = all(
            report.gate_status == "passed" for report in self.gate_reports
        )
        if complete != validated_collection.denominator_complete:
            raise ContrastiveLfProtocolError("denominator_completion_drifted")
        expected_classification = classify_result(
            validated_collection=validated_collection,
            scientific_gates_passed=gates_pass,
        )
        if self.result_classification != expected_classification:
            raise ContrastiveLfProtocolError(
                "result_classification_drifted"
            )
        if not complete and (
            expected_first is not None
            or any(
                report.gate_status != "not_evaluable"
                for report in self.gate_reports
            )
        ):
            raise ContrastiveLfProtocolError("incomplete_gate_report_invalid")
        if complete and self.result_classification == "scientific_failure" and (
            expected_first is None
        ):
            raise ContrastiveLfProtocolError("scientific_failure_result_invalid")
        if self.role_id == SELECTION_ROLE:
            expected_selection_pass = self.result_classification == "success"
            if (
                self.candidate_selection_passed is not expected_selection_pass
                or self.confirmation_passed is not False
                or (
                    expected_selection_pass
                    and self.selected_candidate_id not in CANDIDATE_IDS
                )
                or (
                    not expected_selection_pass
                    and self.selected_candidate_id is not None
                )
            ):
                raise ContrastiveLfProtocolError("selection_result_invalid")
        elif self.candidate_selection_passed is not True or (
            self.confirmation_passed
            is not (self.result_classification == "success")
        ):
            raise ContrastiveLfProtocolError("confirmation_result_invalid")
        if self.role_id == CONFIRMATION_ROLE and (
            self.selected_candidate_id
            != validated_collection.selected_candidate_id
        ):
            raise ContrastiveLfProtocolError(
                "confirmation_collection_candidate_drifted"
            )


def population_standardize(
    null_values: Sequence[float], query: float
) -> tuple[float, float, float, float]:
    """Replay manifest-order binary64 population standardization."""

    if len(null_values) != CLUSTER_COUNT or any(
        not isfinite(value) for value in (*null_values, query)
    ):
        raise ContrastiveLfProtocolError("population_values_invalid")
    mean = 0.0
    for value in null_values:
        mean += float(value)
    mean /= CLUSTER_COUNT
    variance = 0.0
    for value in null_values:
        variance += (float(value) - mean) ** 2
    variance /= CLUSTER_COUNT
    sigma = sqrt(variance)
    if not isfinite(sigma) or sigma <= 0.0:
        raise ContrastiveLfProtocolError("population_sigma_invalid")
    z_score = (float(query) - mean) / sigma
    if not isfinite(z_score):
        raise ContrastiveLfProtocolError("population_z_invalid")
    return mean, variance, sigma, z_score


def provisional_tau(z_values: Sequence[float]) -> float:
    """Return nextafter(fourth-largest-z,+inf), preserving ties."""

    if len(z_values) != CLUSTER_COUNT or any(
        not isfinite(value) for value in z_values
    ):
        raise ContrastiveLfProtocolError("provisional_z_values_invalid")
    ordered = sorted(float(value) for value in z_values)
    return nextafter(ordered[-4], float("inf"))


def branch_key_margin(
    registered_z: float, external_wrong_z: Sequence[float]
) -> float:
    if (
        not isfinite(registered_z)
        or len(external_wrong_z) != 8
        or any(not isfinite(value) for value in external_wrong_z)
    ):
        raise ContrastiveLfProtocolError("key_margin_values_invalid")
    return float(registered_z) - max(float(value) for value in external_wrong_z)


def identity_attribution_passes(
    *,
    registered_z: float,
    primary_null_z: float,
    external_wrong_z: Sequence[float],
) -> bool:
    if not isfinite(primary_null_z):
        raise ContrastiveLfProtocolError("primary_null_z_invalid")
    margin = branch_key_margin(registered_z, external_wrong_z)
    return registered_z > primary_null_z and margin > 0.0


def identity_attribution_gate_passes(outcomes: Sequence[bool]) -> bool:
    """Require at least 28 strict paired-attribution successes of fixed 32."""

    if len(outcomes) != CLUSTER_COUNT or any(
        type(outcome) is not bool for outcome in outcomes
    ):
        raise ContrastiveLfProtocolError("identity_attribution_outcomes_invalid")
    return sum(outcomes) >= 28


def quality_gate_passes(
    candidate_mse: Sequence[float], hf_only_mse: Sequence[float]
) -> bool:
    if (
        len(candidate_mse) != CLUSTER_COUNT
        or len(hf_only_mse) != CLUSTER_COUNT
        or any(not isfinite(value) or value < 0.0 for value in candidate_mse)
        or any(not isfinite(value) or value < 0.0 for value in hf_only_mse)
    ):
        raise ContrastiveLfProtocolError("quality_values_invalid")
    candidate_mean = 0.0
    hf_mean = 0.0
    for candidate_value, hf_value in zip(
        candidate_mse, hf_only_mse, strict=True
    ):
        candidate_mean += float(candidate_value)
        hf_mean += float(hf_value)
    candidate_mean /= CLUSTER_COUNT
    hf_mean /= CLUSTER_COUNT
    return candidate_mean <= hf_mean + float.fromhex(
        QUALITY_EPSILON_FLOAT64_HEX
    )


def blur_complement_passes(
    *, success_count: int, clopper_pearson_lower_value: float
) -> bool:
    """Consume the lower bound from the frozen shared exact callable identity."""

    if (
        type(success_count) is not int
        or success_count not in range(CLUSTER_COUNT + 1)
        or not isfinite(clopper_pearson_lower_value)
        or clopper_pearson_lower_value < 0.0
        or clopper_pearson_lower_value > 1.0
    ):
        raise ContrastiveLfProtocolError("blur_complement_values_invalid")
    return success_count >= 24 and clopper_pearson_lower_value > 0.5


def condition_false_positive_gate_passes(
    *,
    primary_null_positive_count: int,
    external_wrong_positive_counts: Sequence[int],
) -> bool:
    """Apply separate, non-pooled three-of-thirty-two condition caps."""

    if (
        type(primary_null_positive_count) is not int
        or primary_null_positive_count not in range(CLUSTER_COUNT + 1)
        or len(external_wrong_positive_counts) != 8
        or any(
            type(value) is not int or value not in range(CLUSTER_COUNT + 1)
            for value in external_wrong_positive_counts
        )
    ):
        raise ContrastiveLfProtocolError("condition_positive_count_invalid")
    return (
        primary_null_positive_count <= 3
        and all(value <= 3 for value in external_wrong_positive_counts)
    )


def choose_selection_winner(
    *, multiscale_passed: bool, single_scale_passed: bool
) -> str | None:
    if type(multiscale_passed) is not bool or type(single_scale_passed) is not bool:
        raise ContrastiveLfProtocolError("candidate_gate_value_invalid")
    if multiscale_passed:
        return MULTISCALE_CANDIDATE_ID
    if single_scale_passed:
        return SINGLE_SCALE_CANDIDATE_ID
    return None


def classify_result(
    *,
    validated_collection: ValidatedContrastiveLfRecordCollection,
    scientific_gates_passed: bool,
) -> str:
    """Apply the exact operational/insufficient/scientific/success partition."""

    if (
        type(validated_collection) is not ValidatedContrastiveLfRecordCollection
        or type(scientific_gates_passed) is not bool
    ):
        raise ContrastiveLfProtocolError("classification_input_invalid")
    validated_collection.validate()
    if validated_collection.operational_failure_observed:
        if validated_collection.denominator_complete:
            raise ContrastiveLfProtocolError(
                "completed_denominator_operational_failure_invalid"
            )
        return "operational_failure"
    if not validated_collection.denominator_complete:
        return "insufficient_evidence"
    if not scientific_gates_passed:
        return "scientific_failure"
    return "success"


def authenticate_selection_artifact(
    artifact: Mapping[str, object], *, expected_artifact_digest: str
) -> str:
    """Authorize confirmation from one exact passed selection winner only."""

    required = {
        "schema_version",
        "protocol_id",
        "selection_manifest_digest",
        "candidate_selection_passed",
        "selected_candidate_id",
        "candidate_null_asset_digest",
        "provisional_threshold_digest",
        "diagnostic_only",
        "formal_tau_created",
        "formal_fpr_created",
        "candidate_promoted",
        "full_ceg_wm_eligible",
    }
    if (
        type(artifact) is not dict
        or set(artifact) != required
        or _DIGEST.fullmatch(expected_artifact_digest) is None
        or canonical_digest(artifact) != expected_artifact_digest
        or artifact["schema_version"] != SCHEMA_VERSION
        or artifact["protocol_id"] != PROTOCOL_ID
        or artifact["selection_manifest_digest"]
        != MANIFEST_DIGESTS[SELECTION_ROLE]
        or artifact["candidate_selection_passed"] is not True
        or artifact["selected_candidate_id"] not in CANDIDATE_IDS
        or _DIGEST.fullmatch(artifact["candidate_null_asset_digest"])
        is None
        or _DIGEST.fullmatch(artifact["provisional_threshold_digest"])
        is None
        or artifact["diagnostic_only"] is not True
        or artifact["formal_tau_created"] is not False
        or artifact["formal_fpr_created"] is not False
        or artifact["candidate_promoted"] is not False
        or artifact["full_ceg_wm_eligible"] is not False
    ):
        raise ContrastiveLfProtocolError("selection_artifact_authority_drifted")
    return artifact["selected_candidate_id"]


__all__ = [
    "ATTACKS",
    "BLIND_DETECTOR_INPUTS",
    "CANDIDATE_IDS",
    "CANDIDATE_ROLE_LABELS",
    "CLUSTER_COUNT",
    "CONFIG_DIGEST",
    "CONFIG_PATH",
    "CONFIRMATION_ROLE",
    "ContrastiveLfManifest",
    "ContrastiveLfManifestEntry",
    "ContrastiveLfProtocolError",
    "ContrastiveLfProtocolResult",
    "ContrastiveLfRecord",
    "ContrastiveLfRecordTemplate",
    "DENOMINATORS_BY_ROLE",
    "DenominatorReport",
    "ENTRIES_DIGESTS",
    "EXTERNAL_WRONG_KEY_INDEXES",
    "FORBIDDEN_DETECTOR_INPUTS",
    "GATE_ORDER",
    "GateReport",
    "INTERNAL_DECOY_INDEXES",
    "MANIFEST_DIGESTS",
    "MANIFEST_PATHS",
    "MULTISCALE_CANDIDATE_ID",
    "NULL_FIT_ROLE",
    "PILLOW_VERSION",
    "PROMPT_ROSTER_DIGEST",
    "PROMPT_ROSTER_PATH",
    "PROTOCOL_ID",
    "RESULT_CLASSIFICATIONS",
    "SCHEMA_VERSION",
    "SELECTION_ROLE",
    "SINGLE_SCALE_CANDIDATE_ID",
    "SOURCE_ROSTER_ROWS_DIGEST",
    "SOURCE_SNAPSHOT_SHA256",
    "SOURCE_SNAPSHOT_PATH",
    "ValidatedContrastiveLfRecordCollection",
    "authenticate_selection_artifact",
    "blur_complement_passes",
    "branch_key_margin",
    "build_record_templates",
    "canonical_digest",
    "choose_selection_winner",
    "classify_result",
    "condition_false_positive_gate_passes",
    "identity_attribution_passes",
    "identity_attribution_gate_passes",
    "load_configuration",
    "load_manifest",
    "load_prompt_roster",
    "population_standardize",
    "provisional_tau",
    "quality_gate_passes",
    "validate_failure_tail",
    "validate_record_collection",
    "validate_split_disjointness",
]
