"""Frozen development protocol for the salient-local-LF mask/write pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import re
from struct import pack, unpack
import subprocess
from typing import Mapping

from experiments.protocol.development_exploration import DevelopmentStudyUnit
from experiments.protocol.internal_splits import AnalysisUnitIdentity, derive_source_cluster_id


RUN_ID = "ceg_wm_salient_local_lf_mask_write_remote_authority_correction_validation"
PROTOCOL_ID = "ceg_wm_salient_local_lf_mask_write_validation"
PROTOCOL_VERSION = "1.0.0"
OPERATIONAL_UNIT_COUNT = 2
SCIENTIFIC_UNIT_COUNT = 8
MAXIMUM_TOTAL_UNITS = 10
MAXIMUM_ATTEMPTS_PER_UNIT = 1
MAXIMUM_DURATION_SECONDS = 2700
CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250
CANONICAL_CONTENT_RELATIVE_L2_LIMIT = unpack(
    ">f", pack(">f", CONTENT_RELATIVE_L2_NUMERATOR / CONTENT_RELATIVE_L2_DENOMINATOR)
)[0]
QUALITY_PIXEL_COUNT = 786432
QUALITY_SQUARED_CODE_DELTA_LIMIT = 786432
MINIMUM_MECHANISM_SUCCESS_COUNT = 7
REQUIRED_QUALITY_SUCCESS_COUNT = 8
SCIENTIFIC_ROSTER_AUTHORITY_DIGEST = (
    "193702e27508436b4b5545470faeaf7eb1cd494533f1355b6c6cf9d70f16e0de"
)
REGISTERED_KEY_FAMILY_DIGEST = (
    "524a6c2ff671220c5e318d818db5e75488741ff83189cb74e2e5c0acdc3adcf5"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class SalientLocalLfMaskWriteProtocolError(ValueError):
    """The frozen protocol, roster, or producer authority drifted."""


_DENY_AXIS_NAMES = (
    "prompt_digests", "generation_seeds", "cluster_identities",
    "source_cluster_ids", "key_lineage_digests", "image_lineage_digests",
    "namespaces", "lineage_authorities",
)


def canonical_digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _collect_deny_axes(value: object, axes: dict[str, set[object]]) -> None:
    if tuple(axes) != _DENY_AXIS_NAMES:
        raise SalientLocalLfMaskWriteProtocolError("deny-axis collector identity drifted")
    if type(value) is dict:
        for key, item in value.items():
            lowered = key.lower()
            scalars = item if type(item) is list else [item]
            for scalar in scalars:
                if lowered in {"prompt", "prompt_text"} and type(scalar) is str:
                    axes["prompt_digests"].add(sha256(scalar.encode("utf-8")).hexdigest())
                elif lowered == "prompt_digest" and type(scalar) is str:
                    axes["prompt_digests"].add(scalar)
                elif lowered == "generation_seed" and type(scalar) is int:
                    axes["generation_seeds"].add(scalar)
                elif lowered == "cluster_identity" and type(scalar) is str:
                    axes["cluster_identities"].add(scalar)
                elif lowered == "source_cluster_id" and type(scalar) is str:
                    axes["source_cluster_ids"].add(scalar)
                elif lowered == "key_lineage_digest" and type(scalar) is str:
                    axes["key_lineage_digests"].add(scalar)
                elif lowered == "image_lineage_digest" and type(scalar) is str:
                    axes["image_lineage_digests"].add(scalar)
                elif lowered.endswith("_namespace") and type(scalar) is str:
                    axes["namespaces"].add(scalar)
                elif lowered in {
                    "key_lineage_identity", "image_lineage_identity",
                    "registered_key_derivation_identity", "registered_key_family_digest",
                } and type(scalar) is str:
                    axes["lineage_authorities"].add(scalar)
            _collect_deny_axes(item, axes)
    elif type(value) is list:
        for item in value:
            _collect_deny_axes(item, axes)


@dataclass(frozen=True, slots=True)
class HistoricalProducerPathBinding:
    path: str
    package_member_path: str
    git_blob_sha: str
    raw_sha256: str

    def validate(self) -> None:
        path = PurePosixPath(self.path)
        package_path = PurePosixPath(self.package_member_path)
        if (path.is_absolute() or ".." in path.parts or not self.path.startswith("configs/experiments/")
                or package_path.is_absolute() or ".." in package_path.parts
                or not self.package_member_path.startswith("historical_authorities/")
                or re.fullmatch(r"[0-9a-f]{40}", self.git_blob_sha) is None
                or _DIGEST.fullmatch(self.raw_sha256) is None):
            raise SalientLocalLfMaskWriteProtocolError("historical producer path binding is invalid")


@dataclass(frozen=True, slots=True)
class HistoricalProducerAuthority:
    authority_identity: str
    producer_revision: str
    paths: tuple[HistoricalProducerPathBinding, ...]

    def validate(self, repository_root: Path) -> tuple[dict[str, object], ...]:
        if not self.authority_identity or _REVISION.fullmatch(self.producer_revision) is None or not self.paths:
            raise SalientLocalLfMaskWriteProtocolError("historical producer authority is invalid")
        documents = []
        package_manifest_path = repository_root / "PACKAGE_MANIFEST.json"
        package_available = package_manifest_path.is_file()
        git_available = not package_available and subprocess.run(
            ("git", "rev-parse", "--git-dir"), cwd=repository_root,
            check=False, capture_output=True, text=True,
        ).returncode == 0
        package_manifest = None
        if package_available:
            try:
                package_manifest = json.loads(package_manifest_path.read_text("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise SalientLocalLfMaskWriteProtocolError("packaged historical authority manifest is unavailable") from exc
            if type(package_manifest) is not dict or type(package_manifest.get("entries")) is not list:
                raise SalientLocalLfMaskWriteProtocolError("packaged historical authority manifest is invalid")
        for binding in self.paths:
            binding.validate()
            if not package_available and git_available:
                try:
                    tree = subprocess.run(
                        ("git", "ls-tree", self.producer_revision, "--", binding.path),
                        cwd=repository_root, check=True, capture_output=True, text=True,
                    ).stdout.strip().split()
                    payload = subprocess.run(
                        ("git", "show", f"{self.producer_revision}:{binding.path}"),
                        cwd=repository_root, check=True, capture_output=True,
                    ).stdout
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise SalientLocalLfMaskWriteProtocolError(
                        "historical producer Git authority is unavailable"
                    ) from exc
                if len(tree) != 4 or tree[:2] != ["100644", "blob"] or tree[2] != binding.git_blob_sha:
                    raise SalientLocalLfMaskWriteProtocolError("historical producer Git blob drifted")
            elif package_available:
                member = repository_root / binding.package_member_path
                try:
                    if not member.is_file() or member.is_symlink():
                        raise OSError("packaged historical producer member is not regular")
                    payload = member.read_bytes()
                except OSError as exc:
                    raise SalientLocalLfMaskWriteProtocolError("packaged historical producer bytes are unavailable") from exc
                matching = [item for item in package_manifest["entries"] if type(item) is dict and item.get("path") == binding.package_member_path]
                if (len(matching) != 1 or matching[0].get("git_blob_sha") != binding.git_blob_sha
                        or matching[0].get("raw_sha256", matching[0].get("sha256")) != binding.raw_sha256
                        or matching[0].get("size") != len(payload)):
                    raise SalientLocalLfMaskWriteProtocolError("packaged historical producer binding drifted")
            else:
                raise SalientLocalLfMaskWriteProtocolError("historical producer authority is unavailable")
            if sha256(payload).hexdigest() != binding.raw_sha256:
                raise SalientLocalLfMaskWriteProtocolError("historical producer bytes drifted")
            try:
                document = json.loads(payload.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise SalientLocalLfMaskWriteProtocolError("historical producer JSON is invalid") from exc
            if type(document) is not dict:
                raise SalientLocalLfMaskWriteProtocolError("historical producer JSON must be a mapping")
            documents.append(document)
        return tuple(documents)


@dataclass(frozen=True, slots=True)
class CurrentExperimentPathBinding:
    path: str
    mode: str
    git_blob_sha: str
    raw_sha256: str
    package_member_path: str

    def validate(self, revision: str) -> None:
        source = PurePosixPath(self.path)
        packaged = PurePosixPath(self.package_member_path)
        prefix = f"authority_inputs/current_{revision}/"
        if (source.is_absolute() or ".." in source.parts
                or not self.path.startswith("configs/experiments/")
                or self.mode != "100644" or _REVISION.fullmatch(self.git_blob_sha) is None
                or _DIGEST.fullmatch(self.raw_sha256) is None
                or packaged.is_absolute() or ".." in packaged.parts
                or self.package_member_path != prefix + self.path):
            raise SalientLocalLfMaskWriteProtocolError("current experiment path binding is invalid")


def _package_manifest(repository_root: Path) -> dict[str, object] | None:
    path = repository_root / "PACKAGE_MANIFEST.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SalientLocalLfMaskWriteProtocolError("execution package manifest is invalid") from exc
    if type(value) is not dict or type(value.get("entries")) is not list:
        raise SalientLocalLfMaskWriteProtocolError("execution package manifest is invalid")
    return value


@dataclass(frozen=True, slots=True)
class CurrentExperimentAuthority:
    authority_identity: str
    producer_revision: str
    configs_experiments_tree_oid: str
    tracked_path_count: int
    parti_prompt_asset_path: str
    parti_prompt_asset_data_row_count: int
    parti_prompt_asset_unique_prompt_digest_count: int
    current_unique_prompt_digest_count: int
    paths: tuple[CurrentExperimentPathBinding, ...]

    def validate_and_read(self, repository_root: Path) -> tuple[tuple[str, bytes], ...]:
        if (self.authority_identity != "current_experiment_inputs_at_salient_local_lf_authorization_base"
                or self.producer_revision != "061991c67bb0ceb3fbfe3359a2d86b78f301f171"
                or self.configs_experiments_tree_oid != "829bf5200754c6f54c6ee422188e9184776793ab"
                or self.tracked_path_count != 27 or len(self.paths) != 27
                or self.parti_prompt_asset_data_row_count != 1632
                or self.parti_prompt_asset_unique_prompt_digest_count != 1632
                or self.current_unique_prompt_digest_count != 1724):
            raise SalientLocalLfMaskWriteProtocolError("current experiment authority drifted")
        if tuple(item.path for item in self.paths) != tuple(sorted(item.path for item in self.paths)):
            raise SalientLocalLfMaskWriteProtocolError("current experiment inventory is not sorted")
        package = _package_manifest(repository_root)
        payloads: list[tuple[str, bytes]] = []
        if package is None:
            try:
                tree_line = subprocess.run(
                    ("git", "ls-tree", self.producer_revision, "--", "configs/experiments"),
                    cwd=repository_root, check=True, capture_output=True, text=True,
                ).stdout.strip().split()
                tree_bytes = subprocess.run(
                    ("git", "ls-tree", "-r", "-z", self.producer_revision, "--", "configs/experiments"),
                    cwd=repository_root, check=True, capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as exc:
                raise SalientLocalLfMaskWriteProtocolError("current experiment Git authority is unavailable") from exc
            if len(tree_line) != 4 or tree_line[:2] != ["040000", "tree"] or tree_line[2] != self.configs_experiments_tree_oid:
                raise SalientLocalLfMaskWriteProtocolError("current experiment root tree drifted")
            observed: list[tuple[str, str, str]] = []
            for item in tree_bytes.split(b"\0"):
                if not item:
                    continue
                metadata, encoded = item.split(b"\t", 1)
                mode, kind, blob = metadata.decode("ascii").split()
                if kind != "blob":
                    raise SalientLocalLfMaskWriteProtocolError("current experiment tree contains a non-blob")
                observed.append((encoded.decode("utf-8"), mode, blob))
            expected = [(item.path, item.mode, item.git_blob_sha) for item in self.paths]
            if observed != expected:
                raise SalientLocalLfMaskWriteProtocolError("current experiment Git inventory drifted")
            for binding in self.paths:
                binding.validate(self.producer_revision)
                try:
                    payload = subprocess.run(
                        ("git", "cat-file", "blob", binding.git_blob_sha), cwd=repository_root,
                        check=True, capture_output=True,
                    ).stdout
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise SalientLocalLfMaskWriteProtocolError("current experiment Git blob is unavailable") from exc
                payloads.append((binding.path, payload))
        else:
            roots = package.get("authority_root_tree_oids")
            if type(roots) is not dict or roots.get(self.producer_revision) != self.configs_experiments_tree_oid:
                raise SalientLocalLfMaskWriteProtocolError("packaged current authority root tree drifted")
            entries = package["entries"]
            for binding in self.paths:
                binding.validate(self.producer_revision)
                matches = [item for item in entries if type(item) is dict and item.get("path") == binding.package_member_path]
                member = repository_root / binding.package_member_path
                try:
                    if not member.is_file() or member.is_symlink():
                        raise OSError("packaged current authority member is not regular")
                    payload = member.read_bytes()
                except OSError as exc:
                    raise SalientLocalLfMaskWriteProtocolError("packaged current authority bytes are unavailable") from exc
                if (len(matches) != 1 or matches[0].get("mode") != binding.mode
                        or matches[0].get("git_blob_sha") != binding.git_blob_sha
                        or matches[0].get("raw_sha256", matches[0].get("sha256")) != binding.raw_sha256
                        or matches[0].get("size") != len(payload)):
                    raise SalientLocalLfMaskWriteProtocolError("packaged current authority binding drifted")
                payloads.append((binding.path, payload))
        if any(sha256(payload).hexdigest() != binding.raw_sha256
               for binding, (_path, payload) in zip(self.paths, payloads, strict=True)):
            raise SalientLocalLfMaskWriteProtocolError("current experiment bytes drifted")
        asset = dict(payloads).get(self.parti_prompt_asset_path)
        if asset is None:
            raise SalientLocalLfMaskWriteProtocolError("current prompt asset is unavailable")
        try:
            rows = list(csv.reader(io.StringIO(asset.decode("utf-8")), delimiter="\t"))
        except UnicodeError as exc:
            raise SalientLocalLfMaskWriteProtocolError("current prompt asset is invalid") from exc
        if (not rows or rows[0] != ["Prompt", "Category", "Challenge", "Note"]
                or len(rows[1:]) != self.parti_prompt_asset_data_row_count
                or any(len(row) != 4 or not row[0] for row in rows[1:])
                or len({sha256(row[0].encode("utf-8")).hexdigest() for row in rows[1:]})
                != self.parti_prompt_asset_unique_prompt_digest_count):
            raise SalientLocalLfMaskWriteProtocolError("current prompt asset authority drifted")
        return tuple(payloads)


@dataclass(frozen=True, slots=True)
class FutureSplitDenyAuthority:
    schema_version: str
    exclusion_roles: tuple[str, ...]
    prompt_digests: tuple[str, ...]
    generation_seeds: tuple[int, ...]
    cluster_identities: tuple[str, ...]
    source_cluster_ids: tuple[str, ...]
    key_lineage_digests: tuple[str, ...]
    image_lineage_digests: tuple[str, ...]
    seed_namespace: str
    source_cluster_namespace: str
    image_lineage_namespace: str
    image_lineage_identity: str
    key_lineage_namespace: str
    key_lineage_identity: str
    registered_key_derivation_identity: str
    registered_key_family_digest: str
    scientific_roster_authority_digest: str

    def validate(self, roster: "SalientLocalLfScientificRoster", expected_digest: str) -> None:
        entries = roster.entries
        expected = {
            "schema_version": "ceg_wm_salient_local_lf_future_split_deny_authority_v1",
            "exclusion_roles": ("masked_lf_whitening_fit", "independent_confirmation", "candidate_selection", "calibration", "evaluation"),
            "prompt_digests": tuple(item.prompt_digest for item in entries),
            "generation_seeds": tuple(item.generation_seed for item in entries),
            "cluster_identities": tuple(item.cluster_identity for item in entries),
            "source_cluster_ids": tuple(item.source_cluster_id for item in entries),
            "key_lineage_digests": tuple(item.key_lineage_digest for item in entries),
            "image_lineage_digests": tuple(item.image_lineage_digest for item in entries),
            "seed_namespace": roster.seed_namespace,
            "source_cluster_namespace": roster.source_cluster_namespace,
            "image_lineage_namespace": roster.image_lineage_namespace,
            "image_lineage_identity": roster.image_lineage_identity,
            "key_lineage_namespace": roster.key_lineage_namespace,
            "key_lineage_identity": roster.key_lineage_identity,
            "registered_key_derivation_identity": roster.registered_key_derivation_identity,
            "registered_key_family_digest": roster.registered_key_family_digest,
            "scientific_roster_authority_digest": roster.scientific_roster_authority_digest,
        }
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise SalientLocalLfMaskWriteProtocolError("future split deny authority drifted")
        if _DIGEST.fullmatch(expected_digest) is None or canonical_digest(asdict(self)) != expected_digest:
            raise SalientLocalLfMaskWriteProtocolError("future split deny authority digest drifted")


@dataclass(frozen=True, slots=True)
class SalientLocalLfScientificRosterEntry:
    entry_role: str
    cluster_ordinal: int
    cluster_identity: str
    source_cluster_id: str
    prompt: str
    prompt_digest: str
    generation_seed: int
    key_lineage_identity: str
    key_lineage_digest: str
    image_lineage_identity: str
    image_lineage_digest: str

    def validate(self, manifest: "SalientLocalLfScientificRoster") -> None:
        if (self.entry_role != "scientific_mask_write_validation"
                or type(self.cluster_ordinal) is not int
                or not 0 <= self.cluster_ordinal < SCIENTIFIC_UNIT_COUNT
                or type(self.generation_seed) is not int
                or any(type(value) is not str or not value for value in (
                    self.cluster_identity, self.prompt, self.key_lineage_identity,
                    self.image_lineage_identity,
                ))):
            raise SalientLocalLfMaskWriteProtocolError("scientific roster entry is invalid")
        if self.prompt_digest != sha256(self.prompt.encode("utf-8")).hexdigest():
            raise SalientLocalLfMaskWriteProtocolError("scientific prompt digest drifted")
        if self.key_lineage_identity != manifest.key_lineage_identity or self.image_lineage_identity != manifest.image_lineage_identity:
            raise SalientLocalLfMaskWriteProtocolError("scientific lineage identity drifted")
        expected_image = canonical_digest({
            "cluster_identity": self.cluster_identity,
            "entry_role": self.entry_role,
            "generation_seed": self.generation_seed,
            "image_lineage_identity": self.image_lineage_identity,
            "image_lineage_namespace": manifest.image_lineage_namespace,
            "prompt_digest": self.prompt_digest,
        })
        expected_key = canonical_digest({
            "cluster_identity": self.cluster_identity,
            "generation_seed": self.generation_seed,
            "key_lineage_identity": self.key_lineage_identity,
            "key_lineage_namespace": manifest.key_lineage_namespace,
            "prompt_digest": self.prompt_digest,
            "registered_key_family_digest": manifest.registered_key_family_digest,
        })
        expected_source = derive_source_cluster_id(
            prompt_digest=self.prompt_digest,
            generation_seed=self.generation_seed,
            image_lineage_digest=self.image_lineage_digest,
            registered_key_family_digest=manifest.registered_key_family_digest,
        )
        if (self.image_lineage_digest != expected_image or self.key_lineage_digest != expected_key
                or self.source_cluster_id != expected_source):
            raise SalientLocalLfMaskWriteProtocolError("scientific roster derived identity drifted")


@dataclass(frozen=True, slots=True)
class SalientLocalLfScientificRoster:
    schema_version: str
    manifest_id: str
    role_id: str
    split: str
    seed_namespace: str
    source_cluster_namespace: str
    image_lineage_namespace: str
    image_lineage_identity: str
    key_lineage_namespace: str
    key_lineage_identity: str
    registered_key_derivation_identity: str
    registered_key_family_digest: str
    scientific_roster_authority_digest: str
    future_split_deny_authority_digest: str
    future_split_deny_authority: FutureSplitDenyAuthority
    entries: tuple[SalientLocalLfScientificRosterEntry, ...]

    def authority_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed_namespace": self.seed_namespace,
            "source_cluster_namespace": self.source_cluster_namespace,
            "image_lineage_namespace": self.image_lineage_namespace,
            "key_lineage_namespace": self.key_lineage_namespace,
            "registered_key_family_digest": self.registered_key_family_digest,
            "entries": [asdict(item) for item in self.entries],
        }

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def validate(self) -> None:
        if (self.schema_version != "ceg_wm_salient_local_lf_mask_write_scientific_roster_v1"
                or self.manifest_id != "salient_local_lf_mask_write_validation_scientific_roster"
                or self.role_id != "salient_local_lf_mask_write_validation"
                or self.split != "development"
                or self.registered_key_family_digest != REGISTERED_KEY_FAMILY_DIGEST
                or self.scientific_roster_authority_digest != SCIENTIFIC_ROSTER_AUTHORITY_DIGEST
                or canonical_digest(self.authority_payload()) != SCIENTIFIC_ROSTER_AUTHORITY_DIGEST
                or tuple(item.cluster_ordinal for item in self.entries) != tuple(range(SCIENTIFIC_UNIT_COUNT))):
            raise SalientLocalLfMaskWriteProtocolError("scientific roster authority drifted")
        for entry in self.entries:
            entry.validate(self)
        self.future_split_deny_authority.validate(
            self, self.future_split_deny_authority_digest,
        )
        axes = (
            tuple(item.cluster_identity for item in self.entries),
            tuple(item.source_cluster_id for item in self.entries),
            tuple(item.prompt_digest for item in self.entries),
            tuple(item.generation_seed for item in self.entries),
            tuple(item.key_lineage_digest for item in self.entries),
            tuple(item.image_lineage_digest for item in self.entries),
        )
        if any(len(set(axis)) != SCIENTIFIC_UNIT_COUNT for axis in axes):
            raise SalientLocalLfMaskWriteProtocolError("scientific roster axes collide")


@dataclass(frozen=True, slots=True)
class SalientLocalLfMaskWriteProtocol:
    raw: Mapping[str, object]
    manifest: SalientLocalLfScientificRoster
    current_experiment_authority: CurrentExperimentAuthority
    historical_prior_authorities: tuple[HistoricalProducerAuthority, ...]
    unit_roster: tuple[DevelopmentStudyUnit, ...]

    @property
    def run_id(self) -> str: return str(self.raw["run_id"])
    @property
    def protocol_id(self) -> str: return str(self.raw["protocol_id"])
    @property
    def protocol_version(self) -> str: return str(self.raw["protocol_version"])
    @property
    def operational_prompt(self) -> str: return str(self.raw["operational_prompt"])
    @property
    def operational_prompt_digest(self) -> str: return str(self.raw["operational_prompt_digest"])
    @property
    def operational_generation_seed(self) -> int: return int(self.raw["operational_generation_seed"])
    @property
    def unit_roster_digest(self) -> str: return canonical_digest([asdict(item) for item in self.unit_roster])
    def digest(self) -> str: return canonical_digest(dict(self.raw))

    def validate(self, repository_root: Path) -> None:
        expected = {
            "run_id": RUN_ID, "protocol_id": PROTOCOL_ID,
            "operational_unit_count": OPERATIONAL_UNIT_COUNT,
            "scientific_unit_count": SCIENTIFIC_UNIT_COUNT,
            "maximum_total_units": MAXIMUM_TOTAL_UNITS,
            "maximum_attempts_per_unit": MAXIMUM_ATTEMPTS_PER_UNIT,
            "quality_pixel_count": QUALITY_PIXEL_COUNT,
            "quality_squared_code_delta_limit": QUALITY_SQUARED_CODE_DELTA_LIMIT,
            "minimum_mechanism_success_count": MINIMUM_MECHANISM_SUCCESS_COUNT,
            "required_quality_success_count": REQUIRED_QUALITY_SUCCESS_COUNT,
        }
        if any(self.raw.get(key) != value for key, value in expected.items()):
            raise SalientLocalLfMaskWriteProtocolError("protocol frozen scalar drifted")
        if self.raw.get("content_relative_l2_numerator") != 3 or self.raw.get("content_relative_l2_denominator") != 250:
            raise SalientLocalLfMaskWriteProtocolError("actual-dtype budget drifted")
        if self.raw.get("operational_prompt_digest") != sha256(self.operational_prompt.encode()).hexdigest():
            raise SalientLocalLfMaskWriteProtocolError("operational prompt digest drifted")
        self.manifest.validate()
        for key in ("manifest_path", "runtime_configuration_path", "internal_execution_components_path", "gpu_requirements_path"):
            path = repository_root / str(self.raw[key])
            digest_key = key.replace("_path", "_file_sha256")
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != self.raw[digest_key]:
                raise SalientLocalLfMaskWriteProtocolError(f"{key} bytes drifted")
        current_payloads = self.current_experiment_authority.validate_and_read(repository_root)
        if len(self.historical_prior_authorities) != 2:
            raise SalientLocalLfMaskWriteProtocolError("historical producer authority count drifted")
        historical_documents = tuple(
            document
            for authority in self.historical_prior_authorities
            for document in authority.validate(repository_root)
        )
        prior_axes = {name: set() for name in _DENY_AXIS_NAMES}

        for document in historical_documents:
            _collect_deny_axes(document, prior_axes)
        current_prompt_digests: set[str] = set()
        for path, payload in current_payloads:
            if path.endswith(".json"):
                try:
                    document = json.loads(payload.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise SalientLocalLfMaskWriteProtocolError("current experiment JSON is invalid") from exc
                _collect_deny_axes(document, prior_axes)

                def collect_explicit_prompt_digests(value: object) -> None:
                    if type(value) is dict:
                        for key, item in value.items():
                            if key == "prompt_digest" and type(item) is str:
                                current_prompt_digests.add(item)
                            collect_explicit_prompt_digests(item)
                    elif type(value) is list:
                        for item in value:
                            collect_explicit_prompt_digests(item)

                collect_explicit_prompt_digests(document)
            elif path == self.current_experiment_authority.parti_prompt_asset_path:
                rows = list(csv.reader(io.StringIO(payload.decode("utf-8")), delimiter="\t"))[1:]
                current_prompt_digests.update(sha256(row[0].encode("utf-8")).hexdigest() for row in rows)
        if len(current_prompt_digests) != self.current_experiment_authority.current_unique_prompt_digest_count:
            raise SalientLocalLfMaskWriteProtocolError("current prompt digest authority count drifted")
        new_axes = {
            "prompt_digests": {item.prompt_digest for item in self.manifest.entries},
            "generation_seeds": {item.generation_seed for item in self.manifest.entries},
            "cluster_identities": {item.cluster_identity for item in self.manifest.entries},
            "source_cluster_ids": {item.source_cluster_id for item in self.manifest.entries},
            "key_lineage_digests": {item.key_lineage_digest for item in self.manifest.entries},
            "image_lineage_digests": {item.image_lineage_digest for item in self.manifest.entries},
            "namespaces": {
                self.manifest.seed_namespace, self.manifest.source_cluster_namespace,
                self.manifest.image_lineage_namespace, self.manifest.key_lineage_namespace,
            },
            "lineage_authorities": {
                self.manifest.image_lineage_identity, self.manifest.key_lineage_identity,
                self.manifest.registered_key_derivation_identity,
                self.manifest.registered_key_family_digest,
            },
        }
        if any(new_axes[name] & prior_axes[name] for name in new_axes):
            raise SalientLocalLfMaskWriteProtocolError("scientific roster overlaps historical authority")

    def analysis_identity(self, unit_index: int) -> AnalysisUnitIdentity:
        unit = self.unit_roster[unit_index]
        if unit_index < OPERATIONAL_UNIT_COUNT:
            image_digest = canonical_digest({"operational_unit_index": unit_index, "prompt_digest": self.operational_prompt_digest})
            key_digest = canonical_digest({"operational_unit_index": unit_index, "run_id": self.run_id})
            return AnalysisUnitIdentity(
                unit_id=f"salient_local_lf_operational_{unit_index}", case_id=unit.content_branch_id,
                source_cluster_id=derive_source_cluster_id(prompt_digest=self.operational_prompt_digest,
                    generation_seed=self.operational_generation_seed + unit_index,
                    image_lineage_digest=image_digest, registered_key_family_digest=key_digest),
                prompt_digest=self.operational_prompt_digest,
                generation_seed=self.operational_generation_seed + unit_index,
                image_lineage_digest=image_digest, registered_key_family_digest=key_digest,
            )
        entry = self.manifest.entries[unit.source_cluster_ordinal]
        return AnalysisUnitIdentity(
            unit_id=f"salient_local_lf_mask_write_scientific_{entry.cluster_ordinal}",
            case_id="global_hf_local_lf_mask_write_public_rgb8",
            source_cluster_id=entry.source_cluster_id,
            prompt_digest=entry.prompt_digest, generation_seed=entry.generation_seed,
            image_lineage_digest=entry.image_lineage_digest,
            registered_key_family_digest=self.manifest.registered_key_family_digest,
        )


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SalientLocalLfMaskWriteProtocolError("checked-in JSON is unreadable") from exc
    if type(value) is not dict:
        raise SalientLocalLfMaskWriteProtocolError("checked-in JSON must be a mapping")
    return value


def load_salient_local_lf_mask_write_validation_protocol(
    path: str | Path, *, repository_root: str | Path,
) -> SalientLocalLfMaskWriteProtocol:
    root = Path(repository_root)
    raw = _load_json(Path(path))
    manifest_raw = _load_json(root / str(raw["manifest_path"]))
    future_raw = manifest_raw["future_split_deny_authority"]
    manifest = SalientLocalLfScientificRoster(**{
        **manifest_raw,
        "future_split_deny_authority": FutureSplitDenyAuthority(**{
            **future_raw,
            **{key: tuple(future_raw[key]) for key in (
                "exclusion_roles", "prompt_digests", "generation_seeds", "cluster_identities",
                "source_cluster_ids", "key_lineage_digests", "image_lineage_digests",
            )},
        }),
        "entries": tuple(SalientLocalLfScientificRosterEntry(**item) for item in manifest_raw["entries"]),
    })
    current_raw = raw["current_experiment_authority"]
    current_authority = CurrentExperimentAuthority(**{
        **current_raw,
        "paths": tuple(CurrentExperimentPathBinding(**item) for item in current_raw["paths"]),
    })
    authorities = tuple(
        HistoricalProducerAuthority(**{**item, "paths": tuple(HistoricalProducerPathBinding(**binding) for binding in item["paths"])})
        for item in raw["historical_prior_authorities"]
    )
    units = (
        DevelopmentStudyUnit(0, "development_environment_preflight", "development_environment_preflight", 0,
                             "inspyrenet_checkpoint_runtime_preflight", "geometry_case_not_applicable", 1, MAXIMUM_DURATION_SECONDS),
        DevelopmentStudyUnit(1, "development_environment_preflight", "development_environment_preflight", 1,
                             "salient_local_lf_public_write_observation_preflight", "geometry_case_not_applicable", 1, MAXIMUM_DURATION_SECONDS),
        *(DevelopmentStudyUnit(index + OPERATIONAL_UNIT_COUNT, "development_scientific_responsibility_case", "content_embedder", index,
                               "global_hf_local_lf_mask_write_public_rgb8", "geometry_case_not_applicable", 1, MAXIMUM_DURATION_SECONDS)
          for index in range(SCIENTIFIC_UNIT_COUNT)),
    )
    protocol = SalientLocalLfMaskWriteProtocol(raw, manifest, current_authority, authorities, tuple(units))
    protocol.validate(root)
    return protocol


__all__ = [
    "SalientLocalLfMaskWriteProtocolError", "SalientLocalLfMaskWriteProtocol",
    "SalientLocalLfScientificRoster", "SalientLocalLfScientificRosterEntry",
    "CurrentExperimentAuthority", "CurrentExperimentPathBinding", "FutureSplitDenyAuthority",
    "load_salient_local_lf_mask_write_validation_protocol", "canonical_digest",
    "OPERATIONAL_UNIT_COUNT", "SCIENTIFIC_UNIT_COUNT", "MAXIMUM_TOTAL_UNITS",
    "CANONICAL_CONTENT_RELATIVE_L2_LIMIT",
]
