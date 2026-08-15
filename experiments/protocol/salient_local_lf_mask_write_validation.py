"""Frozen development protocol for the salient-local-LF mask/write pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Mapping

from experiments.protocol.development_exploration import DevelopmentStudyUnit
from experiments.protocol.internal_splits import AnalysisUnitIdentity, derive_source_cluster_id


RUN_ID = "ceg_wm_salient_local_lf_mask_write_validation"
PROTOCOL_ID = RUN_ID
PROTOCOL_VERSION = "1.0.0"
OPERATIONAL_UNIT_COUNT = 2
SCIENTIFIC_UNIT_COUNT = 8
MAXIMUM_TOTAL_UNITS = 10
MAXIMUM_ATTEMPTS_PER_UNIT = 1
MAXIMUM_DURATION_SECONDS = 2700
CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250
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


def canonical_digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


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
        git_available = subprocess.run(
            ("git", "rev-parse", "--git-dir"), cwd=repository_root,
            check=False, capture_output=True, text=True,
        ).returncode == 0
        package_manifest = None
        if not git_available:
            try:
                package_manifest = json.loads((repository_root / "PACKAGE_MANIFEST.json").read_text("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise SalientLocalLfMaskWriteProtocolError("packaged historical authority manifest is unavailable") from exc
            if type(package_manifest) is not dict or type(package_manifest.get("entries")) is not list:
                raise SalientLocalLfMaskWriteProtocolError("packaged historical authority manifest is invalid")
        for binding in self.paths:
            binding.validate()
            if git_available:
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
            else:
                member = repository_root / binding.package_member_path
                try:
                    payload = member.read_bytes()
                except OSError as exc:
                    raise SalientLocalLfMaskWriteProtocolError("packaged historical producer bytes are unavailable") from exc
                matching = [item for item in package_manifest["entries"] if type(item) is dict and item.get("path") == binding.package_member_path]
                if (len(matching) != 1 or matching[0].get("git_blob_sha") != binding.git_blob_sha
                        or matching[0].get("sha256") != binding.raw_sha256
                        or matching[0].get("size") != len(payload)):
                    raise SalientLocalLfMaskWriteProtocolError("packaged historical producer binding drifted")
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
        if len(self.historical_prior_authorities) != 2:
            raise SalientLocalLfMaskWriteProtocolError("historical producer authority count drifted")
        historical_documents = tuple(
            document
            for authority in self.historical_prior_authorities
            for document in authority.validate(repository_root)
        )
        prior_axes = {name: set() for name in (
            "prompt_digests", "generation_seeds", "source_clusters",
            "image_lineages", "key_lineages",
        )}

        def collect(value: object) -> None:
            if type(value) is dict:
                for key, item in value.items():
                    lowered = key.lower()
                    scalars = item if type(item) is list else [item]
                    for scalar in scalars:
                        if lowered in {"prompt", "prompt_text"} and type(scalar) is str:
                            prior_axes["prompt_digests"].add(sha256(scalar.encode("utf-8")).hexdigest())
                        elif lowered == "prompt_digest" and type(scalar) is str:
                            prior_axes["prompt_digests"].add(scalar)
                        elif lowered == "generation_seed" and type(scalar) is int:
                            prior_axes["generation_seeds"].add(scalar)
                        elif ("source_cluster" in lowered or lowered == "cluster_identity") and type(scalar) is str:
                            prior_axes["source_clusters"].add(scalar)
                        elif "image_lineage" in lowered and type(scalar) is str:
                            prior_axes["image_lineages"].add(scalar)
                        elif ("key_lineage" in lowered or "key_family" in lowered) and type(scalar) is str:
                            prior_axes["key_lineages"].add(scalar)
                    collect(item)
            elif type(value) is list:
                for item in value:
                    collect(item)

        for document in historical_documents:
            collect(document)
        new_axes = {
            "prompt_digests": {item.prompt_digest for item in self.manifest.entries},
            "generation_seeds": {item.generation_seed for item in self.manifest.entries},
            "source_clusters": {
                *(item.cluster_identity for item in self.manifest.entries),
                *(item.source_cluster_id for item in self.manifest.entries),
                self.manifest.source_cluster_namespace,
            },
            "image_lineages": {
                *(item.image_lineage_digest for item in self.manifest.entries),
                self.manifest.image_lineage_identity,
                self.manifest.image_lineage_namespace,
            },
            "key_lineages": {
                *(item.key_lineage_digest for item in self.manifest.entries),
                self.manifest.key_lineage_identity,
                self.manifest.key_lineage_namespace,
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
    manifest = SalientLocalLfScientificRoster(
        **{**manifest_raw, "entries": tuple(SalientLocalLfScientificRosterEntry(**item) for item in manifest_raw["entries"])}
    )
    authorities = tuple(
        HistoricalProducerAuthority(**{**item, "paths": tuple(HistoricalProducerPathBinding(**binding) for binding in item["paths"])})
        for item in raw["historical_prior_authorities"]
    )
    units = (
        DevelopmentStudyUnit(0, "development_environment_preflight", "development_environment_preflight", 0,
                             "inspyrenet_checkpoint_runtime_preflight", "geometry_case_not_applicable", 1, MAXIMUM_DURATION_SECONDS),
        DevelopmentStudyUnit(1, "development_full_chain_wiring", "development_full_chain_wiring", 1,
                             "salient_local_lf_public_write_observation_preflight", "geometry_case_not_applicable", 1, MAXIMUM_DURATION_SECONDS),
        *(DevelopmentStudyUnit(index + OPERATIONAL_UNIT_COUNT, "development_scientific_responsibility_case", "content_embedder", index,
                               "global_hf_local_lf_mask_write_public_rgb8", "geometry_case_not_applicable", 1, MAXIMUM_DURATION_SECONDS)
          for index in range(SCIENTIFIC_UNIT_COUNT)),
    )
    protocol = SalientLocalLfMaskWriteProtocol(raw, manifest, authorities, tuple(units))
    protocol.validate(root)
    return protocol


__all__ = [
    "SalientLocalLfMaskWriteProtocolError", "SalientLocalLfMaskWriteProtocol",
    "SalientLocalLfScientificRoster", "SalientLocalLfScientificRosterEntry",
    "load_salient_local_lf_mask_write_validation_protocol", "canonical_digest",
    "OPERATIONAL_UNIT_COUNT", "SCIENTIFIC_UNIT_COUNT", "MAXIMUM_TOTAL_UNITS",
]
