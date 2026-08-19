"""Fixed soft-route mechanism-validation runner and production seam."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import stat
from typing import Literal, Protocol, Sequence

import torch

from experiments.protocol.semantic_texture_soft_route_mechanism_validation import (
    ARMS,
    ATTACKS,
    CLUSTER_COUNT,
    CONFIRMATION_ROLE,
    SoftRouteMechanismManifest,
    SoftRouteMechanismManifestEntry,
    SELECTION_ROLE,
    SoftRouteMechanismProtocolError,
    canonical_digest,
    provisional_tau,
)


ExecutionStatus = Literal["completed", "failed", "unstarted"]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class SoftRouteMechanismRunnerError(RuntimeError):
    """One fixed soft-route mechanism validation unit could not be completed safely."""


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismBranchScores:
    hf_score: float
    lf_score: float
    max_score: float
    detector_identity: str
    hf_detector_identity: str
    lf_detector_identity: str
    registered: bool


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismStandardizedScores:
    hf_standardized_score: float
    lf_standardized_score: float
    max_standardized_score: float
    detector_identity: str


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismNullScoreRecord:
    source_cluster_id: str
    sample_id: str
    score_float64_hex: str


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismGeneration:
    image: object
    arm_id: str
    materialization_replay_identity: str | None
    budget_identity: str | None
    paired_rgb8_mse: float


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismGenerationRecord:
    record_attempt_index: int
    source_cluster_id: str
    image_lineage_digest: str
    arm_id: str
    execution_status: ExecutionStatus
    materialization_replay_identity: str | None = None
    budget_identity: str | None = None
    paired_rgb8_mse: float | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismCaseRecord:
    record_attempt_index: int
    source_cluster_id: str
    image_lineage_digest: str
    arm_id: str
    attack_id: str
    key_role: str
    wrong_key_index: int | None
    execution_status: ExecutionStatus
    hf_score: float | None = None
    lf_score: float | None = None
    hf_standardized_score: float | None = None
    lf_standardized_score: float | None = None
    max_standardized_score: float | None = None
    materialization_replay_identity: str | None = None
    budget_identity: str | None = None
    paired_rgb8_mse: float | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismProvisionalCalibration:
    selection_manifest_digest: str
    hf_detector_identity: str
    lf_detector_identity: str
    hf_null_identity: str
    lf_null_identity: str
    tau_hf_provisional: float
    tau_lf_provisional: float
    tau_max_provisional: float
    hf_records: tuple[SoftRouteMechanismNullScoreRecord, ...] = ()
    lf_records: tuple[SoftRouteMechanismNullScoreRecord, ...] = ()
    retired: bool = False

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class SoftRouteMechanismSplitResult:
    protocol_id: str
    manifest_digest: str
    role_id: str
    generations: tuple[SoftRouteMechanismGenerationRecord, ...]
    records: tuple[SoftRouteMechanismCaseRecord, ...]
    provisional_calibration: SoftRouteMechanismProvisionalCalibration | None
    passed: bool
    blocked_class: str | None = None
    diagnostic_only: bool = True
    science_started: bool = False
    scientific_unit_count: int = 0
    candidate_promoted: bool = False
    formal_tau_created: bool = False


class SoftRouteMechanismOperations(Protocol):
    def clean(self, entry: SoftRouteMechanismManifestEntry) -> SoftRouteMechanismGeneration: ...
    def write(self, entry: SoftRouteMechanismManifestEntry, arm_id: str) -> SoftRouteMechanismGeneration: ...
    def attack(self, entry: SoftRouteMechanismManifestEntry, generation: SoftRouteMechanismGeneration, attack_id: str) -> object: ...
    def observe(self, image: object, *, wrong_key_index: int | None) -> SoftRouteMechanismBranchScores: ...
    def build_calibration(self, primary_null: Sequence[tuple[SoftRouteMechanismManifestEntry, SoftRouteMechanismBranchScores]], *, partition_identity: str) -> tuple[str, str, str, str]: ...
    def install_calibration(self, calibration: SoftRouteMechanismProvisionalCalibration) -> None: ...
    def standardize(self, scores: SoftRouteMechanismBranchScores) -> SoftRouteMechanismStandardizedScores: ...
    def close(self) -> None: ...


def _bounded_failure(error: BaseException) -> str:
    return type(error).__name__[:80]


def _generation_templates(manifest: SoftRouteMechanismManifest) -> list[SoftRouteMechanismGenerationRecord]:
    return [SoftRouteMechanismGenerationRecord(1, entry.source_cluster_id, entry.image_lineage_digest, arm, "unstarted") for entry in manifest.entries for arm in ARMS]


def _case_templates(manifest: SoftRouteMechanismManifest) -> list[SoftRouteMechanismCaseRecord]:
    result: list[SoftRouteMechanismCaseRecord] = []
    for ordinal, entry in enumerate(manifest.entries):
        for arm in ARMS:
            for attack in ATTACKS:
                result.append(SoftRouteMechanismCaseRecord(1, entry.source_cluster_id, entry.image_lineage_digest, arm, attack, "registered", None, "unstarted"))
                if arm == "semantic_texture_soft_routed":
                    wrong = ordinal if manifest.role_id == SELECTION_ROLE else CLUSTER_COUNT + ordinal
                    result.append(SoftRouteMechanismCaseRecord(1, entry.source_cluster_id, entry.image_lineage_digest, arm, attack, "wrong", wrong, "unstarted"))
    return result


def _failed_result(manifest: SoftRouteMechanismManifest, generations: list[SoftRouteMechanismGenerationRecord], records: list[SoftRouteMechanismCaseRecord], calibration: SoftRouteMechanismProvisionalCalibration | None) -> SoftRouteMechanismSplitResult:
    return SoftRouteMechanismSplitResult(manifest.protocol_id, manifest.digest(), manifest.role_id, tuple(generations), tuple(records), calibration, False, "implementation_blocked")


def _failed_record(record: SoftRouteMechanismCaseRecord, error: BaseException) -> SoftRouteMechanismCaseRecord:
    return SoftRouteMechanismCaseRecord(**{**asdict(record), "execution_status": "failed", "failure_reason": _bounded_failure(error)})


def _completed_record(record: SoftRouteMechanismCaseRecord, generation: SoftRouteMechanismGeneration, scores: SoftRouteMechanismBranchScores, standardized: SoftRouteMechanismStandardizedScores | None) -> SoftRouteMechanismCaseRecord:
    values = [scores.hf_score, scores.lf_score, generation.paired_rgb8_mse]
    if standardized is not None:
        values.extend((standardized.hf_standardized_score, standardized.lf_standardized_score, standardized.max_standardized_score))
    if any(not isfinite(value) for value in values):
        raise SoftRouteMechanismRunnerError("soft-route mechanism validation observed a non-finite value")
    return SoftRouteMechanismCaseRecord(**{
        **asdict(record), "execution_status": "completed",
        "hf_score": scores.hf_score, "lf_score": scores.lf_score,
        "hf_standardized_score": None if standardized is None else standardized.hf_standardized_score,
        "lf_standardized_score": None if standardized is None else standardized.lf_standardized_score,
        "max_standardized_score": None if standardized is None else standardized.max_standardized_score,
        "materialization_replay_identity": generation.materialization_replay_identity,
        "budget_identity": generation.budget_identity,
        "paired_rgb8_mse": generation.paired_rgb8_mse,
    })


def execute_soft_route_mechanism_split(manifest: SoftRouteMechanismManifest, operations: SoftRouteMechanismOperations, *, provisional_calibration: SoftRouteMechanismProvisionalCalibration | None = None) -> SoftRouteMechanismSplitResult:
    """Execute one attempt while retaining all 160/384 denominator slots."""
    expected_role = SELECTION_ROLE if provisional_calibration is None else CONFIRMATION_ROLE
    try:
        manifest.validate(expected_role=expected_role)
    except SoftRouteMechanismProtocolError as exc:
        raise SoftRouteMechanismRunnerError("soft-route mechanism validation manifest authority is invalid") from exc
    if provisional_calibration is not None and (
        provisional_calibration.retired
        or len(provisional_calibration.hf_records) != CLUSTER_COUNT
        or len(provisional_calibration.lf_records) != CLUSTER_COUNT
    ):
        raise SoftRouteMechanismRunnerError("selection calibration authority is invalid")
    generation_records, case_records = _generation_templates(manifest), _case_templates(manifest)
    generated: dict[tuple[str, str], SoftRouteMechanismGeneration] = {}
    pairs = [(entry, arm) for entry in manifest.entries for arm in ARMS]
    for index, (entry, arm) in enumerate(pairs):
        try:
            value = operations.clean(entry) if arm == ARMS[0] else operations.write(entry, arm)
            if value.arm_id != arm or not isfinite(value.paired_rgb8_mse):
                raise SoftRouteMechanismRunnerError("generation identity drifted")
            generated[(entry.source_cluster_id, arm)] = value
            generation_records[index] = SoftRouteMechanismGenerationRecord(1, entry.source_cluster_id, entry.image_lineage_digest, arm, "completed", value.materialization_replay_identity, value.budget_identity, value.paired_rgb8_mse)
        except Exception as exc:
            generation_records[index] = SoftRouteMechanismGenerationRecord(1, entry.source_cluster_id, entry.image_lineage_digest, arm, "failed", failure_reason=_bounded_failure(exc))
            return _failed_result(manifest, generation_records, case_records, provisional_calibration)
    entry_by_cluster = {entry.source_cluster_id: entry for entry in manifest.entries}
    primary_indexes = [index for index, record in enumerate(case_records) if record.arm_id == ARMS[0] and record.attack_id == "identity"]
    primary: list[tuple[SoftRouteMechanismManifestEntry, SoftRouteMechanismBranchScores]] = []
    for index in primary_indexes:
        record = case_records[index]
        generation = generated[(record.source_cluster_id, record.arm_id)]
        try:
            scores = operations.observe(generation.image, wrong_key_index=None)
            primary.append((entry_by_cluster[record.source_cluster_id], scores))
            case_records[index] = _completed_record(record, generation, scores, None)
        except Exception as exc:
            case_records[index] = _failed_record(record, exc)
            return _failed_result(manifest, generation_records, case_records, provisional_calibration)
    calibration = provisional_calibration
    try:
        if calibration is None:
            hf_detector, lf_detector, hf_null, lf_null = operations.build_calibration(tuple(primary), partition_identity=manifest.digest())
            standardized = tuple(operations.standardize(scores) for _entry, scores in primary)
            calibration = SoftRouteMechanismProvisionalCalibration(
                manifest.digest(), hf_detector, lf_detector, hf_null, lf_null,
                provisional_tau([value.hf_standardized_score for value in standardized]),
                provisional_tau([value.lf_standardized_score for value in standardized]),
                provisional_tau([value.max_standardized_score for value in standardized]),
                tuple(SoftRouteMechanismNullScoreRecord(entry.source_cluster_id, entry.image_lineage_digest, scores.hf_score.hex()) for entry, scores in primary),
                tuple(SoftRouteMechanismNullScoreRecord(entry.source_cluster_id, entry.image_lineage_digest, scores.lf_score.hex()) for entry, scores in primary),
            )
        else:
            operations.install_calibration(calibration)
            standardized = tuple(operations.standardize(scores) for _entry, scores in primary)
        for index, score in zip(primary_indexes, standardized, strict=True):
            record = case_records[index]
            case_records[index] = SoftRouteMechanismCaseRecord(**{**asdict(record), "hf_standardized_score": score.hf_standardized_score, "lf_standardized_score": score.lf_standardized_score, "max_standardized_score": score.max_standardized_score})
    except Exception as exc:
        case_records[primary_indexes[0]] = _failed_record(case_records[primary_indexes[0]], exc)
        return _failed_result(manifest, generation_records, case_records, calibration)
    primary_set = set(primary_indexes)
    for index, record in enumerate(case_records):
        if index in primary_set:
            continue
        entry = entry_by_cluster[record.source_cluster_id]
        generation = generated[(record.source_cluster_id, record.arm_id)]
        try:
            image = operations.attack(entry, generation, record.attack_id)
            scores = operations.observe(image, wrong_key_index=record.wrong_key_index)
            case_records[index] = _completed_record(record, generation, scores, operations.standardize(scores))
        except Exception as exc:
            case_records[index] = _failed_record(record, exc)
            return _failed_result(manifest, generation_records, case_records, calibration)
    completed = tuple(case_records)
    return SoftRouteMechanismSplitResult(manifest.protocol_id, manifest.digest(), manifest.role_id, tuple(generation_records), completed, calibration, _passes_fixed_mechanism_gates(completed, calibration))


def _positives(records: Sequence[SoftRouteMechanismCaseRecord], field: str, threshold: float) -> int:
    return sum(float(getattr(record, field)) >= threshold for record in records)


def _passes_fixed_mechanism_gates(records: tuple[SoftRouteMechanismCaseRecord, ...], calibration: SoftRouteMechanismProvisionalCalibration) -> bool:
    if len(records) != CLUSTER_COUNT * 12 or any(record.execution_status != "completed" for record in records):
        return False
    def selected(arm: str, attack: str, key: str = "registered") -> tuple[SoftRouteMechanismCaseRecord, ...]:
        return tuple(record for record in records if record.arm_id == arm and record.attack_id == attack and record.key_role == key)
    for attack in ATTACKS:
        clean = selected(ARMS[0], attack)
        if len(clean) != CLUSTER_COUNT or any(_positives(clean, field, tau) > 3 for field, tau in (("hf_standardized_score", calibration.tau_hf_provisional), ("lf_standardized_score", calibration.tau_lf_provisional), ("max_standardized_score", calibration.tau_max_provisional))):
            return False
        registered, wrong = selected("semantic_texture_soft_routed", attack), selected("semantic_texture_soft_routed", attack, "wrong")
        paired = {record.source_cluster_id: record for record in wrong}
        if len(registered) != CLUSTER_COUNT or len(wrong) != CLUSTER_COUNT:
            return False
        if sum(float(record.hf_score) > float(paired[record.source_cluster_id].hf_score) and float(record.lf_score) > float(paired[record.source_cluster_id].lf_score) for record in registered) < 21:
            return False
        if _positives(wrong, "max_standardized_score", calibration.tau_max_provisional) > 3:
            return False
    identity_soft, identity_hf = selected("semantic_texture_soft_routed", "identity"), selected("hf_only", "identity")
    crop_soft, crop_hf = selected("semantic_texture_soft_routed", "crop_0_75"), selected("hf_only", "crop_0_75")
    crop_disabled = selected("semantic_texture_route_disabled", "crop_0_75")
    if _positives(identity_soft, "max_standardized_score", calibration.tau_max_provisional) < _positives(identity_hf, "max_standardized_score", calibration.tau_max_provisional) - 1:
        return False
    if _positives(crop_soft, "max_standardized_score", calibration.tau_max_provisional) < _positives(crop_hf, "max_standardized_score", calibration.tau_max_provisional) + 4:
        return False
    if _positives(crop_soft, "max_standardized_score", calibration.tau_max_provisional) < _positives(crop_disabled, "max_standardized_score", calibration.tau_max_provisional) + 1:
        return False
    writes = tuple(record for record in records if record.arm_id != ARMS[0])
    if any(record.materialization_replay_identity is None or record.budget_identity != "combined_relative_l2_3_250" for record in writes):
        return False
    for attack in ATTACKS:
        soft = [float(record.paired_rgb8_mse) for record in selected("semantic_texture_soft_routed", attack)]
        hf = [float(record.paired_rgb8_mse) for record in selected("hf_only", attack)]
        if sum(soft) / CLUSTER_COUNT > sum(hf) / CLUSTER_COUNT + (1.0 / 255.0) ** 2:
            return False
    return True


class AdapterBackedSoftRouteMechanismOperations:
    """Live SD3.5 implementation using only public runtime/method/attack APIs."""

    def __init__(
        self,
        *,
        backend: object,
        runtime_adapter: object,
        session: object,
        semantic_runtime: object,
        adapter: object,
        whitening_asset: object,
        root_key: str,
        attack_registry: object,
    ) -> None:
        from main import identify_root_key

        self._backend = backend
        self._runtime = runtime_adapter
        self._session = session
        self._semantic_runtime = semantic_runtime
        self._adapter = adapter
        self._whitening = whitening_asset
        self._root_key = root_key
        self._root_key_identity = identify_root_key(root_key)
        self._attack_registry = attack_registry
        self._hf_null: object | None = None
        self._lf_null: object | None = None
        self._branches: dict[int, object] = {}

    def _latent(self, entry: SoftRouteMechanismManifestEntry) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(entry.generation_seed)
        latent = torch.randn(
            (1, 16, self._session.image_height // 8, self._session.image_width // 8),
            dtype=torch.float32,
            generator=generator,
            device="cpu",
        )
        return latent.to(device=self._session.selected_device, dtype=torch.float16)

    def clean(self, entry: SoftRouteMechanismManifestEntry) -> SoftRouteMechanismGeneration:
        from runtime import materialize_ordinary_rgb8_snapshot

        self._backend.set_development_generation_prompts(entry.prompt_text, "")
        result = self._runtime.execute_clean_image_and_vae_observation(self._latent(entry))
        image = materialize_ordinary_rgb8_snapshot(result.clean_image)
        return SoftRouteMechanismGeneration(image, ARMS[0], None, None, 0.0)

    def write(self, entry: SoftRouteMechanismManifestEntry, arm_id: str) -> SoftRouteMechanismGeneration:
        from runtime import materialize_ordinary_rgb8_snapshot

        self._backend.set_development_generation_prompts(entry.prompt_text, "")
        observation = self._adapter.execute_semantic_texture_content_arm_write_and_vae(
            self._latent(entry),
            self._root_key,
            self._semantic_runtime,
            arm_id=arm_id,
        )
        result = observation.result.content_write_result
        clean = materialize_ordinary_rgb8_snapshot(result.clean_image)
        written = self._adapter.materialize_semantic_texture_written_rgb8(observation)
        measurement = result.content_materialization
        if measurement.realized_relative_l2 > 3.0 / 250.0:
            raise SoftRouteMechanismRunnerError(
                "actual-dtype budget exceeded"
            )
        mse = float(
            torch.mean(
                (
                    (written.to(torch.float32) - clean.to(torch.float32))
                    / 255.0
                )
                ** 2
            ).item()
        )
        return SoftRouteMechanismGeneration(
            written,
            arm_id,
            measurement.materialization_replay_identity,
            "combined_relative_l2_3_250",
            mse,
        )

    def attack(
        self,
        entry: SoftRouteMechanismManifestEntry,
        generation: SoftRouteMechanismGeneration,
        attack_id: str,
    ) -> object:
        from experiments.attacks.geometric import (
            AttackArtifact,
            GeometricAttackSpec,
            apply_geometric_attack,
        )
        from experiments.protocol.internal_splits import AnalysisUnitIdentity

        identity = AnalysisUnitIdentity(
            unit_id=(
                f"{entry.source_cluster_id}:{generation.arm_id}:{attack_id}"
            ),
            case_id="semantic_texture_soft_route_mechanism_validation",
            source_cluster_id=entry.source_cluster_id,
            prompt_digest=entry.prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=entry.image_lineage_digest,
            registered_key_family_digest=entry.registered_key_family_digest,
        )
        specification = (
            GeometricAttackSpec("identity")
            if attack_id == "identity"
            else GeometricAttackSpec("crop", crop_fraction=0.75)
        )
        return apply_geometric_attack(
            AttackArtifact(identity, generation.image),
            specification,
            registry=self._attack_registry,
        ).attacked_artifact.image

    def observe(
        self,
        image: object,
        *,
        wrong_key_index: int | None,
    ) -> SoftRouteMechanismBranchScores:
        key = (
            self._root_key
            if wrong_key_index is None
            else self._adapter.derive_semantic_texture_wrong_key_material(
                self._root_key_identity.root_key_public_digest,
                wrong_key_index,
            )
        )
        branches = self._adapter.observe_semantic_texture_candidate_branches(
            image,
            key,
            self._semantic_runtime,
            self._whitening,
        )
        scores = SoftRouteMechanismBranchScores(
            float(branches.hf_result.hf_score),
            float(branches.lf_result.lf_score),
            max(
                float(branches.hf_result.hf_score),
                float(branches.lf_result.lf_score),
            ),
            "unstandardized",
            branches.hf_result.detector_identity,
            branches.lf_result.detector_identity,
            wrong_key_index is None,
        )
        self._branches[id(scores)] = branches
        return scores

    def build_calibration(
        self,
        primary_null: Sequence[tuple[SoftRouteMechanismManifestEntry, SoftRouteMechanismBranchScores]],
        *,
        partition_identity: str,
    ) -> tuple[str, str, str, str]:
        observations = tuple(
            (
                entry.source_cluster_id,
                entry.image_lineage_digest,
                self._branches[id(scores)],
            )
            for entry, scores in primary_null
        )
        self._hf_null, self._lf_null = (
            self._adapter.build_semantic_texture_provisional_calibrations(
                observations,
                partition_identity=partition_identity,
            )
        )
        first = primary_null[0][1]
        return (
            first.hf_detector_identity,
            first.lf_detector_identity,
            self._hf_null.calibration_identity,
            self._lf_null.calibration_identity,
        )

    def install_calibration(
        self,
        calibration: SoftRouteMechanismProvisionalCalibration,
    ) -> None:
        self._hf_null, self._lf_null = (
            self._adapter.materialize_semantic_texture_provisional_calibrations(
                hf_detector_identity=calibration.hf_detector_identity,
                lf_detector_identity=calibration.lf_detector_identity,
                partition_identity=calibration.selection_manifest_digest,
                hf_records=tuple(
                    (
                        record.source_cluster_id,
                        record.sample_id,
                        float.fromhex(record.score_float64_hex),
                    )
                    for record in calibration.hf_records
                ),
                lf_records=tuple(
                    (
                        record.source_cluster_id,
                        record.sample_id,
                        float.fromhex(record.score_float64_hex),
                    )
                    for record in calibration.lf_records
                ),
            )
        )
        if (
            self._hf_null.calibration_identity != calibration.hf_null_identity
            or self._lf_null.calibration_identity != calibration.lf_null_identity
        ):
            raise SoftRouteMechanismRunnerError(
                "provisional calibration identity drifted"
            )

    def standardize(
        self,
        scores: SoftRouteMechanismBranchScores,
    ) -> SoftRouteMechanismStandardizedScores:
        if self._hf_null is None or self._lf_null is None:
            raise SoftRouteMechanismRunnerError(
                "provisional calibration is unavailable"
            )
        branches = self._branches.pop(id(scores))
        result = self._adapter.combine_semantic_texture_candidate_branches(
            branches,
            hf_null=self._hf_null,
            lf_null=self._lf_null,
        ).result
        return SoftRouteMechanismStandardizedScores(
            result.hf_standardization.z_score,
            result.lf_standardization.z_score,
            result.content_score,
            result.detector_identity,
        )

    def close(self) -> None:
        self._runtime.close()


def _required_absolute_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value or not Path(value).is_absolute():
        raise SoftRouteMechanismRunnerError(
            "required production environment is incomplete"
        )
    return Path(value).resolve()


def create_adapter_backed_soft_route_mechanism_operations(
    *,
    configuration_path: str | Path,
    detector_asset_bundle: str | Path,
) -> AdapterBackedSoftRouteMechanismOperations:
    """Authenticate the bundle and construct the public production chain."""
    from experiments.attacks.geometric import load_attack_registry
    from experiments.methods import (
        CegWmExperimentAdapter,
        load_ceg_wm_experiment_adapter_configuration,
    )
    from experiments.methods.ceg_wm import (
        materialize_semantic_texture_soft_detector_asset_bundle,
    )
    from experiments.protocol.semantic_texture_soft_detector_assets import (
        SemanticTextureSoftDetectorAssetBundle,
    )
    from runtime import (
        InspyrenetSemanticRuntime,
        Sd35PipelineBackend,
        create_runtime_adapter,
    )

    from experiments.protocol.semantic_texture_soft_route_mechanism_validation import (
        load_soft_route_mechanism_configuration,
    )

    configuration = load_soft_route_mechanism_configuration(configuration_path)
    bundle_path = Path(detector_asset_bundle)
    before = bundle_path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise SoftRouteMechanismRunnerError(
            "asset bundle is not regular"
        )
    descriptor = os.open(
        bundle_path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        after = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            blob = handle.read()
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or sha256(blob).hexdigest() != configuration["asset_bundle_sha256"]
    ):
        raise SoftRouteMechanismRunnerError(
            "asset bundle outer digest drifted"
        )
    bundle = SemanticTextureSoftDetectorAssetBundle.from_mapping(
        json.loads(blob)
    )
    if bundle.bundle_digest != configuration["asset_bundle_digest"]:
        raise SoftRouteMechanismRunnerError(
            "asset bundle inner digest drifted"
        )
    whitening, _historical_hf, _historical_lf = (
        materialize_semantic_texture_soft_detector_asset_bundle(bundle)
    )
    hf_token, root_key = os.environ.get("HF_TOKEN"), os.environ.get(
        "CEG_WM_ROOT_KEY"
    )
    if not hf_token or not root_key:
        raise SoftRouteMechanismRunnerError(
            "required production environment is incomplete"
        )
    cache_root = _required_absolute_environment_path("CEG_WM_CACHE_ROOT")
    persistent_root = _required_absolute_environment_path(
        "CEG_WM_PERSISTENT_ROOT"
    )
    checkpoint = _required_absolute_environment_path(
        "CEG_WM_INSPYRENET_CHECKPOINT_PATH"
    )
    backend = Sd35PipelineBackend(
        cache_root=cache_root,
        persistent_root=persistent_root,
        hf_token=hf_token,
        prompt="soft_route_mechanism",
        negative_prompt="",
    )
    runtime_adapter = create_runtime_adapter(
        backend,
        PACKAGE_ROOT / "configs/runtime/runtime_sd35_flowmatch.json",
    )
    session = runtime_adapter.initialize("cuda")
    semantic_runtime = InspyrenetSemanticRuntime(
        checkpoint,
        selected_device=session.selected_device,
    )
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(
            PACKAGE_ROOT
            / "configs/experiments/internal_execution_components.json"
        ),
        runtime_adapter,
    )
    return AdapterBackedSoftRouteMechanismOperations(
        backend=backend,
        runtime_adapter=runtime_adapter,
        session=session,
        semantic_runtime=semantic_runtime,
        adapter=adapter,
        whitening_asset=whitening,
        root_key=root_key,
        attack_registry=load_attack_registry(
            PACKAGE_ROOT
            / "configs/experiments/internal_execution_components.json"
        ),
    )


__all__ = [
    "AdapterBackedSoftRouteMechanismOperations",
    "SoftRouteMechanismBranchScores",
    "SoftRouteMechanismCaseRecord",
    "SoftRouteMechanismGeneration",
    "SoftRouteMechanismGenerationRecord",
    "SoftRouteMechanismNullScoreRecord",
    "SoftRouteMechanismOperations",
    "SoftRouteMechanismProvisionalCalibration",
    "SoftRouteMechanismSplitResult",
    "SoftRouteMechanismStandardizedScores",
    "SoftRouteMechanismRunnerError",
    "create_adapter_backed_soft_route_mechanism_operations",
    "execute_soft_route_mechanism_split",
]
