"""Production input identities and replay-derived development fit material.

This module owns no method algorithm.  It turns the checked-in prompt roster,
the runtime secret key, and verified COMMITTED development records into the
ordinary values consumed by :mod:`experiments.runners.development_exploration`.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
from typing import Sequence

import torch

from experiments.protocol.development_exploration import (
    DEVELOPMENT_SPLIT,
    DevelopmentPrimaryNullKeyBinding,
    FrozenDevelopmentExplorationProtocol,
    derive_development_primary_null_key_family_digest,
)
from experiments.protocol.internal_splits import (
    AnalysisUnitIdentity,
    FrozenSplitManifest,
    INTERNAL_VALIDATION_PROTOCOL_ID,
    INTERNAL_VALIDATION_PROTOCOL_VERSION,
    SplitAssignment,
    derive_source_cluster_id,
)
from main import (
    BranchNullCalibration,
    NullScoreRecord,
    SpatialRoutingObservation,
    identify_root_key,
)


PROMPT_ROSTER_SCHEMA = "ceg_wm_development_prompt_roster"
SEMANTIC_MODEL_ID = "openai/clip-vit-base-patch32"
SEMANTIC_MODEL_COMMIT = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


class DevelopmentInputError(RuntimeError):
    """A frozen input or replay-derived development fit is unavailable."""


@dataclass(frozen=True, slots=True)
class DevelopmentPrompt:
    cluster_ordinal: int
    generation_seed: int
    prompt: str


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentPromptRoster:
    roster_id: str
    seed_namespace: str
    entries: tuple[DevelopmentPrompt, ...]
    digest: str


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_development_prompt_roster(path: str | Path) -> FrozenDevelopmentPromptRoster:
    try:
        document = json.loads(Path(path).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentInputError("development prompt roster is unreadable") from exc
    if type(document) is not dict or set(document) != {
        "schema_version", "roster_id", "seed_namespace", "entries"
    }:
        raise DevelopmentInputError("development prompt roster fields drifted")
    if document["schema_version"] != PROMPT_ROSTER_SCHEMA:
        raise DevelopmentInputError("development prompt roster schema drifted")
    if any(type(document[key]) is not str or not document[key] for key in ("roster_id", "seed_namespace")):
        raise DevelopmentInputError("development prompt roster identity is invalid")
    raw_entries = document["entries"]
    if type(raw_entries) is not list or len(raw_entries) != 64:
        raise DevelopmentInputError("development prompt roster must freeze 64 clusters")
    entries: list[DevelopmentPrompt] = []
    for raw in raw_entries:
        if type(raw) is not dict or set(raw) != {"cluster_ordinal", "generation_seed", "prompt"}:
            raise DevelopmentInputError("development prompt entry fields drifted")
        entry = DevelopmentPrompt(**raw)
        if (
            type(entry.cluster_ordinal) is not int
            or type(entry.generation_seed) is not int
            or type(entry.prompt) is not str
            or not entry.prompt.strip()
        ):
            raise DevelopmentInputError("development prompt entry is invalid")
        entries.append(entry)
    if tuple(item.cluster_ordinal for item in entries) != tuple(range(64)):
        raise DevelopmentInputError("development prompt cluster order drifted")
    return FrozenDevelopmentPromptRoster(
        roster_id=document["roster_id"],
        seed_namespace=document["seed_namespace"],
        entries=tuple(entries),
        digest=_canonical_digest(document),
    )


def build_development_manifest_and_key_roster(
    protocol: FrozenDevelopmentExplorationProtocol,
    prompt_roster: FrozenDevelopmentPromptRoster,
    registered_root_key: str,
) -> tuple[FrozenSplitManifest, tuple[DevelopmentPrimaryNullKeyBinding, ...]]:
    if type(protocol) is not FrozenDevelopmentExplorationProtocol or protocol.validate():
        raise DevelopmentInputError("development protocol is invalid")
    public_digest = identify_root_key(registered_root_key).root_key_public_digest
    assignments: list[SplitAssignment] = []
    bindings: list[DevelopmentPrimaryNullKeyBinding] = []
    for entry in prompt_roster.entries:
        prompt_digest = sha256(entry.prompt.encode("utf-8")).hexdigest()
        image_lineage_digest = _canonical_digest(
            {
                "generation_seed": entry.generation_seed,
                "prompt_digest": prompt_digest,
                "roster_digest": prompt_roster.digest,
            }
        )
        key_family_digest = derive_development_primary_null_key_family_digest(
            protocol.threshold_detector_authority,
            registered_key_public_digest=public_digest,
            detection_key_public_digest=public_digest,
        )
        source_cluster_id = derive_source_cluster_id(
            prompt_digest=prompt_digest,
            generation_seed=entry.generation_seed,
            image_lineage_digest=image_lineage_digest,
            registered_key_family_digest=key_family_digest,
        )
        assignments.append(
            SplitAssignment(
                identity=AnalysisUnitIdentity(
                    unit_id=f"development_source_cluster_{entry.cluster_ordinal:02d}",
                    case_id="development_primary_null_threshold_fit",
                    source_cluster_id=source_cluster_id,
                    prompt_digest=prompt_digest,
                    generation_seed=entry.generation_seed,
                    image_lineage_digest=image_lineage_digest,
                    registered_key_family_digest=key_family_digest,
                ),
                split=DEVELOPMENT_SPLIT,
            )
        )
        bindings.append(
            DevelopmentPrimaryNullKeyBinding(
                source_cluster_id=source_cluster_id,
                registered_key_family_digest=key_family_digest,
                registered_key_public_digest=public_digest,
                detection_key_public_digest=public_digest,
            )
        )
    manifest = FrozenSplitManifest(
        protocol_id=INTERNAL_VALIDATION_PROTOCOL_ID,
        protocol_version=INTERNAL_VALIDATION_PROTOCOL_VERSION,
        manifest_id="development_exploration_runtime_input_manifest",
        manifest_revision=prompt_roster.digest,
        assignments=tuple(assignments),
    )
    violations = manifest.validate(require_all_splits=False)
    if violations:
        raise DevelopmentInputError(",".join(violations))
    return manifest, tuple(bindings)


def exact_positive_nearest_rank_p95(values: Sequence[float]) -> float:
    normalized = sorted(
        float(value)
        for value in values
        if not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
        and float(value) > 0.0
    )
    if not normalized:
        raise DevelopmentInputError("routing reference fit has no strictly positive values")
    return normalized[ceil(0.95 * len(normalized)) - 1]


class DevelopmentSemanticObservationProducer:
    """Real frozen-revision CLIP patch/prompt semantic observation."""

    def __init__(self, *, cache_root: Path, hf_token: str | None, device: str) -> None:
        try:
            from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
        except ImportError as exc:
            raise DevelopmentInputError("frozen CLIP runtime is unavailable") from exc
        self._processor = CLIPImageProcessor.from_pretrained(
            SEMANTIC_MODEL_ID,
            revision=SEMANTIC_MODEL_COMMIT,
            token=hf_token,
            cache_dir=str(cache_root),
        )
        self._tokenizer = CLIPTokenizerFast.from_pretrained(
            SEMANTIC_MODEL_ID,
            revision=SEMANTIC_MODEL_COMMIT,
            token=hf_token,
            cache_dir=str(cache_root),
        )
        self._model = CLIPModel.from_pretrained(
            SEMANTIC_MODEL_ID,
            revision=SEMANTIC_MODEL_COMMIT,
            token=hf_token,
            cache_dir=str(cache_root),
        ).to(device)
        self._model.eval()
        self._device = torch.device(device)

    def observe(self, routing_rgb: torch.Tensor, prompt: str) -> SpatialRoutingObservation:
        if (
            not isinstance(routing_rgb, torch.Tensor)
            or routing_rgb.ndim != 4
            or tuple(routing_rgb.shape[:2]) != (1, 3)
            or type(prompt) is not str
            or not prompt
        ):
            raise DevelopmentInputError("semantic observation input is invalid")
        rgb = routing_rgb.detach().to(device=self._device, dtype=torch.float32)
        if bool((rgb < 0.0).any().item()) or bool((rgb > 1.0).any().item()):
            raise DevelopmentInputError("semantic RGB is outside the closed unit interval")
        image_inputs = self._processor(
            images=rgb,
            return_tensors="pt",
            do_rescale=False,
        )
        text_inputs = self._tokenizer(
            prompt,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            vision = self._model.vision_model(
                pixel_values=image_inputs["pixel_values"].to(self._device)
            ).last_hidden_state[:, 1:, :]
            patches = self._model.visual_projection(
                self._model.vision_model.post_layernorm(vision)
            )
            text = self._model.text_model(
                input_ids=text_inputs["input_ids"].to(self._device),
                attention_mask=text_inputs["attention_mask"].to(self._device),
            ).pooler_output
            text = self._model.text_projection(text)
            patches = torch.nn.functional.normalize(patches, dim=-1)
            text = torch.nn.functional.normalize(text, dim=-1)
            semantic = torch.clamp(
                ((patches * text[:, None, :]).sum(dim=-1) + 1.0) / 2.0,
                0.0,
                1.0,
            )
        if tuple(semantic.shape) != (1, 49):
            raise DevelopmentInputError("CLIP semantic patch grid is not seven by seven")
        rgb_bytes = rgb.detach().to(device="cpu").contiguous().numpy().tobytes()
        return SpatialRoutingObservation(
            values=tuple(float(item) for item in semantic.to(device="cpu").reshape(-1)),
            spatial_shape=(7, 7),
            source_identity_digest=_canonical_digest(
                {
                    "model_id": SEMANTIC_MODEL_ID,
                    "model_revision": SEMANTIC_MODEL_COMMIT,
                    "prompt_digest": sha256(prompt.encode("utf-8")).hexdigest(),
                    "rgb_float32_sha256": sha256(rgb_bytes).hexdigest(),
                    "semantic_rule": "clip_patch_prompt_cosine_unit_interval",
                }
            ),
        )


def replay_branch_null_calibration(
    evidence: Sequence[tuple[object, object]],
    *,
    branch: str,
) -> BranchNullCalibration:
    responsibility = f"{branch}_detector"
    records: list[NullScoreRecord] = []
    detector_identity: str | None = None
    for record, _marker in evidence:
        if record.responsibility_id != responsibility:
            continue
        if record.content_branch_id != "clean_control" or record.execution_status != "success":
            continue
        result = record.operation_result_payload
        score = result.get(f"{branch}_score")
        observed_identity = result.get("detector_identity")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not isfinite(float(score)):
            raise DevelopmentInputError("committed primary-null score is invalid")
        if type(observed_identity) is not str or not observed_identity:
            raise DevelopmentInputError("committed detector identity is invalid")
        if detector_identity is not None and detector_identity != observed_identity:
            raise DevelopmentInputError("committed primary-null detector identity drifted")
        detector_identity = observed_identity
        identity = AnalysisUnitIdentity(**record.analysis_unit_identity)
        records.append(
            NullScoreRecord(
                float(score),
                identity.source_cluster_id,
                f"{responsibility}_{record.unit_index:04d}",
            )
        )
    if detector_identity is None or len(records) < 2:
        raise DevelopmentInputError("verified COMMITTED primary-null evidence is incomplete")
    return BranchNullCalibration(
        branch=branch,
        detector_identity=detector_identity,
        partition_identity="development_exploratory_primary_null_committed_replay",
        records=tuple(records),
    )
