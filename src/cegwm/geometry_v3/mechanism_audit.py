"""Artifact-bound, identity-only Geometry-V3 P1M0 mechanism audit.

P1M0 is a diagnostic node.  It retains bounded public scores and contract
facts only; images, latents, Q/K, anchors, keys and prompt text stay transient.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from cegwm.geometry_v3.active_writer import (
    ActiveQKWriterSession,
    P0_ANCHOR_POINT_COUNT,
    P0_IMAGE_SIZE,
    P0_INFERENCE_STEPS,
    P0_MODEL_ID,
    WriterScalarObservation,
    canonical_qk_pattern,
    normalized_pattern_correlation,
)
from cegwm.geometry_v3.confirmation import (
    P1_CONFIG_ID,
    P1_CONTROL_IDS,
    P1_KIND_IDS,
    SOURCE_EXECUTION_EXACT as P0_EXECUTION_EXACT,
    SOURCE_PLAN_DIGEST as P0_PLAN_DIGEST,
    SOURCE_PROTOCOL as P0_PROTOCOL,
    SOURCE_ROSTER_DIGEST as P0_ROSTER_DIGEST,
    SOURCE_RUN_ID as P0_RUN_ID,
    fixed_config,
    validate_p0_source,
    validate_p0_source_identity,
)
from cegwm.geometry_v3.contracts import CanonicalRelationAnchor, derive_canonical_relation_anchor
from cegwm.geometry_v3.operational import (
    _config_number,
    _fresh_observation_scheduler,
    _module_device_dtype,
    _module_pair,
)
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key


P1M0_PROTOCOL_ID = "geometry-v3-keyed-qk-canonical-anchor-p1m0-mechanism-audit-v1"
P1M0_PROMPT_ID = "geometry-v3-p1m0-public-prompt-01"
P1M0_PROMPT_TEXT = (
    "A brass compass beside a closed field notebook on pale stone, diffuse morning light"
)
P1M0_GENERATION_SEED = 273
P1M0_OBSERVATION_NOISE_SEED = 29073
P1M0_OBSERVATION_TIMESTEP = 500
P1M0_OBSERVATION_TEXT_TOKENS = 333
P1M0_STAGES = ("writer_step18_latent", "final_predecode_latent", "final_rgb_reencode")
P1M0_GENERATION_ROLES = ("no_writer", "writer")
P1M0_UNIT_COUNT = 24
P1M0_STATUS_MISMATCH = "P1M0_IMPLEMENTATION_MISMATCH_INDICATED"
P1M0_STATUS_INSUFFICIENT = "P1M0_OBSERVABILITY_INSUFFICIENCY_INDICATED"
P1M0_STATUS_INCONCLUSIVE = "P1M0_INCONCLUSIVE"
P1M0_STATUS_STOPPED = "P1M0_STOPPED"
P1M0_SCIENCE_DENOMINATOR = 0
P1M0_ARTIFACT_MAX_BYTES = 2 * 1024 * 1024

P0_SOURCE_DIRECTORY = (
    "/content/drive/MyDrive/CEG-WM/Geometry-V3/P0/"
    "Geometry-V3-P0-9b5085c805b6-20260828T122005Z"
)
P1_SOURCE_DIRECTORY = (
    "/content/drive/MyDrive/CEG-WM/Geometry-V3/P1/"
    "Geometry-V3-P1-517ba73993f1-20260828T131759Z"
)
P1_EXECUTION_EXACT = "517ba73993f11f51ade27fee181814294fe53797"
P1_RUN_ID = "geometry-v3-qk-p1-517ba73993f1"
P1_PROTOCOL = "geometry-v3-keyed-qk-active-writer-p1-confirmation-v1"
P1_PLAN_DIGEST = "daf83c679bb49b97a04f5b2b83716a8c6215310850472aad5544788531cf4a89"
P1_ROSTER_DIGEST = "df44b82f8649730bdf5a3c797a292cc8c43b145d06b2170aac6f1af5422db250"
P1_SOURCE_STATUS = "P1_UNRESOLVED"
P1_SOURCE_UNIT_COUNT = 24

_WRONG_KEY_DOMAIN = b"CEG-WM/geometry-v3/p1/wrong-key-control/v1\x00"
_SOURCE_FILENAMES = {"receipt.json", "manifest.json", "terminal.json", "metrics.jsonl"}
_PRIVATE_VALUE = re.compile(
    r"(?:raw\s*(?:q\s*/\s*k|qk|query|key|token)|hf[_ -]?token|access[_ -]?token|"
    r"auth[_ -]?token|api[_ -]?key|bearer\s+[a-z0-9._-]+|secret|credential|"
    r"model\s+weights?|weight\s+tensors?|prompt\s+text|image\s+bytes|latent\s+tensors?)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class ValidatedSources:
    p0_identity: dict[str, Any]
    p1_identity: dict[str, Any]
    p0_selected_scores: tuple[dict[str, Any], ...]
    p1_scores: tuple[dict[str, Any], ...]
    two_instance_displacement: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class P1M0ExecutionResult:
    status: str
    records: tuple[dict[str, Any], ...]
    writer_hook_scalars: tuple[dict[str, Any], ...]
    stage_decay: tuple[dict[str, Any], ...]
    contract_audit: tuple[dict[str, Any], ...]
    operational_failure_point: str | None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_public_leak(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("P1M0 source nesting exceeds bound")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("P1M0 source field name differs")
            lowered = key.lower()
            if any(term in lowered for term in (
                "raw_qk", "raw_query", "raw_key", "image_bytes", "prompt_text",
                "geometry_key", "hf_token", "access_token", "secret", "latent_tensor",
                "model_weights", "weight_tensor", "private_path",
            )):
                raise ValueError("P1M0 source contains forbidden field")
            _reject_public_leak(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _reject_public_leak(child, depth + 1)
    elif isinstance(value, str):
        normalized = value.lower().replace("\\", "/")
        embedded_path = (
            normalized.startswith("//") or normalized.startswith("~/")
            or "file://" in normalized or bool(re.search(r"\b[a-z]:/", normalized))
            or bool(re.search(r"(?<![:/a-z0-9._-])//[a-z0-9_.-]+/[a-z0-9_.-]+", normalized))
            or any(
                match.group(0) != "/content/drive"
                and not match.group(0).startswith("/content/drive/")
                for match in re.finditer(
                    r"(?<![:/a-z0-9._-])/[a-z0-9_.-]+(?:/[a-z0-9_.-]+)*", normalized
                )
            )
        )
        if embedded_path or _PRIVATE_VALUE.search(value):
            raise ValueError("P1M0 source contains forbidden value")


def _read_json(path: Path, maximum: int = 512 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("P1M0 source sidecar differs")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("P1M0 source sidecar root differs")
    _reject_public_leak(value)
    return value


def _read_metrics(path: Path, expected_count: int) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size < 2 * 1024 * 1024:
        raise ValueError("P1M0 source metrics differ")
    records: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        if not line or len(line) > 512 * 1024:
            raise ValueError("P1M0 source metric line differs")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("P1M0 source metric root differs")
        _reject_public_leak(value)
        records.append(value)
    if len(records) != expected_count:
        raise ValueError("P1M0 source metric count differs")
    return tuple(records)


def _validate_manifest(root: Path, manifest: Mapping[str, Any], expected_identity: tuple[str, ...]) -> None:
    if (
        manifest.get("run_id"), manifest.get("protocol"), manifest.get("execution_exact"),
        manifest.get("plan_digest"), manifest.get("roster_digest"),
    ) != expected_identity:
        raise ValueError("P1M0 source manifest identity differs")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("P1M0 source manifest roster differs")
    names: set[str] = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "bytes", "sha256"}:
            raise ValueError("P1M0 source manifest entry differs")
        name = entry.get("name")
        if name not in {"metrics.jsonl", "receipt.json", "terminal.json"} or name in names:
            raise ValueError("P1M0 source manifest filename differs")
        payload = (root / name).read_bytes()
        if entry.get("bytes") != len(payload) or entry.get("sha256") != _digest(payload):
            raise ValueError("P1M0 source payload binding differs")
        names.add(name)
        total += len(payload)
    if names != {"metrics.jsonl", "receipt.json", "terminal.json"}:
        raise ValueError("P1M0 source payload roster differs")
    if manifest.get("total_payload_bytes") != total:
        raise ValueError("P1M0 source aggregate differs")


def _public_score_record(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("config_id", "attack_id", "feature_kind", "control", "score", "margin")
    public = {field: record.get(field) for field in fields}
    for field in ("score", "margin"):
        value = public[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("P1M0 source public score is nonfinite")
        public[field] = float(value)
    return public


def validate_sources(p0_root: Path, p1_root: Path) -> ValidatedSources:
    """Validate both immutable source artifacts before any model load."""

    p0_identity = validate_p0_source(p0_root)
    if p0_identity != validate_p0_source_identity(p0_identity):
        raise ValueError("P1M0 P0 compact identity differs")
    p0_metrics = _read_metrics(p0_root / "metrics.jsonl", 144)
    p0_selected = tuple(
        _public_score_record(record) for record in p0_metrics
        if record.get("config_id") == P1_CONFIG_ID
    )
    if len(p0_selected) != 24:
        raise ValueError("P1M0 P0 selected score roster differs")

    if p1_root.is_symlink() or not p1_root.is_dir():
        raise ValueError("P1M0 P1 source root differs")
    children = tuple(p1_root.iterdir())
    if {path.name for path in children} != _SOURCE_FILENAMES:
        raise ValueError("P1M0 P1 source file roster differs")
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("P1M0 P1 source contains non-file")
    if sum(path.stat().st_size for path in children) >= P1M0_ARTIFACT_MAX_BYTES:
        raise ValueError("P1M0 P1 source aggregate exceeds bound")
    receipt = _read_json(p1_root / "receipt.json")
    manifest = _read_json(p1_root / "manifest.json")
    terminal = _read_json(p1_root / "terminal.json", 1024)
    p1_metrics = _read_metrics(p1_root / "metrics.jsonl", P1_SOURCE_UNIT_COUNT)
    identity_tuple = (
        P1_RUN_ID, P1_PROTOCOL, P1_EXECUTION_EXACT, P1_PLAN_DIGEST, P1_ROSTER_DIGEST,
    )
    _validate_manifest(p1_root, manifest, identity_tuple)
    if (
        receipt.get("run_id"), receipt.get("protocol"), receipt.get("execution_exact"),
        receipt.get("plan_digest"), receipt.get("roster_digest"), receipt.get("status"),
        receipt.get("artifact_status"), receipt.get("fixed_config_id"),
        receipt.get("fixed_unit_count"), receipt.get("calculated_unit_count"),
        receipt.get("failed_unit_count"), receipt.get("science_denominator"),
    ) != (
        P1_RUN_ID, P1_PROTOCOL, P1_EXECUTION_EXACT, P1_PLAN_DIGEST, P1_ROSTER_DIGEST,
        P1_SOURCE_STATUS, "complete", P1_CONFIG_ID, 24, 24, 0, 0,
    ):
        raise ValueError("P1M0 P1 receipt identity differs")
    if receipt.get("source_p0_artifact_identity") != p0_identity:
        raise ValueError("P1M0 P1 nested P0 identity differs")
    if (
        terminal.get("run_id"), terminal.get("status"), terminal.get("artifact_status"),
        terminal.get("fixed_config_id"), terminal.get("science_denominator"),
    ) != (P1_RUN_ID, P1_SOURCE_STATUS, "complete", P1_CONFIG_ID, 0):
        raise ValueError("P1M0 P1 terminal identity differs")
    expected_roster = tuple(
        (attack, kind, control)
        for attack in ("identity", "rotate270", "similarity", "crop_rescale")
        for kind in P1_KIND_IDS for control in P1_CONTROL_IDS
    )
    observed_roster = tuple(
        (record.get("attack_id"), record.get("feature_kind"), record.get("control"))
        for record in p1_metrics
    )
    if observed_roster != expected_roster:
        raise ValueError("P1M0 P1 metric roster differs")
    if any(
        record.get("config_id") != P1_CONFIG_ID
        or record.get("status") != "calculated"
        or record.get("error_class") is not None
        for record in p1_metrics
    ):
        raise ValueError("P1M0 P1 metric status differs")
    p1_scores = tuple(_public_score_record(record) for record in p1_metrics)

    p0_identity_scores = {
        (item["feature_kind"], item["control"]): item
        for item in p0_selected if item["attack_id"] == "identity"
    }
    p1_identity_scores = {
        (item["feature_kind"], item["control"]): item
        for item in p1_scores if item["attack_id"] == "identity"
    }
    if set(p0_identity_scores) != set(p1_identity_scores) or len(p0_identity_scores) != 6:
        raise ValueError("P1M0 identity score controls differ")
    displacement = tuple({
        "feature_kind": key[0],
        "control": key[1],
        "p0_score": p0_identity_scores[key]["score"],
        "p1_score": p1_identity_scores[key]["score"],
        "two_instance_displacement": p1_identity_scores[key]["score"] - p0_identity_scores[key]["score"],
    } for key in sorted(p0_identity_scores))
    p1_identity = {
        "run_id": P1_RUN_ID, "protocol": P1_PROTOCOL,
        "execution_exact": P1_EXECUTION_EXACT, "plan_digest": P1_PLAN_DIGEST,
        "roster_digest": P1_ROSTER_DIGEST, "status": P1_SOURCE_STATUS,
        "artifact_status": "complete", "fixed_config_id": P1_CONFIG_ID,
        "fixed_unit_count": 24, "calculated_unit_count": 24,
        "failed_unit_count": 0, "science_denominator": 0,
    }
    return ValidatedSources(
        dict(p0_identity), p1_identity, p0_selected, p1_scores, displacement,
    )


def public_plan() -> dict[str, Any]:
    return {
        "protocol": P1M0_PROTOCOL_ID,
        "model_id": P0_MODEL_ID,
        "prompt_id": P1M0_PROMPT_ID,
        "generation_seed": P1M0_GENERATION_SEED,
        "observation_noise_seed": P1M0_OBSERVATION_NOISE_SEED,
        "observation_timestep": P1M0_OBSERVATION_TIMESTEP,
        "fixed_config_id": P1_CONFIG_ID,
        "writer_step_index": 18,
        "relative_rms_budget": 0.0025,
        "generation_roles": list(P1M0_GENERATION_ROLES),
        "stages": list(P1M0_STAGES),
        "fixed_unit_count": P1M0_UNIT_COUNT,
        "decision_rule": {
            "implementation_mismatch": "any_public_contract_or_positive_injection_sign_check_false",
            "observability_insufficiency": "all_contracts_true_and_qk_hook_lifts_positive_and_qk_final_rgb_separations_nonpositive",
            "otherwise": "inconclusive",
        },
        "science_denominator": 0,
    }


def _generator_for(pipeline: Any) -> torch.Generator:
    device, _ = _module_device_dtype(getattr(pipeline, "transformer", None))
    return torch.Generator(device=device.type).manual_seed(P1M0_GENERATION_SEED)


class _LatentCapture:
    def __init__(self, writer_session: ActiveQKWriterSession | None = None) -> None:
        self.writer_session = writer_session
        self.steps: list[int] = []
        self.latents: dict[str, torch.Tensor] = {}

    def __call__(
        self, pipeline: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if self.writer_session is not None:
            callback_kwargs = self.writer_session.callback_on_step_end(
                pipeline, step_index, timestep, callback_kwargs,
            )
        if step_index != len(self.steps):
            raise RuntimeError("P1M0 callback order differs")
        self.steps.append(step_index)
        latent = callback_kwargs.get("latents")
        if not isinstance(latent, torch.Tensor) or not bool(torch.isfinite(latent).all()):
            raise RuntimeError("P1M0 callback latent contract differs")
        if step_index == 18:
            self.latents["writer_step18_latent"] = latent.detach().clone()
        if step_index == 19:
            self.latents["final_predecode_latent"] = latent.detach().clone()
        return callback_kwargs

    def assert_complete(self) -> dict[str, torch.Tensor]:
        if self.steps != list(range(P0_INFERENCE_STEPS)) or set(self.latents) != {
            "writer_step18_latent", "final_predecode_latent",
        }:
            raise RuntimeError("P1M0 latent capture did not complete")
        return self.latents


def _generate_no_writer(pipeline: Any) -> tuple[Any, dict[str, torch.Tensor]]:
    capture = _LatentCapture()
    result = pipeline(
        prompt=P1M0_PROMPT_TEXT, num_inference_steps=P0_INFERENCE_STEPS,
        height=P0_IMAGE_SIZE, width=P0_IMAGE_SIZE,
        generator=_generator_for(pipeline), output_type="pil",
        callback_on_step_end=capture,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("P1M0 no-writer generation differs")
    return require_ordinary_rgb_image(images[0]), capture.assert_complete()


def _generate_writer(
    pipeline: Any, correct_anchor: CanonicalRelationAnchor, wrong_anchor: CanonicalRelationAnchor,
) -> tuple[Any, dict[str, torch.Tensor], tuple[dict[str, Any], ...]]:
    observations: list[WriterScalarObservation] = []
    session = ActiveQKWriterSession(
        getattr(pipeline, "transformer", None), fixed_config(), correct_anchor,
        scalar_observer=observations.append, scalar_wrong_anchor=wrong_anchor,
    )
    capture = _LatentCapture(session)
    with session:
        result = pipeline(
            prompt=P1M0_PROMPT_TEXT, num_inference_steps=P0_INFERENCE_STEPS,
            height=P0_IMAGE_SIZE, width=P0_IMAGE_SIZE,
            generator=_generator_for(pipeline), output_type="pil",
            callback_on_step_end=capture,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    session.assert_complete()
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("P1M0 writer generation differs")
    if [item.feature_kind for item in observations] != ["q", "k"]:
        raise RuntimeError("P1M0 writer scalar observation roster differs")
    return (
        require_ordinary_rgb_image(images[0]), capture.assert_complete(),
        tuple(asdict(item) for item in observations),
    )


def _observe_latent(
    pipeline: Any, latent: torch.Tensor, correct_anchor: CanonicalRelationAnchor,
    wrong_anchor: CanonicalRelationAnchor,
) -> dict[str, tuple[float, float]]:
    transformer = getattr(pipeline, "transformer", None)
    if not isinstance(transformer, torch.nn.Module):
        raise RuntimeError("P1M0 transformer is unavailable")
    device, dtype = _module_device_dtype(transformer)
    latent = latent.detach().to(device=device, dtype=dtype)
    scheduler = _fresh_observation_scheduler(pipeline)
    generator = torch.Generator(device=device.type).manual_seed(P1M0_OBSERVATION_NOISE_SEED)
    noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)
    timestep = torch.tensor((P1M0_OBSERVATION_TIMESTEP,), device=device, dtype=torch.long)
    noisy = scheduler.scale_noise(latent, timestep, noise)
    if not isinstance(noisy, torch.Tensor) or noisy.shape != latent.shape or not bool(torch.isfinite(noisy).all()):
        raise RuntimeError("P1M0 observation noise contract differs")
    config_object = getattr(transformer, "config", None)
    encoder = torch.zeros(
        (1, P1M0_OBSERVATION_TEXT_TOKENS, _config_number(config_object, "joint_attention_dim")),
        device=device, dtype=dtype,
    )
    pooled = torch.zeros(
        (1, _config_number(config_object, "pooled_projection_dim")), device=device, dtype=dtype,
    )
    q_module, k_module = _module_pair(transformer, fixed_config().block_index)
    captured: dict[str, tuple[float, float]] = {}

    def capture(kind: str, module_path: str):
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            if kind in captured or not isinstance(output, torch.Tensor):
                raise RuntimeError("P1M0 fresh Q/K hook contract differs")
            correct = canonical_qk_pattern(correct_anchor, output, module_path=module_path)
            wrong = canonical_qk_pattern(wrong_anchor, output, module_path=module_path)
            captured[kind] = (
                normalized_pattern_correlation(output, correct),
                normalized_pattern_correlation(output, wrong),
            )
            return output
        return hook

    handles = (
        q_module.register_forward_hook(capture("q", f"{fixed_config().layer_path}.to_q")),
        k_module.register_forward_hook(capture("k", f"{fixed_config().layer_path}.to_k")),
    )
    try:
        with torch.no_grad():
            transformer(
                hidden_states=noisy, encoder_hidden_states=encoder,
                pooled_projections=pooled, timestep=timestep, return_dict=False,
            )
    finally:
        for handle in reversed(handles):
            handle.remove()
    if set(captured) != {"q", "k"}:
        raise RuntimeError("P1M0 fresh observer roster differs")
    return captured


def _rgb_latent(pipeline: Any, image: Any) -> torch.Tensor:
    ordinary = require_ordinary_rgb_image(image)
    return encode_final_rgb_image(
        ordinary, getattr(pipeline, "image_processor", None), getattr(pipeline, "vae", None),
    )


def _records_for(
    generation_role: str, stage: str, scores: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, Any], ...]:
    return tuple({
        "generation_role": generation_role,
        "stage": stage,
        "feature_kind": kind,
        "anchor_control": control,
        "score": float(scores[kind][index]),
        "status": "calculated",
        "error_class": None,
    } for kind in P1_KIND_IDS for index, control in enumerate(("correct_key_anchor", "wrong_key_anchor")))


def run_p1m0(pipeline: Any, geometry_key: str | bytes | bytearray | memoryview) -> P1M0ExecutionResult:
    key = normalize_detection_key(geometry_key)
    correct_anchor = derive_canonical_relation_anchor(key, point_count=P0_ANCHOR_POINT_COUNT)
    wrong_anchor = derive_canonical_relation_anchor(
        hashlib.sha256(_WRONG_KEY_DOMAIN + key).digest(), point_count=P0_ANCHOR_POINT_COUNT,
    )
    try:
        baseline_rgb, baseline_latents = _generate_no_writer(pipeline)
        writer_rgb, writer_latents, hook_scalars = _generate_writer(
            pipeline, correct_anchor, wrong_anchor,
        )
        records: list[dict[str, Any]] = []
        score_maps: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
        for role, rgb, latents in (
            ("no_writer", baseline_rgb, baseline_latents),
            ("writer", writer_rgb, writer_latents),
        ):
            for stage in P1M0_STAGES:
                latent = _rgb_latent(pipeline, rgb) if stage == "final_rgb_reencode" else latents[stage]
                scores = _observe_latent(pipeline, latent, correct_anchor, wrong_anchor)
                score_maps[(role, stage)] = scores
                records.extend(_records_for(role, stage, scores))
        if len(records) != P1M0_UNIT_COUNT:
            raise RuntimeError("P1M0 retained score roster differs")
        hook_by_kind = {item["feature_kind"]: item for item in hook_scalars}
        contract_audit: list[dict[str, Any]] = []
        stage_decay: list[dict[str, Any]] = []
        for kind in P1_KIND_IDS:
            hook = hook_by_kind[kind]
            hook_lift = float(hook["post_correct_correlation"] - hook["pre_correct_correlation"])
            sign_consistent = hook_lift > 0.0 and bool(hook["contract_pass"])
            contract_audit.append({
                "feature_kind": kind,
                "module_path": hook["module_path"],
                "spatial_axis": hook["spatial_axis"],
                "normalization": hook["normalization"],
                "injection_sign": hook["injection_sign"],
                "token_grid_side": hook["token_grid_side"],
                "token_count": hook["token_count"],
                "channel_count": hook["channel_count"],
                "contract_pass": bool(hook["contract_pass"]),
                "positive_injection_sign_consistent": sign_consistent,
            })
            previous = float(hook["post_correct_correlation"])
            for stage in P1M0_STAGES:
                writer_scores = score_maps[("writer", stage)][kind]
                baseline_scores = score_maps[("no_writer", stage)][kind]
                separation = float(writer_scores[0] - max(writer_scores[1], baseline_scores[0]))
                stage_decay.append({
                    "feature_kind": kind,
                    "stage": stage,
                    "writer_correct_score": float(writer_scores[0]),
                    "writer_wrong_score": float(writer_scores[1]),
                    "no_writer_correct_score": float(baseline_scores[0]),
                    "no_writer_wrong_score": float(baseline_scores[1]),
                    "correct_score_change_from_previous_stage": float(writer_scores[0] - previous),
                    "writer_separation": separation,
                })
                previous = float(writer_scores[0])
        mismatch = any(
            not item["contract_pass"] or not item["positive_injection_sign_consistent"]
            for item in contract_audit
        )
        rgb_separations = {
            item["feature_kind"]: item["writer_separation"]
            for item in stage_decay if item["stage"] == "final_rgb_reencode"
        }
        if mismatch:
            status = P1M0_STATUS_MISMATCH
        elif all(rgb_separations[kind] <= 0.0 for kind in P1_KIND_IDS):
            status = P1M0_STATUS_INSUFFICIENT
        else:
            status = P1M0_STATUS_INCONCLUSIVE
        return P1M0ExecutionResult(
            status, tuple(records), hook_scalars, tuple(stage_decay),
            tuple(contract_audit), None,
        )
    except Exception:  # noqa: BLE001 - bounded stopped result, no exception text
        return P1M0ExecutionResult(P1M0_STATUS_STOPPED, (), (), (), (), "mechanism_audit")


def package_p1m0_artifacts(
    output_directory: Path, *, exact: str, sources: ValidatedSources,
    result: P1M0ExecutionResult,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError("P1M0 output directory already exists")
    output_directory.mkdir(parents=True, exist_ok=False)
    run_id = f"geometry-v3-qk-p1m0-{exact[:12]}"
    stage_scores = b"".join(_json_bytes(record) + b"\n" for record in result.records)
    source_scores = _json_bytes({
        "p0_selected_config_public_scores": list(sources.p0_selected_scores),
        "p1_public_scores": list(sources.p1_scores),
        "identity_two_instance_displacement": list(sources.two_instance_displacement),
        "displacement_interpretation": "two_instance_displacement_not_population_variance",
    })
    plan_digest = _digest(_json_bytes(public_plan()))
    receipt = {
        "run_id": run_id, "protocol": P1M0_PROTOCOL_ID, "execution_exact": exact,
        "model_id": P0_MODEL_ID, "prompt_id": P1M0_PROMPT_ID,
        "source_p0_artifact_identity": sources.p0_identity,
        "source_p1_artifact_identity": sources.p1_identity,
        "plan_digest": plan_digest, "status": result.status,
        "artifact_status": "complete", "fixed_config_id": P1_CONFIG_ID,
        "fixed_unit_count": P1M0_UNIT_COUNT,
        "calculated_unit_count": len(result.records),
        "failed_unit_count": P1M0_UNIT_COUNT - len(result.records),
        "writer_hook_scalars": list(result.writer_hook_scalars),
        "stage_decay": list(result.stage_decay),
        "contract_audit": list(result.contract_audit),
        "operational_failure_point": result.operational_failure_point,
        "science_denominator": 0,
    }
    terminal = {
        "run_id": run_id, "status": result.status, "artifact_status": "complete",
        "fixed_config_id": P1_CONFIG_ID, "science_denominator": 0,
    }
    payloads = {
        "source_scores.json": source_scores,
        "stage_scores.jsonl": stage_scores,
        "receipt.json": _json_bytes(receipt),
        "terminal.json": _json_bytes(terminal),
    }
    manifest = {
        "run_id": run_id, "protocol": P1M0_PROTOCOL_ID, "execution_exact": exact,
        "plan_digest": plan_digest,
        "files": [
            {"name": name, "bytes": len(data), "sha256": _digest(data)}
            for name, data in sorted(payloads.items())
        ],
        "total_payload_bytes": sum(len(data) for data in payloads.values()),
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    if sum(len(data) for data in payloads.values()) >= P1M0_ARTIFACT_MAX_BYTES:
        raise RuntimeError("P1M0 artifact exceeds bound")
    for name, data in payloads.items():
        with (output_directory / name).open("xb") as stream:
            stream.write(data)
    return {
        "run_id": run_id, "status": result.status, "artifact_status": "complete",
        "fixed_config_id": P1_CONFIG_ID, "science_denominator": 0,
    }


def load_real_pipeline(model_id: str, token: str) -> Any:
    if model_id != P0_MODEL_ID:
        raise ValueError("P1M0 model identity differs")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_geometry_v3_p1m0")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=token)
    pipeline.to("cuda")
    return pipeline


def execute_plan(
    plan: Mapping[str, Any], *, geometry_key: str, hf_token: str,
    sources: ValidatedSources,
    preloader: Callable[[str, str], Any] = load_real_pipeline,
) -> dict[str, Any]:
    if set(plan) != {
        "expected_exact", "execution_exact", "p0_source_directory",
        "p1_source_directory", "output_directory",
    }:
        raise ValueError("P1M0 plan fields differ")
    expected, execution = plan["expected_exact"], plan["execution_exact"]
    if not isinstance(expected, str) or expected != execution or len(expected) != 40:
        raise ValueError("P1M0 execution identity differs")
    if plan["p0_source_directory"] != P0_SOURCE_DIRECTORY:
        raise ValueError("P1M0 P0 source path differs")
    if plan["p1_source_directory"] != P1_SOURCE_DIRECTORY:
        raise ValueError("P1M0 P1 source path differs")
    output = plan["output_directory"]
    if not isinstance(output, str) or not output.startswith(
        "/content/drive/MyDrive/CEG-WM/Geometry-V3/P1M0/Geometry-V3-P1M0-"
    ):
        raise ValueError("P1M0 output namespace differs")
    if not geometry_key.strip() or not hf_token.strip():
        raise ValueError("P1M0 runtime credentials are required")
    pipeline = preloader(P0_MODEL_ID, hf_token)
    result = run_p1m0(pipeline, geometry_key)
    return package_p1m0_artifacts(Path(output), exact=execution, sources=sources, result=result)
