"""Independent Geometry-V3 P1 active-anchor confirmation.

P1 consumes only the bounded public P0 artifact, freezes one discovered
writer configuration, and uses new generation and attack instances.  Images,
keys, anchors, latents, and Q/K tensors remain transient process memory.
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

import numpy as np
import torch
from PIL import Image

from cegwm.geometry_v3.active_writer import (
    ActiveQKWriterSession,
    P0_ANCHOR_POINT_COUNT,
    P0_CONFIGS,
    P0_IMAGE_SIZE,
    P0_INFERENCE_STEPS,
    P0_MODEL_ID,
    P0WriterConfig,
    WriterInjectionMeasurement,
    canonical_qk_pattern,
    normalized_pattern_correlation,
)
from cegwm.geometry_v3.contracts import CanonicalRelationAnchor, derive_canonical_relation_anchor
from cegwm.geometry_v3.operational import (
    ObservationScores,
    _config_number,
    _fresh_observation_scheduler,
    _module_device_dtype,
    _module_pair,
    _rgb_interference,
    _transform_points,
)
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key


P1_PROTOCOL_ID = "geometry-v3-keyed-qk-active-writer-p1-confirmation-v1"
P1_PROMPT_ID = "geometry-v3-p1-public-prompt-01"
P1_PROMPT_TEXT = (
    "A blue ceramic teapot beside a folded linen cloth on a wooden table, soft window light"
)
P1_GENERATION_SEED = 173
P1_OBSERVATION_NOISE_SEED = 19073
P1_OBSERVATION_TIMESTEP = 500
P1_OBSERVATION_TEXT_TOKENS = 333
P1_CONFIG_ID = "block12-qk-rms0p0025"
P1_ATTACK_IDS = ("identity", "rotate270", "similarity", "crop_rescale")
P1_KIND_IDS = ("q", "k")
P1_CONTROL_IDS = ("correct_key_anchor", "wrong_key_anchor", "no_writer")
P1_UNIT_COUNT = 24
P1_STATUS_STOPPED = "P1_STOPPED"
P1_STATUS_UNRESOLVED = "P1_UNRESOLVED"
P1_STATUS_CONFIRMED = "P1_ACTIVE_ANCHOR_CONFIRMED"
P1_SCIENCE_DENOMINATOR = 0
P1_ARTIFACT_MAX_BYTES = 2 * 1024 * 1024

SOURCE_EXECUTION_EXACT = "9b5085c805b6e3580fadc153598aac93fcc41eab"
SOURCE_RUN_ID = "geometry-v3-qk-p0-9b5085c805b6"
SOURCE_PROTOCOL = "geometry-v3-keyed-qk-active-writer-p0-v1"
SOURCE_PLAN_DIGEST = "672e93ebae97407976cfb42bf0f73ddc39753651514925c2fd55b5c7016b0c96"
SOURCE_ROSTER_DIGEST = "e84b7d3d746627f46e4bc9355fe0764500523da67f1f76ae82c0b60598adcdfd"
SOURCE_STATUS = "P0_WRITER_CANDIDATE_FROZEN"
SOURCE_SELECTED_CONFIG_ID = P1_CONFIG_ID
SOURCE_UNIT_COUNT = 144
SOURCE_ROOT_MAX_BYTES = 2 * 1024 * 1024
SOURCE_SIDECAR_MAX_BYTES = 512 * 1024
SOURCE_TERMINAL_MAX_BYTES = 1024

_WRONG_KEY_DOMAIN = b"CEG-WM/geometry-v3/p1/wrong-key-control/v1\x00"
_SOURCE_FILENAMES = {"receipt.json", "manifest.json", "terminal.json", "metrics.jsonl"}
_PRIVATE_VALUE_PATTERNS = (
    re.compile(r"\braw\s*(?:q\s*/\s*k|qk|query|key|token)\b", re.I),
    re.compile(r"\b(?:hf[_ -]?token|access[_ -]?token|auth[_ -]?token|api[_ -]?key|bearer\s+[a-z0-9._-]+|secret|credential)\b", re.I),
    re.compile(r"\b(?:model\s+weights?|weight\s+tensors?|prompt\s+text|image\s+bytes|latent\s+tensors?)\b", re.I),
)


@dataclass(frozen=True, slots=True)
class AttackResult:
    image: Image.Image
    homography: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class GeneratedConfig:
    image: Image.Image
    measurements: tuple[WriterInjectionMeasurement, ...]


@dataclass(frozen=True, slots=True)
class P1ExecutionResult:
    status: str
    records: tuple[dict[str, Any], ...]
    q_four_attack_equal_weight_median_margin: float | None
    k_four_attack_equal_weight_median_margin: float | None
    per_transform_audit: tuple[dict[str, Any], ...]
    interference: tuple[dict[str, Any], ...]
    writer_measurements: tuple[dict[str, Any], ...]
    operational_failure_point: str | None


def fixed_config() -> P0WriterConfig:
    matches = tuple(config for config in P0_CONFIGS if config.config_id == P1_CONFIG_ID)
    if len(matches) != 1:
        raise RuntimeError("P1 frozen writer configuration differs")
    config = matches[0]
    if config.block_index != 12 or config.relative_rms_budget != 0.0025:
        raise RuntimeError("P1 placement or budget differs")
    return config


def fixed_roster() -> tuple[tuple[str, str, str], ...]:
    roster = tuple(
        (attack, kind, control)
        for attack in P1_ATTACK_IDS
        for kind in P1_KIND_IDS
        for control in P1_CONTROL_IDS
    )
    if len(roster) != P1_UNIT_COUNT or len(set(roster)) != P1_UNIT_COUNT:
        raise RuntimeError("P1 fixed roster differs")
    return roster


def public_plan() -> dict[str, Any]:
    return {
        "protocol": P1_PROTOCOL_ID,
        "model_id": P0_MODEL_ID,
        "prompt_id": P1_PROMPT_ID,
        "image_size": [P0_IMAGE_SIZE, P0_IMAGE_SIZE],
        "inference_steps": P0_INFERENCE_STEPS,
        "generation_seed": P1_GENERATION_SEED,
        "observation_noise_seed": P1_OBSERVATION_NOISE_SEED,
        "observation_timestep": P1_OBSERVATION_TIMESTEP,
        "fixed_config": {
            "config_id": P1_CONFIG_ID,
            "block_index": 12,
            "feature_kinds": list(P1_KIND_IDS),
            "writer_step_index": 18,
            "relative_rms_budget": 0.0025,
        },
        "attacks": [
            {"attack_id": "identity"},
            {"attack_id": "rotate270", "pillow_transpose": "ROTATE_270"},
            {"attack_id": "similarity", "angle_degrees": -11.0, "scale": 0.89,
             "translation": [-17.0, 9.0], "centre": [256.0, 256.0]},
            {"attack_id": "crop_rescale", "box": [46, 28, 470, 482],
             "output_size": [512, 512], "resampler": "BICUBIC"},
        ],
        "controls": list(P1_CONTROL_IDS),
        "fixed_unit_count": P1_UNIT_COUNT,
        "confirmation_rule": "q_and_k_four_attack_equal_weight_median_margin_strictly_positive",
        "science_denominator": 0,
    }


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path, maximum: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("P1 source sidecar is not a regular file")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError("P1 source sidecar exceeds its public bound")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("P1 source sidecar is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("P1 source sidecar root must be an object")
    return value


def _reject_public_leak(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("P1 public value nesting exceeds the bound")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("P1 public field name must be a string")
            lowered = key.lower()
            if any(term in lowered for term in (
                "raw_qk", "raw_query", "raw_key", "image_bytes", "prompt_text",
                "geometry_key", "hf_token", "access_token", "secret", "latent_tensor",
                "model_weights", "weight_tensor", "private_path",
            )):
                raise ValueError("P1 source contains a forbidden public field")
            _reject_public_leak(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _reject_public_leak(child, depth + 1)
    elif isinstance(value, str):
        lowered = value.lower()
        normalized = lowered.replace("\\", "/")
        embedded_private_path = (
            normalized.startswith("//")
            or normalized.startswith("~/")
            or "file://" in normalized
            or bool(re.search(r"\b[a-z]:/", normalized))
            or bool(re.search(r"(?<![:/a-z0-9._-])//[a-z0-9_.-]+/[a-z0-9_.-]+", normalized))
            or any(
                match.group(0) != "/content/drive"
                and not match.group(0).startswith("/content/drive/")
                for match in re.finditer(
                    r"(?<![:/a-z0-9._-])/[a-z0-9_.-]+(?:/[a-z0-9_.-]+)*", normalized
                )
            )
        )
        if embedded_private_path or any(pattern.search(lowered) for pattern in _PRIVATE_VALUE_PATTERNS):
            raise ValueError("P1 source contains a forbidden public value")


def _read_metrics(path: Path) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > SOURCE_ROOT_MAX_BYTES:
        raise ValueError("P1 source metrics exceed the public bound")
    records: list[dict[str, Any]] = []
    for raw_line in path.read_bytes().splitlines():
        if not raw_line or len(raw_line) > SOURCE_SIDECAR_MAX_BYTES:
            raise ValueError("P1 source metric line differs")
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("P1 source metric line is invalid") from error
        if not isinstance(value, dict):
            raise ValueError("P1 source metric line must be an object")
        _reject_public_leak(value)
        records.append(value)
    return tuple(records)


def _recompute_source_selection(
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Independently replay the frozen P0 margin, gate, and winner rule."""

    summaries: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[float, float, float, int], str]] = []
    for config in P0_CONFIGS:
        q_margins: list[float] = []
        k_margins: list[float] = []
        calculated = 0
        for attack in ("identity", "rotate90", "similarity", "crop_rescale"):
            for kind, destination in (("q", q_margins), ("k", k_margins)):
                group = [
                    record for record in metrics
                    if record["config_id"] == config.config_id
                    and record["attack_id"] == attack
                    and record["feature_kind"] == kind
                ]
                if len(group) != 3 or {record["control"] for record in group} != set(P1_CONTROL_IDS):
                    raise ValueError("P1 source control score structure differs")
                by_control = {record["control"]: record for record in group}
                scores = {
                    control: float(by_control[control]["score"])
                    for control in P1_CONTROL_IDS
                }
                recomputed = scores["correct_key_anchor"] - max(
                    scores["wrong_key_anchor"], scores["no_writer"]
                )
                if not math.isfinite(recomputed):
                    raise ValueError("P1 source recomputed margin is nonfinite")
                for record in group:
                    if float(record["margin"]) != recomputed:
                        raise ValueError("P1 source recorded margin differs from scores")
                destination.append(recomputed)
                calculated += len(group)
        if len(q_margins) != 4 or len(k_margins) != 4 or calculated != 24:
            raise ValueError("P1 source configuration metric roster differs")
        q_median = float(np.median(np.asarray(q_margins, dtype=np.float64)))
        k_median = float(np.median(np.asarray(k_margins, dtype=np.float64)))
        complete = calculated == 24
        is_eligible = complete and q_median > 0.0 and k_median > 0.0
        summaries.append({
            "config_id": config.config_id,
            "block_index": config.block_index,
            "relative_rms_budget": config.relative_rms_budget,
            "calculated_unit_count": calculated,
            "q_four_attack_equal_weight_median_margin": q_median,
            "k_four_attack_equal_weight_median_margin": k_median,
            "eligible": is_eligible,
        })
        if is_eligible:
            worst = min(q_median, k_median)
            centre = float(np.median(np.asarray((q_median, k_median), dtype=np.float64)))
            eligible.append((
                (-worst, -centre, config.relative_rms_budget, config.block_index),
                config.config_id,
            ))
    if len(eligible) != 1:
        raise ValueError("P1 source must retain exactly one eligible P0 candidate")
    eligible.sort(key=lambda item: item[0])
    return tuple(summaries), eligible[0][1]


def validate_p0_source(root: Path) -> dict[str, Any]:
    """Validate the immutable public P0 artifact before any model work."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("P1 source root must be a real directory")
    children = tuple(root.iterdir())
    if {path.name for path in children} != _SOURCE_FILENAMES:
        raise ValueError("P1 source file roster differs")
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("P1 source contains a non-regular file")
    if sum(path.stat().st_size for path in children) >= SOURCE_ROOT_MAX_BYTES:
        raise ValueError("P1 source artifact exceeds its aggregate bound")

    receipt = _read_json(root / "receipt.json", SOURCE_SIDECAR_MAX_BYTES)
    manifest = _read_json(root / "manifest.json", SOURCE_SIDECAR_MAX_BYTES)
    terminal = _read_json(root / "terminal.json", SOURCE_TERMINAL_MAX_BYTES)
    _reject_public_leak(receipt)
    _reject_public_leak(manifest)
    _reject_public_leak(terminal)
    metrics = _read_metrics(root / "metrics.jsonl")

    if set(receipt) != {
        "run_id", "protocol", "execution_exact", "model_id", "prompt_id",
        "plan_digest", "roster_digest", "status", "artifact_status",
        "fixed_unit_count", "calculated_unit_count", "failed_unit_count",
        "selected_config_id", "operational_failure_point", "science_denominator",
        "config_summaries", "interference", "writer_measurements",
    }:
        raise ValueError("P1 source receipt fields differ")
    if set(manifest) != {
        "run_id", "protocol", "execution_exact", "plan_digest", "roster_digest",
        "files", "total_payload_bytes",
    }:
        raise ValueError("P1 source manifest fields differ")
    if set(terminal) != {
        "run_id", "status", "artifact_status", "selected_config_id", "science_denominator",
    }:
        raise ValueError("P1 source terminal fields differ")

    receipt_identity = (
        receipt.get("run_id"), receipt.get("protocol"), receipt.get("execution_exact"),
        receipt.get("plan_digest"), receipt.get("roster_digest"), receipt.get("artifact_status"),
        receipt.get("fixed_unit_count"), receipt.get("calculated_unit_count"),
        receipt.get("failed_unit_count"), receipt.get("status"),
        receipt.get("selected_config_id"), receipt.get("science_denominator"),
    )
    expected_receipt_identity = (
        SOURCE_RUN_ID, SOURCE_PROTOCOL, SOURCE_EXECUTION_EXACT, SOURCE_PLAN_DIGEST,
        SOURCE_ROSTER_DIGEST, "complete", SOURCE_UNIT_COUNT, SOURCE_UNIT_COUNT, 0,
        SOURCE_STATUS, SOURCE_SELECTED_CONFIG_ID, 0,
    )
    if receipt_identity != expected_receipt_identity:
        raise ValueError("P1 source receipt identity differs")

    if (
        manifest.get("run_id"), manifest.get("protocol"), manifest.get("execution_exact"),
        manifest.get("plan_digest"), manifest.get("roster_digest"),
    ) != (
        SOURCE_RUN_ID, SOURCE_PROTOCOL, SOURCE_EXECUTION_EXACT,
        SOURCE_PLAN_DIGEST, SOURCE_ROSTER_DIGEST,
    ):
        raise ValueError("P1 source manifest identity differs")
    if (
        terminal.get("run_id"), terminal.get("status"), terminal.get("artifact_status"),
        terminal.get("selected_config_id"), terminal.get("science_denominator"),
    ) != (SOURCE_RUN_ID, SOURCE_STATUS, "complete", SOURCE_SELECTED_CONFIG_ID, 0):
        raise ValueError("P1 source terminal identity differs")

    file_entries = manifest.get("files")
    if not isinstance(file_entries, list) or len(file_entries) != 3:
        raise ValueError("P1 source manifest file roster differs")
    expected_payload_names = {"metrics.jsonl", "receipt.json", "terminal.json"}
    observed_payload_names: set[str] = set()
    payload_total = 0
    for entry in file_entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "bytes", "sha256"}:
            raise ValueError("P1 source manifest file entry differs")
        name = entry["name"]
        if not isinstance(name, str) or name not in expected_payload_names or name in observed_payload_names:
            raise ValueError("P1 source manifest filename differs")
        payload = (root / name).read_bytes()
        if entry["bytes"] != len(payload) or entry["sha256"] != _digest(payload):
            raise ValueError("P1 source payload binding differs")
        observed_payload_names.add(name)
        payload_total += len(payload)
    if observed_payload_names != expected_payload_names or manifest.get("total_payload_bytes") != payload_total:
        raise ValueError("P1 source manifest aggregate differs")

    if len(metrics) != SOURCE_UNIT_COUNT:
        raise ValueError("P1 source metric count differs")
    expected_roster = tuple(
        (config.config_id, attack, kind, control)
        for config in P0_CONFIGS
        for attack in ("identity", "rotate90", "similarity", "crop_rescale")
        for kind in P1_KIND_IDS
        for control in P1_CONTROL_IDS
    )
    observed_roster: list[tuple[str, str, str, str]] = []
    for record in metrics:
        if set(record) != {
            "config_id", "attack_id", "feature_kind", "control", "status",
            "error_class", "score", "margin",
        }:
            raise ValueError("P1 source metric fields differ")
        identity = (
            record.get("config_id"), record.get("attack_id"),
            record.get("feature_kind"), record.get("control"),
        )
        observed_roster.append(identity)
        if record.get("status") != "calculated" or record.get("error_class") is not None:
            raise ValueError("P1 source retains a non-calculated unit")
        for field in ("score", "margin"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("P1 source metric is nonfinite")
    if tuple(observed_roster) != expected_roster:
        raise ValueError("P1 source metric roster differs")

    summaries = receipt.get("config_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(P0_CONFIGS):
        raise ValueError("P1 source configuration summaries differ")
    if any(
        not isinstance(item, dict) or set(item) != {
            "config_id", "block_index", "relative_rms_budget", "calculated_unit_count",
            "q_four_attack_equal_weight_median_margin",
            "k_four_attack_equal_weight_median_margin", "eligible",
        }
        for item in summaries
    ):
        raise ValueError("P1 source configuration summary fields differ")
    if [item["config_id"] for item in summaries] != [config.config_id for config in P0_CONFIGS]:
        raise ValueError("P1 source configuration summary roster differs")
    recomputed_summaries, recomputed_winner = _recompute_source_selection(metrics)
    if tuple(summaries) != recomputed_summaries:
        raise ValueError("P1 source receipt summaries differ from metrics")
    if recomputed_winner != receipt.get("selected_config_id"):
        raise ValueError("P1 source selected candidate differs from frozen P0 replay")
    selected = [item for item in summaries if isinstance(item, dict) and item.get("config_id") == P1_CONFIG_ID]
    if len(selected) != 1 or selected[0].get("eligible") is not True or selected[0].get("calculated_unit_count") != 24:
        raise ValueError("P1 source selected configuration summary differs")
    if sum(item.get("eligible") is True for item in summaries) != 1:
        raise ValueError("P1 source candidate uniqueness differs")
    measurements = receipt.get("writer_measurements")
    interference = receipt.get("interference")
    if not isinstance(measurements, list) or len(measurements) != 12:
        raise ValueError("P1 source writer measurement count differs")
    if not isinstance(interference, list) or len(interference) != 6:
        raise ValueError("P1 source interference count differs")
    if [item.get("config_id") for item in interference if isinstance(item, dict)] != [
        config.config_id for config in P0_CONFIGS
    ]:
        raise ValueError("P1 source interference roster differs")
    if [
        (item.get("config_id"), item.get("feature_kind"))
        for item in measurements if isinstance(item, dict)
    ] != [
        (config.config_id, kind) for config in P0_CONFIGS for kind in P1_KIND_IDS
    ]:
        raise ValueError("P1 source writer measurement roster differs")

    return {
        "run_id": SOURCE_RUN_ID,
        "protocol": SOURCE_PROTOCOL,
        "execution_exact": SOURCE_EXECUTION_EXACT,
        "plan_digest": SOURCE_PLAN_DIGEST,
        "roster_digest": SOURCE_ROSTER_DIGEST,
        "status": SOURCE_STATUS,
        "artifact_status": "complete",
        "fixed_unit_count": SOURCE_UNIT_COUNT,
        "calculated_unit_count": SOURCE_UNIT_COUNT,
        "failed_unit_count": 0,
        "selected_config_id": SOURCE_SELECTED_CONFIG_ID,
        "science_denominator": 0,
    }


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("P1 attack homography must be finite 3x3")
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _pillow_inverse_affine(homography: np.ndarray) -> tuple[float, ...]:
    inverse = np.linalg.inv(homography)
    linear, offset = inverse[:2, :2], inverse[:2, 2]
    pixel_offset = linear @ np.array((0.5, 0.5)) + offset - 0.5
    return (
        float(linear[0, 0]), float(linear[0, 1]), float(pixel_offset[0]),
        float(linear[1, 0]), float(linear[1, 1]), float(pixel_offset[1]),
    )


def apply_attack(image: Any, attack_id: str) -> AttackResult:
    rgb = require_ordinary_rgb_image(image)
    if rgb.size != (P0_IMAGE_SIZE, P0_IMAGE_SIZE):
        raise ValueError("P1 attacks require exactly 512x512 RGB")
    if attack_id == "identity":
        return AttackResult(rgb.copy(), _matrix_tuple(np.eye(3, dtype=np.float64)))
    if attack_id == "rotate270":
        output = rgb.transpose(Image.Transpose.ROTATE_270)
        h = np.array(((0.0, -1.0, 512.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        return AttackResult(output, _matrix_tuple(h))
    if attack_id == "similarity":
        angle = math.radians(-11.0)
        scale = 0.89
        cosine, sine = math.cos(angle), math.sin(angle)
        linear = np.array(((scale * cosine, -scale * sine), (scale * sine, scale * cosine)))
        centre = np.array((256.0, 256.0))
        translation = np.array((-17.0, 9.0))
        offset = centre + translation - linear @ centre
        h = np.array(
            ((linear[0, 0], linear[0, 1], offset[0]),
             (linear[1, 0], linear[1, 1], offset[1]),
             (0.0, 0.0, 1.0)), dtype=np.float64,
        )
        output = rgb.transform(
            rgb.size, Image.Transform.AFFINE, _pillow_inverse_affine(h),
            resample=Image.Resampling.BICUBIC,
        )
        return AttackResult(output, _matrix_tuple(h))
    if attack_id == "crop_rescale":
        left, top, right, bottom = 46, 28, 470, 482
        output = rgb.crop((left, top, right, bottom)).resize(rgb.size, Image.Resampling.BICUBIC)
        sx, sy = 512.0 / (right - left), 512.0 / (bottom - top)
        h = np.array(((sx, 0.0, -left * sx), (0.0, sy, -top * sy), (0.0, 0.0, 1.0)))
        return AttackResult(output, _matrix_tuple(h))
    raise ValueError("P1 attack is outside the fixed roster")


def _generator_for(pipeline: Any) -> torch.Generator:
    device, _ = _module_device_dtype(getattr(pipeline, "transformer", None))
    return torch.Generator(device=device.type).manual_seed(P1_GENERATION_SEED)


def generate_no_writer(pipeline: Any) -> Image.Image:
    result = pipeline(
        prompt=P1_PROMPT_TEXT, num_inference_steps=P0_INFERENCE_STEPS,
        height=P0_IMAGE_SIZE, width=P0_IMAGE_SIZE,
        generator=_generator_for(pipeline), output_type="pil",
    )
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("P1 baseline generation must return one final RGB")
    return require_ordinary_rgb_image(images[0])


def generate_writer_config(
    pipeline: Any, config: P0WriterConfig, anchor: CanonicalRelationAnchor,
) -> GeneratedConfig:
    if config != fixed_config():
        raise ValueError("P1 cannot switch writer configuration")
    with ActiveQKWriterSession(getattr(pipeline, "transformer", None), config, anchor) as session:
        result = pipeline(
            prompt=P1_PROMPT_TEXT, num_inference_steps=P0_INFERENCE_STEPS,
            height=P0_IMAGE_SIZE, width=P0_IMAGE_SIZE,
            generator=_generator_for(pipeline), output_type="pil",
            callback_on_step_end=session.callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    measurements = session.assert_complete()
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("P1 writer generation must return one final RGB")
    return GeneratedConfig(require_ordinary_rgb_image(images[0]), measurements)


def observe_fresh_attacked_rgb(
    pipeline: Any,
    image: Any,
    config: P0WriterConfig,
    correct_anchor: CanonicalRelationAnchor,
    wrong_anchor: CanonicalRelationAnchor,
    homography: Sequence[Sequence[float]],
) -> ObservationScores:
    ordinary = require_ordinary_rgb_image(image)
    transformer = getattr(pipeline, "transformer", None)
    if not isinstance(transformer, torch.nn.Module):
        raise RuntimeError("P1 pipeline transformer is unavailable")
    scheduler = _fresh_observation_scheduler(pipeline)
    latent = encode_final_rgb_image(ordinary, getattr(pipeline, "image_processor", None), getattr(pipeline, "vae", None))
    device, dtype = _module_device_dtype(transformer)
    latent = latent.to(device=device, dtype=dtype)
    generator = torch.Generator(device=device.type).manual_seed(P1_OBSERVATION_NOISE_SEED)
    noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)
    timestep = torch.tensor((P1_OBSERVATION_TIMESTEP,), device=device, dtype=torch.long)
    noisy = scheduler.scale_noise(latent, timestep, noise)
    if (
        not isinstance(noisy, torch.Tensor) or noisy.shape != latent.shape
        or noisy.dtype != latent.dtype or noisy.device != latent.device
        or not bool(torch.isfinite(noisy).all())
    ):
        raise RuntimeError("P1 observation noise contract differs")
    config_object = getattr(transformer, "config", None)
    encoder = torch.zeros(
        (1, P1_OBSERVATION_TEXT_TOKENS, _config_number(config_object, "joint_attention_dim")),
        device=device, dtype=dtype,
    )
    pooled = torch.zeros(
        (1, _config_number(config_object, "pooled_projection_dim")), device=device, dtype=dtype
    )
    q_module, k_module = _module_pair(transformer, config.block_index)
    correct_points = _transform_points(correct_anchor.points, homography)
    wrong_points = _transform_points(wrong_anchor.points, homography)
    captured: dict[str, tuple[float, float]] = {}

    def capture(kind: str, module_path: str):
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            if kind in captured:
                raise RuntimeError("P1 fresh observer Q/K hook repeated")
            if not isinstance(output, torch.Tensor):
                raise TypeError("P1 fresh observer requires tensor Q/K")
            correct_pattern = canonical_qk_pattern(
                correct_anchor, output, module_path=module_path, transformed_points=correct_points
            )
            wrong_pattern = canonical_qk_pattern(
                wrong_anchor, output, module_path=module_path, transformed_points=wrong_points
            )
            captured[kind] = (
                normalized_pattern_correlation(output, correct_pattern),
                normalized_pattern_correlation(output, wrong_pattern),
            )
            return output
        return hook

    handles = (
        q_module.register_forward_hook(capture("q", f"{config.layer_path}.to_q")),
        k_module.register_forward_hook(capture("k", f"{config.layer_path}.to_k")),
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
        raise RuntimeError("P1 fresh observer did not capture Q and K")
    return ObservationScores(
        q_correct=captured["q"][0], q_wrong=captured["q"][1],
        k_correct=captured["k"][0], k_wrong=captured["k"][1],
    )


def _failure_record(identity: tuple[str, str, str], error_class: str) -> dict[str, Any]:
    attack, kind, control = identity
    return {
        "config_id": P1_CONFIG_ID, "attack_id": attack, "feature_kind": kind,
        "control": control, "status": "failed", "error_class": error_class,
        "score": None, "margin": None,
    }


def _public_error(error: BaseException) -> str:
    if isinstance(error, (TypeError, ValueError)):
        return "validation_error"
    if isinstance(error, RuntimeError):
        return "runtime_error"
    return "operational_error"


def confirm_active_anchor(
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, float | None, float | None, tuple[dict[str, Any], ...]]:
    if len(records) != P1_UNIT_COUNT:
        raise ValueError("P1 confirmation requires all 24 retained units")
    identities = tuple(
        (record.get("attack_id"), record.get("feature_kind"), record.get("control"))
        for record in records
    )
    if identities != fixed_roster():
        raise ValueError("P1 confirmation roster order differs")
    if any(record.get("status") != "calculated" for record in records):
        return P1_STATUS_STOPPED, None, None, ()
    audits: list[dict[str, Any]] = []
    medians: dict[str, float] = {}
    for kind in P1_KIND_IDS:
        margins: list[float] = []
        for attack in P1_ATTACK_IDS:
            matches = [
                record for record in records
                if record["attack_id"] == attack and record["feature_kind"] == kind
                and record["control"] == "correct_key_anchor"
            ]
            if len(matches) != 1:
                raise ValueError("P1 per-transform confirmation record differs")
            value = matches[0].get("margin")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("P1 confirmation margin is nonfinite")
            margin = float(value)
            margins.append(margin)
            audits.append({"attack_id": attack, "feature_kind": kind, "margin": margin})
        medians[kind] = float(np.median(np.asarray(margins, dtype=np.float64)))
    status = (
        P1_STATUS_CONFIRMED if medians["q"] > 0.0 and medians["k"] > 0.0
        else P1_STATUS_UNRESOLVED
    )
    return status, medians["q"], medians["k"], tuple(audits)


def run_p1(pipeline: Any, geometry_key: str | bytes | bytearray | memoryview) -> P1ExecutionResult:
    key = normalize_detection_key(geometry_key)
    correct_anchor = derive_canonical_relation_anchor(key, point_count=P0_ANCHOR_POINT_COUNT)
    wrong_anchor = derive_canonical_relation_anchor(
        hashlib.sha256(_WRONG_KEY_DOMAIN + key).digest(), point_count=P0_ANCHOR_POINT_COUNT
    )
    roster = fixed_roster()
    records_by_id: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        baseline = generate_no_writer(pipeline)
    except Exception as error:  # noqa: BLE001 - retain the complete roster
        public = _public_error(error)
        return P1ExecutionResult(
            P1_STATUS_STOPPED, tuple(_failure_record(identity, public) for identity in roster),
            None, None, (), (), (), "baseline_generation",
        )
    try:
        generated = generate_writer_config(pipeline, fixed_config(), correct_anchor)
        interference = (_rgb_interference(generated.image, baseline, P1_CONFIG_ID),)
        measurements = tuple(asdict(item) for item in generated.measurements)
    except Exception as error:  # noqa: BLE001 - retain the complete roster
        public = _public_error(error)
        return P1ExecutionResult(
            P1_STATUS_STOPPED, tuple(_failure_record(identity, public) for identity in roster),
            None, None, (), (), (), "writer_generation",
        )
    for attack_id in P1_ATTACK_IDS:
        identities = tuple(
            (attack_id, kind, control) for kind in P1_KIND_IDS for control in P1_CONTROL_IDS
        )
        try:
            writer_attack = apply_attack(generated.image, attack_id)
            baseline_attack = apply_attack(baseline, attack_id)
            writer_scores = observe_fresh_attacked_rgb(
                pipeline, writer_attack.image, fixed_config(), correct_anchor, wrong_anchor,
                writer_attack.homography,
            )
            no_writer_scores = observe_fresh_attacked_rgb(
                pipeline, baseline_attack.image, fixed_config(), correct_anchor, wrong_anchor,
                baseline_attack.homography,
            )
            for kind in P1_KIND_IDS:
                correct = float(getattr(writer_scores, f"{kind}_correct"))
                wrong = float(getattr(writer_scores, f"{kind}_wrong"))
                no_writer = float(getattr(no_writer_scores, f"{kind}_correct"))
                values = (correct, wrong, no_writer)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("P1 observation score is nonfinite")
                margin = correct - max(wrong, no_writer)
                for control, score in zip(P1_CONTROL_IDS, values, strict=True):
                    identity = (attack_id, kind, control)
                    records_by_id[identity] = {
                        "config_id": P1_CONFIG_ID, "attack_id": attack_id,
                        "feature_kind": kind, "control": control,
                        "status": "calculated", "error_class": None,
                        "score": score, "margin": margin,
                    }
        except Exception as error:  # noqa: BLE001 - retain all six attack units
            public = _public_error(error)
            for identity in identities:
                records_by_id[identity] = _failure_record(identity, public)
    ordered = tuple(records_by_id.get(identity, _failure_record(identity, "runtime_error")) for identity in roster)
    status, q_median, k_median, audits = confirm_active_anchor(ordered)
    return P1ExecutionResult(
        status, ordered, q_median, k_median, audits, interference, measurements,
        "fresh_observation" if status == P1_STATUS_STOPPED else None,
    )


def package_p1_artifacts(
    output_directory: Path, *, exact: str, source_identity: Mapping[str, Any], result: P1ExecutionResult,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError("P1 output directory already exists")
    output_directory.mkdir(parents=True, exist_ok=False)
    run_id = f"geometry-v3-qk-p1-{exact[:12]}"
    metrics = b"".join(_json_bytes(record) + b"\n" for record in result.records)
    plan_digest = _digest(_json_bytes(public_plan()))
    roster_digest = _digest(_json_bytes(fixed_roster()))
    receipt = {
        "run_id": run_id, "protocol": P1_PROTOCOL_ID, "execution_exact": exact,
        "model_id": P0_MODEL_ID, "prompt_id": P1_PROMPT_ID,
        "source_p0_artifact_identity": dict(source_identity),
        "plan_digest": plan_digest, "roster_digest": roster_digest,
        "status": result.status, "artifact_status": "complete",
        "fixed_config_id": P1_CONFIG_ID, "fixed_unit_count": P1_UNIT_COUNT,
        "calculated_unit_count": sum(record["status"] == "calculated" for record in result.records),
        "failed_unit_count": sum(record["status"] == "failed" for record in result.records),
        "q_four_attack_equal_weight_median_margin": result.q_four_attack_equal_weight_median_margin,
        "k_four_attack_equal_weight_median_margin": result.k_four_attack_equal_weight_median_margin,
        "per_transform_audit": list(result.per_transform_audit),
        "interference": list(result.interference),
        "writer_measurements": list(result.writer_measurements),
        "operational_failure_point": result.operational_failure_point,
        "science_denominator": 0,
    }
    terminal = {
        "run_id": run_id, "status": result.status, "artifact_status": "complete",
        "fixed_config_id": P1_CONFIG_ID, "science_denominator": 0,
    }
    payloads = {
        "metrics.jsonl": metrics,
        "receipt.json": _json_bytes(receipt),
        "terminal.json": _json_bytes(terminal),
    }
    manifest = {
        "run_id": run_id, "protocol": P1_PROTOCOL_ID, "execution_exact": exact,
        "source_p0_artifact_identity": dict(source_identity),
        "plan_digest": plan_digest, "roster_digest": roster_digest,
        "files": [
            {"name": name, "bytes": len(data), "sha256": _digest(data)}
            for name, data in sorted(payloads.items())
        ],
        "total_payload_bytes": sum(len(data) for data in payloads.values()),
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    if sum(len(data) for data in payloads.values()) >= P1_ARTIFACT_MAX_BYTES:
        raise RuntimeError("P1 bounded artifact exceeds two MiB")
    for name, data in payloads.items():
        with (output_directory / name).open("xb") as stream:
            stream.write(data)
    return {
        "run_id": run_id, "status": result.status, "artifact_status": "complete",
        "fixed_config_id": P1_CONFIG_ID, "science_denominator": 0,
    }


def load_real_pipeline(model_id: str, token: str) -> Any:
    if model_id != P0_MODEL_ID:
        raise ValueError("P1 model identity differs")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_geometry_v3_p1")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=token)
    pipeline.to("cuda")
    return pipeline


def execute_plan(
    plan: Mapping[str, Any], *, geometry_key: str, hf_token: str,
    source_identity: Mapping[str, Any], preloader: Callable[[str, str], Any] = load_real_pipeline,
) -> dict[str, Any]:
    if set(plan) != {"expected_exact", "execution_exact", "source_directory", "output_directory"}:
        raise ValueError("P1 plan fields differ")
    expected, execution = plan["expected_exact"], plan["execution_exact"]
    if not isinstance(expected, str) or expected != execution or len(expected) != 40:
        raise ValueError("P1 execution identity differs")
    source = plan["source_directory"]
    if not isinstance(source, str) or source != (
        "/content/drive/MyDrive/CEG-WM/Geometry-V3/P0/"
        "Geometry-V3-P0-9b5085c805b6-20260828T122005Z"
    ):
        raise ValueError("P1 source path differs")
    output = plan["output_directory"]
    if not isinstance(output, str) or not output.startswith(
        "/content/drive/MyDrive/CEG-WM/Geometry-V3/P1/Geometry-V3-P1-"
    ):
        raise ValueError("P1 output must use its create-only Drive namespace")
    if dict(source_identity) != validate_p0_source_identity(source_identity):
        raise ValueError("P1 validated source identity differs")
    if not geometry_key.strip() or not hf_token.strip():
        raise ValueError("P1 runtime credentials are required")
    pipeline = preloader(P0_MODEL_ID, hf_token)
    result = run_p1(pipeline, geometry_key)
    return package_p1_artifacts(Path(output), exact=execution, source_identity=source_identity, result=result)


def validate_p0_source_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "run_id": SOURCE_RUN_ID, "protocol": SOURCE_PROTOCOL,
        "execution_exact": SOURCE_EXECUTION_EXACT, "plan_digest": SOURCE_PLAN_DIGEST,
        "roster_digest": SOURCE_ROSTER_DIGEST, "status": SOURCE_STATUS,
        "artifact_status": "complete", "fixed_unit_count": SOURCE_UNIT_COUNT,
        "calculated_unit_count": SOURCE_UNIT_COUNT, "failed_unit_count": 0,
        "selected_config_id": SOURCE_SELECTED_CONFIG_ID, "science_denominator": 0,
    }
    if dict(value) != expected:
        raise ValueError("P1 compact source identity differs")
    return expected
