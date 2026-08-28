"""Operational Geometry-V3 P0 active-writer discovery path.

The production path emits bounded derived scalars only.  Keys, prompts,
images, latents, canonical tensors, model weights, and Q/K tensors remain
transient process memory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
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
    P0_PROTOCOL_ID,
    P0_SEED,
    P0WriterConfig,
    WriterInjectionMeasurement,
    canonical_qk_pattern,
    normalized_pattern_correlation,
)
from cegwm.geometry_v3.contracts import (
    CanonicalRelationAnchor,
    derive_canonical_relation_anchor,
)
from cegwm.runtime.diffusers_sd35 import load_sd35_pipeline
from cegwm.runtime.observation import encode_final_rgb_image, require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key


P0_PROMPT_ID = "geometry-v3-p0-public-prompt-01"
P0_PROMPT_TEXT = "A small red lighthouse on a rocky island at dawn, calm sea, clear sky"
P0_OBSERVATION_NOISE_SEED = 9073
P0_OBSERVATION_TIMESTEP = 500
P0_OBSERVATION_TEXT_TOKENS = 333
P0_STATUS_STOPPED = "P0_STOPPED"
P0_STATUS_UNRESOLVED = "P0_UNRESOLVED"
P0_STATUS_FROZEN = "P0_WRITER_CANDIDATE_FROZEN"
P0_SCIENCE_DENOMINATOR = 0
P0_UNIT_COUNT = 144
P0_ARTIFACT_MAX_BYTES = 2 * 1024 * 1024
ATTACK_IDS = ("identity", "rotate90", "similarity", "crop_rescale")
CONTROL_IDS = ("correct_key_anchor", "wrong_key_anchor", "no_writer")
KIND_IDS = ("q", "k")
_WRONG_KEY_DOMAIN = b"CEG-WM/geometry-v3/p0/wrong-key-control/v1\x00"


@dataclass(frozen=True, slots=True)
class AttackResult:
    image: Image.Image
    homography: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class ObservationScores:
    q_correct: float
    q_wrong: float
    k_correct: float
    k_wrong: float


@dataclass(frozen=True, slots=True)
class GeneratedConfig:
    image: Image.Image
    measurements: tuple[WriterInjectionMeasurement, ...]


@dataclass(frozen=True, slots=True)
class P0ExecutionResult:
    status: str
    selected_config_id: str | None
    records: tuple[dict[str, Any], ...]
    config_summaries: tuple[dict[str, Any], ...]
    interference: tuple[dict[str, Any], ...]
    writer_measurements: tuple[dict[str, Any], ...]
    operational_failure_point: str | None


def fixed_roster() -> tuple[tuple[str, str, str, str], ...]:
    roster = tuple(
        (config.config_id, attack, kind, control)
        for config in P0_CONFIGS
        for attack in ATTACK_IDS
        for kind in KIND_IDS
        for control in CONTROL_IDS
    )
    if len(roster) != P0_UNIT_COUNT or len(set(roster)) != P0_UNIT_COUNT:
        raise RuntimeError("P0 fixed roster construction differs")
    return roster


def public_plan() -> dict[str, Any]:
    """Return the complete public P0 method identity without private inputs."""

    return {
        "protocol": P0_PROTOCOL_ID,
        "model_id": P0_MODEL_ID,
        "prompt_id": P0_PROMPT_ID,
        "image_size": [P0_IMAGE_SIZE, P0_IMAGE_SIZE],
        "inference_steps": P0_INFERENCE_STEPS,
        "generation_seed": P0_SEED,
        "writer_step_index": 18,
        "placement_groups": [
            {
                "block_index": block,
                "module_paths": [
                    f"transformer_blocks.{block}.attn.to_q",
                    f"transformer_blocks.{block}.attn.to_k",
                ],
            }
            for block in (4, 12, 20)
        ],
        "relative_rms_budgets": [0.0025, 0.005],
        "attacks": list(ATTACK_IDS),
        "controls": list(CONTROL_IDS),
        "observation_noise_seed": P0_OBSERVATION_NOISE_SEED,
        "observation_timestep": P0_OBSERVATION_TIMESTEP,
        "fixed_unit_count": P0_UNIT_COUNT,
        "science_denominator": 0,
    }


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("attack homography must be finite 3x3")
    return tuple(tuple(float(value) for value in row) for row in matrix)


def apply_attack(image: Any, attack_id: str) -> AttackResult:
    """Apply one frozen Pillow attack and return its forward RGB homography."""

    rgb = require_ordinary_rgb_image(image)
    if rgb.size != (P0_IMAGE_SIZE, P0_IMAGE_SIZE):
        raise ValueError("P0 attacks require exactly 512x512 RGB")
    if attack_id == "identity":
        return AttackResult(rgb.copy(), _matrix_tuple(np.eye(3, dtype=np.float64)))
    if attack_id == "rotate90":
        output = rgb.transpose(Image.Transpose.ROTATE_90)
        h = np.array(((0.0, 1.0, 0.0), (-1.0, 0.0, 512.0), (0.0, 0.0, 1.0)))
        return AttackResult(output, _matrix_tuple(h))
    if attack_id == "similarity":
        angle = math.radians(7.0)
        scale = 0.93
        cosine, sine = math.cos(angle), math.sin(angle)
        linear = np.array(((scale * cosine, -scale * sine), (scale * sine, scale * cosine)))
        centre = np.array((256.0, 256.0))
        translation = np.array((13.0, 17.0))
        offset = centre + translation - linear @ centre
        h = np.array(
            ((linear[0, 0], linear[0, 1], offset[0]),
             (linear[1, 0], linear[1, 1], offset[1]),
             (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        inverse = np.linalg.inv(h)
        coefficients = tuple(float(value) for value in inverse[:2].reshape(-1))
        output = rgb.transform(
            rgb.size,
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
        )
        return AttackResult(output, _matrix_tuple(h))
    if attack_id == "crop_rescale":
        left, top, right, bottom = 32, 44, 476, 468
        output = rgb.crop((left, top, right, bottom)).resize(
            rgb.size, Image.Resampling.BICUBIC
        )
        sx, sy = 512.0 / (right - left), 512.0 / (bottom - top)
        h = np.array(((sx, 0.0, -left * sx), (0.0, sy, -top * sy), (0.0, 0.0, 1.0)))
        return AttackResult(output, _matrix_tuple(h))
    raise ValueError("P0 attack is not in the fixed roster")


def _transform_points(
    points: Sequence[Sequence[float]],
    homography: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    h = np.asarray(homography, dtype=np.float64)
    transformed: list[tuple[float, float]] = []
    for x_normalized, y_normalized in points:
        source = np.array((float(x_normalized) * 512.0, float(y_normalized) * 512.0, 1.0))
        destination = h @ source
        if not np.isfinite(destination).all() or abs(float(destination[2])) <= 1e-12:
            raise ValueError("attack truth homography cannot transport anchor points")
        transformed.append(
            (float(destination[0] / destination[2] / 512.0),
             float(destination[1] / destination[2] / 512.0))
        )
    return tuple(transformed)


def _module_pair(transformer: Any, block_index: int) -> tuple[torch.nn.Module, torch.nn.Module]:
    blocks = getattr(transformer, "transformer_blocks", None)
    if not isinstance(blocks, (torch.nn.ModuleList, list, tuple)) or len(blocks) <= block_index:
        raise RuntimeError("SD3 transformer block topology differs")
    attention = getattr(blocks[block_index], "attn", None)
    q_module, k_module = getattr(attention, "to_q", None), getattr(attention, "to_k", None)
    if not isinstance(q_module, torch.nn.Module) or not isinstance(k_module, torch.nn.Module):
        raise RuntimeError("SD3 sample-side Q/K module topology differs")
    return q_module, k_module


def _module_device_dtype(module: Any) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(module.parameters())
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("SD3 transformer device/dtype is unavailable") from error
    if not parameter.dtype.is_floating_point:
        raise RuntimeError("SD3 transformer requires a floating dtype")
    return parameter.device, parameter.dtype


def _config_number(config: Any, name: str) -> int:
    value = config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"SD3 transformer config {name} is unavailable")
    return value


def observe_fresh_attacked_rgb(
    pipeline: Any,
    image: Any,
    config: P0WriterConfig,
    correct_anchor: CanonicalRelationAnchor,
    wrong_anchor: CanonicalRelationAnchor,
    homography: Sequence[Sequence[float]],
) -> ObservationScores:
    """Recompute Q/K from the current attacked RGB and return correlations only."""

    ordinary = require_ordinary_rgb_image(image)
    transformer = getattr(pipeline, "transformer", None)
    vae = getattr(pipeline, "vae", None)
    processor = getattr(pipeline, "image_processor", None)
    scheduler = getattr(pipeline, "scheduler", None)
    if not isinstance(transformer, torch.nn.Module):
        raise RuntimeError("pipeline transformer is unavailable")
    add_noise = getattr(scheduler, "add_noise", None)
    if not callable(add_noise):
        raise RuntimeError("pipeline scheduler add_noise is unavailable")
    latent = encode_final_rgb_image(ordinary, processor, vae)
    device, dtype = _module_device_dtype(transformer)
    latent = latent.to(device=device, dtype=dtype)
    generator = torch.Generator(device=device.type).manual_seed(P0_OBSERVATION_NOISE_SEED)
    noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)
    timestep = torch.tensor((P0_OBSERVATION_TIMESTEP,), device=device, dtype=torch.long)
    noisy = add_noise(latent, noise, timestep)
    if not isinstance(noisy, torch.Tensor) or noisy.shape != latent.shape:
        raise RuntimeError("frozen observation noise contract differs")
    config_object = getattr(transformer, "config", None)
    joint_dimension = _config_number(config_object, "joint_attention_dim")
    pooled_dimension = _config_number(config_object, "pooled_projection_dim")
    encoder = torch.zeros(
        (1, P0_OBSERVATION_TEXT_TOKENS, joint_dimension), device=device, dtype=dtype
    )
    pooled = torch.zeros((1, pooled_dimension), device=device, dtype=dtype)
    q_module, k_module = _module_pair(transformer, config.block_index)
    correct_points = _transform_points(correct_anchor.points, homography)
    wrong_points = _transform_points(wrong_anchor.points, homography)
    captured: dict[str, tuple[float, float]] = {}

    def capture(kind: str, module_path: str):
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            if kind in captured:
                raise RuntimeError("fresh observer Q/K module was called more than once")
            if not isinstance(output, torch.Tensor):
                raise TypeError("fresh observer Q/K projection must return a tensor")
            correct_pattern = canonical_qk_pattern(
                correct_anchor,
                output,
                module_path=module_path,
                transformed_points=correct_points,
            )
            wrong_pattern = canonical_qk_pattern(
                wrong_anchor,
                output,
                module_path=module_path,
                transformed_points=wrong_points,
            )
            captured[kind] = (
                normalized_pattern_correlation(output, correct_pattern),
                normalized_pattern_correlation(output, wrong_pattern),
            )
            return output

        return hook

    handles = [
        q_module.register_forward_hook(capture("q", f"{config.layer_path}.to_q")),
        k_module.register_forward_hook(capture("k", f"{config.layer_path}.to_k")),
    ]
    try:
        with torch.no_grad():
            transformer(
                hidden_states=noisy,
                encoder_hidden_states=encoder,
                pooled_projections=pooled,
                timestep=timestep,
                return_dict=False,
            )
    finally:
        for handle in reversed(handles):
            handle.remove()
    if set(captured) != {"q", "k"}:
        raise RuntimeError("fresh observer did not capture both Q and K")
    return ObservationScores(
        q_correct=captured["q"][0],
        q_wrong=captured["q"][1],
        k_correct=captured["k"][0],
        k_wrong=captured["k"][1],
    )


def _generator_for(pipeline: Any) -> torch.Generator:
    transformer = getattr(pipeline, "transformer", None)
    device, _ = _module_device_dtype(transformer)
    return torch.Generator(device=device.type).manual_seed(P0_SEED)


def generate_no_writer(pipeline: Any) -> Image.Image:
    result = pipeline(
        prompt=P0_PROMPT_TEXT,
        num_inference_steps=P0_INFERENCE_STEPS,
        height=P0_IMAGE_SIZE,
        width=P0_IMAGE_SIZE,
        generator=_generator_for(pipeline),
        output_type="pil",
    )
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("P0 baseline generation must return one final RGB")
    return require_ordinary_rgb_image(images[0])


def generate_writer_config(
    pipeline: Any,
    config: P0WriterConfig,
    anchor: CanonicalRelationAnchor,
) -> GeneratedConfig:
    transformer = getattr(pipeline, "transformer", None)
    with ActiveQKWriterSession(transformer, config, anchor) as session:
        result = pipeline(
            prompt=P0_PROMPT_TEXT,
            num_inference_steps=P0_INFERENCE_STEPS,
            height=P0_IMAGE_SIZE,
            width=P0_IMAGE_SIZE,
            generator=_generator_for(pipeline),
            output_type="pil",
            callback_on_step_end=session.callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    measurements = session.assert_complete()
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("P0 writer generation must return one final RGB")
    return GeneratedConfig(require_ordinary_rgb_image(images[0]), measurements)


def _rgb_interference(writer: Image.Image, baseline: Image.Image, config_id: str) -> dict[str, Any]:
    left = np.asarray(writer, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired P0 final RGB shapes differ")
    mse = float(np.mean((left - right) ** 2))
    psnr = None if mse <= 0.0 else float(10.0 * math.log10(255.0**2 / mse))
    return {
        "config_id": config_id,
        "rgb_mse": mse,
        "rgb_psnr_db": psnr,
        "content_detector_hook_status": "not_invoked_record_only",
    }


def _failure_record(identity: tuple[str, str, str, str], error_class: str) -> dict[str, Any]:
    config_id, attack, kind, control = identity
    return {
        "config_id": config_id,
        "attack_id": attack,
        "feature_kind": kind,
        "control": control,
        "status": "failed",
        "error_class": error_class,
        "score": None,
        "margin": None,
    }


def _public_error(error: BaseException) -> str:
    if isinstance(error, (TypeError, ValueError)):
        return "validation_error"
    if isinstance(error, RuntimeError):
        return "runtime_error"
    return "operational_error"


def _median(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("P0 aggregate requires finite fixed values")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def select_writer_candidate(records: Sequence[Mapping[str, Any]]) -> tuple[str, tuple[dict[str, Any], ...]]:
    if len(records) != P0_UNIT_COUNT:
        raise ValueError("P0 selection requires the complete 144-unit roster")
    summaries: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[float, float, float, int], str]] = []
    by_config = {config.config_id: config for config in P0_CONFIGS}
    for config in P0_CONFIGS:
        subset = [record for record in records if record["config_id"] == config.config_id]
        complete = len(subset) == 24 and all(record["status"] == "calculated" for record in subset)
        q_values = [
            float(record["margin"])
            for record in subset
            if record["feature_kind"] == "q" and record["control"] == "correct_key_anchor"
        ]
        k_values = [
            float(record["margin"])
            for record in subset
            if record["feature_kind"] == "k" and record["control"] == "correct_key_anchor"
        ]
        q_median = _median(q_values) if complete and len(q_values) == 4 else None
        k_median = _median(k_values) if complete and len(k_values) == 4 else None
        is_eligible = bool(
            complete and q_median is not None and k_median is not None
            and q_median > 0.0 and k_median > 0.0
        )
        summary = {
            "config_id": config.config_id,
            "block_index": config.block_index,
            "relative_rms_budget": config.relative_rms_budget,
            "calculated_unit_count": sum(record["status"] == "calculated" for record in subset),
            "q_four_attack_equal_weight_median_margin": q_median,
            "k_four_attack_equal_weight_median_margin": k_median,
            "eligible": is_eligible,
        }
        summaries.append(summary)
        if is_eligible:
            worst = min(float(q_median), float(k_median))
            centre = _median((float(q_median), float(k_median)))
            eligible.append(((-worst, -centre, config.relative_rms_budget, config.block_index), config.config_id))
    if not eligible:
        return P0_STATUS_UNRESOLVED, tuple(summaries)
    eligible.sort(key=lambda item: item[0])
    winner = eligible[0][1]
    if winner not in by_config:
        raise RuntimeError("P0 selected config is outside the fixed roster")
    return winner, tuple(summaries)


def run_p0(
    pipeline: Any,
    geometry_key: str | bytes | bytearray | memoryview,
) -> P0ExecutionResult:
    key = normalize_detection_key(geometry_key)
    correct_anchor = derive_canonical_relation_anchor(key, point_count=P0_ANCHOR_POINT_COUNT)
    wrong_key = hashlib.sha256(_WRONG_KEY_DOMAIN + key).digest()
    wrong_anchor = derive_canonical_relation_anchor(wrong_key, point_count=P0_ANCHOR_POINT_COUNT)
    roster = fixed_roster()
    records_by_id: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    interference: list[dict[str, Any]] = []
    writer_measurements: list[dict[str, Any]] = []
    generated: dict[str, GeneratedConfig] = {}
    failure_point: str | None = None
    try:
        baseline = generate_no_writer(pipeline)
    except Exception as error:  # noqa: BLE001 - retain the complete fixed roster
        error_class = _public_error(error)
        return P0ExecutionResult(
            P0_STATUS_STOPPED,
            None,
            tuple(_failure_record(identity, error_class) for identity in roster),
            (),
            (),
            (),
            "baseline_generation",
        )
    for config in P0_CONFIGS:
        try:
            generated[config.config_id] = generate_writer_config(pipeline, config, correct_anchor)
            writer_measurements.extend(
                asdict(measurement)
                for measurement in generated[config.config_id].measurements
            )
            interference.append(
                _rgb_interference(generated[config.config_id].image, baseline, config.config_id)
            )
        except Exception as error:  # noqa: BLE001 - retain this config's 24 units
            failure_point = failure_point or "writer_generation"
            error_class = _public_error(error)
            for identity in roster:
                if identity[0] == config.config_id:
                    records_by_id[identity] = _failure_record(identity, error_class)
    baseline_scores: dict[tuple[int, str], ObservationScores] = {}
    for config in P0_CONFIGS:
        if config.config_id not in generated:
            continue
        for attack_id in ATTACK_IDS:
            identities = tuple(
                (config.config_id, attack_id, kind, control)
                for kind in KIND_IDS for control in CONTROL_IDS
            )
            try:
                writer_attack = apply_attack(generated[config.config_id].image, attack_id)
                writer_scores = observe_fresh_attacked_rgb(
                    pipeline,
                    writer_attack.image,
                    config,
                    correct_anchor,
                    wrong_anchor,
                    writer_attack.homography,
                )
                baseline_key = (config.block_index, attack_id)
                if baseline_key not in baseline_scores:
                    baseline_attack = apply_attack(baseline, attack_id)
                    baseline_scores[baseline_key] = observe_fresh_attacked_rgb(
                        pipeline,
                        baseline_attack.image,
                        config,
                        correct_anchor,
                        wrong_anchor,
                        baseline_attack.homography,
                    )
                no_writer = baseline_scores[baseline_key]
                for kind in KIND_IDS:
                    correct = getattr(writer_scores, f"{kind}_correct")
                    wrong = getattr(writer_scores, f"{kind}_wrong")
                    null = getattr(no_writer, f"{kind}_correct")
                    margin = correct - max(wrong, null)
                    scores = {
                        "correct_key_anchor": correct,
                        "wrong_key_anchor": wrong,
                        "no_writer": null,
                    }
                    for control, score in scores.items():
                        identity = (config.config_id, attack_id, kind, control)
                        records_by_id[identity] = {
                            "config_id": config.config_id,
                            "attack_id": attack_id,
                            "feature_kind": kind,
                            "control": control,
                            "status": "calculated",
                            "error_class": None,
                            "score": float(score),
                            "margin": float(margin),
                        }
            except Exception as error:  # noqa: BLE001 - retain all six observation units
                failure_point = failure_point or "fresh_observation"
                error_class = _public_error(error)
                for identity in identities:
                    records_by_id[identity] = _failure_record(identity, error_class)
    ordered = tuple(records_by_id.get(identity, _failure_record(identity, "runtime_error")) for identity in roster)
    if any(record["status"] != "calculated" for record in ordered):
        return P0ExecutionResult(
            P0_STATUS_STOPPED,
            None,
            ordered,
            (),
            tuple(interference),
            tuple(writer_measurements),
            failure_point or "retained_units",
        )
    selected_or_status, summaries = select_writer_candidate(ordered)
    if selected_or_status == P0_STATUS_UNRESOLVED:
        return P0ExecutionResult(
            P0_STATUS_UNRESOLVED,
            None,
            ordered,
            summaries,
            tuple(interference),
            tuple(writer_measurements),
            None,
        )
    return P0ExecutionResult(
        P0_STATUS_FROZEN,
        selected_or_status,
        ordered,
        summaries,
        tuple(interference),
        tuple(writer_measurements),
        None,
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_p0_artifacts(
    output_directory: Path,
    *,
    exact: str,
    result: P0ExecutionResult,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError("P0 output directory already exists")
    output_directory.mkdir(parents=True, exist_ok=False)
    run_id = f"geometry-v3-qk-p0-{exact[:12]}"
    metrics = b"".join(_json_bytes(record) + b"\n" for record in result.records)
    plan_digest = _digest(_json_bytes(public_plan()))
    roster_digest = _digest(_json_bytes(fixed_roster()))
    receipt = {
        "run_id": run_id,
        "protocol": P0_PROTOCOL_ID,
        "execution_exact": exact,
        "model_id": P0_MODEL_ID,
        "prompt_id": P0_PROMPT_ID,
        "plan_digest": plan_digest,
        "roster_digest": roster_digest,
        "status": result.status,
        "artifact_status": "complete",
        "fixed_unit_count": P0_UNIT_COUNT,
        "calculated_unit_count": sum(record["status"] == "calculated" for record in result.records),
        "failed_unit_count": sum(record["status"] == "failed" for record in result.records),
        "selected_config_id": result.selected_config_id,
        "operational_failure_point": result.operational_failure_point,
        "science_denominator": P0_SCIENCE_DENOMINATOR,
        "config_summaries": list(result.config_summaries),
        "interference": list(result.interference),
        "writer_measurements": list(result.writer_measurements),
    }
    terminal = {
        "run_id": run_id,
        "status": result.status,
        "artifact_status": "complete",
        "selected_config_id": result.selected_config_id,
        "science_denominator": 0,
    }
    payloads = {
        "metrics.jsonl": metrics,
        "receipt.json": _json_bytes(receipt),
        "terminal.json": _json_bytes(terminal),
    }
    manifest = {
        "run_id": run_id,
        "protocol": P0_PROTOCOL_ID,
        "execution_exact": exact,
        "plan_digest": plan_digest,
        "roster_digest": roster_digest,
        "files": [
            {"name": name, "bytes": len(data), "sha256": _digest(data)}
            for name, data in sorted(payloads.items())
        ],
        "total_payload_bytes": sum(len(data) for data in payloads.values()),
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    total = sum(len(data) for data in payloads.values())
    if total >= P0_ARTIFACT_MAX_BYTES:
        raise RuntimeError("P0 bounded artifact exceeds two MiB")
    for name, data in payloads.items():
        path = output_directory / name
        with path.open("xb") as stream:
            stream.write(data)
    return {
        "run_id": run_id,
        "status": result.status,
        "artifact_status": "complete",
        "selected_config_id": result.selected_config_id,
        "science_denominator": 0,
    }


def load_real_pipeline(model_id: str, token: str) -> Any:
    if model_id != P0_MODEL_ID:
        raise ValueError("P0 model identity differs")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_real_geometry_v3_p0")
    pipeline = load_sd35_pipeline(model_id, torch_dtype=torch.float16, token=token)
    pipeline.to("cuda")
    return pipeline


def execute_plan(
    plan: Mapping[str, Any],
    *,
    geometry_key: str,
    hf_token: str,
    preloader: Callable[[str, str], Any] = load_real_pipeline,
) -> dict[str, Any]:
    if set(plan) != {"expected_exact", "execution_exact", "output_directory"}:
        raise ValueError("P0 plan fields differ")
    expected, execution = plan["expected_exact"], plan["execution_exact"]
    if not isinstance(expected, str) or expected != execution or len(expected) != 40:
        raise ValueError("P0 execution identity differs")
    output = plan["output_directory"]
    if not isinstance(output, str) or not output.startswith("/content/drive/MyDrive/CEG-WM/Geometry-V3/P0/"):
        raise ValueError("P0 output must use its create-only Drive namespace")
    if not geometry_key.strip() or not hf_token.strip():
        raise ValueError("P0 runtime credentials are required")
    pipeline = preloader(P0_MODEL_ID, hf_token)
    result = run_p0(pipeline, geometry_key)
    return package_p0_artifacts(Path(output), exact=execution, result=result)
