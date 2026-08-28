"""Real PyTorch/Pillow N0 training and operational evaluation for Geometry-V2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from cegwm.geometry_v2.contracts import (
    GeometryEstimate,
    PROTOCOL_IDENTITY,
    derive_keyed_sync_target,
)
from cegwm.geometry_v2.neural_sync import (
    BlindCornerExtractor,
    IMAGE_SIZE,
    KeyedResidualEmbedder,
    MAX_RESIDUAL,
    n0_joint_loss,
)


TRAIN_SEEDS = tuple(range(1000, 1128))
VALIDATION_SEEDS = tuple(range(2000, 2032))
CONFIRMATION_SEEDS = tuple(range(3000, 3032))
ATTACKS = ("identity", "rotate90", "similarity", "crop_rescale")
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 1.0e-3
TRAINING_SEED = 73
MINIMUM_SUPPORT = 1.0
RELIABILITY_THRESHOLD = 0.5
CONFIRMATION_UNIT_COUNT = 128
STATUS_STOPPED = "N0_STOPPED"
STATUS_UNRESOLVED = "N0_UNRESOLVED"
STATUS_CANDIDATE = "N0_GEOMETRY_CANDIDATE"


def _stable_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _procedural_rgb(seed: int) -> Tensor:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("procedural seed must be an integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    axis = torch.linspace(0.0, 1.0, IMAGE_SIZE)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    phase = (seed % 97) / 97.0
    image = torch.stack(
        (
            0.18 + 0.58 * xx + 0.08 * torch.sin((yy + phase) * math.tau * 3),
            0.18 + 0.58 * yy + 0.08 * torch.cos((xx + phase) * math.tau * 4),
            0.20 + 0.30 * xx + 0.30 * yy + 0.07 * torch.sin((xx - yy + phase) * math.tau * 5),
        )
    )
    image = image + (torch.rand((3, IMAGE_SIZE, IMAGE_SIZE), generator=generator) - 0.5) * 0.04
    return image.clamp(0.05, 0.95).to(torch.float32)


def _target_code(geometry_key: bytes, seed: int, *, device: torch.device) -> Tensor:
    context = f"geometry-v2-n0/sample={seed}".encode("ascii")
    target = derive_keyed_sync_target(geometry_key, context, code_length=64)
    return torch.tensor(target.bipolar_code, dtype=torch.float32, device=device)


def _tensor_to_pil(image: Tensor) -> Image.Image:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(array * 255.0).clip(0, 255).astype(np.uint8), "RGB")


def _pil_to_tensor(image: Image.Image, *, device: torch.device) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).to(device=device)


def _similarity_h_pixels() -> np.ndarray:
    angle = np.deg2rad(7.0)
    scale = 0.93
    linear = scale * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    centre = np.array([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0])
    translation = np.array([13.0, 17.0]) * (IMAGE_SIZE / 512.0)
    offset = centre + translation - linear @ centre
    return np.array(
        [[linear[0, 0], linear[0, 1], offset[0]], [linear[1, 0], linear[1, 1], offset[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _pillow_inverse_affine(h: np.ndarray) -> tuple[float, float, float, float, float, float]:
    inverse = np.linalg.inv(h)
    linear, offset = inverse[:2, :2], inverse[:2, 2]
    index_offset = linear @ np.array([0.5, 0.5]) + offset - 0.5
    return (
        float(linear[0, 0]), float(linear[0, 1]), float(index_offset[0]),
        float(linear[1, 0]), float(linear[1, 1]), float(index_offset[1]),
    )


def _normalize_pixel_h(h: np.ndarray) -> np.ndarray:
    scale = np.diag([float(IMAGE_SIZE), float(IMAGE_SIZE), 1.0])
    return np.linalg.inv(scale) @ h @ scale


def _corners_from_h(h: np.ndarray) -> np.ndarray:
    canonical = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)))
    projected = (h @ canonical.T).T
    return projected[:, :2] / projected[:, 2:3]


def apply_pillow_attack(image: Tensor, label: str, *, device: torch.device | None = None) -> tuple[Tensor, np.ndarray, np.ndarray]:
    """Apply the frozen actual Pillow attack and return normalized H/corner truth."""

    output_device = device or image.device
    source = _tensor_to_pil(image)
    if label == "identity":
        attacked, h_pixels = source.copy(), np.eye(3, dtype=np.float64)
    elif label == "rotate90":
        attacked = source.transpose(Image.Transpose.ROTATE_90)
        h_pixels = np.array(((0.0, 1.0, 0.0), (-1.0, 0.0, float(IMAGE_SIZE)), (0.0, 0.0, 1.0)))
    elif label == "similarity":
        h_pixels = _similarity_h_pixels()
        attacked = source.transform(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Transform.AFFINE,
            _pillow_inverse_affine(h_pixels),
            resample=Image.Resampling.BICUBIC,
        )
    elif label == "crop_rescale":
        left, top, right, bottom = 8, 11, 119, 117
        attacked = source.crop((left, top, right, bottom)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
        )
        h_pixels = np.array(
            ((IMAGE_SIZE / (right - left), 0.0, -left * IMAGE_SIZE / (right - left)),
             (0.0, IMAGE_SIZE / (bottom - top), -top * IMAGE_SIZE / (bottom - top)),
             (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
    else:
        raise ValueError("unknown N0 attack")
    h_normalized = _normalize_pixel_h(h_pixels)
    return _pil_to_tensor(attacked, device=output_device), h_normalized, _corners_from_h(h_normalized)


def _estimate_h(corners: np.ndarray) -> np.ndarray:
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("predicted corners must be finite 4x2")
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), corners, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]); rhs.append(float(u))
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]); rhs.append(float(v))
    values = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    return np.array(((values[0], values[1], values[2]), (values[3], values[4], values[5]), (values[6], values[7], 1.0)))


def _percentile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("percentile requires finite values")
    return float(np.percentile(array, quantile, method="linear"))


@dataclass(frozen=True, slots=True)
class N0RunResult:
    summary: dict[str, Any]
    units: tuple[dict[str, Any], ...]


def decide_n0_status(units: Iterable[dict[str, Any]], *, actual_residual_max: float) -> tuple[str, dict[str, Any]]:
    records = tuple(units)
    if len(records) != CONFIRMATION_UNIT_COUNT:
        raise ValueError("N0 confirmation roster must contain exactly 128 units")
    calculated_records = tuple(record for record in records if record.get("status") == "calculated")
    failed = len(records) - len(calculated_records)
    errors = tuple(float(record["mean_corner_error"]) for record in calculated_records)
    if any(not math.isfinite(value) or value < 0.0 for value in errors):
        raise ValueError("N0 corner errors must be finite and nonnegative")
    reliable = sum(record.get("reliable") is True for record in calculated_records)
    median_error = statistics.median(errors) if errors else None
    p95_error = _percentile(errors, 95.0) if errors else None
    reliable_fraction = reliable / CONFIRMATION_UNIT_COUNT
    if failed:
        status = STATUS_STOPPED
    elif (
        median_error is not None and median_error < 0.05
        and p95_error is not None and p95_error < 0.10
        and reliable_fraction >= 0.75
        and math.isfinite(actual_residual_max)
        and actual_residual_max <= MAX_RESIDUAL + 1.0e-7
    ):
        status = STATUS_CANDIDATE
    else:
        status = STATUS_UNRESOLVED
    return status, {
        "declared_unit_count": CONFIRMATION_UNIT_COUNT,
        "calculated_unit_count": len(calculated_records),
        "failed_unit_count": failed,
        "median_corner_error": median_error,
        "p95_corner_error": p95_error,
        "reliable_fraction": reliable_fraction,
        "actual_residual_max": float(actual_residual_max),
    }


def _embed_one(embedder: KeyedResidualEmbedder, image: Tensor, code: Tensor) -> tuple[Tensor, float]:
    output = embedder(image.unsqueeze(0), code.unsqueeze(0))
    return output.image[0], float(output.residual.detach().abs().max().cpu())


def _evaluate_split(
    seeds: tuple[int, ...],
    geometry_key: bytes,
    embedder: KeyedResidualEmbedder,
    extractor: BlindCornerExtractor,
    device: torch.device,
    *,
    retain_units: bool,
) -> tuple[list[dict[str, Any]], list[float], float]:
    records: list[dict[str, Any]] = []
    errors: list[float] = []
    residual_max = 0.0
    embedder.eval(); extractor.eval()
    with torch.no_grad():
        for seed in seeds:
            clean = _procedural_rgb(seed).to(device)
            code = _target_code(geometry_key, seed, device=device)
            embedded, residual = _embed_one(embedder, clean, code)
            residual_max = max(residual_max, residual)
            for attack_label in ATTACKS:
                record: dict[str, Any] = {"seed": seed, "attack": attack_label, "status": "failed"}
                try:
                    attacked, truth_h, truth_corners = apply_pillow_attack(embedded, attack_label, device=device)
                    prediction = extractor(attacked.unsqueeze(0))
                    predicted = prediction.corners[0].detach().cpu().numpy().astype(np.float64)
                    predicted_h = _estimate_h(predicted)
                    GeometryEstimate(tuple(map(tuple, predicted)), tuple(map(tuple, predicted_h)))
                    error = float(np.linalg.norm(predicted - truth_corners, axis=1).mean())
                    if not math.isfinite(error):
                        raise ValueError("corner error is non-finite")
                    score = float(np.clip(1.0 - error / 0.25, 0.0, 1.0))
                    support = float(prediction.support[0].detach().cpu())
                    record.update(
                        status="calculated",
                        mean_corner_error=error,
                        reliability_score=score,
                        reliable=bool(score >= RELIABILITY_THRESHOLD and support >= MINIMUM_SUPPORT),
                        support=support,
                        extractor_confidence=float(prediction.confidence[0].detach().cpu()),
                        truth_h_finite=bool(np.isfinite(truth_h).all()),
                    )
                    errors.append(error)
                except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                    record["failure_class"] = "geometry_estimation_error"
                if retain_units:
                    records.append(record)
    return records, errors, residual_max


def _record_only_audits(
    geometry_key: bytes,
    embedder: KeyedResidualEmbedder,
    extractor: BlindCornerExtractor,
    device: torch.device,
) -> dict[str, Any]:
    alternate_key = hashlib.sha256(b"CEG-WM/geometry-v2/n0/key-separation\x00" + geometry_key).digest()
    key_distances: list[float] = []
    no_sync_errors: list[float] = []
    residual_mse: list[float] = []
    embedder.eval(); extractor.eval()
    with torch.no_grad():
        for seed in CONFIRMATION_SEEDS[:8]:
            clean = _procedural_rgb(seed).to(device)
            code = _target_code(geometry_key, seed, device=device)
            other = _target_code(alternate_key, seed, device=device)
            first = embedder(clean.unsqueeze(0), code.unsqueeze(0)).image
            second = embedder(clean.unsqueeze(0), other.unsqueeze(0)).image
            key_distances.append(float((first - second).abs().mean().cpu()))
            residual_mse.append(float((first[0] - clean).square().mean().cpu()))
            attacked, _, truth = apply_pillow_attack(clean, "similarity", device=device)
            predicted = extractor(attacked.unsqueeze(0)).corners[0].cpu().numpy()
            no_sync_errors.append(float(np.linalg.norm(predicted - truth, axis=1).mean()))
    mean_mse = statistics.fmean(residual_mse)
    return {
        "audit_seed_count": 8,
        "embedded_rgb_psnr_db": 10.0 * math.log10(1.0 / max(mean_mse, 1.0e-12)),
        "key_separation_mean_rgb_l1": statistics.fmean(key_distances),
        "no_sync_similarity_mean_corner_error": statistics.fmean(no_sync_errors),
        "gate_role": "record_only",
    }


def run_n0(geometry_key: bytes, *, device_name: str = "cpu") -> N0RunResult:
    """Train once, freeze reliability, and evaluate the independent confirmation split."""

    if not isinstance(geometry_key, bytes) or not 16 <= len(geometry_key) <= 4096:
        raise ValueError("geometry key must be 16..4096 bytes")
    if set(TRAIN_SEEDS) & set(VALIDATION_SEEDS) or set(TRAIN_SEEDS) & set(CONFIRMATION_SEEDS) or set(VALIDATION_SEEDS) & set(CONFIRMATION_SEEDS):
        raise RuntimeError("N0 splits overlap")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)
    embedder = KeyedResidualEmbedder().to(device)
    extractor = BlindCornerExtractor().to(device)
    optimizer = torch.optim.Adam((*embedder.parameters(), *extractor.parameters()), lr=LEARNING_RATE)
    loss_totals = {"total": 0.0, "corner": 0.0, "sync_reconstruction": 0.0, "residual_l2": 0.0}
    steps = 0
    train_residual_max = 0.0
    embedder.train(); extractor.train()
    for epoch in range(EPOCHS):
        for start in range(0, len(TRAIN_SEEDS), BATCH_SIZE):
            seeds = TRAIN_SEEDS[start:start + BATCH_SIZE]
            clean = torch.stack([_procedural_rgb(seed) for seed in seeds]).to(device)
            code = torch.stack([_target_code(geometry_key, seed, device=device) for seed in seeds])
            embedded = embedder(clean, code)
            attacked_items: list[Tensor] = []
            corner_items: list[Tensor] = []
            for index, (seed, image) in enumerate(zip(seeds, embedded.image, strict=True)):
                attack_label = ATTACKS[(seed - TRAIN_SEEDS[0] + epoch) % len(ATTACKS)]
                attacked, _, corners = apply_pillow_attack(image, attack_label, device=device)
                attacked_items.append(attacked)
                corner_items.append(torch.tensor(corners, dtype=torch.float32, device=device))
            prediction = extractor(torch.stack(attacked_items))
            loss, components = n0_joint_loss(prediction, torch.stack(corner_items), embedded, code)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_residual_max = max(train_residual_max, float(embedded.residual.detach().abs().max().cpu()))
            loss_totals["total"] += float(loss.detach().cpu())
            for name, value in components.items():
                loss_totals[name] += float(value.detach().cpu())
            steps += 1

    validation_records, validation_errors, validation_residual_max = _evaluate_split(
        VALIDATION_SEEDS, geometry_key, embedder, extractor, device, retain_units=False
    )
    del validation_records
    units, confirmation_errors, confirmation_residual_max = _evaluate_split(
        CONFIRMATION_SEEDS, geometry_key, embedder, extractor, device, retain_units=True
    )
    actual_residual_max = max(train_residual_max, validation_residual_max, confirmation_residual_max)
    status, confirmation = decide_n0_status(units, actual_residual_max=actual_residual_max)
    summary = {
        "protocol": PROTOCOL_IDENTITY,
        "n0_status": status,
        "science_denominator": 0,
        "training": {
            "seed": TRAINING_SEED, "seed_count": len(TRAIN_SEEDS), "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
            "attack_sampling": "four_class_equal_weight_fixed", "mean_losses": {key: value / steps for key, value in loss_totals.items()},
        },
        "validation": {
            "seed_count": len(VALIDATION_SEEDS), "observation_count": len(VALIDATION_SEEDS) * len(ATTACKS),
            "reliability_formula": "clamp(1-mean_corner_error/0.25,0,1)",
            "threshold": RELIABILITY_THRESHOLD, "minimum_support": MINIMUM_SUPPORT,
            "calculated_count": len(validation_errors),
        },
        "confirmation": confirmation,
        "record_only": _record_only_audits(geometry_key, embedder, extractor, device),
        "weights_persisted": False,
        "images_persisted": False,
        "raw_geometry_key_persisted": False,
        "geometry_authority": "coordinates_only",
    }
    return N0RunResult(summary=summary, units=tuple(units))


def package_n0(result: N0RunResult, output_root: Path, *, execution_exact: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("N0 output root is create-only")
    output_root.mkdir(parents=True, exist_ok=False)
    metrics = b"".join(_stable_json(record) + b"\n" for record in result.units)
    if len(metrics) > 524288:
        raise ValueError("N0 metrics artifact exceeds bound")
    receipt = dict(result.summary)
    receipt.update(
        run_id=f"geometry-v2-neural-corner-sync-n0-{execution_exact[:12]}",
        execution_identity={"commit": execution_exact},
        artifact_status="complete",
        metrics_sha256=hashlib.sha256(metrics).hexdigest(),
    )
    manifest = {
        "run_id": receipt["run_id"], "protocol": PROTOCOL_IDENTITY, "execution_exact": execution_exact,
        "n0_status": receipt["n0_status"], "science_denominator": 0,
        "declared_unit_count": CONFIRMATION_UNIT_COUNT, "files": ["metrics.jsonl", "receipt.json", "terminal.json"],
        "metrics_sha256": receipt["metrics_sha256"],
    }
    terminal = {
        "run_id": receipt["run_id"], "n0_status": receipt["n0_status"],
        "artifact_status": "complete", "science_denominator": 0,
        "execution_identity": {"commit": execution_exact},
    }
    files = {
        "metrics.jsonl": metrics,
        "receipt.json": _stable_json(receipt),
        "manifest.json": _stable_json(manifest),
        "terminal.json": _stable_json(terminal),
    }
    if any(len(value) > 262144 for name, value in files.items() if name != "metrics.jsonl"):
        raise ValueError("N0 root sidecar exceeds bound")
    for name, value in files.items():
        with (output_root / name).open("xb") as handle:
            handle.write(value)
    return {"artifact_status": "complete", "receipt_filename": "receipt.json", "manifest_filename": "manifest.json"}


__all__ = [
    "ATTACKS", "BATCH_SIZE", "CONFIRMATION_SEEDS", "CONFIRMATION_UNIT_COUNT", "EPOCHS",
    "LEARNING_RATE", "MINIMUM_SUPPORT", "N0RunResult", "RELIABILITY_THRESHOLD",
    "STATUS_CANDIDATE", "STATUS_STOPPED", "STATUS_UNRESOLVED", "TRAINING_SEED",
    "TRAIN_SEEDS", "VALIDATION_SEEDS", "apply_pillow_attack", "decide_n0_status", "package_n0", "run_n0",
]
