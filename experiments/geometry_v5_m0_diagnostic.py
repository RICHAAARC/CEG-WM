"""Small, nonformal Geometry-V5 M0 Colab diagnostic (four cases only)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from cegwm.protocol.geometry_v5_m0 import GeometryV5M0RawRecord, load_geometry_v5_m0_contract


_DIAGNOSTIC_ID = "geometry_v5_m0_sd21_small_canary_v1"
_METHOD_SOURCE_EXACT = "fb48ad94a1aedd8b30adbc90918cdebe4973b20b"
_CASE_IDS = ("identity", "rotation_+10", "scale_1.1", "translation_x_+0.08")


@dataclass(frozen=True)
class _Bindings:
    """Private test seam; CLI construction is always concrete and lazy."""

    load_pipeline: Callable[[], Any]
    initial_z_t: Callable[[Any, int], Any]
    generate: Callable[[Any, str, Any], Any]
    attack: Callable[[Any, Mapping[str, Any]], Any]
    detect: Callable[[Any, Any], GeometryV5M0RawRecord]
    method_preflight: Callable[[Path], tuple[list[dict[str, Any]], bool]] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_json.exists():
        print(_summary({"diagnostic_id": _DIAGNOSTIC_ID, "status": "output_exists", "error_class": "FileExistsError"}))
        return 2
    try:
        result, complete = run_diagnostic(args.repo_root, args.output_json, _concrete_bindings())
    except Exception as error:
        print(_summary({"diagnostic_id": _DIAGNOSTIC_ID, "status": "setup_failed", "error_class": type(error).__name__}))
        return 2
    print(_summary({"diagnostic_id": _DIAGNOSTIC_ID, "status": "complete" if complete else "incomplete", "cases": len(result["cases"]), "science_denominator": 0}))
    return 0 if complete else 1


def _concrete_bindings() -> _Bindings:
    """Bind the three existing concrete runtime functions after CLI entry."""
    torch = __import__("torch")
    from PIL import Image
    from cegwm.runtime.geometry_v5_m0_sd21 import SD21M0Identity, generate_bound_sd21, load_bound_sd21_pipeline, recover_and_estimate_bound_sd21

    identity = SD21M0Identity()

    def initial_z_t(pipeline: Any, seed: int) -> Any:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        return torch.randn((1, 4, 64, 64), generator=generator, device=pipeline.device, dtype=pipeline.unet.dtype)

    def generate(pipeline: Any, prompt: str, initial: Any) -> Any:
        return generate_bound_sd21(pipeline, prompt, initial, identity)

    def attack(final_rgb: Any, case: Mapping[str, Any]) -> Any:
        return _apply_forward_attack_pil(final_rgb, case, Image)

    def detect(pipeline: Any, attacked_rgb: Any) -> GeometryV5M0RawRecord:
        return recover_and_estimate_bound_sd21(pipeline, attacked_rgb, identity)

    return _Bindings(load_bound_sd21_pipeline, initial_z_t, generate, attack, detect)


def run_diagnostic(repo_root: Path, output_json: Path, bindings: _Bindings, event_sink: Callable[[str], None] | None = None) -> tuple[dict[str, Any], bool]:
    """Run one seed/generation and four independent attacked-RGB diagnostics."""
    if output_json.exists():
        raise FileExistsError("diagnostic output already exists")
    contract = load_geometry_v5_m0_contract(repo_root)
    unit = contract.units[0]
    cases = tuple(case for case in contract.config["development"]["attacks"] if case["attack_id"] in _CASE_IDS)
    if unit.seed != 7501 or tuple(case["attack_id"] for case in cases) != _CASE_IDS:
        raise RuntimeError("diagnostic roster differs")
    from cegwm.runtime.geometry_v5_m0_sd21 import SD21M0Identity
    identity = SD21M0Identity()
    records: list[dict[str, Any]] = []
    try:
        preflight, preflight_passed = (bindings.method_preflight or _run_method_preflight)(repo_root)
    except Exception as error:
        preflight, preflight_passed = [_preflight_failure("method_preflight_setup", None, error)], False
    if not preflight_passed:
        records = [_preflight_blocked_case(case["attack_id"]) for case in cases]
        return _write_result(output_json, unit, identity, records, preflight)
    try:
        pipeline = bindings.load_pipeline()
    except Exception as error:
        records = [_failed_case(case["attack_id"], "model_load", error, event_sink) for case in cases]
        return _write_result(output_json, unit, identity, records, preflight)
    try:
        initial_z_t = bindings.initial_z_t(pipeline, unit.seed)
        final_rgb = _extract_single_rgb_image(bindings.generate(pipeline, unit.prompt, initial_z_t))
    except Exception as error:
        records = [_failed_case(case["attack_id"], "generation", error, event_sink) for case in cases]
        return _write_result(output_json, unit, identity, records, preflight)
    for case in cases:
        try:
            attacked_rgb = bindings.attack(final_rgb, case)
        except Exception as error:
            records.append(_failed_case(case["attack_id"], "attack", error, event_sink))
            continue
        try:
            # The concrete detector sees only frozen pipeline closure plus attacked RGB.
            raw = bindings.detect(pipeline, attacked_rgb)
            if not isinstance(raw, GeometryV5M0RawRecord):
                raise TypeError("detector must return GeometryV5M0RawRecord")
            records.append(_record_after_raw_freeze(case, raw, event_sink))
        except Exception as error:
            records.append(_failed_case(case["attack_id"], "detector", error, event_sink))
    return _write_result(output_json, unit, identity, records, preflight)


def _run_method_preflight(repo_root: Path) -> tuple[list[dict[str, Any]], bool]:
    """Exercise only the deterministic initial-zT method before model load.

    These are engineering preflights, not image evidence: each result is kept
    under ``method_preflight`` and the science denominator remains zero.
    """

    method = _load_method_module(repo_root)
    records: list[dict[str, Any]] = []
    try:
        size = 16
        latent = tuple(
            tuple(tuple(math.sin((channel + 1) * (row + 1) * (column + 2)) for column in range(size)) for row in range(size))
            for channel in range(4)
        )
        injected = method.inject_initial_z_t_x_template(latent, method.build_hermitian_x_template())
        estimate = method.estimate_rotation_scale_from_recovered_z_t(
            injected, ((0.0, 1.0), (10.0, 1.0 / 1.1), (-10.0, 1.0)),
        )
        if estimate.rotation_degrees != 0.0 or estimate.scale != 1.0:
            raise ValueError("direct writer detector identity closure differs")
        records.append(_preflight_success("direct_initial_z_t_identity", "identity", estimate, 0.0, 0.0, {"latent_side": size}))
    except Exception as error:
        records.append(_preflight_failure("direct_initial_z_t_identity", "identity", error))
    for case_id, spectral_rotation, spectral_scale, expected_rotation, expected_scale in (
        ("rotation_+10", 10.0, 1.0, -10.0, 1.0),
        ("scale_1.1", 0.0, 1.0 / 1.1, 0.0, 1.0 / 1.1),
    ):
        try:
            recovered = _known_spectral_latent(method, 32, spectral_rotation, spectral_scale)
            estimate = method.estimate_rotation_scale_from_recovered_z_t(
                recovered, ((0.0, 1.0), (spectral_rotation, spectral_scale), (spectral_rotation - 2.0, spectral_scale)),
            )
            if estimate.rotation_degrees != expected_rotation or estimate.scale != expected_scale:
                raise ValueError("known latent attacked_to_canonical R/S differs")
            records.append(_preflight_success("known_latent_rst", case_id, estimate, 0.0, 0.0, {"latent_side": 32}))
        except Exception as error:
            records.append(_preflight_failure("known_latent_rst", case_id, error))
    try:
        side, requested_tx, grid_shift = 64, 0.08, 5
        spectrum = [[0j for _ in range(side)] for _ in range(side)]
        for y, x in method._template_support(method.build_hermitian_x_template(), side, side):
            spectrum[y][x] = 1.0 + 0j
        canonical = tuple(tuple(value.real for value in row) for row in method._idft2(spectrum))
        observed = tuple(
            tuple(canonical[row][(column + grid_shift) % side] for column in range(side))
            for row in range(side)
        )
        tx, ty = method.estimate_translation_phase_correlation(canonical, observed)
        if abs(tx - requested_tx) > 1.0 / side or ty != 0.0:
            raise ValueError("known latent attacked_to_canonical T differs")
        H = method.assemble_attacked_to_canonical_similarity(0.0, 1.0, tx, ty)
        records.append({
            "stage": "known_latent_rst", "case_id": "translation_x_+0.08", "status": "METHOD_PREFLIGHT_PASSED",
            "raw_estimates": {"rotation_degrees": 0.0, "scale": 1.0, "tx": tx, "ty": ty, "H_hat": H},
            "diagnostics": {"requested_tx": requested_tx, "translation_grid_resolution": 1.0 / side},
        })
    except Exception as error:
        records.append(_preflight_failure("known_latent_rst", "translation_x_+0.08", error))
    return records, all(record["status"] == "METHOD_PREFLIGHT_PASSED" for record in records)


def _load_method_module(repo_root: Path) -> Any:
    """Load the isolated pure method file without importing optional torch code."""

    source = repo_root / "src" / "cegwm" / "method" / "geometry_v5_m0.py"
    spec = importlib.util.spec_from_file_location("_geometry_v5_m0_preflight", source)
    if spec is None or spec.loader is None:
        raise ImportError("method preflight source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _known_spectral_latent(method: Any, side: int, rotation_degrees: float, scale: float) -> tuple[tuple[tuple[float, ...], ...], ...]:
    spectrum = [[0j for _ in range(side)] for _ in range(side)]
    angle = math.radians(rotation_degrees)
    for point in method.build_hermitian_x_template():
        x = scale * (math.cos(angle) * point.frequency_x - math.sin(angle) * point.frequency_y)
        y = scale * (math.sin(angle) * point.frequency_x + math.cos(angle) * point.frequency_y)
        if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
            spectrum[method._frequency_bin(y, side)][method._frequency_bin(x, side)] = 100.0 + 0j
    plane = tuple(tuple(value.real for value in row) for row in method._idft2(spectrum))
    return tuple(plane if channel == 3 else tuple(tuple(0.0 for _ in range(side)) for _ in range(side)) for channel in range(4))


def _preflight_success(stage: str, case_id: str, estimate: Any, tx: float, ty: float, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    from_method = estimate.diagnostics
    H = _load_h_from_estimate(estimate, tx, ty)
    return {
        "stage": stage, "case_id": case_id, "status": "METHOD_PREFLIGHT_PASSED",
        "raw_estimates": {"rotation_degrees": estimate.rotation_degrees, "scale": estimate.scale, "tx": tx, "ty": ty, "H_hat": H},
        "diagnostics": {**dict(from_method), **dict(diagnostics)},
    }


def _load_h_from_estimate(estimate: Any, tx: float, ty: float) -> tuple[tuple[float, float, float], ...]:
    rotation = math.radians(float(estimate.rotation_degrees))
    scale = float(estimate.scale)
    cosine, sine = scale * math.cos(rotation), scale * math.sin(rotation)
    return ((cosine, -sine, tx), (sine, cosine, ty), (0.0, 0.0, 1.0))


def _preflight_failure(stage: str, case_id: str | None, error: Exception) -> dict[str, Any]:
    return {
        "stage": stage, "case_id": case_id, "status": "METHOD_PREFLIGHT_FAILED",
        "raw_estimates": None, "diagnostics": {"error_class": type(error).__name__},
    }


def _preflight_blocked_case(attack_id: str) -> dict[str, Any]:
    """Retain the four final cases without inventing a detector raw result."""

    return {
        "attack_id": attack_id, "raw": None, "truth_errors": _failed_truth_errors(),
        "failure_stage": "method_preflight", "error_class": "MethodPreflightFailed",
    }


def _extract_single_rgb_image(generation_output: Any) -> Any:
    """Strictly unwrap the one ordinary RGB image from a diffusers output."""
    images = getattr(generation_output, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise ValueError("generation output must contain exactly one image")
    image = images[0]
    if getattr(image, "mode", None) != "RGB" or getattr(image, "size", None) != (512, 512):
        raise ValueError("generation output image must be ordinary 512x512 RGB")
    return image


def _apply_forward_attack_pil(final_rgb: Any, case: Mapping[str, Any], image_module: Any) -> Any:
    """Apply forward A=sR,t with PIL's output-to-input A^-1 sampling map."""
    if getattr(final_rgb, "mode", None) != "RGB" or getattr(final_rgb, "size", None) != (512, 512):
        raise ValueError("generation must return ordinary 512x512 RGB")
    truth = _truth_h(case)
    b00, b01, u0 = truth[0]; b10, b11, u1 = truth[1]; center = 255.5
    coefficients = (b00, b01, center - b00 * center - b01 * center + 512 * u0, b10, b11, center - b10 * center - b11 * center + 512 * u1)
    return final_rgb.transform((512, 512), image_module.Transform.AFFINE, coefficients, resample=image_module.Resampling.BILINEAR, fillcolor=(0, 0, 0))


def _truth_h(case: Mapping[str, Any]) -> tuple[tuple[float, float, float], ...]:
    scale, theta = float(case["scale"]), math.radians(float(case["rotation_degrees"]))
    tx, ty = float(case["tx"]), float(case["ty"])
    if not math.isfinite(scale) or scale <= 0.0 or not all(math.isfinite(value) for value in (theta, tx, ty)):
        raise ValueError("diagnostic attack parameters differ")
    cosine, sine = math.cos(theta) / scale, math.sin(theta) / scale
    b00, b01, b10, b11 = cosine, sine, -sine, cosine
    return ((b00, b01, -(b00 * tx + b01 * ty)), (b10, b11, -(b10 * tx + b11 * ty)), (0.0, 0.0, 1.0))


def _record_after_raw_freeze(case: Mapping[str, Any], raw: GeometryV5M0RawRecord, event_sink: Callable[[str], None] | None) -> dict[str, Any]:
    raw_bytes = _canonical_json(_raw_payload(raw))
    frozen_raw = json.loads(raw_bytes)
    if event_sink is not None:
        event_sink("raw_frozen")
    truth_errors = _diagnostic_truth_errors(raw, _truth_h(case))
    if event_sink is not None:
        event_sink("truth_evaluated")
    return {"attack_id": case["attack_id"], "raw": frozen_raw, "truth_errors": truth_errors, "failure_stage": None, "error_class": None}


def _failed_case(attack_id: str, stage: str, error: Exception, event_sink: Callable[[str], None] | None) -> dict[str, Any]:
    raw = GeometryV5M0RawRecord("FAILED", None, None, None, None, None, {})
    raw_bytes = _canonical_json(_raw_payload(raw))
    if event_sink is not None:
        event_sink("raw_frozen")
    return {"attack_id": attack_id, "raw": json.loads(raw_bytes), "truth_errors": _failed_truth_errors(), "failure_stage": stage, "error_class": type(error).__name__}


def _raw_payload(raw: GeometryV5M0RawRecord) -> dict[str, Any]:
    return {"status": raw.status.value, "rotation_degrees": raw.rotation_degrees, "scale": raw.scale, "tx": raw.tx, "ty": raw.ty, "H_hat": raw.H_hat, "diagnostics": dict(raw.diagnostics)}


def _diagnostic_truth_errors(raw: GeometryV5M0RawRecord, truth: tuple[tuple[float, float, float], ...]) -> dict[str, Any]:
    if raw.status.value == "FAILED":
        return _failed_truth_errors()
    assert raw.rotation_degrees is not None and raw.scale is not None and raw.tx is not None and raw.ty is not None and raw.H_hat is not None
    rotation = math.degrees(math.atan2(truth[1][0], truth[0][0]))
    scale = math.hypot(truth[0][0], truth[1][0])
    errors = {"rotation_abs_degrees": abs(_wrapped_degrees(raw.rotation_degrees - rotation)), "log_scale_abs": abs(math.log(raw.scale) - math.log(scale)), "tx_abs": abs(raw.tx - truth[0][2]), "ty_abs": abs(raw.ty - truth[1][2]), "corner_rms": _corner_rms(raw.H_hat, truth)}
    return {"diagnostic_only_not_a_gate": True, **errors, "within_existing_m0_tolerances_diagnostic_only_not_a_gate": {"rotation": errors["rotation_abs_degrees"] <= 2.5, "log_scale": errors["log_scale_abs"] <= 0.04, "tx": errors["tx_abs"] <= 0.025, "ty": errors["ty_abs"] <= 0.025, "corner_rms": errors["corner_rms"] <= 0.03}}


def _failed_truth_errors() -> dict[str, Any]:
    return {"diagnostic_only_not_a_gate": True, "rotation_abs_degrees": None, "log_scale_abs": None, "tx_abs": None, "ty_abs": None, "corner_rms": None, "within_existing_m0_tolerances_diagnostic_only_not_a_gate": None}


def _wrapped_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _corner_rms(estimated: tuple[tuple[float, float, float], ...], truth: tuple[tuple[float, float, float], ...]) -> float:
    squared = 0.0
    for x, y in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        ex, ey = estimated[0][0] * x + estimated[0][1] * y + estimated[0][2], estimated[1][0] * x + estimated[1][1] * y + estimated[1][2]
        tx, ty = truth[0][0] * x + truth[0][1] * y + truth[0][2], truth[1][0] * x + truth[1][1] * y + truth[1][2]
        squared += (ex - tx) ** 2 + (ey - ty) ** 2
    return math.sqrt(squared / 4.0)


def _write_result(output_json: Path, unit: Any, identity: Any, cases: list[dict[str, Any]], preflight: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    result = {"diagnostic_id": _DIAGNOSTIC_ID, "method_source_exact": _METHOD_SOURCE_EXACT, "model": {"id": identity.model_family, "revision": identity.model_revision}, "seed": unit.seed, "unit_id": unit.unit_id, "method_preflight": preflight, "cases": cases, "diagnostic_denominator": 4, "science_denominator": 0, "claim": "nonformal_colab_method_diagnostic_only"}
    if len(cases) != 4:
        raise RuntimeError("diagnostic retention differs")
    _exclusive_json(output_json, result)
    return result, all(case["raw"] is not None and case["raw"]["status"] == "ESTIMATE_AVAILABLE" for case in cases)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_json(value))


def _summary(value: Mapping[str, Any]) -> str:
    return _canonical_json(value).decode("utf-8").strip()


if __name__ == "__main__":
    raise SystemExit(main())
