"""Shared development attacks; no model execution or formal threshold fitting.

H maps reference RGB integer pixel centres to observed RGB integer pixel centres.
Positive angle is clockwise on a screen (x right, y down). This convention is
explicitly separate from historical normalized observed-to-canonical matrices.
"""
from dataclasses import asdict, dataclass
import json
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Attack:
    name: str
    angle_deg: float = 0.0
    scale: float = 1.0
    noise_sigma: float = 0.0


CORE_ATTACKS = (
    Attack("clean"), Attack("rotation_plus10", 10),
    Attack("rotation_minus10", -10), Attack("scale_075", scale=.75),
    Attack("scale_125", scale=1.25), Attack("awgn_002", noise_sigma=.02),
    Attack("awgn_005", noise_sigma=.05),
    Attack("rotation_plus10_scale075", 10, .75),
    Attack("rotation_minus10_scale125", -10, 1.25),
)


def reference_to_observed(size, angle_deg=0., scale=1., translation=(0., 0.)):
    """Scale then rotate about pixel-centre canvas centre, then translate."""
    w, h = size
    if min(w, h) < 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("positive canvas and finite positive scale required")
    if not np.isfinite([angle_deg, *translation]).all():
        raise ValueError("finite angle/translation required")
    a = math.radians(angle_deg)
    linear = scale * np.array([[math.cos(a), -math.sin(a)],
                               [math.sin(a), math.cos(a)]])
    center = np.array([(w - 1) / 2, (h - 1) / 2])
    result = np.eye(3)
    result[:2, :2] = linear
    result[:2, 2] = center - linear @ center + translation
    return result


def _sample(image, output_to_input):
    # Pillow expresses coordinates at pixel edges: integer-centre coordinates
    # need a half-pixel conjugation, including when scale/rotation are combined.
    offset = np.eye(3)
    offset[:2, 2] = .5
    matrix = offset @ np.asarray(output_to_input) @ np.linalg.inv(offset)
    matrix = matrix / matrix[2, 2]
    return image.transform(image.size, Image.Transform.PERSPECTIVE,
                           tuple(matrix.ravel()[:8]),
                           resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))


def render_attack(image, attack, *, noise_seed=0):
    """Returns RGB and diagnostic truth H. Never pass truth to blind detection."""
    rgb = image.convert("RGB")
    H = reference_to_observed(rgb.size, attack.angle_deg, attack.scale)
    result = rgb.copy() if np.array_equal(H, np.eye(3)) else _sample(rgb, np.linalg.inv(H))
    if not math.isfinite(attack.noise_sigma) or attack.noise_sigma < 0:
        raise ValueError("noise sigma must be finite and nonnegative")
    if attack.noise_sigma:
        values = np.asarray(result, dtype=np.float64) / 255.
        noise = np.random.default_rng(noise_seed).normal(0., attack.noise_sigma, values.shape)
        result = Image.fromarray(np.rint(np.clip(values + noise, 0., 1.) * 255).astype(np.uint8))
    return result, H


def rectify_once(observed_rgb, reference_to_observed_H):
    """One RGB resampling; no geometry score or validity mask enters content."""
    return _sample(observed_rgb.convert("RGB"), reference_to_observed_H)


def latent_to_rgb_matrix(rgb_size, latent_size):
    """Pixel-centre mapping; H_latent = inv(S) @ H_rgb @ S."""
    ratios = np.asarray(rgb_size, dtype=float) / np.asarray(latent_size, dtype=float)
    result = np.diag([*ratios, 1.])
    result[:2, 2] = (ratios - 1.) / 2.
    return result


def protocol_summary():
    return {"user_suggested_envelope": {"fit_pairs": 8, "independent_validation_pairs": 24,
            "image_variants": ["unmarked", "v2", "A_geometry", "A_tolerance",
                               "B_simple", "B_survival"],
            "base_images": 192, "B_extra_continuations": 72,
            "suggested_total_images": 264},
            "actual_counts": "Use each executable entrypoint plan with its selected arguments; this envelope is not an execution plan",
            "core_attacks": [asdict(a) for a in CORE_ATTACKS],
            "geometry_direction": "reference_pixel_centres_to_observed_pixel_centres",
            "render": "inverse H; bilinear; black; fixed canvas; one sample",
            "noise": "independent RGB-channel Gaussian; RGB[0,1]; clip then round uint8",
            "claim": "development only; 24 validation negatives cannot support 0.1% FPR"}


if __name__ == "__main__":
    print(json.dumps(protocol_summary(), indent=2))
