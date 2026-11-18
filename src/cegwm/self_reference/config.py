"""Versioned pilot constants; changes require a new protocol identity."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

DCT_ORDER = tuple((u, v) for u in range(4) for v in range(4) if (u, v) != (0, 0))
SHIFTS = ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2))


@dataclass(frozen=True)
class Config:
    version: str = "self-reference-relation-v1"
    bits: int = 32
    width: int = 64
    latent_channels: int = 16
    rgb_size: int = 512
    downsample: int = 8
    mask_border: int = 8
    conv_layers: int = 3
    residual_cap: float = 0.05
    epsilon: float = 1e-8
    updates: int = 10000
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20261118
    reference_weight: float = 0.1
    rgb_weight: float = 10.0
    residual_weight: float = 0.01
    checkpoint_every: int = 1000
    bootstrap_samples: int = 2000
    a_top1_min: float = 0.95
    a_max_baseline_gap: float = 0.02
    b_gain_target: float = 0.10
    psnr_min: float = 30.0
    lpips_max: float = 0.05
    psnr_gap_min: float = -0.5
    lpips_gap_max: float = 0.005
    dct_order: tuple = DCT_ORDER
    attacks_dx_dy: tuple = SHIFTS

    @property
    def latent_size(self):
        return self.rgb_size // self.downsample

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


ARMS = ("B0", "B1", "C")
SPLITS = ("train", "dev", "model_selection", "calibration", "test")


def load_config(path=None):
    config = Config()
    if path is not None and json.loads(Path(path).read_text()) != json.loads(json.dumps(asdict(config))):
        raise ValueError("config differs from frozen v1; create a new protocol, do not tune v1")
    return config
