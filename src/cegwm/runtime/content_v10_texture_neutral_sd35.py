"""V10's explicit internal production path over the unchanged V6 pair mechanics."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, math
from typing import Any
from cegwm.method.content_weighted_joint_v9 import LFHFScorePair
from cegwm.method.content_whitening_v4 import score_content_v4_lf_image
from cegwm.method.hf import score_hf_image
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key
from cegwm.shared.prg import prg_bytes
from cegwm.runtime.content_iss_sd35_v6 import ContentV6EvaluationAssets, _generator, _run_content_v6_pass2, content_v6_h, iss_beta, require_ordinary_rgb_image, run_sd35_plain

@dataclass(frozen=True)
class ContentV10RunOutput:
    image: Any
    primary_null: Any
    measurement: Any
    texture_summary: Any

@dataclass(frozen=True)
class ContentV10PairedEvaluationOutput:
    v9_image: Any; v10_image: Any; primary_null: Any
    v9_measurement: Any; v10_measurement: Any
    plain_rgb_sha256: str; texture_scalar: float; texture_raw_digest: str

class ContentV10PairedArmFailure(RuntimeError):
    def __init__(self, arm: str):
        self.arm=arm; super().__init__(f"Content V10 paired {arm} arm failed")

V10_CALIBRATION_WRONG_KEY_DOMAIN = "stage-a/content-adaptive-v2-external-wrong-key/v1"

def derive_v10_calibration_wrong_keys(calibration_key: bytes) -> tuple[bytes, ...]:
    key = normalize_detection_key(calibration_key)
    return tuple(prg_bytes(key, f"{V10_CALIBRATION_WRONG_KEY_DOMAIN}/index={index}", 32) for index in range(16))

def _blind_calibration_pair(image: Any, key: bytes, assets: ContentV6EvaluationAssets) -> LFHFScorePair:
    ordinary = require_ordinary_rgb_image(image)
    lf = float(score_content_v4_lf_image(ordinary, key, assets.lf_public_assets))
    hf = float(score_hf_image(ordinary, key, assets.hf_public_assets))
    if not all(math.isfinite(score) and -1.0 <= score <= 1.0 for score in (lf, hf)):
        raise ValueError("Content V10 calibration branch scores must be finite in [-1, 1]")
    return LFHFScorePair(lf, hf)

def require_v10_calibration_asset(asset: Any) -> Any:
    from cegwm.method.content_v10_texture_neutral import V10CalibrationAsset
    if not isinstance(asset, V10CalibrationAsset):
        raise ValueError("Content V10 requires its own accepted calibration asset")
    return asset

def run_content_v10_evaluation_pair(pipeline: Any, prompt: str, detection_key: Any, assets: Any, *, height: int, width: int, seed: int) -> ContentV10RunOutput:
    """Keep V6 primary-null/ISS behavior but route pass-2 through V10 at step 18."""
    from cegwm.method.content_v10_texture_neutral import allocate_texture_neutral
    if not isinstance(assets, ContentV6EvaluationAssets):
        raise TypeError("Content V10 evaluation pair requires frozen evaluation assets")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Content V10 evaluation seed must be a nonnegative integer")
    primary_null=require_ordinary_rgb_image(run_sd35_plain(pipeline,prompt,height=height,width=width,generator=_generator(seed)))
    beta=iss_beta(content_v6_h(primary_null,detection_key,assets.lf_public_assets),assets.iss_asset)
    summary=[]
    def allocator(signals: Any) -> Any:
        value=allocate_texture_neutral(signals); summary.append(value.texture_summary); return value.allocation
    image, measurement=_run_content_v6_pass2(pipeline,prompt,detection_key,assets,beta,height=height,width=width,generator=_generator(seed),allocation_factory=allocator)
    if len(summary)!=1: raise RuntimeError("Content V10 step-18 allocation did not occur exactly once")
    return ContentV10RunOutput(image,primary_null,measurement,summary[0])

def run_content_v10_paired_evaluation(pipeline: Any, prompt: str, detection_key: Any, assets: Any, *, height: int, width: int, seed: int) -> ContentV10PairedEvaluationOutput:
    """One common plain/beta, then default V3 and V10-private step-18 passes."""
    from cegwm.method.content_adaptive_v3 import allocate_content
    from cegwm.method.content_v10_texture_neutral import allocate_texture_neutral
    if not isinstance(assets, ContentV6EvaluationAssets): raise TypeError("Content V10 paired evaluation requires frozen V6 assets")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0: raise ValueError("Content V10 paired evaluation seed must be a nonnegative integer")
    primary_null=require_ordinary_rgb_image(run_sd35_plain(pipeline,prompt,height=height,width=width,generator=_generator(seed)))
    beta=iss_beta(content_v6_h(primary_null,detection_key,assets.lf_public_assets),assets.iss_asset)
    try:
        v9_image,v9_measurement=_run_content_v6_pass2(pipeline,prompt,detection_key,assets,beta,height=height,width=width,generator=_generator(seed),allocation_factory=allocate_content)
    except BaseException as error:
        raise ContentV10PairedArmFailure("v9") from error
    summaries=[]
    def v10_allocator(signals: Any) -> Any:
        value=allocate_texture_neutral(signals); summaries.append(value.texture_summary); return value.allocation
    try:
        v10_image,v10_measurement=_run_content_v6_pass2(pipeline,prompt,detection_key,assets,beta,height=height,width=width,generator=_generator(seed),allocation_factory=v10_allocator)
    except BaseException as error:
        raise ContentV10PairedArmFailure("v10") from error
    if len(summaries)!=1: raise RuntimeError("Content V10 paired step-18 allocation did not occur exactly once")
    if not callable(getattr(primary_null,"tobytes",None)): raise TypeError("Content V10 paired primary null must be ordinary RGB")
    summary=summaries[0]
    return ContentV10PairedEvaluationOutput(v9_image,v10_image,primary_null,v9_measurement,v10_measurement,hashlib.sha256(primary_null.tobytes()).hexdigest(),float(summary.mean),summary.digest)

def run_content_v10_calibration_unit(pipeline: Any, unit: Any, calibration_key: bytes, assets: ContentV6EvaluationAssets) -> tuple[LFHFScorePair, ...]:
    """Real V10 step-18 pair followed by the frozen 33 calibration observations."""
    required = ("prompt", "height", "width", "seed")
    if any(not hasattr(unit, field) for field in required) or not isinstance(assets, ContentV6EvaluationAssets):
        raise TypeError("Content V10 calibration requires validated unit and V6 assets")
    wrong_keys = derive_v10_calibration_wrong_keys(calibration_key)
    if len(wrong_keys) != 16 or len(set(wrong_keys)) != 16:
        raise RuntimeError("Content V10 calibration requires exactly 16 ordered wrong keys")
    output = run_content_v10_evaluation_pair(pipeline, unit.prompt, calibration_key, assets, height=unit.height, width=unit.width, seed=unit.seed)
    if not isinstance(output, ContentV10RunOutput):
        raise TypeError("Content V10 calibration requires real V10 evaluation output")
    pairs = tuple(
        [*(_blind_calibration_pair(output.image, key, assets) for key in wrong_keys),
         _blind_calibration_pair(output.primary_null, calibration_key, assets),
         *(_blind_calibration_pair(output.primary_null, key, assets) for key in wrong_keys)]
    )
    if len(pairs) != 33:
        raise RuntimeError("Content V10 calibration unit must yield exactly 33 pairs")
    return pairs
