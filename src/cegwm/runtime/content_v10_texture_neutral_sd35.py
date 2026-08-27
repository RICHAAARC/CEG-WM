"""V10's explicit internal production path over the unchanged V6 pair mechanics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from cegwm.runtime.content_iss_sd35_v6 import ContentV6RunOutput, _generator, _run_content_v6_pass2, content_v6_h, iss_beta, require_ordinary_rgb_image, run_sd35_plain

@dataclass(frozen=True)
class ContentV10RunOutput:
    image: Any
    primary_null: Any
    measurement: Any
    texture_summary: Any

def require_v10_calibration_asset(asset: Any) -> Any:
    from cegwm.method.content_v10_texture_neutral import V10CalibrationAsset
    if not isinstance(asset, V10CalibrationAsset):
        raise ValueError("Content V10 requires its own accepted calibration asset")
    return asset

def run_content_v10_evaluation_pair(pipeline: Any, prompt: str, detection_key: Any, assets: Any, *, height: int, width: int, seed: int) -> ContentV10RunOutput:
    """Keep V6 primary-null/ISS behavior but route pass-2 through V10 at step 18."""
    from cegwm.method.content_v10_texture_neutral import allocate_texture_neutral
    primary_null=require_ordinary_rgb_image(run_sd35_plain(pipeline,prompt,height=height,width=width,generator=_generator(seed)))
    beta=iss_beta(content_v6_h(primary_null,detection_key,assets.lf_public_assets),assets.iss_asset)
    summary=[]
    def allocator(signals: Any) -> Any:
        value=allocate_texture_neutral(signals); summary.append(value.texture_summary); return value.allocation
    image, measurement=_run_content_v6_pass2(pipeline,prompt,detection_key,assets,beta,height=height,width=width,generator=_generator(seed),allocation_factory=allocator)
    if len(summary)!=1: raise RuntimeError("Content V10 step-18 allocation did not occur exactly once")
    return ContentV10RunOutput(image,primary_null,measurement,summary[0])
