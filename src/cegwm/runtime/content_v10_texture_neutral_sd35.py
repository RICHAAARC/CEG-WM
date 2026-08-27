"""V10 callback helpers; generation remains delegated to the frozen V6 runtime."""
from __future__ import annotations
from typing import Any

def v10_allocation_from_signals(signals: Any) -> Any:
    # Keep the public fail-closed asset boundary importable without Torch.
    from cegwm.method.content_v10_texture_neutral import allocate_texture_neutral
    return allocate_texture_neutral(signals)

def require_v10_calibration_asset(asset: Any) -> Any:
    if asset is None:
        raise ValueError("Content V10 requires its own accepted calibration asset")
    return asset
