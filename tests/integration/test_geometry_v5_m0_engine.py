from __future__ import annotations

from pathlib import Path

import pytest

from cegwm.protocol.geometry_v5_m0 import GeometryV5M0RawRecord
from cegwm.runtime.geometry_v5_m0_sd21 import (
    public_runtime_capabilities,
    recover_and_estimate_from_attacked_rgb,
    run_generation_with_initial_z_t,
)


@pytest.mark.integration
def test_injected_fake_adapters_exercise_only_fixed_boundaries_not_real_mechanism() -> None:
    generated: dict[str, object] = {}

    def fake_generator(**kwargs: object) -> str:
        generated.update(kwargs)
        return "fake-final-rgb"

    assert run_generation_with_initial_z_t(fake_generator, "fake-z", "manifest prompt") == "fake-final-rgb"
    assert generated["prompt"] == "manifest prompt" and generated["num_inference_steps"] == 50

    def fake_inverter(image: object, **kwargs: object) -> str:
        assert image == "attacked-rgb" and kwargs["prompt"] == ""
        return "fake-recovered-z"

    def fake_estimator(recovered: object) -> GeometryV5M0RawRecord:
        assert recovered == "fake-recovered-z"
        return GeometryV5M0RawRecord("FAILED", None, None, None, None, None, {})

    raw = recover_and_estimate_from_attacked_rgb("attacked-rgb", fake_inverter, fake_estimator)
    assert raw.status.value == "FAILED"
    capabilities = public_runtime_capabilities()
    assert capabilities["real_model_adapter_bound"] is False
    assert capabilities["fake_injected_adapter_is_real_evidence"] is False


@pytest.mark.integration
def test_real_combined_entry_uses_concrete_empty_prompt_inversion_and_blind_estimator() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/cegwm/runtime/geometry_v5_m0_sd21.py").read_text(encoding="utf-8")
    assert "def invert_bound_sd21_attacked_rgb" in source
    assert "prompt=\"\", guidance_scale=1.0" in source
    assert "def estimate_bound_blind_rst" in source
    assert "def recover_and_estimate_bound_sd21" in source
    assert "return GeometryV5M0RawRecord(\"FAILED\"" in source
