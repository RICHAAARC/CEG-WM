from __future__ import annotations

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

    assert run_generation_with_initial_z_t(fake_generator, "fake-z") == "fake-final-rgb"
    assert generated["prompt"] == "" and generated["num_inference_steps"] == 50

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
