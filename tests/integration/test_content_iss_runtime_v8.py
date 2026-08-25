from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cegwm.method.content_iss_v8 import (
    ISSDevelopmentMeasurement,
    build_iss_asset,
    fit_iss_gain_target,
)
from cegwm.protocol.content_chain_v8 import ContentV8Unit
from cegwm.runtime import content_iss_sd35_v8 as runtime
from cegwm.runtime.content_adaptive_sd35_v2 import ContentEmbedAssets


def _assets() -> ContentEmbedAssets:
    value = object.__new__(ContentEmbedAssets)
    object.__setattr__(value, "hf_public_assets", object())
    object.__setattr__(value, "lf_public_assets", object())
    return value


def _unit(split: str = "content_v6_iss_development_v1") -> ContentV8Unit:
    return ContentV8Unit("u", split, "s", "prompt", 17, 512, 512)


def _asset():
    fit = fit_iss_gain_target(
        ISSDevelopmentMeasurement(-0.2, 0.1, 0.2 + index / 1000)
        for index in range(32)
    )
    return build_iss_asset("1" * 40, "a" * 64, b"d" * 32, fit)


@pytest.mark.integration
def test_single_parametric_control_is_plain_then_write_with_host_selected_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    generators: list[object] = []

    def generator(seed: int) -> object:
        value = object()
        generators.append(value)
        calls.append(("generator", seed, value))
        return value

    monkeypatch.setattr(runtime, "_generator", generator)
    monkeypatch.setattr(
        runtime,
        "run_sd35_plain",
        lambda pipeline, prompt, **kwargs: calls.append(("plain", kwargs["generator"]))
        or Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB"),
    )
    monkeypatch.setattr(
        runtime,
        "content_v8_h",
        lambda image, key, assets: calls.append(("h", image, key)) or 0.2,
    )
    monkeypatch.setattr(
        runtime,
        "_run_write",
        lambda pipeline, unit, key, assets, beta: calls.append(("write", beta))
        or ("write-image", "measurement"),
    )
    output, host = runtime._run_two_pass(
        object(), _unit(), b"d" * 32, _assets(), lambda h: h + 1.0
    )
    assert [item[0] for item in calls] == ["generator", "plain", "h", "write"]
    assert calls[0][1] == 17
    assert calls[-1] == ("write", 1.2)
    assert host == 0.2 and output.primary_null.mode == "RGB"


@pytest.mark.integration
def test_development_pair_uses_beta_one_and_exactly_16_wrong_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[float] = []

    def two_pass(pipeline, unit, key, assets, controller):
        observed.append(controller(-0.4))
        return SimpleNamespace(image="write"), -0.4

    monkeypatch.setattr(runtime, "_run_two_pass", two_pass)
    monkeypatch.setattr(
        runtime, "derive_wrong_keys",
        lambda key: tuple(bytes([index + 1]) * 32 for index in range(16)),
    )
    monkeypatch.setattr(
        runtime, "content_v8_h",
        lambda image, key, assets: 0.3 if key == b"d" * 32 else 0.2,
    )
    measurement = runtime.run_content_v8_development_pair(
        object(), _unit(), b"d" * 32, _assets()
    )
    assert observed == [1.0]
    assert measurement == ISSDevelopmentMeasurement(-0.4, 0.3, 0.2)


@pytest.mark.integration
def test_evaluation_uses_asset_controller_and_rejects_nonformal_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    betas: list[float] = []

    def two_pass(pipeline, unit, key, assets, controller):
        betas.append(controller(-0.4))
        return SimpleNamespace(image="write"), -0.4

    monkeypatch.setattr(runtime, "_run_two_pass", two_pass)
    output = runtime.run_content_v8_evaluation_pair(
        object(),
        _unit("content_v6_iss_clean_v1"),
        b"d" * 32,
        _assets(),
        _asset(),
    )
    assert output.image == "write"
    assert betas == [2.0]
    with pytest.raises(TypeError, match="outside both"):
        runtime.run_content_v8_evaluation_pair(
            object(), _unit("not-formal"), b"d" * 32, _assets(), _asset()
        )
