from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
import pytest

from cegwm.protocol.content_chain_v9 import CONTENT_V9_CALIBRATION_SPLIT, ContentV9Unit
from cegwm.runtime import content_weighted_joint_sd35_v9 as runtime
from cegwm.runtime.content_iss_sd35_v6 import ContentV6RunOutput


def _assets() -> runtime.ContentV9CalibrationAssets:
    instance = object.__new__(runtime.ContentV9CalibrationAssets)
    v6 = SimpleNamespace(lf_public_assets=object(), hf_public_assets=object())
    object.__setattr__(instance, "v6_assets", v6)
    return instance


@pytest.mark.integration
def test_real_v6_pair_delegation_excludes_candidate_registered_and_orders_33_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Image.new("RGB", (512, 512), "gray")
    primary_null = Image.new("RGB", (512, 512), "white")
    calls: list[tuple[str, object]] = []
    wrong_keys = tuple(bytes([index + 1]) * 32 for index in range(16))
    monkeypatch.setattr(runtime, "derive_calibration_wrong_keys", lambda key: wrong_keys)
    monkeypatch.setattr(
        runtime,
        "run_content_v6_evaluation_pair",
        lambda *args, **kwargs: calls.append(("v6_pair", kwargs))
        or ContentV6RunOutput(candidate, primary_null, object()),
    )
    monkeypatch.setattr(runtime, "require_ordinary_rgb_image", lambda image: image)
    monkeypatch.setattr(
        runtime,
        "score_content_v4_lf_image",
        lambda image, key, assets: calls.append(("lf", (image, key))) or key[0] / 255,
    )
    monkeypatch.setattr(
        runtime,
        "score_hf_image",
        lambda image, key, assets: calls.append(("hf", (image, key))) or -key[0] / 255,
    )
    unit = ContentV9Unit(
        "content-v9-calibration-0001", CONTENT_V9_CALIBRATION_SPLIT,
        "content-v9-calibration-source-0001", "prompt", 2026091000, 512, 512,
    )
    registered = b"r" * 32
    pairs = runtime.run_content_v9_calibration_unit(object(), unit, registered, _assets())
    assert len(pairs) == 33
    assert [name for name, _ in calls].count("v6_pair") == 1
    scorer_calls = [payload for name, payload in calls if name == "lf"]
    assert len(scorer_calls) == 33
    assert all(key != registered for image, key in scorer_calls if image is candidate)
    assert [key for image, key in scorer_calls if image is candidate] == list(wrong_keys)
    assert [key for image, key in scorer_calls if image is primary_null] == [registered, *wrong_keys]


@pytest.mark.integration
def test_runtime_rejects_non_calibration_unit_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime, "run_content_v6_evaluation_pair", lambda *a, **k: pytest.fail("must not run")
    )
    unit = ContentV9Unit("u", "wrong", "s", "p", 1, 512, 512)
    with pytest.raises(TypeError, match="validated calibration unit"):
        runtime.run_content_v9_calibration_unit(object(), unit, b"k" * 32, _assets())
