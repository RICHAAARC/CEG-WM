from __future__ import annotations

from types import SimpleNamespace

import pytest

from cegwm.protocol.content_chain_v6 import ContentV6Unit, V6_DEVELOPMENT_SPLIT
from cegwm.runtime import content_iss_sd35_v6 as runtime


def _assets() -> runtime.ContentV6DevelopmentAssets:
    instance = object.__new__(runtime.ContentV6DevelopmentAssets)
    object.__setattr__(instance, "embed_assets", object())
    object.__setattr__(instance, "lf_public_assets", object())
    return instance


@pytest.mark.integration
def test_real_pair_orchestration_is_plain_then_v4_beta_one_with_reset_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(runtime, "_generator", lambda seed: f"generator:{seed}")
    monkeypatch.setattr(
        runtime,
        "run_sd35_plain",
        lambda pipeline, prompt, **kwargs: calls.append(("plain", pipeline, prompt, kwargs)) or "plain-image",
    )
    monkeypatch.setattr(
        runtime,
        "run_sd35_content_v3",
        lambda pipeline, prompt, key, assets, **kwargs: calls.append(("v4-beta-one", pipeline, prompt, key, assets, kwargs)) or SimpleNamespace(image="joint-image"),
    )
    wrong_keys = tuple(bytes([index + 1]) * 32 for index in range(16))
    monkeypatch.setattr(runtime, "derive_development_wrong_keys", lambda key: wrong_keys)
    monkeypatch.setattr(
        runtime,
        "content_v6_h",
        lambda image, key, assets: 0.1 if image == "plain-image" else (0.4 if key == b"d"*32 else 0.2),
    )
    unit = ContentV6Unit(
        "content-v6-iss-dev-0001", V6_DEVELOPMENT_SPLIT,
        "content-v6-iss-dev-source-0001", "prompt", 2026082400, 512, 512,
    )
    assets = _assets()
    measurement = runtime.run_content_v6_development_pair(object(), unit, b"d"*32, assets)
    assert [call[0] for call in calls] == ["plain", "v4-beta-one"]
    assert calls[0][-1]["generator"] == calls[1][-1]["generator"] == "generator:2026082400"
    assert calls[1][4] is assets.embed_assets
    assert measurement.host_score == 0.1
    assert measurement.beta_one_score == 0.4
    assert measurement.competition_score == 0.2


@pytest.mark.integration
def test_runtime_rejects_non_development_unit_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "run_sd35_plain", lambda *args, **kwargs: pytest.fail("must not run"))
    unit = ContentV6Unit("u", "wrong", "s", "p", 1, 512, 512)
    with pytest.raises(TypeError, match="validated dev unit"):
        runtime.run_content_v6_development_pair(object(), unit, b"d"*32, _assets())
