from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
import pytest

from cegwm.protocol.content_chain_v7 import ContentV7Unit, V7_DEVELOPMENT_SPLIT
from cegwm.runtime import content_iss_sd35_v7 as runtime


def _development_assets() -> runtime.ContentV7DevelopmentAssets:
    instance = object.__new__(runtime.ContentV7DevelopmentAssets)
    object.__setattr__(instance, "embed_assets", object())
    object.__setattr__(instance, "lf_public_assets", object())
    return instance


def _evaluation_assets() -> runtime.ContentV7EvaluationAssets:
    instance = object.__new__(runtime.ContentV7EvaluationAssets)
    object.__setattr__(instance, "embed_assets", object())
    object.__setattr__(instance, "lf_public_assets", object())
    object.__setattr__(instance, "iss_asset", object())
    return instance


@pytest.mark.integration
def test_development_pair_is_plain_then_unweighted_writer_with_reset_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    primary = Image.new("RGB", (512, 512), "navy")
    joint = Image.new("RGB", (512, 512), "gray")
    monkeypatch.setattr(runtime, "_generator", lambda seed: f"generator:{seed}")
    monkeypatch.setattr(
        runtime,
        "run_sd35_plain",
        lambda pipeline, prompt, **kwargs: calls.append(
            ("plain", pipeline, prompt, kwargs)
        ) or primary,
    )
    monkeypatch.setattr(
        runtime,
        "run_sd35_content_v3",
        lambda pipeline, prompt, key, assets, **kwargs: calls.append(
            ("unweighted", pipeline, prompt, key, assets, kwargs)
        ) or SimpleNamespace(image=joint),
    )
    wrong_keys = tuple(bytes([index + 1]) * 32 for index in range(16))
    monkeypatch.setattr(runtime, "derive_development_wrong_keys", lambda key: wrong_keys)
    monkeypatch.setattr(
        runtime,
        "score_content_v7_lf",
        lambda image, key, assets: (
            0.1
            if image.getpixel((0, 0)) == primary.getpixel((0, 0))
            else (0.4 if key == b"d" * 32 else 0.2)
        ),
    )
    unit = ContentV7Unit(
        "content-v6-iss-dev-0001",
        V7_DEVELOPMENT_SPLIT,
        "content-v6-iss-dev-source-0001",
        "prompt",
        2026082400,
        512,
        512,
    )
    assets = _development_assets()
    measurement = runtime.run_content_v7_development_pair(
        object(), unit, b"d" * 32, assets
    )
    assert [call[0] for call in calls] == ["plain", "unweighted"]
    assert calls[0][-1]["generator"] == calls[1][-1]["generator"] == (
        "generator:2026082400"
    )
    assert measurement.host_registered_score == 0.1
    assert measurement.beta_one_registered_score == 0.4
    assert measurement.beta_one_competition_score == 0.2


@pytest.mark.integration
def test_evaluation_pair_reuses_sole_pass1_null_then_same_seed_pass2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    primary = Image.new("RGB", (512, 512), "navy")
    joint = Image.new("RGB", (512, 512), "gray")
    measurement = object()
    monkeypatch.setattr(runtime, "_generator", lambda seed: f"generator:{seed}")
    monkeypatch.setattr(
        runtime,
        "run_sd35_plain",
        lambda pipeline, prompt, **kwargs: calls.append(
            ("pass1", pipeline, prompt, kwargs)
        ) or primary,
    )
    monkeypatch.setattr(
        runtime,
        "score_content_v7_lf",
        lambda image, key, assets: calls.append(("host", image, key, assets)) or 0.01,
    )
    monkeypatch.setattr(
        runtime,
        "iss_beta",
        lambda host, asset: calls.append(("beta", host, asset)) or 1.75,
    )
    monkeypatch.setattr(
        runtime,
        "_run_content_v7_pass2",
        lambda pipeline, prompt, key, assets, beta, **kwargs: calls.append(
            ("pass2", pipeline, prompt, key, assets, beta, kwargs)
        ) or (joint, measurement),
    )
    output = runtime.run_content_v7_evaluation_pair(
        object(),
        "prompt",
        b"d" * 32,
        _evaluation_assets(),
        height=512,
        width=512,
        seed=2026082500,
    )
    assert [call[0] for call in calls] == ["pass1", "host", "beta", "pass2"]
    assert calls[0][-1]["generator"] == calls[-1][-1]["generator"] == (
        "generator:2026082500"
    )
    assert calls[-1][5] == 1.75
    assert output.primary_null.tobytes() == primary.tobytes()
    assert output.image is joint
    assert output.measurement is measurement


@pytest.mark.integration
def test_development_rejects_wrong_role_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime, "run_sd35_plain", lambda *args, **kwargs: pytest.fail("must not run")
    )
    unit = ContentV7Unit("u", "wrong", "s", "p", 1, 512, 512)
    with pytest.raises(TypeError, match="validated dev unit"):
        runtime.run_content_v7_development_pair(
            object(), unit, b"d" * 32, _development_assets()
        )
