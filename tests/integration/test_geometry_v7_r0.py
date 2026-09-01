from __future__ import annotations

import json

from PIL import Image
import pytest

from cegwm.geometry_v7.contracts import CANONICAL_CORNERS_NORMALIZED, estimate_geometry
from cegwm.geometry_v7.r0 import (
    ContentScore,
    ImageQuality,
    R0Arm,
    R0NumericGates,
    r0_record_payload,
    run_r0_four_arm_unit,
)


@pytest.mark.integration
def test_four_arm_routing_records_raw_deltas_false_positive_quality_and_denominators() -> None:
    unwatermarked = Image.new("RGB", (512, 512), (10, 10, 10))
    content = Image.new("RGB", (512, 512), (20, 20, 20))
    sync_calls: list[tuple[int, float]] = []
    score_calls: list[int] = []
    geometry_calls: list[int] = []
    quality_calls: list[tuple[int, int]] = []

    def sync_embedder(image: Image.Image, strength: float) -> Image.Image:
        value = image.getpixel((0, 0))[0]
        sync_calls.append((value, strength))
        return Image.new("RGB", (512, 512), (value + 1,) * 3)

    def content_scorer(image: Image.Image) -> ContentScore:
        value = image.getpixel((0, 0))[0]
        score_calls.append(value)
        return ContentScore(value / 100.0, value / 200.0, value / 50.0, value >= 20)

    def geometry_detector(image: Image.Image):
        value = image.getpixel((0, 0))[0]
        geometry_calls.append(value)
        return estimate_geometry(value / 10.0, CANONICAL_CORNERS_NORMALIZED)

    def quality_scorer(reference: Image.Image, candidate: Image.Image) -> ImageQuality:
        pair = (reference.getpixel((0, 0))[0], candidate.getpixel((0, 0))[0])
        quality_calls.append(pair)
        return ImageQuality(40.0, 0.99, 0.01)

    record = run_r0_four_arm_unit(
        unit_id="r0-0001",
        unwatermarked_final_rgb=unwatermarked,
        content_watermarked_final_rgb=content,
        residual_strength=0.5,
        sync_embedder=sync_embedder,
        content_scorer=content_scorer,
        geometry_detector=geometry_detector,
        quality_scorer=quality_scorer,
    )
    assert sync_calls == [(10, 0.5), (20, 0.5)]
    assert score_calls == [10, 11, 20, 21]
    assert geometry_calls == [10, 11, 20, 21]
    assert quality_calls == [(10, 11), (20, 21)]
    assert [arm.arm for arm in record.arms] == [R0Arm.U, R0Arm.G, R0Arm.C, R0Arm.CG]
    assert dict(record.cg_minus_c_raw or ()) == pytest.approx(
        {"lf": 0.01, "hf": 0.005, "weighted_joint": 0.02}
    )
    assert record.cg_c_content_flip is False
    assert record.g_content_false_positive is False
    assert (record.negative_arm_denominator, record.positive_arm_denominator) == (2, 2)
    assert record.failure_arm_denominator == 4 and record.failed_arm_count == 0
    payload = r0_record_payload(record)
    assert json.loads(json.dumps(payload, allow_nan=False))["arms"][3]["geometry"]["legal"] is True


@pytest.mark.integration
def test_failed_cg_stays_in_fixed_denominator_without_retry_or_fallback() -> None:
    calls: list[int] = []

    def failing_sync(image: Image.Image, strength: float) -> Image.Image:
        value = image.getpixel((0, 0))[0]
        calls.append(value)
        if value == 20:
            raise RuntimeError("fixed failure")
        return image.copy()

    record = run_r0_four_arm_unit(
        unit_id="r0-0002",
        unwatermarked_final_rgb=Image.new("RGB", (512, 512), (10, 10, 10)),
        content_watermarked_final_rgb=Image.new("RGB", (512, 512), (20, 20, 20)),
        residual_strength=1.0,
        sync_embedder=failing_sync,
        content_scorer=lambda image: ContentScore(0.0, 0.0, 0.0, False),
        geometry_detector=lambda image: estimate_geometry(0.0, CANONICAL_CORNERS_NORMALIZED),
        quality_scorer=lambda left, right: ImageQuality(40.0, 0.99, 0.01),
    )
    assert calls == [10, 20]
    cg = record.arms[3]
    assert cg.arm is R0Arm.CG and cg.image is None
    assert cg.errors == ("sync_embed:RuntimeError:fixed failure",)
    assert record.failure_arm_denominator == 4 and record.failed_arm_count == 1
    assert record.cg_minus_c_raw is None and record.cg_c_content_flip is None


@pytest.mark.integration
def test_numeric_gates_fail_closed_until_user_freezes_every_value() -> None:
    gates = R0NumericGates()
    assert "residual_strengths" in gates.pending_fields
    with pytest.raises(RuntimeError, match="pending confirmation"):
        gates.require_frozen()
