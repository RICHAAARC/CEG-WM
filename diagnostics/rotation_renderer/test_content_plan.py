from PIL import Image
from cegwm.method.blind_detection import registered_minus_wrong_key_max
from diagnostics.rotation_renderer.content_plan import (
    ALL_ATTACKS, PILOT, REMAINING, _score_routes, plan,
)


def test_fixed_pilot_and_remaining_cover_exact_small_plan():
    assert len(ALL_ATTACKS) == len(set(ALL_ATTACKS)) == 16
    assert len(PILOT) == 2 and len(REMAINING) == 14
    assert set(PILOT).isdisjoint(REMAINING)
    assert set(PILOT+REMAINING) == set(ALL_ATTACKS)
    assert plan()["total_content_calls"] == 3*len(ALL_ATTACKS)+4 == 52


def test_absent_prediction_keeps_pre_and_oracle_no_truth_to_score():
    image = Image.new("RGB", (512,512), (73,45,19))
    seen = []
    def scorer(current):
        seen.append(current)
        return registered_minus_wrong_key_max(2., (0.,)*16)
    rows = _score_routes(image, None, ((1,0,0),(0,1,0),(0,0,1)), scorer, 1.)
    assert len(seen) == 2 and all(isinstance(x,Image.Image) for x in seen)
    assert rows[0]["error"] is None and rows[2]["error"] is None
    assert rows[1]["error"] and rows[1]["statistic"] is None
    assert rows[0]["statistic"] == rows[2]["statistic"]
    assert len(rows[0]["statistic"]["wrong_key_weighted_joint"]) == 16
