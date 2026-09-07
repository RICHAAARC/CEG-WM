"""Focused geometry and dataset separation checks; no content success claim."""
import numpy as np
from PIL import Image
import pytest

from diagnostics.blind_detection_v2.development import ROSTER, PERTURBATIONS, ANGLES, STRENGTHS, rotation, sampler_h
from cegwm.geometry_v7.r1b import rectify_attacked_rgb


@pytest.mark.quick
@pytest.mark.parametrize("angle",[-13.,7.,21.])
def test_truth_sampler_recovers_asymmetric_landmarks(angle):
    image=np.zeros((512,512,3),dtype=np.uint8)
    image[140:165,190:210,0]=255
    image[320:340,350:365,1]=255
    original=Image.fromarray(image)
    recovered=np.asarray(rectify_attacked_rgb(rotation(original,angle),sampler_h(angle)))
    yy,xx=np.mgrid[:512,:512]
    for channel in (0,1):
        before=image[:,:,channel].astype(float)
        after=recovered[:,:,channel].astype(float)
        assert abs((before*xx).sum()/before.sum()-(after*xx).sum()/after.sum()) < .2
        assert abs((before*yy).sum()/before.sum()-(after*yy).sum()/after.sum()) < .2


@pytest.mark.quick
def test_development_is_small_disjoint_and_keeps_all_planned_conditions():
    assert len(ROSTER)==4 and len({r['seed'] for r in ROSTER})==4
    assert all(r['seed'] not in range(2027000000,2027040000) for r in ROSTER)
    assert all(r['seed'] not in range(2026101000,2026101004) for r in ROSTER)
    assert len(ROSTER)*(len(STRENGTHS)*len(ANGLES)+len(ANGLES)+2*len(PERTURBATIONS))==216
    assert len(set(PERTURBATIONS))==21
