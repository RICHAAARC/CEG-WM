import numpy as np
import pytest
from cegwm.method.latent_sync import public_template,warp_field,estimate_rotation
from experiments.latent_rotation_objective_diagnostic import (
    analyze_observation,center_sensitivity,mapped_diagnostic_center,rotation_matrix)

pytestmark=pytest.mark.unit


def test_original_objective_and_optimizer_match_production_exactly():
    observation=warp_field(public_template(4,32,32),rotation_matrix((15.5,15.5),-10)).astype(np.float32)
    current=estimate_rotation(observation)
    result=analyze_observation(observation)
    assert result['optimizers']['original']['angle']==pytest.approx(current['parameters'][0],abs=1e-12)
    assert result['optimizers']['original']['correlation']==pytest.approx(current['correlation'],abs=1e-12)
    assert abs(result['optimizers']['tight']['angle']+10)<.005
    assert len(result['curve'])==121
    assert [result['curve'][i]['angle'] for i in (0,60,120)]==[-15,0,15]
    assert not result['curve_used_to_select_optimizer']
    assert result['optimizers']['tight']['nfev']>0


def test_centers_are_separate_diagnostic_and_truth_is_not_an_optimizer_input():
    observation=public_template(4,32,32)
    original=analyze_observation(observation)
    center=mapped_diagnostic_center((32,32))
    alternative=center_sensitivity(observation,center)
    assert original['center_xy']==[15.5,15.5]
    assert alternative['center_xy']==pytest.approx([15.46875,15.46875])
    assert mapped_diagnostic_center((64,64))==[31.4375,31.4375]
    assert alternative['diagnostic_center_override']
    with pytest.raises(TypeError): analyze_observation(observation,truth_H=np.eye(3))
    with pytest.raises(TypeError): center_sensitivity(observation,center,truth_H=np.eye(3))


def test_empty_signal_reports_identity_without_optimization_and_bad_input_rejected():
    result=analyze_observation(np.zeros((4,16,16)))
    assert not result['nonzero_filtered_observation']
    assert result['optimizers']['original']['nfev']==0
    assert result['optimizers']['tight']['angle']==0
    with pytest.raises(ValueError): analyze_observation(np.full((4,16,16),np.nan))
    with pytest.raises(ValueError): center_sensitivity(np.zeros((4,16,16)),[np.nan,7.5])
