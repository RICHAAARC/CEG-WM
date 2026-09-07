import numpy as np
from PIL import Image
import pytest
import torch
from cegwm.geometry_v7.contracts import GeometryStatus, syncseal_raw_to_public_continuous
from cegwm.geometry_v7.syncseal import SyncSealTorchScript
from diagnostics.continuous_corners_v1.paired_geometry import estimate_pair

pytestmark=pytest.mark.unit


class FractionalFixture(torch.nn.Module):
    """No model: known quarter-pixel translation in model-grid coordinates."""
    def __init__(self):
        super().__init__()
        q=torch.tensor([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        self.register_buffer('points',(((q+1)*127.5+.25-128)/128).reshape(1,8))
        self.calls=0
    def embed(self, image):
        raise AssertionError('embedding must not be used')
    def detect(self, image):
        self.calls+=1
        return dict(preds=torch.cat((self.points.new_tensor([[.25]]),self.points),dim=1),preds_pts=self.points)


def test_explicit_continuous_path_keeps_default_rounded_and_is_stateless():
    fixture=FractionalFixture()
    backend=SyncSealTorchScript(fixture)
    image=Image.new('RGB',(512,512))
    default_before=backend.detect_geometry(image)
    continuous=backend.detect_geometry_continuous(image)
    default_after=backend.detect_geometry(image)
    assert fixture.calls==3
    assert default_before==default_after
    np.testing.assert_allclose(default_before.homography_observed_to_canonical,np.eye(3),atol=1e-14)
    expected=np.eye(3)
    expected[:2,2]=.5/255
    np.testing.assert_allclose(continuous.homography_observed_to_canonical,expected,atol=1e-14)
    assert continuous.raw_syncseal_corners==default_before.raw_syncseal_corners


def test_continuous_rejects_nonfinite_output_like_default():
    fixture=FractionalFixture()
    fixture.points[0,0]=torch.nan
    backend=SyncSealTorchScript(fixture)
    for method in (backend.detect_geometry,backend.detect_geometry_continuous):
        result=method(Image.new('RGB',(512,512)))
        assert result.status is GeometryStatus.ERROR and result.homography_observed_to_canonical is None
    with pytest.raises(ValueError): syncseal_raw_to_public_continuous([[float('nan'),0]]*4)


def test_development_pair_reuses_one_raw_observation():
    fixture=FractionalFixture()
    pair=estimate_pair(SyncSealTorchScript(fixture),Image.new('RGB',(512,512)))
    assert fixture.calls==1
    assert pair['rounded'].raw_syncseal_corners==pair['continuous'].raw_syncseal_corners
    assert pair['rounded'].observed_corners_in_canonical_normalized!=pair['continuous'].observed_corners_in_canonical_normalized
