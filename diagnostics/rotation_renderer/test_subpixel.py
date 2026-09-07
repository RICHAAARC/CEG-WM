import numpy as np
from PIL import Image
from diagnostics.rotation_renderer.subpixel import ROSTER, continuous_h, _one


def test_continuous_conversion_preserves_fractional_model_coordinates():
    q=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    # Encode a quarter model-pixel shift, independently in the model grid.
    target=(q+1)*127.5 + .25
    raw=(target-128)/128
    points,h=continuous_h(raw)
    np.testing.assert_allclose(np.asarray(points),q+.5/255,rtol=0,atol=1e-15)
    mapped=np.column_stack((q,np.ones(4)))@np.asarray(h).T
    np.testing.assert_allclose(mapped[:,:2]/mapped[:,2:],points,atol=1e-14)
    assert len(ROSTER)==len(set(ROSTER))==8


def test_invalid_raw_is_retained_without_scoring_or_fallback():
    calls=[]
    row=_one(Image.new('RGB',(512,512)),{'raw_syncseal_corners':None},lambda x:calls.append(x))
    assert row['error'] and row['statistic'] is None and row['continuous_h'] is None
    assert row['scorer_called'] is False and not calls
