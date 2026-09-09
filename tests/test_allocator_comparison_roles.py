import pytest
from experiments.run_survival_allocator_dev import summarize_separation

pytestmark = pytest.mark.unit


def test_allocator_gain_is_relative_to_uniform_not_historical_original():
    quality = {'psnr':40., 'ssim':.99, 'lpips':.01}
    rows = [dict(id='one',condition='clean',variant=variant,error=None,score=score,
                 registered=score+1.,max16wrong=1.,quality=quality)
            for variant,score in [('clean',0.),('original',1.),('uniform',3.),('survival',2.)]]
    result = {row['variant']:row for row in summarize_separation(rows,1)}
    candidate = result['survival']
    assert candidate['comparisons_to_original'][0]['score_delta'] == 1.
    assert candidate['comparisons_to_uniform'][0]['score_delta'] == -1.
    assert not candidate['quality_matching_claim']
    assert result['original']['comparisons_to_uniform'] == []
