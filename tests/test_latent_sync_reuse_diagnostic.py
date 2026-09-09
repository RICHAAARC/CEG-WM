import numpy as np
import pytest
from PIL import Image

from experiments.run_latent_sync_reuse_diagnostic import (
    SOURCE_NAMES,coordinate_audit,diagnostic_plan,evaluate_images,rotation_about)
from experiments.parallel_method_protocol_v1 import ACTIVE_A_ATTACKS,render_attack
from cegwm.method.latent_sync import rectify_once,latent_to_rgb_h

pytestmark=pytest.mark.unit


def image_fixture():
    y,x=np.mgrid[:512,:512]
    return Image.fromarray(np.stack((x%256,y%256,(3*x+5*y)%256),axis=-1).astype(np.uint8))


def test_actual_v2_sign_center_and_latent_rgb_conjugation():
    audit=coordinate_audit()
    assert audit['forward_screen_degrees']==pytest.approx(-10)
    assert audit['actual_rgb_center']==pytest.approx([255,255])
    assert audit['actual_latent64_center']==pytest.approx([31.4375,31.4375])
    assert audit['reader_rgb_center']==pytest.approx([255.5,255.5])
    assert audit['exact_angle_reader_center_translation_error_pixels']==pytest.approx(.1232568334)
    H=np.array(audit['H_reference_to_observed'])
    assert np.allclose(latent_to_rgb_h(np.array(audit['H_latent64_reference_to_observed']),(64,64),(512,512)),H)
    point=np.array([355,255,1.])
    assert (H@point)[1] < 255  # right-of-center point moves upward, display CCW


def test_reuse_paths_counts_direct_warps_and_no_oracle_selection():
    image=image_fixture(); audit=coordinate_audit(); truth=np.array(audit['H_reference_to_observed'])
    seen=[]; reads=[]
    def reader(current):
        reads.append(current)
        return dict(H_reference_to_observed_pixels=(np.eye(3) if len(reads)%2 else truth).tolist())
    def scorer(current):
        seen.append(current); return float(len(seen))
    report=evaluate_images(dict.fromkeys(SOURCE_NAMES,image),scorer=scorer,reader=reader)
    assert len(reads)==report['actual_reader_calls']==8
    assert len(seen)==report['actual_content_scorer_calls']==32
    assert report['failed_content_rows']==report['failed_reader_rows']==0
    assert report['path_comparisons'][0]['whole_pre_estimated_post']==3
    assert report['path_comparisons'][1]['whole_pre_estimated_post']==6
    assert report['rows'][1]['identity_application_shortcut'] is False
    assert seen[1] is not image  # actual identity transform, not an application copy shortcut
    assert np.array_equal(np.asarray(seen[1]),np.asarray(image))
    attacked,_=render_attack(image,ACTIVE_A_ATTACKS[1])
    for index,delta in ((6,-1),(7,1)):
        expected=rectify_once(attacked,rotation_about(audit['actual_rgb_center'],-10+delta))
        assert np.array_equal(np.asarray(seen[index]),np.asarray(expected))
        assert report['rows'][index]['correction_resamplings']==1


def test_reader_failure_preserves_all_32_rows_and_independent_oracle_scores():
    def fail(_): raise RuntimeError('reader failed')
    report=evaluate_images(dict.fromkeys(SOURCE_NAMES,image_fixture()),scorer=lambda _:1.,reader=fail)
    assert len(report['rows'])==32
    assert report['failed_content_rows']==8
    assert report['failed_reader_rows']==8
    assert report['actual_content_scorer_calls']==24
    assert all(r['error'] is None for r in report['rows'] if r['variant']=='rotation_truth')


def test_missing_sources_retain_denominator_without_model_calls():
    report=evaluate_images({},scorer=None,reader=None)
    assert len(report['rows'])==32
    assert len(report['reader_rows'])==8
    assert report['actual_content_scorer_calls']==report['actual_reader_calls']==0
    assert diagnostic_plan()['generations']==0


def test_invalid_reader_h_is_failed_before_prediction_and_rows_stream_incrementally():
    retained=[]; reader_retained=[]
    report=evaluate_images(dict.fromkeys(SOURCE_NAMES,image_fixture()),scorer=lambda _:1.,
        reader=lambda _:dict(H_reference_to_observed_pixels=np.full((3,3),np.nan).tolist()),
        on_row=lambda r:retained.append(dict(r)),on_reader_row=lambda r:reader_retained.append(dict(r)))
    assert retained==report['rows'] and reader_retained==report['reader_rows']
    assert report['failed_reader_rows']==8
    assert report['failed_content_rows']==8
    assert all('finite invertible' in r['error'] for r in reader_retained)


def test_detailed_score_preserves_all_17_key_branches_without_second_observation(monkeypatch):
    import cegwm.runtime.blind_scoring_v2 as scoring
    from experiments.run_latent_sync_reuse_diagnostic import detailed_content_score
    calls=[]
    labels=['registered']+[f'wrong_{i:02d}' for i in range(16)]
    branches={name:dict(zip(labels,[3.]+list(np.arange(16)/10))) for name in ('lf','hf','weighted_joint')}
    def score(image,key,assets,*,reuse_observation):
        calls.append(reuse_observation); return branches,'shared_candidate_observation'
    monkeypatch.setattr(scoring,'score_branches_v2',score)
    result=detailed_content_score(image_fixture(),b'key',object())
    assert calls==[True]
    assert result['raw_content_score']==pytest.approx(1.5)
    assert result['registered_weighted_joint']==3.
    assert result['wrong_key_max']==1.5
    assert result['branches']==branches
