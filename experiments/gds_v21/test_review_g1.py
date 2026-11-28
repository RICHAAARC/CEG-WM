"""Software tests with explicit synthetic arrays; no generative model or training."""
import numpy as np
import pytest
from geometry import N,MASK,groups,addresses,compose
from review_g1 import retention,relative_geometry,classify_failure,run


def test_duplicate_events_and_unique_endpoint_cohort_are_different():
    group,address=groups(),addresses()
    codes=np.flatnonzero(group==address[0]);a,b=int(codes[0]),int(codes[1])
    wrong=int(np.flatnonzero(group!=address[0])[0])
    z=np.full(N,wrong,dtype=int)
    z60=z.copy();z60[0]=a
    z50=z.copy();z50[0]=b;z50[1]=a
    z10=z.copy();z10[0]=MASK;z10[1]=b
    z0=z.copy();z0[0]=a
    states={60:z60,50:z50,11:z50.copy(),10:z10,0:z0}
    rows=[dict(timestep=60,executed_selected=[0]),dict(timestep=50,executed_selected=[0,1])]
    events,result=retention(states,rows,group,address)
    assert result['successful_events']==3 and result['unique_success_positions']==2
    assert result['event_group_retention_t0']==2/3
    assert result['unique_group_retention_t0']==.5
    assert result['unique_group_retention_t10']==.5
    assert result['unique_exact_token_retention_t0']==0
    assert events[0]['exact_token_retained_t0']==1


def test_no_selected_writes_is_not_zero_retention():
    states={t:np.zeros(N,dtype=int) for t in (0,10,11)}
    events,result=retention(states,[],groups(),addresses())
    assert events==[] and result['unique_success_positions']==0
    assert np.isnan(result['unique_group_retention_t0']) and not result['retention_applicable']


@pytest.mark.parametrize('a,b,c,label',[
    (1,1,.1,'positive_final'),(1,-1,0,'positive_at_11_nonpositive_final'),
    (-1,1,-1,'nonpositive_at_11_recovered_at_10_lost_by_final'),
    (-1,0,-1,'already_nonpositive_at_11_and_10')])
def test_failure_stage(a,b,c,label):
    assert classify_failure(a,b,c)==label


def test_relative_transform_removes_reference_rotation_and_scale():
    r=(9.,1.25,12.,-12.);delta=(0.,1.,12.,0.)
    details=relative_geometry(r,compose(r,delta))
    assert details['relative_type']=='translation'
    assert abs(details['relative_tx']-12)<1e-12 and abs(details['relative_ty'])<1e-12


def test_refuses_output_inside_source_before_writing(tmp_path):
    with pytest.raises(ValueError,match='separate'):
        run(tmp_path,tmp_path/'derived')
    assert not (tmp_path/'derived').exists()
