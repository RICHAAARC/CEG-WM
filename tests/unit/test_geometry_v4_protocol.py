from __future__ import annotations
import hashlib, json
from pathlib import Path
import pytest
from cegwm.protocol import geometry_v4 as g
ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/"configs/geometry_v4"/g.GEOMETRY_V4_CONFIG_NAME

@pytest.mark.unit
def test_contract_freezes_blind_boundary_anchors_roster_and_policy() -> None:
    c=g.load_geometry_v4_p0_contract(ROOT)
    assert hashlib.sha256(CFG.read_bytes()).hexdigest()==g.GEOMETRY_V4_CONFIG_SHA256
    assert CFG.read_bytes()==(json.dumps(c,ensure_ascii=True,indent=2)+"\n").encode("ascii")
    assert c["geometry_key"]["hierarchy"].startswith("geometry_and_content_are_sibling")
    assert c["detector_boundary"]["allowed_input"]=="current_attacked_ordinary_RGB_only"
    assert len(c["p1_attack_roster"]["attacks"])==16
    assert c["global_anchors"]["cycles_per_image"]==[8,16,32]
    assert c["local_anchors"]["centers"]==[.125,.375,.625,.875]
    assert c["residual_budget"]["total_luma_rms_cap"]==2/255
    assert c["content_rejudge"]["geometry_score"]=="never_votes"

@pytest.mark.unit
def test_root_key_digest_and_reliability_fail_closed() -> None:
    root=b"a sufficiently long root key"
    assert g.derive_geometry_v4_key(root)!=g.derive_geometry_v4_key(root,salt=b"other")
    with pytest.raises(TypeError,match="root_key"): g.derive_geometry_v4_key("bad") # type: ignore[arg-type]
    g.require_geometry_v4_contract_digest(g.GEOMETRY_V4_CONFIG_SHA256)
    with pytest.raises(ValueError,match="differs"): g.require_geometry_v4_contract_digest("0"*64)
    good={"PSR":8,"support":6,"inlier_ratio":.5,"spatial_coverage":.75,"macro_regions":3,"reprojection_rms_diagonal":.02,"condition_number":1e4,"cross_scale_rotation_spread_deg":2,"cross_scale_log_scale_spread":.03,"corner_validity":True,"aggregate_reliability":.5}
    assert g.reliability_is_reliable(good)
    good["PSR"]=float("nan")
    assert not g.reliability_is_reliable(good)
    for key,value in (("reprojection_rms_diagonal",-.1),("cross_scale_rotation_spread_deg",-.1),("condition_number",.5),("inlier_ratio",1.1),("support",6.5),("macro_regions",3.5)):
        bad={"PSR":8,"support":6,"inlier_ratio":.5,"spatial_coverage":.75,"macro_regions":3,"reprojection_rms_diagonal":.02,"condition_number":1e4,"cross_scale_rotation_spread_deg":2,"cross_scale_log_scale_spread":.03,"corner_validity":True,"aggregate_reliability":.5}; bad[key]=value
        assert not g.reliability_is_reliable(bad)

@pytest.mark.unit
def test_observation_reliable_and_nonfinite_regressions() -> None:
    corners=((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    assert g.GeometryV4Observation((1.,0.,0.,0.,1.,0.,0.,0.,1.),corners,6,.5,"RELIABLE").status=="RELIABLE"
    with pytest.raises(ValueError): g.GeometryV4Observation(None,(),6,.6,"RELIABLE")
    with pytest.raises(ValueError): g.GeometryV4Observation((1.,0.,0.,0.,1.,0.,0.,0.,1.),corners,7,float("nan"),"RELIABLE")
    with pytest.raises(ValueError): g.GeometryV4Observation(None,(),0,.1,"STOPPED")
    with pytest.raises(ValueError): g.GeometryV4Observation((1.,0.,0.,0.,1.,0.,0.,0.,1.),((0.,0.),(1.,0.),(.5,.2),(0.,1.)),6,.5,"RELIABLE")
    with pytest.raises(ValueError): g.GeometryV4Observation((1.,0.,.1,0.,1.,0.,0.,0.,1.),corners,6,.5,"RELIABLE")
