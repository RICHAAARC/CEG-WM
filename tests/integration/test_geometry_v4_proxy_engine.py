import numpy as np
from experiments.geometry_v4_proxy_engine import run_unit
def test_identity_canary_records_writer_detector_and_budget():
    r=run_unit(np.full((64,64,3),.5),"0123456789abcdef")
    assert r["budget"]["rms"]>0 and r["detection"]["status"]=="UNRELIABLE"
