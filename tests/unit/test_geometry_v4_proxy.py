import numpy as np
from cegwm.method.geometry_v4_proxy import write_proxy,detect_proxy
def test_real_keyed_rgb_flow_and_wrong_key_changes_observation():
    rgb=np.full((64,64,3),.5); marked,budget=write_proxy(rgb,"0123456789abcdef")
    assert budget["rms"]>0 and budget["peak"]<=.01
    assert detect_proxy(marked,"0123456789abcdef")["H_hat"] != detect_proxy(marked,"fedcba9876543210")["H_hat"]
