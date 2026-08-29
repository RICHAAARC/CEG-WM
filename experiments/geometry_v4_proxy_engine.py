"""P1 split runner; P1C is declared but never executed by P1D-A."""
from cegwm.method.geometry_v4_proxy import write_proxy,detect_proxy
def run_unit(rgb,key):
    marked,budget=write_proxy(rgb,key); return {"marked":marked,"budget":budget,"detection":detect_proxy(marked,key)}
