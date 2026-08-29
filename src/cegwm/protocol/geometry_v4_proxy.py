"""P1 RGB-proxy contract loader."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
P1_CONFIG="geometry_v4_p1_proxy_v1.json"
P1_DIGEST="f365628baec9691daecb0477e7f6ad3adab660d8068af5654b12edd11672fc89"
def load_p1_proxy(root: str|Path)->dict:
    raw=(Path(root)/"configs/geometry_v4"/P1_CONFIG).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=P1_DIGEST: raise ValueError("P1 proxy config differs")
    value=json.loads(raw)
    if raw != (json.dumps(value,indent=2,sort_keys=True)+"\n").encode(): raise ValueError("P1 proxy config is noncanonical")
    return value
