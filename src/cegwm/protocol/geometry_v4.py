"""Frozen P0 contract for Geometry-V4 coordinate recovery only."""
from __future__ import annotations
import hashlib, hmac, json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

GEOMETRY_V4_METHOD_ID = "geometry_v4_keyed_multiscale_sync_anchor_v1"
GEOMETRY_V4_PROXY_WRITER_ID = "geometry_v4_rgb_proxy_writer_v1"
GEOMETRY_V4_GENERATED_WRITER_ID = "geometry_v4_generated_writer_v1"
GEOMETRY_V4_PROTOCOL_ID = "cegwm-geometry-v4-p0-contract-v1"
GEOMETRY_V4_PROTOCOL_VERSION = 1
GEOMETRY_V4_CONFIG_NAME = "geometry_v4_p0_contract_v1.json"
GEOMETRY_V4_CONFIG_SHA256 = "0c2eb87056f3c903b672f46bad1a2def197260d20a73482c073e656b195a56d2"
GEOMETRY_V4_STATUS = ("RELIABLE", "UNRELIABLE", "STOPPED")
GEOMETRY_V4_GEOMETRY_KEY_DOMAIN = b"CEG-WM/geometry-v4/keyed-multiscale-sync-anchor/v1"
_HEX64 = re.compile(r"[0-9a-f]{64}")

def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): raise ValueError(f"geometry V4 {name} must be finite")
    return float(value)

def _h(values: tuple[float, ...], reliable: bool) -> None:
    if len(values) != 9 or values[8] != 1.0: raise ValueError("geometry V4 H_hat must be normalized nine-value")
    v = tuple(_finite(x, "H_hat") for x in values)
    det = v[0]*(v[4]*v[8]-v[5]*v[7])-v[1]*(v[3]*v[8]-v[5]*v[6])+v[2]*(v[3]*v[7]-v[4]*v[6])
    if reliable and det == 0.0: raise ValueError("geometry V4 reliable H_hat must be non-singular")

def _corners(points: tuple[tuple[float, float], ...]) -> None:
    if len(points) != 4: raise ValueError("geometry V4 corners must be ordered TL TR BR BL")
    for point in points:
        if not isinstance(point, tuple) or len(point) != 2 or any(not -1 <= _finite(x, "corner") <= 2 for x in point): raise ValueError("geometry V4 corner differs")
    crosses = []
    for i in range(4):
        a,b,c=points[i],points[(i+1)%4],points[(i+2)%4]
        crosses.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
    area = sum(points[i][0]*points[(i+1)%4][1]-points[(i+1)%4][0]*points[i][1] for i in range(4))
    if area <= 0.0 or any(x <= 0.0 for x in crosses): raise ValueError("geometry V4 corners must be strict convex positive area")

def _mapped_corners(values: tuple[float, ...]) -> tuple[tuple[float,float], ...]:
    answer=[]
    for x,y in ((0.,0.),(1.,0.),(1.,1.),(0.,1.)):
        w=values[6]*x+values[7]*y+values[8]
        if not math.isfinite(w) or w == 0.0: raise ValueError("geometry V4 H_hat homogeneous division differs")
        answer.append(((values[0]*x+values[1]*y+values[2])/w,(values[3]*x+values[4]*y+values[5])/w))
    return tuple(answer)

@dataclass(frozen=True, slots=True)
class GeometryV4Observation:
    H_hat: tuple[float, ...] | None
    corners_hat: tuple[tuple[float, float], ...]
    support: int
    reliability: float
    status: str
    def __post_init__(self) -> None:
        if self.status not in GEOMETRY_V4_STATUS or isinstance(self.support, bool) or not isinstance(self.support, int) or self.support < 0: raise ValueError("geometry V4 status or support differs")
        r = _finite(self.reliability, "reliability")
        if not 0 <= r <= 1: raise ValueError("geometry V4 reliability must be in [0,1]")
        if self.status == "STOPPED":
            if self.H_hat is not None or self.corners_hat or self.support or r: raise ValueError("geometry V4 STOPPED output must be empty")
            return
        if self.H_hat is not None: _h(self.H_hat, self.status == "RELIABLE")
        if self.corners_hat: _corners(self.corners_hat)
        if self.status == "RELIABLE":
            if self.H_hat is None or len(self.corners_hat) != 4 or self.support < 6 or r < .5: raise ValueError("geometry V4 RELIABLE fails closed")
            if any(abs(a-b)>1e-9 for got,want in zip(self.corners_hat,_mapped_corners(self.H_hat),strict=True) for a,b in zip(got,want,strict=True)): raise ValueError("geometry V4 H/corners mismatch")

def derive_geometry_v4_key(root_key: bytes, *, length: int = 32, salt: bytes = b"") -> bytes:
    if not isinstance(root_key, bytes) or not root_key: raise TypeError("geometry V4 root_key must be non-empty bytes")
    if not isinstance(salt, bytes): raise TypeError("geometry V4 salt must be bytes")
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= 8160: raise ValueError("geometry V4 length differs")
    prk, out, prior = hmac.new(salt or b"\0"*32, root_key, hashlib.sha256).digest(), b"", b""
    for counter in range(1, math.ceil(length/32)+1):
        prior = hmac.new(prk, prior + GEOMETRY_V4_GEOMETRY_KEY_DOMAIN + bytes((counter,)), hashlib.sha256).digest(); out += prior
    return out[:length]

def reliability_is_reliable(metrics: Mapping[str, object]) -> bool:
    keys = ("PSR","support","inlier_ratio","spatial_coverage","macro_regions","reprojection_rms_diagonal","condition_number","cross_scale_rotation_spread_deg","cross_scale_log_scale_spread","aggregate_reliability")
    try: v = {key:_finite(metrics[key], key) for key in keys}
    except (KeyError, ValueError, TypeError): return False
    if (metrics.get("corner_validity") is not True or v["PSR"]<0 or not isinstance(metrics["support"],int) or isinstance(metrics["support"],bool) or not 0<=v["support"]<=16 or any(not 0<=v[k]<=1 for k in ("inlier_ratio","spatial_coverage","aggregate_reliability")) or not isinstance(metrics["macro_regions"],int) or isinstance(metrics["macro_regions"],bool) or not 0<=v["macro_regions"]<=4 or any(v[k]<0 for k in ("reprojection_rms_diagonal","cross_scale_rotation_spread_deg","cross_scale_log_scale_spread")) or v["condition_number"]<1): return False
    return v["PSR"]>=8 and v["support"]>=6 and v["inlier_ratio"]>=.5 and v["spatial_coverage"]>=.75 and v["macro_regions"]>=3 and v["reprojection_rms_diagonal"]<=.02 and v["condition_number"]<=1e4 and v["cross_scale_rotation_spread_deg"]<=2 and v["cross_scale_log_scale_spread"]<=.03 and v["aggregate_reliability"]>=.5

def _canonical(value: Mapping[str, Any]) -> bytes: return (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)+"\n").encode("ascii")
def load_geometry_v4_p0_contract(repo_root: str | Path) -> Mapping[str, Any]:
    raw=(Path(repo_root)/"configs"/"geometry_v4"/GEOMETRY_V4_CONFIG_NAME).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=GEOMETRY_V4_CONFIG_SHA256: raise ValueError("geometry V4 P0 contract bytes differ")
    try: c=json.loads(raw)
    except (UnicodeDecodeError,json.JSONDecodeError) as e: raise ValueError("geometry V4 P0 contract must be JSON") from e
    if not isinstance(c,dict) or raw!=_canonical(c): raise ValueError("geometry V4 P0 contract encoding differs")
    if c.get("p0_protocol_id")!=GEOMETRY_V4_PROTOCOL_ID or c.get("p0_protocol_version")!=1 or c.get("identities",{}).get("method_id")!=GEOMETRY_V4_METHOD_ID or tuple(c.get("status_enum",()))!=GEOMETRY_V4_STATUS or c.get("detector_boundary",{}).get("allowed_input")!="current_attacked_ordinary_RGB_only" or not c.get("route_invariants") or not c.get("record_only_pre_run"): raise ValueError("geometry V4 P0 identity differs")
    return c
def geometry_v4_contract_digest(contract: Mapping[str, Any]) -> str: return hashlib.sha256(_canonical(contract)).hexdigest()
def require_geometry_v4_contract_digest(digest: str) -> None:
    if not isinstance(digest,str) or _HEX64.fullmatch(digest) is None: raise ValueError("geometry V4 contract digest must be lowercase 64-hex")
    if digest != GEOMETRY_V4_CONFIG_SHA256: raise ValueError("geometry V4 contract digest differs")
