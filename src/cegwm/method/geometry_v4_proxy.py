"""Deterministic NumPy RGB proxy; detector consumes attacked RGB and key only."""
from __future__ import annotations
import hashlib, numpy as np
from cegwm.shared.keys import normalize_detection_key
from cegwm.protocol.geometry_v4 import derive_geometry_v4_key

def _field(shape: tuple[int,int], key: bytes)->np.ndarray:
    h,w=shape; y,x=np.mgrid[:h,:w]; seed=int.from_bytes(hashlib.sha256(key).digest()[:8],"big")
    phase=(seed%10000)/10000*2*np.pi; base=sum(np.sin(2*np.pi*f*x/w+phase*(i+1))+0.7*np.cos(2*np.pi*f*y/h-phase) for i,f in enumerate((8,16,32)))
    tile=((x//max(1,w//4))+4*(y//max(1,h//4))); return base+0.3*np.sin(tile+phase)
def write_proxy(rgb: np.ndarray, detection_key: str|bytes)->tuple[np.ndarray,dict]:
    root=normalize_detection_key(detection_key); key=derive_geometry_v4_key(root); f=_field(rgb.shape[:2],key); f=f/np.sqrt(np.mean(f*f)); delta=f[...,None]*(.004/np.sqrt(3)); out=np.clip(rgb.astype(float)+delta,0,1)
    return out,{"rms":float(np.sqrt(np.mean((out-rgb)**2))),"peak":float(np.max(np.abs(out-rgb))),"key_digest":hashlib.sha256(key).hexdigest()}
def detect_proxy(attacked: np.ndarray,detection_key: str|bytes)->dict:
    root=normalize_detection_key(detection_key); key=derive_geometry_v4_key(root); a=attacked.mean(axis=2)-attacked.mean(); f=_field(a.shape,key); corr=np.fft.ifft2(np.fft.fft2(a)*np.conj(np.fft.fft2(f))).real; iy,ix=np.unravel_index(np.argmax(corr),corr.shape); h,w=a.shape; tx=(ix if ix<=w//2 else ix-w)/w; ty=(iy if iy<=h//2 else iy-h)/h
    return {"H_hat":(1.,0.,-tx,0.,1.,-ty,0.,0.,1.),"corners_hat":((-tx,-ty),(1-tx,-ty),(1-tx,1-ty),(-tx,1-ty)),"support":16,"reliability":0.,"status":"UNRELIABLE","diagnostics":{"PSR":float(corr.max()/(corr.std()+1e-12)),"translation":(tx,ty)}}
