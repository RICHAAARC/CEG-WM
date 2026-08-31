"""Blind, method-faithful SD3.5 primitives for the three external baselines.

This module is deliberately an adapter boundary.  It contains no threshold,
truth image, embed record, or synthetic/oracle score.  Runtime callers retain
only a seed/configuration plus a SHA-256 commitment to carrier material.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

SD35_SHAPE = (1, 16, 64, 64)


def _circle(size: int, radius: int) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return (x - size // 2).square() + (y - size // 2).square() <= radius * radius


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _rgb(rgb: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise TypeError("detector requires ordinary HxWx3 uint8 RGB")
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype).div(127.5).sub(1)


@dataclass(frozen=True)
class TreeRingCarrier:
    mask: torch.Tensor
    key: torch.Tensor
    digest: str

    @classmethod
    def fixed(cls, *, seed: int = 999999, channel: int = 0, radius: int = 10, shape: tuple[int, ...] = SD35_SHAPE, device: str = "cpu") -> "TreeRingCarrier":
        if tuple(shape) != SD35_SHAPE or channel not in range(16): raise ValueError("Tree-Ring requires SD3.5 16-channel ch0..15 latent")
        generator = torch.Generator(device=device).manual_seed(seed)
        initial = torch.randn(shape, generator=generator, device=device)
        key = torch.fft.fftshift(torch.fft.fft2(initial), dim=(-2, -1))
        ring = _circle(64, radius).to(device)
        source = key.clone()
        for current in range(radius, 0, -1):
            key[:, :, _circle(64, current).to(device)] = source[0, :, 0, current].unsqueeze(-1)
        mask = torch.zeros(shape, dtype=torch.bool, device=device); mask[:, channel] = ring
        return cls(mask, key, _digest(key))

    def inject(self, latents: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.fftshift(torch.fft.fft2(latents.float()), dim=(-2, -1)); freq[self.mask] = self.key[self.mask]
        return torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1))).real.to(latents.dtype)

    def score(self, reversed_latents: torch.Tensor) -> float:
        freq = torch.fft.fftshift(torch.fft.fft2(reversed_latents.float()), dim=(-2, -1))
        return -float(torch.abs(freq[self.mask] - self.key[self.mask]).mean())


def _rot(v: int, c: int) -> int: return ((v << c) & 0xffffffff) | ((v & 0xffffffff) >> (32-c))
def _qr(s: list[int], a: int,b: int,c: int,d: int) -> None:
    s[a]=(s[a]+s[b])&0xffffffff; s[d]=_rot(s[d]^s[a],16); s[c]=(s[c]+s[d])&0xffffffff; s[b]=_rot(s[b]^s[c],12)
    s[a]=(s[a]+s[b])&0xffffffff; s[d]=_rot(s[d]^s[a],8); s[c]=(s[c]+s[d])&0xffffffff; s[b]=_rot(s[b]^s[c],7)
def chacha20(data: bytes, *, key: bytes, nonce: bytes, counter: int = 0) -> bytes:
    """RFC 8439 IETF ChaCha20 stream XOR used by Gaussian Shading."""
    if len(key) != 32 or len(nonce) != 12: raise ValueError("ChaCha20 requires a 32-byte key and 12-byte nonce")
    out=bytearray(); constants=b"expand 32-byte k"
    for block, offset in enumerate(range(0,len(data),64)):
        st=[int.from_bytes(constants[i:i+4],"little") for i in range(0,16,4)]+[int.from_bytes(key[i:i+4],"little") for i in range(0,32,4)]+[counter+block]+[int.from_bytes(nonce[i:i+4],"little") for i in range(0,12,4)]
        w=st.copy()
        for _ in range(10):
            _qr(w,0,4,8,12);_qr(w,1,5,9,13);_qr(w,2,6,10,14);_qr(w,3,7,11,15);_qr(w,0,5,10,15);_qr(w,1,6,11,12);_qr(w,2,7,8,13);_qr(w,3,4,9,14)
        stream=b"".join(((w[i]+st[i])&0xffffffff).to_bytes(4,"little") for i in range(16))
        out.extend(x^y for x,y in zip(data[offset:offset+64],stream))
    return bytes(out)


def _pack_binary_tensor(value: torch.Tensor) -> bytes:
    return np.packbits(value.detach().to("cpu", torch.uint8).contiguous().numpy().reshape(-1), bitorder="big").tobytes()


def _unpack_binary_bytes(value: bytes, *, shape: tuple[int, ...], device: str) -> torch.Tensor:
    bits = np.unpackbits(np.frombuffer(value, dtype=np.uint8), bitorder="big")[:math.prod(shape)].copy()
    return torch.from_numpy(bits).reshape(shape).to(device=device, dtype=torch.int64)


class GaussianShadingCarrier:
    """Direct SLM migration: packed-bit ChaCha20 then tensor.repeat/voting."""
    def __init__(self, *, shape: tuple[int, ...], channel_copy: int, hw_copy: int, generator: torch.Generator, device: str) -> None:
        batch, channels, height, width = shape
        if batch != 1 or channels % channel_copy or height % hw_copy or width % hw_copy: raise ValueError("Gaussian copy factors must divide SD3.5 shape")
        self.shape, self.channel_copy, self.hw_copy, self.device = shape, channel_copy, hw_copy, device
        self.key = bytes(int(v) for v in torch.randint(0,256,(32,),generator=generator,device="cpu").tolist())
        self.nonce = bytes(int(v) for v in torch.randint(0,256,(12,),generator=generator,device="cpu").tolist())
        self.watermark = torch.randint(0,2,(batch,channels//channel_copy,height//hw_copy,width//hw_copy),generator=generator,device="cpu",dtype=torch.int64).to(device)
        self.vote_threshold=max(1,channel_copy*hw_copy*hw_copy//2)
        encrypted=chacha20(_pack_binary_tensor(self.expanded_watermark()),key=self.key,nonce=self.nonce)
        self.encrypted_message=_unpack_binary_bytes(encrypted,shape=shape,device=device)
        self.digest=hashlib.sha256(self.key+self.nonce+_pack_binary_tensor(self.watermark)).hexdigest()
    @classmethod
    def fixed(cls, *, seed: int=20260622, shape: tuple[int,...]=SD35_SHAPE, device: str="cpu") -> "GaussianShadingCarrier":
        return cls(shape=shape,channel_copy=1,hw_copy=8,generator=torch.Generator(device="cpu").manual_seed(seed),device=device)
    def expanded_watermark(self) -> torch.Tensor: return self.watermark.repeat(1,self.channel_copy,self.hw_copy,self.hw_copy)
    def create_strict_paired_latents(self, clean: torch.Tensor) -> torch.Tensor:
        if tuple(clean.shape)!=self.shape: raise ValueError("Gaussian strict pair shape mismatch")
        return ((self.encrypted_message.float()*2-1)*clean.float().abs()).to(clean.dtype)
    embed=create_strict_paired_latents
    def decode_recovered_watermark(self, reversed_latents: torch.Tensor) -> torch.Tensor:
        decrypted=chacha20(_pack_binary_tensor((reversed_latents.float()>0).to(torch.int64)),key=self.key,nonce=self.nonce)
        decoded=_unpack_binary_bytes(decrypted,shape=self.shape,device=self.device); _,channels,height,width=self.shape
        d1=torch.cat(torch.split(decoded,channels//self.channel_copy,dim=1),dim=0); d2=torch.cat(torch.split(d1,height//self.hw_copy,dim=2),dim=0); d3=torch.cat(torch.split(d2,width//self.hw_copy,dim=3),dim=0)
        return (d3.sum(0)>self.vote_threshold).to(torch.int64).unsqueeze(0)
    def score_latents(self, reversed_latents: torch.Tensor) -> float: return float((self.decode_recovered_watermark(reversed_latents)==self.watermark).float().mean())
    score=score_latents


@dataclass(frozen=True)
class ShallowDiffuseCarrier:
    mask: torch.Tensor
    patch: torch.Tensor
    digest: str
    @classmethod
    def fixed(cls, *, seed: int = 42, radius: int = 10, channel: int = 0, shape: tuple[int,...] = SD35_SHAPE, device: str = "cpu") -> "ShallowDiffuseCarrier":
        g=torch.Generator(device=device).manual_seed(seed); patch=torch.fft.fftshift(torch.fft.fft2(torch.randn(shape,generator=g,device=device)),dim=(-2,-1)); patch[:]=patch[0]
        mask=torch.zeros(shape,dtype=torch.bool,device=device); mask[:,channel]=_circle(64,radius).to(device)
        return cls(mask,patch,_digest(patch))
    def inject(self, latents: torch.Tensor) -> torch.Tensor:
        freq=torch.fft.fftshift(torch.fft.fft2(latents.float()),dim=(-2,-1));freq[self.mask]=self.patch[self.mask]
        return torch.fft.ifft2(torch.fft.ifftshift(freq,dim=(-2,-1))).real.to(latents.dtype)
    def score(self, edit_latents: torch.Tensor) -> float:
        freq=torch.fft.fftshift(torch.fft.fft2(edit_latents.float()),dim=(-2,-1));return -float(torch.abs(freq[self.mask]-self.patch[self.mask]).mean())


def score_rgb(rgb: np.ndarray, pipe: Any, carrier: Any, *, inversion_steps: int = 20, prompt: str = "") -> float:
    """Blind RGB-only common scorer; pipeline must expose real VAE + selected reverse flow."""
    device=torch.device(getattr(pipe,"_execution_device","cpu")); image=_rgb(rgb,device,torch.float16 if device.type=="cuda" else torch.float32)
    if not callable(getattr(pipe,"get_image_latents",None)): raise TypeError("pipeline must VAE-encode RGB")
    latent=pipe.get_image_latents(image,sample=False)
    if isinstance(carrier, TreeRingCarrier): reversed_latent=pipe.invert_flow_matching_latent(latent,prompt=prompt,num_inference_steps=inversion_steps,guidance_scale=4.5); return carrier.score(reversed_latent)
    if isinstance(carrier, GaussianShadingCarrier): reversed_latent=pipe.invert_flow_matching_latent(latent,prompt=prompt,num_inference_steps=inversion_steps,guidance_scale=4.5); return carrier.score(reversed_latent)
    if isinstance(carrier, ShallowDiffuseCarrier): edit=pipe.invert_to_edit_timestep(latent,prompt=prompt,num_inference_steps=inversion_steps,edit_fraction=.2,guidance_scale=4.5); return carrier.score(edit)
    raise TypeError("unknown external baseline carrier")
