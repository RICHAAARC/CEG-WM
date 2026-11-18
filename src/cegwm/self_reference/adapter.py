"""Differentiable training proxy and separate ordinary-RGB8 evaluation boundary."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from cegwm.runtime.observation import encode_final_rgb_image, _vae_config_number


def rgb8(pixels):
    if not pixels.dtype.is_floating_point or not bool(torch.isfinite(pixels).all()):
        raise ValueError("RGB quantization requires finite floating pixels")
    return (pixels.detach().float().clamp(0, 1) * 255).round().to(torch.uint8)


def pil_image(pixels):
    return Image.fromarray(pixels.detach().cpu().permute(1, 2, 0).numpy(), "RGB")


class Adapter:
    def __init__(self, vae, processor, config):
        self.vae, self.processor, self.config = vae.eval(), processor, config
        vae.requires_grad_(False)
        self.scale = _vae_config_number(vae.config, "scaling_factor", positive=True)
        self.shift = _vae_config_number(vae.config, "shift_factor", positive=False)
        self.check_processor()

    def check_processor(self):
        # Reject unsupported processor configuration, then check actual conversion.
        cfg = getattr(self.processor, "config", None)
        if cfg is not None:
            for name in ("do_binarize", "do_convert_grayscale"):
                if getattr(cfg, name, False):
                    raise ValueError("unsupported processor: " + name)
            if getattr(cfg, "do_normalize", True) is not True:
                raise ValueError("processor must normalize RGB to [-1,1]")
        size = self.config.rgb_size
        pixels = np.arange(size * size * 3, dtype=np.int64).reshape(size, size, 3).astype(np.uint8)
        expected = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float()/127.5-1
        actual = self.processor.preprocess(Image.fromarray(pixels, "RGB"))
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or not torch.allclose(
                actual.cpu().float(), expected, atol=2e-7, rtol=0):
            raise ValueError("unsupported processor: resizing, color or normalization differs")

    def decode(self, z):
        self.check_tensor(z, (self.config.latent_channels, self.config.latent_size, self.config.latent_size), "decode input")
        raw = self.vae.decode(z / self.scale + self.shift).sample
        self.check_tensor(raw, (3, self.config.rgb_size, self.config.rgb_size), "decode output")
        result = (raw / 2 + 0.5).clamp(0, 1)
        if result.shape[1:] != (3, self.config.rgb_size, self.config.rgb_size):
            raise ValueError("decoder output does not match frozen RGB dimensions")
        return result

    def quantize_ste(self, pixels):
        quantized = rgb8(pixels).to(pixels.dtype) / 255
        return pixels + (quantized - pixels).detach()

    def encode_proxy(self, pixels):
        self.check_tensor(pixels, (3, self.config.rgb_size, self.config.rgb_size), "encode input")
        if bool(((pixels < 0) | (pixels > 1)).any()):
            raise ValueError("encode proxy expects [0,1] RGB")
        result = (self.vae.encode(pixels * 2 - 1).latent_dist.mode() - self.shift) * self.scale
        self.check_tensor(result, (self.config.latent_channels, self.config.latent_size, self.config.latent_size), "encode output")
        return result

    def check_tensor(self, value, shape, label):
        if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[1:] != shape or value.shape[0] < 1:
            raise ValueError(label + " shape mismatch")
        if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
            raise ValueError(label + " must be finite floating tensor")

    def observe(self, pixels):
        # Formal path MUST pass through existing PIL/uint8 observer, never proxy.
        if pixels.dtype != torch.uint8:
            raise TypeError("formal observation requires uint8 RGB")
        if pixels.ndim != 4 or pixels.shape[1:] != (3, self.config.rgb_size, self.config.rgb_size):
            raise ValueError("formal RGB shape mismatch")
        observation = torch.cat([encode_final_rgb_image(pil_image(x), self.processor, self.vae) for x in pixels])
        self.check_tensor(observation, (self.config.latent_channels, self.config.latent_size, self.config.latent_size), "formal observation")
        return observation


def load_adapter(model_dir, config, device):
    # Explicitly deferred imports: --plan and package import do not load diffusers.
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    path = Path(model_dir)
    if not path.is_dir():
        raise ValueError("explicit local SD3.5 VAE export directory required")
    vae = AutoencoderKL.from_pretrained(str(path), local_files_only=True, torch_dtype=torch.float32).to(device)
    if int(vae.config.latent_channels) != config.latent_channels:
        raise ValueError("VAE channel mismatch")
    scale = 2 ** (len(vae.config.block_out_channels) - 1)
    if scale != config.downsample:
        raise ValueError("VAE spatial scale mismatch")
    return Adapter(vae, VaeImageProcessor(vae_scale_factor=scale), config)


def load_lpips(path, device):
    if path is None:
        return None
    import lpips
    # No pretrained backbone download: supplied checkpoint contains ALL weights.
    metric = lpips.LPIPS(net="alex", pnet_rand=True, pretrained=False, verbose=False)
    metric.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    metric.eval().requires_grad_(False).to(device)
    return lambda x, y: metric(x * 2 - 1, y * 2 - 1).reshape(-1)
