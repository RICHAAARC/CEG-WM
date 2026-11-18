"""Matched local paths. VAE attention is outside this finite-radius reader."""
import torch
from torch import nn
from .protocol import ordinary_logits, relation


def trunk(inputs, config):
    layers = []
    for i in range(config.conv_layers):
        layers.extend([nn.Conv2d(inputs if i == 0 else config.width, config.width, 3, padding=1), nn.SiLU()])
    return nn.Sequential(*layers)


class Writer(nn.Module):
    def __init__(self, arm, config):
        super().__init__()
        if arm not in ("B0", "B1", "C"):
            raise ValueError(arm)
        self.arm, self.config = arm, config
        self.trunk = trunk(config.latent_channels + config.bits + (arm != "B0"), config)
        self.head = nn.Conv2d(config.width, config.latent_channels, 1)

    def forward(self, z, bits, p=None):
        message = bits[:, :, None, None].expand(-1, -1, *z.shape[-2:])
        if self.arm == "B0":
            if p is not None:
                raise ValueError("B0 has no reference task or reference input")
            fields = message
        else:
            if p is None:
                raise ValueError("reference required on writer side")
            fields = torch.cat([p, message * p if self.arm == "C" else message], 1)
        return z + self.config.residual_cap * self.head(self.trunk(torch.cat([z, fields], 1))).tanh()


class Reader(nn.Module):
    def __init__(self, arm, config):
        super().__init__()
        self.arm, self.config = arm, config
        self.trunk = trunk(config.latent_channels, config)
        self.payload = nn.Conv2d(config.width, config.bits, 1)
        self.reference = nn.Conv2d(config.width, 1, 1) if arm != "B0" else None

    def forward(self, observation):
        # No key, nonce, p, original image, adaptive region or alignment input.
        local = self.trunk(observation)
        q = self.payload(local)
        p = self.reference(local) if self.reference is not None else None
        evidence = relation(p, q, self.config.mask_border, self.config.epsilon) if self.arm == "C" \
            else ordinary_logits(q, self.config.mask_border)
        return {"evidence": evidence, "p": p, "q": q}


def parameter_counts(config):
    return {arm: {"writer": sum(p.numel() for p in Writer(arm, config).parameters()),
                  "reader": sum(p.numel() for p in Reader(arm, config).parameters())}
            for arm in ("B0", "B1", "C")}
