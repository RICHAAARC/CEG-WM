"""Cache manifest validation. No generation or model downloads occur here."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from .config import SPLITS
from .protocol import key_bits


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_path(base, name):
    result = (base / name).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ValueError("manifest asset escapes its directory")
    return result


def require_hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected SHA256 hex digest")


class Cache:
    def __init__(self, manifest, config, *, verify_files=True):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.raw = json.loads(self.path.read_text())
        self.hash = file_hash(self.path)
        self.config = config
        raw = self.raw
        if raw.get("schema") != "self-reference-cache-v1":
            raise ValueError("unsupported manifest schema")
        if raw.get("config_hash") != config.digest:
            raise ValueError("manifest config hash mismatch")
        model = raw["model"]
        if model["family"] != "SD3.5" or raw.get("latent_convention") != "final_z0_scaled_before_decode":
            raise ValueError("requires SD3.5 final scaled z0 cache")
        for name in ("id", "revision", "vae_export"):
            if not model.get(name):
                raise ValueError("missing model provenance: " + name)
        require_hash(model["denoiser_sha256"])
        self.model_dir = safe_path(self.root, model["vae_export"])
        if not model.get("vae_files"):
            raise ValueError("must enumerate complete local VAE export")
        self.rows = raw["units"]
        self.keys = raw["keys"]
        if set(self.keys) != set(SPLITS):
            raise ValueError("all five split key rosters required")
        seen_keys, seen_bits = set(), set()
        for split in SPLITS:
            if not self.keys[split]:
                raise ValueError("empty key roster")
            for key in self.keys[split]:
                bits = tuple(key_bits(key).tolist())
                if key in seen_keys or bits in seen_bits:
                    raise ValueError("key leakage or bit collision across frozen key rosters")
                seen_keys.add(key)
                seen_bits.add(bits)
        owners, units, nonces, pairs = {}, set(), set(), set()
        source_records = {}
        for row in self.rows:
            split = row["split"]
            if split not in SPLITS or row["key"] not in self.keys[split]:
                raise ValueError("unit split/key inconsistent")
            if row["unit_id"] in units:
                raise ValueError("duplicate unit")
            units.add(row["unit_id"])
            for field in ("source_id", "source_group", "source_sha256", "latent_sha256", "clean_sha256"):
                value = row[field]
                if not value:
                    raise ValueError("missing source provenance")
                if field.endswith("sha256"):
                    require_hash(value)
                # Prevent exact content or augmented source leakage, not only ID leakage.
                owner_key = (field, value)
                if owner_key in owners and owners[owner_key] != split:
                    raise ValueError("source leakage across splits: " + field)
                owners[owner_key] = split
            identity = tuple(row[x] for x in ("source_group", "source_sha256", "latent_sha256", "clean_sha256"))
            if row["source_id"] in source_records and source_records[row["source_id"]] != identity:
                raise ValueError("same source_id has inconsistent cache/provenance")
            source_records[row["source_id"]] = identity
            pair = (row["source_id"], row["key"])
            if pair in pairs:
                raise ValueError("duplicate source/key unit")
            pairs.add(pair)
            require_hash(row["reference_nonce"])
            if row["reference_nonce"] in nonces:
                raise ValueError("reference nonce reuse")
            nonces.add(row["reference_nonce"])
            provenance = row["provenance"]
            for name in ("generation_seed", "prompt_sha256", "sampler", "steps", "guidance_scale"):
                if name not in provenance:
                    raise ValueError("missing generation provenance: " + name)
            require_hash(provenance["prompt_sha256"])
        if {row["split"] for row in self.rows} != set(SPLITS):
            raise ValueError("each split needs at least one unit")
        if verify_files:
            expected = model["vae_files"]
            actual = {p.relative_to(self.model_dir).as_posix() for p in self.model_dir.rglob("*") if p.is_file()}
            if actual != set(expected):
                raise ValueError("VAE export file inventory mismatch")
            for name, value in expected.items():
                require_hash(value)
                if file_hash(safe_path(self.model_dir, name)) != value:
                    raise ValueError("VAE hash mismatch: " + name)
            verified = set()
            for row in self.rows:
                for name in ("latent", "clean"):
                    path = safe_path(self.root, row[name])
                    if (path, row[name + "_sha256"]) not in verified:
                        if file_hash(path) != row[name + "_sha256"]:
                            raise ValueError("cache hash mismatch: " + str(path))
                        verified.add((path, row[name + "_sha256"]))
                self.load(row)

    def split(self, split):
        return sorted((r for r in self.rows if r["split"] == split), key=lambda r: r["unit_id"])

    def load(self, row):
        z = np.load(safe_path(self.root, row["latent"]), allow_pickle=False)
        shape = (self.config.latent_channels, self.config.latent_size, self.config.latent_size)
        if z.shape != shape or z.dtype != np.float32 or not np.isfinite(z).all():
            raise ValueError("cache latent must be finite float32 CHW final scaled z0")
        with Image.open(safe_path(self.root, row["clean"])) as image:
            if image.mode != "RGB" or image.size != (self.config.rgb_size, self.config.rgb_size) or image.format != "PNG":
                raise ValueError("clean reference must be lossless ordinary RGB PNG at frozen resolution")
            clean = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        return torch.from_numpy(z.copy()), clean

    def report(self):
        return {"manifest_hash": self.hash, "config_hash": self.config.digest,
                "splits": {s: {"units": len(self.split(s)), "sources": len({r["source_group"] for r in self.split(s)}),
                                "keys": len(self.keys[s])} for s in SPLITS},
                "evidence": "cache engineering validation only; provenance is declared, not independently attested"}
