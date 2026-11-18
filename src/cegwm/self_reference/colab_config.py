"""Frozen Colab generation/assets. This module uses only the standard library."""
import hashlib
import json
from pathlib import Path

MODEL_CARD = "https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
GENERATION = {
    "schema": "sr-colab-generation-v1", "model_id": "stabilityai/stable-diffusion-3.5-medium",
    "revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80", "height": 512, "width": 512,
    "steps": 40, "guidance_scale": 4.5, "max_sequence_length": 256,
    "dtype": "bfloat16", "clean_dtype": "float32", "output_type": "latent",
    "all_three_text_encoders": True, "cpu_offload": True, "generation_process_sources": 640,
    "atomic_flush_each_source": True,
    "diffusers": "0.35.1", "capacity_policy": "report_only",
    "alexnet_url": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    "alexnet_sha256_prefix": "7be5be79",
    "lpips_linear_url": "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/082bb24f84c091ea94de2867d34c4544f68e0963/lpips/weights/v0.1/alex.pth",
    "lpips_linear_git_blob": "1df9dfe62abb1fc89cc7f82b4e5fe886c979708e",
}
DEPENDENCIES = ["diffusers==0.35.1", "transformers==4.56.2", "accelerate==1.10.1",
                "huggingface-hub==0.34.4", "tokenizers==0.22.0", "safetensors==0.6.2",
                "sentencepiece>=0.2.1,<0.3", "protobuf>=4,<7", "lpips==0.1.4",
                "numpy>=1.26,<3", "Pillow>=10", "scipy", "tqdm", "packaging", "psutil",
                "requests", "filelock", "fsspec", "pyyaml", "regex", "importlib-metadata",
                "typing-extensions", "hf-xet>=1.1.3,<2", "httpx", "httpcore", "h11", "anyio",
                "sniffio", "certifi", "idna", "urllib3", "charset-normalizer", "zipp", "setuptools>=68", "wheel"]


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def roster(repo):
    path = Path(repo)/"configs/self_reference_v1_sources.json"
    data = json.loads(path.read_text())
    rows = data["sources"]
    expected = {"train": 256, "dev": 64, "model_selection": 64, "calibration": 128, "test": 128}
    if len(rows) != 640 or {s: sum(r["split"] == s for r in rows) for s in expected} != expected:
        raise ValueError("frozen source counts differ")
    if len({r["source_id"] for r in rows}) != 640 or len({r["prompt"] for r in rows}) != 640:
        raise ValueError("duplicate source or prompt")
    owners = {}
    for row in rows:
        if row["source_group"] != row["template_family"]:
            raise ValueError("source_group must be the template family for cluster CI")
        family = row["template_family"]
        if family in owners and owners[family] != row["split"]:
            raise ValueError("template family crosses splits")
        owners[family] = row["split"]
    if len(owners) != 80:
        raise ValueError("expected 80 families")
    return data, file_sha(path)
