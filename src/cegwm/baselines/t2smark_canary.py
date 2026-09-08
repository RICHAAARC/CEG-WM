"""Existing official T2SMark runtime loader, without canary machinery."""
from pathlib import Path
from typing import Any
import sys

def load_t2smark_sd35_pipeline(
    official_source: Path,
    *,
    model_id: str,
    model_revision: str,
    hf_token: str,
) -> Any:
    """Load the pinned official SD3.5 pipeline used by the proven canary path."""

    if not isinstance(hf_token, str) or not hf_token.strip():
        raise RuntimeError("HF_TOKEN is required")
    source_text = str(official_source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import torch
    from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline

    pipeline = InversionDiffusion3Pipeline.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.float16,
        token=hf_token,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline

