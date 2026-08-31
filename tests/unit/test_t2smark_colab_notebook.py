"""Static contract checks for the unexecuted T2SMark Colab canary handoff."""

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/baseline_v1_t2smark_colab_canary.ipynb")


def test_t2smark_canary_notebook_contract() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "".join(code[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code)
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for required in [
        "https://github.com/0xD009/T2SMark.git", "0c1fbfd50fcd1fba135477a2c016e284d5d7914d",
        "stabilityai/stable-diffusion-3.5-medium", "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        "num_inference_steps=NUM_INFERENCE_STEPS", "GUIDANCE_SCALE, NUM_INFERENCE_STEPS, NUM_INVERSION_STEPS = 4.0, 40, 10",
        "torch.cuda.is_available()", "InversionDiffusion3Pipeline.from_pretrained", "get_image_latents(image, sample=False)",
        "naive_forward_diffusion", "norm1_w", "torch.Generator()", "torch.randperm", "math.erfc",
        "clean_no_attack", "jpeg_q50", "resize_50_bicubic_restore", "center_crop_80_restore",
        "gaussian_blur_sigma_1px", "rotation_10_bicubic_reflect_center_crop_v1", "planned_observations, rows, artifacts = 12",
        "exist_ok=False", "engineering_canary_complete", "local_adapter_exact",
    ]:
        assert required in text
    assert "force_remount" not in text and "rmtree" not in text and "rm -" not in text
    assert "mock" not in text.lower() and "placeholder" not in text.lower()
    assert "TPR" in text and "FPR" in text and "threshold" in text


def test_notebook_embeds_t2s_a_formula_markers() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    source = Path("src/cegwm/baselines/t2smark.py").read_text(encoding="utf-8")
    for marker in ["class T2SMarkCodec", "def embed_t2smark_sd35", "def score_t2smark_rgb", "torch.Generator()", "sorted=False"]:
        assert marker in text
        assert marker in source
