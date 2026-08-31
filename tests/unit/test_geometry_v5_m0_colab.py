from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from cegwm.protocol import geometry_v5_m0_colab as colab


_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_execution_contract_is_byte_bound_and_freezes_model_ddim_prompts_grid_and_claim() -> None:
    contract = colab.load_m0_execution_contract(_ROOT)
    raw = (_ROOT / colab.EXECUTION_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == colab.EXECUTION_SHA256
    assert contract.config["source_m0_exact"] == colab.SOURCE_M0_EXACT
    assert contract.config["model"]["model_id"] == "sd2-community/stable-diffusion-2-1-base"
    assert contract.config["scheduler"]["class"] == "DDIMScheduler"
    assert contract.config["generation"]["prompt"] == "manifest_unit_prompt"
    assert contract.config["inversion"]["prompt"] == "" and contract.config["inversion"]["guidance_scale"] == 1.0
    assert contract.config["spectral_grid"]["direction"] == "attacked_to_canonical_spatial"
    assert contract.config["artifacts"]["mode"] == "create_only" and contract.config["artifacts"]["records"] == 44
    assert contract.config["claim_ceiling"] == colab.CLAIM_CEILING


@pytest.mark.unit
def test_runtime_module_has_no_import_time_torch_diffusers_or_model_network_imports() -> None:
    path = _ROOT / "src/cegwm/runtime/geometry_v5_m0_sd21.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "torch" not in imported and "diffusers" not in imported
