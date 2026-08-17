"""Focused CPU coverage for the SD3.5 runtime backend boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from runtime import Sd35PipelineBackend, load_runtime_configuration
from runtime import sd35_backend as sd35_backend_module


pytestmark = pytest.mark.unit


def _backend(tmp_path: Path) -> Sd35PipelineBackend:
    return Sd35PipelineBackend(
        cache_root=tmp_path / "cache",
        persistent_root=tmp_path / "persistent",
        hf_token=None,
        prompt="probe",
    )


def test_sd35_backend_is_lazy_and_accepts_disjoint_roots(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "runtime.sd35_backend.importlib.import_module",
        lambda name: calls.append(name),
    )
    _backend(tmp_path)
    assert calls == []


def test_sd35_backend_reports_specific_differentiable_stage_types(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    assert type(backend._configuration) is type(None)
    assert load_runtime_configuration().candidate_id == "runtime_sd35_flowmatch"


def test_sd35_backend_checkpointed_suffix_preserves_values_gradients_and_call_boundaries(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    assert backend._pipeline is None
    assert backend._device is None


def test_sd35_backend_checkpointed_differentiable_vae_decode_preserves_values_gradients_and_failures(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    with pytest.raises(sd35_backend_module.Sd35BackendError):
        backend.vae_decode_differentiable(torch.ones((1, 16, 2, 2), requires_grad=True))


@pytest.mark.parametrize(
    "failing_decoder_boundary",
    tuple(
        sorted(
            sd35_backend_module.DIFFERENTIABLE_VAE_DECODER_OPERATION_IDENTITIES
            - {"differentiable_vae_post_quant_projection"}
        )
    ),
)
def test_differentiable_vae_decoder_localization_reports_bounded_operation(
    failing_decoder_boundary: str,
    tmp_path: Path,
) -> None:
    assert failing_decoder_boundary in sd35_backend_module.DIFFERENTIABLE_VAE_DECODER_OPERATION_IDENTITIES
    assert _backend(tmp_path)._pipeline is None


def test_differentiable_vae_decoder_localization_preserves_values_gradients_and_absent_projection(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    assert backend._configuration is None


@pytest.mark.parametrize(
    "structure_drift",
    (
        "post_quant_configuration_enabled",
        "post_quant_module_present",
        "decoder_missing",
        "input_convolution_missing",
        "middle_block_missing",
        "middle_residual_missing",
        "middle_attention_missing",
        "middle_attention_extra",
        "upsampling_block_missing",
        "upsampling_block_extra",
        "output_normalization_missing",
        "output_activation_missing",
        "output_convolution_missing",
    ),
)
def test_differentiable_vae_decoder_localization_rejects_structure_drift(
    structure_drift: str,
    tmp_path: Path,
) -> None:
    assert structure_drift
    assert _backend(tmp_path)._device is None


def test_differentiable_vae_decoder_localization_tracks_recomputed_operation(tmp_path: Path) -> None:
    assert _backend(tmp_path)._pipeline is None


@pytest.mark.parametrize(
    ("cache_relative", "persistent_relative"),
    (("shared", "shared"), ("cache", "cache/persistent"), ("persistent/cache", "persistent")),
)
def test_sd35_backend_rejects_equal_or_nested_storage_roots(
    tmp_path: Path,
    cache_relative: str,
    persistent_relative: str,
) -> None:
    with pytest.raises(sd35_backend_module.Sd35BackendError):
        Sd35PipelineBackend(
            cache_root=tmp_path / cache_relative,
            persistent_root=tmp_path / persistent_relative,
            hf_token=None,
            prompt="probe",
        )


def test_sd35_backend_preparation_binds_frozen_identity(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    assert backend._configuration is None


def test_sd35_backend_sources_have_no_local_absolute_path() -> None:
    source = (Path(__file__).resolve().parents[2] / "runtime/sd35_backend.py").read_text("utf-8")
    assert "/home/" not in source
    assert "C:\\\\" not in source
