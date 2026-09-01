from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
import pytest
import torch

from cegwm.geometry_v7.contracts import GeometryStatus
from cegwm.geometry_v7.syncseal import (
    SYNCSEAL_TORCHSCRIPT_URL,
    SyncSealTorchScript,
    download_official_syncseal_torchscript,
)


class _TorchScriptContractFixture:
    """Shape/call tracer only; it is not a SyncSeal implementation."""

    def __init__(
        self,
        *,
        malformed: bool = False,
        preds_w_shape: tuple[int, ...] = (1, 1, 512, 512),
        nonfinite_preds_w: bool = False,
    ) -> None:
        self.malformed = malformed
        self.preds_w_shape = preds_w_shape
        self.nonfinite_preds_w = nonfinite_preds_w
        self.embed_calls = 0
        self.detect_calls = 0

    def to(self, device: torch.device) -> "_TorchScriptContractFixture":
        return self

    def eval(self) -> "_TorchScriptContractFixture":
        return self

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        self.embed_calls += 1
        official_residual = torch.full(
            (1, 1, 512, 512), 1.0 / 255.0, dtype=image.dtype, device=image.device
        )
        residual = torch.full(
            self.preds_w_shape, 1.0 / 255.0, dtype=image.dtype, device=image.device
        )
        if self.nonfinite_preds_w:
            residual.reshape(-1)[0] = torch.nan
        return {
            "preds_w": residual,
            "imgs_w": torch.clamp(image + official_residual, 0.0, 1.0),
        }

    def detect(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        self.detect_calls += 1
        if self.malformed:
            return {"preds": torch.zeros((1, 8)), "preds_pts": torch.zeros((1, 8))}
        hi = 127.0 / 128.0
        points = torch.tensor([[-1.0, -1.0, hi, -1.0, hi, hi, -1.0, hi]])
        return {"preds": torch.cat((torch.tensor([[0.25]]), points), dim=1), "preds_pts": points}


@pytest.mark.unit
def test_official_embed_adapter_scales_only_the_final_rgb_residual() -> None:
    fixture = _TorchScriptContractFixture()
    adapter = SyncSealTorchScript(fixture)
    source = Image.new("RGB", (512, 512), (100, 100, 100))
    output = adapter.embed_final_rgb(source, 0.5)
    assert fixture.embed_calls == 1
    # Half of a one-code-value residual rounds back to the original byte.
    assert output.getpixel((0, 0)) == (100, 100, 100)
    assert source.getpixel((0, 0)) == (100, 100, 100)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("shape", "nonfinite", "message"),
    (
        ((2, 1, 512, 512), False, "1x1x512x512"),
        ((1, 1, 511, 512), False, "1x1x512x512"),
        ((1, 1, 512, 512), True, "finite floating point"),
    ),
)
def test_embed_rejects_wrong_batch_wrong_spatial_and_nonfinite_preds_w(
    shape: tuple[int, ...], nonfinite: bool, message: str
) -> None:
    adapter = SyncSealTorchScript(
        _TorchScriptContractFixture(preds_w_shape=shape, nonfinite_preds_w=nonfinite)
    )
    with pytest.raises(ValueError, match=message):
        adapter.embed_final_rgb(Image.new("RGB", (512, 512)), 1.0)


@pytest.mark.unit
def test_official_detect_adapter_names_logit_and_validates_all_nine_values() -> None:
    fixture = _TorchScriptContractFixture()
    estimate = SyncSealTorchScript(fixture).detect_geometry(Image.new("RGB", (512, 512)))
    assert fixture.detect_calls == 1
    assert estimate.status is GeometryStatus.UNRELIABLE
    assert estimate.uncalibrated_sync_logit == pytest.approx(0.25)
    assert estimate.raw_syncseal_corners == (
        (-1.0, -1.0), (127.0 / 128.0, -1.0),
        (127.0 / 128.0, 127.0 / 128.0), (-1.0, 127.0 / 128.0),
    )
    assert estimate.homography_current_to_canonical == (
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    )
    malformed = SyncSealTorchScript(_TorchScriptContractFixture(malformed=True)).detect_geometry(
        Image.new("RGB", (512, 512))
    )
    assert malformed.status is GeometryStatus.ERROR
    assert "1x9" in (malformed.error or "")


@pytest.mark.unit
def test_official_url_download_is_create_only(tmp_path: Path) -> None:
    seen: list[str] = []

    def opener(url: str) -> io.BytesIO:
        seen.append(url)
        return io.BytesIO(b"torchscript-placeholder")

    destination = tmp_path / "syncmodel.jit.pt"
    assert download_official_syncseal_torchscript(destination, opener=opener) == destination
    assert destination.read_bytes() == b"torchscript-placeholder"
    assert seen == [SYNCSEAL_TORCHSCRIPT_URL]
    with pytest.raises(FileExistsError):
        download_official_syncseal_torchscript(destination, opener=opener)
    assert destination.read_bytes() == b"torchscript-placeholder"
