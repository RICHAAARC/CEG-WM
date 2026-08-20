from __future__ import annotations

from main import content_embedder, contrastive_lf_carrier
import pytest
from struct import pack, unpack


pytestmark = pytest.mark.quick


def test_real_contrastive_carrier_enters_existing_three_over_two_fifty_embedder() -> None:
    shape = (1, 16, 3, 3)
    carrier = contrastive_lf_carrier(
        "functional-stage-a-root",
        shape,
        candidate_id="lf_multiscale_lowpass_contrastive",
    )
    latent = tuple(1.0 + index / 1000.0 for index in range(144))
    result = content_embedder(
        latent, lf_carrier_result=carrier.as_embedding_carrier()
    )
    assert result.mode == "lf_only"
    assert result.target_relative_l2 == unpack(">f", pack(">f", 3 / 250))[0]
    assert result.lf_carrier_config_digest == carrier.carrier_config_digest
    assert result.active_hf_direction is None
