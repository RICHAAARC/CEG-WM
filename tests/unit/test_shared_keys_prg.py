from __future__ import annotations

import numpy as np
import pytest

from cegwm.shared.keys import normalize_detection_key, public_key_digest
from cegwm.shared.prg import prg_bytes, prg_normal, prg_rademacher


@pytest.mark.unit
def test_text_key_normalization_has_one_public_identity() -> None:
    composed = "stage-a-detection-kéy-001"
    decomposed = "stage-a-detection-ke\u0301y-001"

    assert normalize_detection_key(composed) == normalize_detection_key(decomposed)
    assert public_key_digest(composed) == public_key_digest(decomposed)
    assert len(public_key_digest(composed)) == 64


@pytest.mark.unit
def test_key_validation_does_not_trim_or_accept_short_material() -> None:
    assert normalize_detection_key(b" 0123456789abcdef ") != normalize_detection_key(
        b"0123456789abcdef"
    )
    with pytest.raises(ValueError, match="at least 16"):
        normalize_detection_key("short")
    with pytest.raises(ValueError, match="control"):
        normalize_detection_key("0123456789abcdef\n")


@pytest.mark.unit
def test_prg_is_deterministic_and_domain_separated_without_global_rng() -> None:
    key = b"0123456789abcdef0123456789abcdef"
    before = np.random.get_state()

    first = prg_bytes(key, "hf/carrier/unit-0001", 96)
    second = prg_bytes(key, "hf/carrier/unit-0001", 96)
    other = prg_bytes(key, "lf/carrier/unit-0001", 96)
    rademacher = prg_rademacher(key, "lf/carrier/unit-0001", (4, 7))
    normal = prg_normal(key, "hf/carrier/unit-0001", (4, 7))
    after = np.random.get_state()

    assert first == second
    assert first != other
    assert set(np.unique(rademacher)) == {-1.0, 1.0}
    assert normal.shape == (4, 7)
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.unit
def test_public_identity_and_prg_have_fixed_vectors() -> None:
    key = b"0123456789abcdef0123456789abcdef"

    assert public_key_digest(key) == (
        "2ffbf244826d00bd72d120cbb5a25e1d92a7f01a6f0672cadf8ef3aa2ec5a0ea"
    )
    assert prg_bytes(key, "hf/carrier/unit-0001", 32).hex() == (
        "f2c0c52a345e1607649d2618bf279d434386808ba7f0ca3b27bef7edd5ac9a54"
    )


@pytest.mark.unit
def test_array_prg_binds_kind_shape_and_explicit_domain() -> None:
    key = b"abcdef0123456789abcdef0123456789"

    one = prg_rademacher(key, "candidate-a/unit-1", (32,), dtype=np.float64)
    two = prg_rademacher(key, "candidate-a/unit-2", (32,), dtype=np.float64)
    reshaped = prg_rademacher(key, "candidate-a/unit-1", (4, 8), dtype=np.float64)

    assert not np.array_equal(one, two)
    assert not np.array_equal(one, reshaped.ravel())
