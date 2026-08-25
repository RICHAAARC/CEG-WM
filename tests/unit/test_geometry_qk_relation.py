import numpy as np
import pytest

from cegwm.geometry.qk_relation import keyed_qk_relation


KEY = "geometry-test-key-0001"
OTHER = "geometry-test-key-0002"


def test_relation_consumes_numeric_qk_values_and_is_keyed() -> None:
    q = np.eye(4)
    k = np.eye(4)
    result = keyed_qk_relation(q, k, KEY, comparison_key=OTHER)
    changed = keyed_qk_relation(np.roll(q, 1, axis=0), k, KEY)
    assert result.relation.shape == (4, 4)
    assert result.gap > 0
    assert result.coverage == 1.0
    assert result.wrong_key_margin != 0.0
    assert result.projection != changed.projection


def test_relation_rejects_shape_only_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError):
        keyed_qk_relation(np.ones((1, 3)), np.ones((1, 3)), KEY)
    with pytest.raises(ValueError):
        keyed_qk_relation(np.array([[np.nan, 1], [1, 1]]), np.ones((2, 2)), KEY)
