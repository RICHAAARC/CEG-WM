import math

import numpy as np
import pytest

from cegwm.geometry.transform import apply_h, estimate_bounded_similarity, transform_corners


POINTS = np.array(((2, 2), (12, 2), (12, 12), (2, 12), (7, 7)), dtype=float)


def _h(angle=0.0, scale=1.0, tx=0.0, ty=0.0):
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    return np.array(((c, -s, tx), (s, c, ty), (0, 0, 1)), dtype=float)


@pytest.mark.parametrize("angle,scale,tx,ty", [(0, 1, 0, 0), (0.2, 1, 3, -2), (-0.15, 1.4, 4, 5), (0.1, 0.7, 2, 3)])
def test_similarity_recovers_identity_rotation_scale_translation_and_crop_rescale(angle, scale, tx, ty) -> None:
    target = apply_h(POINTS, _h(angle, scale, tx, ty))
    fitted = estimate_bounded_similarity(POINTS, target, (20, 20))
    assert np.allclose(apply_h(POINTS, fitted.h_canonical_to_observed), target, atol=1e-8)
    assert fitted.residual < 1e-8


def test_similarity_recovers_flip_and_h_direction_corners() -> None:
    flip = np.array(((-1, 0, 18), (0, 1, 1), (0, 0, 1)), dtype=float)
    fitted = estimate_bounded_similarity(POINTS, apply_h(POINTS, flip), (20, 20))
    assert np.allclose(apply_h(POINTS, fitted.h_canonical_to_observed), apply_h(POINTS, flip))
    assert np.allclose(transform_corners(fitted.h_canonical_to_observed, (20, 20)), apply_h(np.array(((0,0),(19,0),(19,19),(0,19))), flip))


def test_similarity_recovers_d4_ninety_degree_rotation() -> None:
    quarter_turn = np.array(((0, -1, 18), (1, 0, 0), (0, 0, 1)), dtype=float)
    fitted = estimate_bounded_similarity(POINTS, apply_h(POINTS, quarter_turn), (20, 20))
    assert fitted.d4_index == 1
    assert np.allclose(apply_h(POINTS, fitted.h_canonical_to_observed), apply_h(POINTS, quarter_turn))


def test_out_of_bounds_and_ambiguity_are_reported_not_hidden() -> None:
    moved = apply_h(POINTS, _h(tx=100, ty=100))
    fitted = estimate_bounded_similarity(POINTS, moved, (20, 20), max_translation=200)
    assert not fitted.valid_corners
    with pytest.raises(ValueError):
        estimate_bounded_similarity(POINTS, moved, (20, 20), max_translation=10)


def test_d4_boundary_rotation_candidates_are_marked_ambiguous() -> None:
    target = apply_h(POINTS, _h(angle=math.pi / 4, tx=1, ty=1))
    fitted = estimate_bounded_similarity(POINTS, target, (20, 20), ambiguity_tolerance=1e-8)
    assert fitted.uniqueness_gap <= 1e-8
    assert not fitted.valid_corners


def test_asymmetric_reflection_uses_one_d4_family_without_negative_scale_alias() -> None:
    asymmetric = np.array(((2, 3), (13, 4), (9, 15), (4, 11), (11, 9)), dtype=float)
    reflection = np.array(((-1, 0, 39), (0, 1, 0), (0, 0, 1)), dtype=float)
    fitted = estimate_bounded_similarity(asymmetric, apply_h(asymmetric, reflection), (40, 40))
    d4_determinants = (1, 1, 1, 1, -1, -1, -1, -1)
    residual_determinant = np.linalg.det(fitted.h_canonical_to_observed[:2, :2]) / d4_determinants[fitted.d4_index]
    assert fitted.scale > 0
    assert residual_determinant > 0
    assert fitted.uniqueness_gap > 1e-8
    assert fitted.valid_corners


def test_crop_uses_visible_correspondences_and_reports_coverage() -> None:
    visible = POINTS[:4]
    target = apply_h(visible, _h(scale=0.75, tx=2, ty=2))
    fitted = estimate_bounded_similarity(visible, target, (20, 20), total_reference_points=len(POINTS))
    assert fitted.coverage == 0.8
    assert np.allclose(apply_h(visible, fitted.h_canonical_to_observed), target)
