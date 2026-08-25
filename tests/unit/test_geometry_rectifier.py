import numpy as np

from cegwm.geometry.rectifier import inverse_rectify


def test_inverse_rectify_identity_and_support_mask() -> None:
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    result = inverse_rectify(image, np.eye(3), (4, 4))
    assert np.array_equal(result.image, image)
    assert result.valid_support.all()


def test_inverse_rectify_marks_deleted_crop_content_without_inpaint() -> None:
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    h = np.array(((1, 0, -1), (0, 1, 0), (0, 0, 1)), dtype=float)
    result = inverse_rectify(image, h, (3, 3))
    assert not result.valid_support[:, 0].any()
    assert np.all(result.image[:, 0] == 0)
