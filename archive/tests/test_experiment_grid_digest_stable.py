"""
文件目的：验证实验矩阵 digest 稳定性。
Module type: General module
"""

from __future__ import annotations

from main.evaluation.experiment_matrix import build_experiment_grid


def test_experiment_grid_digest_stable() -> None:
    """
    功能：验证同一输入可复现相同 grid_item_digest。

    Assert stable grid_item_digest values for identical input config.

    Args:
        None.

    Returns:
        None.
    """
    base_cfg = {
        "model_id": "model_a",
        "seed": 7,
        "experiment_matrix": {
            "models": ["model_a", "model_b"],
            "seeds": [11, 13],
            "attack_protocol_families": ["rotate", "resize"],
            "ablation_variants": [
                {"enable_geometry": True},
                {"enable_geometry": False},
            ],
        },
    }

    grid_1 = build_experiment_grid(base_cfg)
    grid_2 = build_experiment_grid(base_cfg)

    digest_list_1 = [item["grid_item_digest"] for item in grid_1]
    digest_list_2 = [item["grid_item_digest"] for item in grid_2]

    assert digest_list_1 == digest_list_2
    assert len(digest_list_1) == 16
