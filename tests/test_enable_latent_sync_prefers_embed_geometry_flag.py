"""
File purpose: 验证 resolve_enable_latent_sync 优先读取 embed.geometry.enable_latent_sync
Module type: General module
"""

from __future__ import annotations

import pytest

from main.watermarking.geometry_chain.sync import resolve_enable_latent_sync


def test_enable_latent_sync_prefers_embed_geometry_flag() -> None:
    """
    功能：当同时存在 embed/detect geometry 开关时，embed 优先。

    When both embed.geometry.enable_latent_sync and detect.geometry.enable_latent_sync exist,
    prefer embed.geometry.enable_latent_sync.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If priority is incorrect.
    """
    # (1) embed=True, detect=False → 应返回 True（embed 优先）
    cfg_embed_true_detect_false = {
        "embed": {
            "geometry": {
                "enable_latent_sync": True
            }
        },
        "detect": {
            "geometry": {
                "enable_latent_sync": False
            }
        }
    }
    assert resolve_enable_latent_sync(cfg_embed_true_detect_false) is True
    
    # (2) embed=False, detect=True → 应返回 False（embed 优先）
    cfg_embed_false_detect_true = {
        "embed": {
            "geometry": {
                "enable_latent_sync": False
            }
        },
        "detect": {
            "geometry": {
                "enable_latent_sync": True
            }
        }
    }
    assert resolve_enable_latent_sync(cfg_embed_false_detect_true) is False
    
    # (3) embed 缺失，detect=True → 回退到 detect，应返回 True
    cfg_no_embed_detect_true = {
        "detect": {
            "geometry": {
                "enable_latent_sync": True
            }
        }
    }
    assert resolve_enable_latent_sync(cfg_no_embed_detect_true) is True
    
    # (4) embed=True, detect 缺失 → 应返回 True（embed 唯一有效）
    cfg_embed_true_no_detect = {
        "embed": {
            "geometry": {
                "enable_latent_sync": True
            }
        }
    }
    assert resolve_enable_latent_sync(cfg_embed_true_no_detect) is True
    
    # (5) 都缺失 → 应返回 False
    cfg_empty = {}
    assert resolve_enable_latent_sync(cfg_empty) is False
