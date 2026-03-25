"""
测试用例：embed 侧 LF 参数解析的 ecc 双语义兼容。

功能说明：
- 验证 ecc="sparse_ldpc" 时，embed 参数构建不会抛错且采用 image 路径默认 redundancy。
- 验证 ecc=int 时，embed 参数构建保持 legacy redundancy 语义。
"""

from __future__ import annotations

from typing import Any, Dict

from main.watermarking.embed import orchestrator as embed_orchestrator


def test_embed_lf_params_accept_sparse_ldpc_mode() -> None:
    """
    功能：验证 sparse_ldpc 模式不会触发 int 转换异常。

    Verify that sparse_ldpc mode does not trigger int-cast failures in embed LF params.

    Args:
        None.

    Returns:
        None.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "lf": {
                "ecc": "sparse_ldpc",
                "strength": 1.5,
                "variance": 1.5,
            }
        }
    }

    params = embed_orchestrator._build_lf_image_embed_params(cfg)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(params, dict)
    assert params.get("redundancy") == 1
    assert params.get("alpha") == 1.5


def test_embed_lf_params_keep_int_ecc_redundancy() -> None:
    """
    功能：验证 legacy int ecc 仍映射为 redundancy。

    Verify that legacy int ecc is still mapped to redundancy in embed LF params.

    Args:
        None.

    Returns:
        None.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "lf": {
                "ecc": 3,
                "strength": 1.0,
                "variance": 2.0,
            }
        }
    }

    params = embed_orchestrator._build_lf_image_embed_params(cfg)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(params, dict)
    assert params.get("redundancy") == 3
    assert params.get("alpha") == 1.0
    assert params.get("variance") == 2.0
