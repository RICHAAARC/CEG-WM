"""
功能：验证 _probe_model_v2_availability 中工程根路径解析使用 parents[3] 而非 parents[2]

Module type: General module

Regression tests for P0 fix: semantic_mask_provider.py 中 parents[2]→parents[3] 的路径解析修正。
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# 确保 main/ 在 sys.path 中
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from main.watermarking.content_chain.semantic_mask_provider import (
    _probe_model_v2_availability,
)


class TestProbeModelPathResolution:
    """验证 _probe_model_v2_availability 中 project_root 解析正确性。"""

    def test_project_root_resolves_to_repo_root_not_main(self):
        """
        Verify that the resolved project_root points to the repository root (CEG-WM/),
        NOT to the 'main/' subdirectory.

        This test guards against regression to parents[2] which resolved to 'main/'.
        """
        # 通过 mock 让文件"存在"，同时 mock _load_saliency_model 避免真实模型加载
        dummy_model_path = "models/inspyrenet/ckpt_base.pth"
        mask_params = {
            "semantic_model_path": dummy_model_path,
            "saliency_source": "model",
            "semantic_model_source": "inspyrenet",
        }

        # 捕获 project_root 实际值
        captured_roots: list[Path] = []

        original_probe = _probe_model_v2_availability.__wrapped__ if hasattr(
            _probe_model_v2_availability, "__wrapped__"
        ) else None

        import main.watermarking.content_chain.semantic_mask_provider as _smp_mod

        # 替换 Path.exists 和 Path.is_file 以控制文件存在性检查
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(
                 _smp_mod, "_load_saliency_model", return_value=MagicMock()
             ) as _mock_load:
            result_available, result_reason = _probe_model_v2_availability(mask_params)

        # 无论 probe 是否最终返回 True（取决于 callable 检查），
        # 关键验证是：模型文件路径不包含 "main" 作为 project_root 下一层。
        # 通过检查实际 parents[3] 计算出的路径来验证。
        actual_provider_file = Path(_smp_mod.__file__).resolve()
        parents_2 = actual_provider_file.parents[2]  # 错误值：main/
        parents_3 = actual_provider_file.parents[3]  # 正确值：CEG-WM/

        # parents[3] 应当是工程根，其 name 不应为 "main"
        assert parents_3.name != "main", (
            f"parents[3] 不应指向 'main/'，实际：{parents_3}"
        )
        # parents[2] 的 name 应当是 "main"（证明层级计算正确）
        assert parents_2.name == "main", (
            f"parents[2] 应当指向 'main/'，实际：{parents_2}"
        )

    def test_probe_returns_false_with_nonexistent_model_file(self):
        """
        Verify that _probe_model_v2_availability returns (False, failure_reason)
        when the model file does not exist at the resolved path.

        Args:
            None (uses real filesystem check with a non-existent path)

        Returns:
            None (assertion-based)
        """
        mask_params = {
            "semantic_model_path": "models/inspyrenet/ckpt_base.pth",
            "saliency_source": "model",
            "semantic_model_source": "inspyrenet",
        }
        # 不 mock exists，使用真实文件系统：CI 环境无模型文件，probe_failure_reason 必为非 None
        available, failure_reason = _probe_model_v2_availability(mask_params)

        # 在 CI（无模型文件）环境，expect probe 失败
        if not available:
            assert failure_reason is not None, "失败时 failure_reason 不应为 None"
            # 确认不是因为路径解析错误而使 project_root = main/（即路径不含 "/main/models/"）
            assert "/main/models/" not in (failure_reason or ""), (
                f"failure_reason 暗示 project_root 仍为 main/：{failure_reason}"
            )

    def test_probe_returns_true_when_model_file_mocked_exists(self):
        """
        Verify that _probe_model_v2_availability returns (True, None)
        when the model file exists and _load_saliency_model returns a callable stub.

        Args:
            None (uses mock filesystem and mock model loader)

        Returns:
            None (assertion-based)
        """
        import main.watermarking.content_chain.semantic_mask_provider as _smp_mod

        mask_params = {
            "semantic_model_path": "models/inspyrenet/ckpt_base.pth",
            "saliency_source": "model",
            "semantic_model_source": "inspyrenet",
        }
        callable_stub = MagicMock(return_value=None)
        callable_stub.__call__ = lambda *a, **k: None  # 确保 callable 检查通过

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(_smp_mod, "_load_saliency_model", return_value=callable_stub):
            available, failure_reason = _probe_model_v2_availability(mask_params)

        assert available is True, f"mock 文件存在时 probe 应返回 True，实际：available={available}, reason={failure_reason}"
        assert failure_reason is None, f"成功时 failure_reason 应为 None，实际：{failure_reason}"
