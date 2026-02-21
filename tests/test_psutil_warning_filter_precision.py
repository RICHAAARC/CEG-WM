"""
File purpose: Validate psutil RuntimeWarning filter precision in tests.
Module type: General module
"""

from __future__ import annotations

import warnings

import pytest


def test_non_psutil_runtime_warning_is_not_filtered() -> None:
    with pytest.warns(RuntimeWarning, match="custom_runtime_warning_should_remain_visible"):
        warnings.warn("custom_runtime_warning_should_remain_visible", RuntimeWarning)


def test_psutil_like_message_from_non_psutil_module_is_not_filtered() -> None:
    with pytest.warns(RuntimeWarning, match=r"/proc/vmstat"):
        warnings.warn("mock warning while reading /proc/vmstat", RuntimeWarning)
