"""
功能：records_io 原子写盘在 Windows WinError 5 下的受限回退测试

Module type: General module

验证边界：
1. 主路径 replace 成功时正常写入。
2. 仅在 Windows + WinError 5 时走回退写入。
3. 非目标异常不回退并继续抛出。
"""

from pathlib import Path
import sys

import pytest

from main.core import records_io


def _build_permission_error_with_winerror(winerror: int) -> PermissionError:
    """
    功能：构造带 winerror 属性的 PermissionError。 

    Build PermissionError carrying a specific winerror value.

    Args:
        winerror: WinError code.

    Returns:
        PermissionError instance.
    """
    error = PermissionError("replace failed")
    error.winerror = winerror
    return error


def _replace_like_success(self: Path, target: Path) -> Path:
    """
    功能：模拟 replace 成功路径，避免依赖当前环境 ACL 行为。

    Simulate a successful replace by copying bytes and removing source.
    """
    target.write_bytes(self.read_bytes())
    self.unlink()
    return target


def test_atomic_replace_write_bytes_primary_path_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：主路径 replace 成功时应完成写入。 

    Verify atomic replace path succeeds without fallback when replace works.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    dst = tmp_path / "artifact.bin"
    payload = b"primary-path-ok"

    monkeypatch.setattr(Path, "replace", _replace_like_success)
    records_io._atomic_replace_write_bytes(dst, payload)

    assert dst.exists()
    assert dst.read_bytes() == payload
    assert not list(tmp_path.glob("*.writing"))


@pytest.mark.skipif(sys.platform != "win32", reason="WinError 5 回退路径仅适用于 Windows")
def test_atomic_replace_write_bytes_winerror_5_fallback_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：Windows WinError 5 时应触发受限回退并写入成功。 

    Verify fallback path writes destination successfully on Windows WinError 5.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    dst = tmp_path / "artifact.bin"
    payload = b"fallback-ok"

    def _raise_winerror_5(self: Path, target: Path) -> Path:
        raise _build_permission_error_with_winerror(5)

    monkeypatch.setattr(records_io.os, "name", "nt", raising=False)
    monkeypatch.setattr(Path, "replace", _raise_winerror_5)

    records_io._atomic_replace_write_bytes(dst, payload)

    assert dst.exists()
    assert dst.read_bytes() == payload
    assert not list(tmp_path.glob("*.writing"))


def test_atomic_replace_write_bytes_non_target_permission_error_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：非 WinError 5 的 PermissionError 必须继续抛出。 

    Verify non-target PermissionError does not trigger fallback.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    dst = tmp_path / "artifact.bin"
    payload = b"should-raise"

    def _raise_winerror_32(self: Path, target: Path) -> Path:
        raise _build_permission_error_with_winerror(32)

    monkeypatch.setattr(records_io.os, "name", "nt", raising=False)
    monkeypatch.setattr(Path, "replace", _raise_winerror_32)

    with pytest.raises(PermissionError):
        records_io._atomic_replace_write_bytes(dst, payload)

    assert not dst.exists()


def test_atomic_replace_write_bytes_mkstemp_error_must_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：临时文件创建失败时必须抛出异常，不允许静默成功。

    Verify mkstemp failure is propagated.
    """
    dst = tmp_path / "artifact.bin"

    def _raise_mkstemp(*_args, **_kwargs):
        raise OSError("mkstemp-failed")

    monkeypatch.setattr(records_io.tempfile, "mkstemp", _raise_mkstemp)

    with pytest.raises(OSError, match="mkstemp-failed"):
        records_io._atomic_replace_write_bytes(dst, b"payload")

    assert not dst.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="WinError 5 回退路径仅适用于 Windows")
def test_atomic_replace_write_bytes_winerror_5_fallback_write_error_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：WinError 5 回退写入失败时必须抛出异常。

    Verify fallback write failure is not swallowed.

    Notes:
        monkeypatch must NOT globally replace os.open with a blanket raiser:
        tempfile._mkstemp_inner on Windows catches PermissionError and retries
        indefinitely when os.access(dir, W_OK) is True, causing an infinite loop.
        Instead, distinguish mkstemp calls (which carry os.O_EXCL) from the
        fallback open (O_WRONLY|O_CREAT|O_TRUNC, no O_EXCL) and only raise for
        the latter.
    """
    import os as _os_module

    dst = tmp_path / "artifact.bin"
    payload = b"fallback-write-error"

    def _raise_winerror_5(self: Path, target: Path) -> Path:
        raise _build_permission_error_with_winerror(5)

    _original_os_open = _os_module.open

    def _raise_on_fallback_open(path, flags, mode: int = 0o666, **kwargs):
        # mkstemp 使用 O_EXCL（独占创建）；回退路径使用 O_WRONLY|O_CREAT|O_TRUNC，不带 O_EXCL
        # 仅对回退路径抛出异常，避免破坏 tempfile._mkstemp_inner 的内部 open 调用
        if flags & _os_module.O_EXCL:
            return _original_os_open(path, flags, mode, **kwargs)
        raise PermissionError("fallback-open-failed")

    monkeypatch.setattr(records_io.os, "name", "nt", raising=False)
    monkeypatch.setattr(Path, "replace", _raise_winerror_5)
    monkeypatch.setattr(records_io.os, "open", _raise_on_fallback_open)

    with pytest.raises(PermissionError, match="fallback-open-failed"):
        records_io._atomic_replace_write_bytes(dst, payload)
