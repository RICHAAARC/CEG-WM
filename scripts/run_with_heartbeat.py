#!/usr/bin/env python3
"""
长耗时命令心跳执行器

功能说明：
- 运行任意外部命令并透传其标准输出/错误输出。
- 在命令执行期间按固定间隔打印 heartbeat，避免长时间静默看起来像卡住。
- 保留并返回子进程真实退出码，不改变原始命令语义。

Module type: General module
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """
    功能：解析命令行参数。

    Parse command-line arguments for heartbeat runner.

    Args:
        argv: Raw argument list without program name.

    Returns:
        Parsed argparse namespace.

    Raises:
        TypeError: If argv is not a list of strings.
    """
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        # argv 类型非法，必须 fail-fast。
        raise TypeError("argv must be List[str]")

    parser = argparse.ArgumentParser(
        description="Run a command with periodic heartbeat logs."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Heartbeat interval in seconds. Default: 10.0",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Optional working directory for child process.",
    )
    parser.add_argument(
        "--show-command",
        action="store_true",
        help="Print full command before execution.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Use '--' before command, e.g. -- python -m pytest ...",
    )
    return parser.parse_args(argv)


def _normalize_command(command_parts: List[str]) -> List[str]:
    """
    功能：标准化命令参数列表。

    Normalize command parts parsed by argparse.REMAINDER.

    Args:
        command_parts: Raw remainder arguments.

    Returns:
        Normalized command list.

    Raises:
        TypeError: If command_parts is invalid.
        ValueError: If command is empty.
    """
    if not isinstance(command_parts, list) or any(not isinstance(item, str) for item in command_parts):
        # command_parts 类型非法，必须 fail-fast。
        raise TypeError("command_parts must be List[str]")

    normalized = list(command_parts)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]

    if not normalized:
        # 命令为空时无法执行，必须 fail-fast。
        raise ValueError("command must not be empty; pass command after '--'")
    return normalized


def _resolve_cwd(cwd_value: Optional[str]) -> Optional[Path]:
    """
    功能：解析并校验工作目录。

    Resolve and validate optional working directory.

    Args:
        cwd_value: Raw cwd string from CLI.

    Returns:
        Resolved Path or None.

    Raises:
        TypeError: If cwd_value type is invalid.
        ValueError: If cwd path does not exist or is not a directory.
    """
    if cwd_value is None:
        return None
    if not isinstance(cwd_value, str):
        # cwd 类型非法，必须 fail-fast。
        raise TypeError("cwd must be str or None")

    cwd_path = Path(cwd_value).resolve()
    if not cwd_path.exists() or not cwd_path.is_dir():
        # cwd 不存在或不是目录，必须 fail-fast。
        raise ValueError(f"invalid cwd: {cwd_path}")
    return cwd_path


def _run_command_with_heartbeat(
    command: List[str],
    heartbeat_interval_seconds: float,
    working_dir: Optional[Path],
    show_command: bool,
) -> int:
    """
    功能：执行命令并按固定间隔输出心跳。

    Execute a child process and print heartbeat lines periodically.

    Args:
        command: Command and arguments.
        heartbeat_interval_seconds: Heartbeat interval in seconds.
        working_dir: Optional working directory.
        show_command: Whether to print command before execution.

    Returns:
        Child process return code.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If interval is not positive.
    """
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        # command 类型非法，必须 fail-fast。
        raise TypeError("command must be List[str]")
    if not isinstance(heartbeat_interval_seconds, (int, float)):
        # interval 类型非法，必须 fail-fast。
        raise TypeError("heartbeat_interval_seconds must be numeric")
    if heartbeat_interval_seconds <= 0:
        # interval 非正数时语义不成立，必须 fail-fast。
        raise ValueError("heartbeat_interval_seconds must be > 0")
    if working_dir is not None and not isinstance(working_dir, Path):
        # working_dir 类型非法，必须 fail-fast。
        raise TypeError("working_dir must be Path or None")
    if not isinstance(show_command, bool):
        # show_command 类型非法，必须 fail-fast。
        raise TypeError("show_command must be bool")

    cwd_for_child = str(working_dir) if working_dir is not None else None
    if show_command:
        print(f"[heartbeat] command: {' '.join(command)}", flush=True)

    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd_for_child,
        stdout=None,
        stderr=None,
        env=os.environ.copy(),
        shell=False,
    )

    print(
        f"[heartbeat] started pid={process.pid} at {datetime.now().isoformat(timespec='seconds')}",
        flush=True,
    )

    next_heartbeat = start + heartbeat_interval_seconds
    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                elapsed = time.monotonic() - start
                print(
                    f"[heartbeat] finished rc={return_code} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                return return_code

            now = time.monotonic()
            if now >= next_heartbeat:
                elapsed = now - start
                print(
                    f"[heartbeat] running pid={process.pid} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                next_heartbeat = now + heartbeat_interval_seconds

            time.sleep(min(0.5, heartbeat_interval_seconds))
    except KeyboardInterrupt:
        # 用户中断时需要显式终止子进程，避免孤儿进程残留。
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            # terminate 失败时强制 kill，避免子进程泄漏。
            process.kill()
        return 130


def main() -> int:
    """
    功能：心跳执行脚本入口。

    Entry point for heartbeat command runner.

    Args:
        None.

    Returns:
        Process exit code.
    """
    args = _parse_args(sys.argv[1:])
    command = _normalize_command(args.command)
    working_dir = _resolve_cwd(args.cwd)
    return _run_command_with_heartbeat(
        command=command,
        heartbeat_interval_seconds=float(args.interval),
        working_dir=working_dir,
        show_command=bool(args.show_command),
    )


if __name__ == "__main__":
    sys.exit(main())
