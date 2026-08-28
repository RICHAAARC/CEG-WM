"""Bounded CLI for the Geometry-V3 P1M0 posterior mechanism audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from cegwm.geometry_v3.mechanism_audit import (
    execute_plan,
    load_real_pipeline,
    validate_sources,
)


MAX_PLAN_BYTES = 32_768
MAX_CONTROL_BYTES = 1_024
TOKEN_ENV = "HF_TOKEN"
KEY_ENV = "CEGWM_GEOMETRY_KEY"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_plan(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("P1M0 plan is not a regular file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_PLAN_BYTES:
        raise ValueError("P1M0 plan size differs")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or len(_json_bytes(value)) > MAX_PLAN_BYTES:
        raise ValueError("P1M0 plan root or canonical size differs")
    return value


def _git_exact(expected: str) -> str:
    root = Path(__file__).resolve().parents[1]
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout
    if exact != expected or dirty:
        raise RuntimeError("P1M0 checkout identity or cleanliness differs")
    return exact


def _error_class(error: BaseException) -> str:
    if isinstance(error, (TypeError, ValueError, json.JSONDecodeError)):
        return "validation_error"
    if isinstance(error, FileExistsError):
        return "artifact_exists"
    return "runtime_error"


def _emit(fd: int, value: dict[str, Any]) -> None:
    payload = _json_bytes(value)
    if len(payload) > MAX_CONTROL_BYTES:
        raise RuntimeError("P1M0 control receipt exceeds bound")
    os.write(fd, payload)


def _main(
    argv: list[str] | None = None,
    *,
    preloader: Callable[[str, str], Any] = load_real_pipeline,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--control-fd", required=True, type=int)
    args = parser.parse_args(argv)
    stage = "plan"
    token = os.environ.pop(TOKEN_ENV, "")
    key = os.environ.pop(KEY_ENV, "")
    try:
        plan = _read_plan(Path(args.plan))
        expected = plan.get("expected_exact")
        if not isinstance(expected, str):
            raise ValueError("P1M0 expected exact is missing")
        stage = "source_validation"
        p0_value = plan.get("p0_source_directory")
        p1_value = plan.get("p1_source_directory")
        if not isinstance(p0_value, str) or not isinstance(p1_value, str):
            raise ValueError("P1M0 source directories are missing")
        sources = validate_sources(Path(p0_value), Path(p1_value))
        _git_exact(expected)
        stage = "mechanism_audit"
        try:
            control = execute_plan(
                plan, geometry_key=key, hf_token=token,
                sources=sources, preloader=preloader,
            )
        finally:
            token = ""
            key = ""
        stage = "control_channel"
        _emit(args.control_fd, {
            "status": "success", "p1m0_status": control["status"],
            **{name: value for name, value in control.items() if name != "status"},
        })
        return 0
    except Exception as error:  # noqa: BLE001 - finite public receipt only
        token = ""
        key = ""
        if stage == "control_channel":
            return 1
        try:
            _emit(args.control_fd, {
                "status": "failure", "failure_point": stage,
                "error_class": _error_class(error), "science_denominator": 0,
            })
        except Exception:  # noqa: BLE001 - no stdout/stderr fallback
            pass
        return 1
    finally:
        try:
            os.close(args.control_fd)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(_main())
