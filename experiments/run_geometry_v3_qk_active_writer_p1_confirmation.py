"""Bounded CLI for Geometry-V3 P1 independent active-anchor confirmation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from cegwm.geometry_v3.confirmation import execute_plan, load_real_pipeline, validate_p0_source


MAX_PLAN_BYTES = 32_768
MAX_CONTROL_BYTES = 1_024
TOKEN_ENV = "HF_TOKEN"
KEY_ENV = "CEGWM_GEOMETRY_KEY"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_plan(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_PLAN_BYTES:
        raise ValueError("P1 plan size differs")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError("P1 plan read is incomplete")
    value = json.loads(raw)
    if not isinstance(value, dict) or len(_json_bytes(value)) > MAX_PLAN_BYTES:
        raise ValueError("P1 plan root or canonical size differs")
    return value


def _git_exact(expected: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout
    if exact != expected or dirty:
        raise RuntimeError("P1 checkout identity or cleanliness differs")
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
        raise RuntimeError("P1 control receipt exceeds the public bound")
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
            raise ValueError("P1 expected exact is missing")
        stage = "source_validation"
        source_value = plan.get("source_directory")
        if not isinstance(source_value, str):
            raise ValueError("P1 source directory is missing")
        source_identity = validate_p0_source(Path(source_value))
        _git_exact(expected)
        stage = "execution"
        try:
            control = execute_plan(
                plan, geometry_key=key, hf_token=token,
                source_identity=source_identity, preloader=preloader,
            )
        finally:
            token = ""
            key = ""
        stage = "control_channel"
        _emit(args.control_fd, {
            "status": "success", "p1_status": control["status"],
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
