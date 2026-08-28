"""Bounded Geometry-V3 P0D1 inner-Q single-configuration writer diagnostic.

The diagnostic reuses the frozen P0 writer implementation and exposes only
finite public stages and integer counters.  It never serializes exception
text, credentials, prompts, tensors, anchors, Q/K, latents, or model state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, NamedTuple

import torch

from cegwm.geometry_v3.active_writer import (
    ActiveQKWriterSession,
    P0_ANCHOR_POINT_COUNT,
    P0_IMAGE_SIZE,
    P0_INFERENCE_STEPS,
    P0_MODEL_ID,
    P0_Q_DIAGNOSTIC_CHECKPOINTS,
    P0_SEED,
    P0WriterConfig,
)
from cegwm.geometry_v3.contracts import derive_canonical_relation_anchor
from cegwm.geometry_v3.operational import P0_PROMPT_TEXT, load_real_pipeline
from cegwm.runtime.observation import require_ordinary_rgb_image
from cegwm.shared.keys import normalize_detection_key


P0D1_PROTOCOL_ID = "geometry-v3-keyed-qk-active-writer-p0d1-v1"
P0D1_CONFIG = P0WriterConfig(4, 0.0025)
P0D1_STATUS_STOPPED = "P0D1_STOPPED"
P0D1_STATUS_COMPLETE = "P0D1_DIAGNOSTIC_COMPLETE"
P0D1_SCIENCE_DENOMINATOR = 0
P0D1_DRIVE_ROOT = "/content/drive/MyDrive/CEG-WM/Geometry-V3/P0D1/"
P0D1_ARTIFACT_MAX_BYTES = 32 * 1024
MAX_PLAN_BYTES = 32_768
MAX_CONTROL_BYTES = 1_024
TOKEN_ENV = "HF_TOKEN"
KEY_ENV = "CEGWM_GEOMETRY_KEY"
PUBLIC_FAILURE_POINTS = (
    "pipeline_load",
    "session_setup",
    "pipeline_callback",
    "session_completion",
    "rgb_validation",
    "none",
)
COUNTER_NAMES = (
    "pipeline_load_count",
    "session_setup_count",
    "pipeline_callback_count",
    "step18_reached_count",
    "transformer_root_call_count",
    "block4_to_q_hook_hit_count",
    "block4_to_k_hook_hit_count",
    "block4_to_q_injection_count",
    "block4_to_k_injection_count",
    *(f"{checkpoint}_count" for checkpoint in P0_Q_DIAGNOSTIC_CHECKPOINTS),
    "session_completion_count",
    "final_rgb_validation_count",
)


class P0D1DiagnosticResult(NamedTuple):
    status: str
    failure_point: str
    error_class: str | None
    counters: dict[str, int]


def _zero_counts() -> dict[str, int]:
    return {name: 0 for name in COUNTER_NAMES}


def _public_error(error: BaseException) -> str:
    if isinstance(error, (TypeError, ValueError, json.JSONDecodeError)):
        return "validation_error"
    return "runtime_error"


class DiagnosticWriterSession(ActiveQKWriterSession):
    """Count the existing production session without changing writer semantics."""

    def __init__(self, transformer: Any, counts: dict[str, int], anchor: Any) -> None:
        self._diagnostic_counts = counts
        super().__init__(
            transformer,
            P0D1_CONFIG,
            anchor,
            q_diagnostic_observer=self._record_q_checkpoint,
        )

    def _record_q_checkpoint(self, checkpoint: str) -> None:
        name = f"{checkpoint}_count"
        if checkpoint not in P0_Q_DIAGNOSTIC_CHECKPOINTS or name not in self._diagnostic_counts:
            raise RuntimeError("P0D1 Q checkpoint differs from the finite public roster")
        if self._diagnostic_counts[name] != 0:
            raise RuntimeError("P0D1 Q checkpoint was repeated")
        self._diagnostic_counts[name] = 1

    def _root_pre_hook(self, module: Any, inputs: tuple[Any, ...]) -> None:
        call_index = self._root_call_count
        super()._root_pre_hook(module, inputs)
        self._diagnostic_counts["transformer_root_call_count"] += 1
        if call_index == 18:
            self._diagnostic_counts["step18_reached_count"] = 1

    def _feature_hook(self, kind: str, module_path: str):
        production_hook = super()._feature_hook(kind, module_path)

        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            hit_name = f"block4_to_{kind}_hook_hit_count"
            injection_name = f"block4_to_{kind}_injection_count"
            self._diagnostic_counts[hit_name] += 1
            was_injected = kind in self._measurements
            result = production_hook(module, inputs, output)
            if not was_injected and kind in self._measurements:
                self._diagnostic_counts[injection_name] += 1
            return result

        return hook

    def callback_on_step_end(
        self,
        pipeline: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        self._diagnostic_counts["pipeline_callback_count"] += 1
        return super().callback_on_step_end(
            pipeline, step_index, timestep, callback_kwargs
        )


def _generator_for(pipeline: Any) -> torch.Generator:
    transformer = getattr(pipeline, "transformer", None)
    try:
        parameter = next(transformer.parameters())
    except (AttributeError, StopIteration, TypeError) as error:
        raise RuntimeError("P0D1 transformer device is unavailable") from error
    return torch.Generator(device=parameter.device.type).manual_seed(P0_SEED)


def run_p0d1(
    *,
    geometry_key: str,
    hf_token: str,
    preloader: Callable[[str, str], Any] = load_real_pipeline,
) -> P0D1DiagnosticResult:
    """Run the one fixed writer configuration and retain bounded diagnostics."""

    counts = _zero_counts()
    failure_point = "pipeline_load"
    try:
        pipeline = preloader(P0_MODEL_ID, hf_token)
        counts["pipeline_load_count"] = 1
        failure_point = "session_setup"
        key = normalize_detection_key(geometry_key)
        anchor = derive_canonical_relation_anchor(
            key, point_count=P0_ANCHOR_POINT_COUNT
        )
        transformer = getattr(pipeline, "transformer", None)
        session = DiagnosticWriterSession(transformer, counts, anchor)
        with session:
            counts["session_setup_count"] = 1
            failure_point = "pipeline_callback"
            output = pipeline(
                prompt=P0_PROMPT_TEXT,
                num_inference_steps=P0_INFERENCE_STEPS,
                height=P0_IMAGE_SIZE,
                width=P0_IMAGE_SIZE,
                generator=_generator_for(pipeline),
                output_type="pil",
                callback_on_step_end=session.callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        failure_point = "session_completion"
        session.assert_complete()
        counts["session_completion_count"] = 1
        failure_point = "rgb_validation"
        images = getattr(output, "images", None)
        if not isinstance(images, (list, tuple)) or len(images) != 1:
            raise RuntimeError("P0D1 generation did not return one RGB")
        image = require_ordinary_rgb_image(images[0])
        if image.size != (P0_IMAGE_SIZE, P0_IMAGE_SIZE):
            raise ValueError("P0D1 final RGB dimensions differ")
        counts["final_rgb_validation_count"] = 1
        return P0D1DiagnosticResult(
            P0D1_STATUS_COMPLETE, "none", None, dict(counts)
        )
    except Exception as error:  # noqa: BLE001 - public class only; no message escapes
        return P0D1DiagnosticResult(
            P0D1_STATUS_STOPPED,
            failure_point,
            _public_error(error),
            dict(counts),
        )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def public_plan() -> dict[str, Any]:
    return {
        "protocol": P0D1_PROTOCOL_ID,
        "model_id": P0_MODEL_ID,
        "config_id": P0D1_CONFIG.config_id,
        "block_index": 4,
        "feature_kinds": ["q", "k"],
        "relative_rms_budget": 0.0025,
        "writer_step_index": 18,
        "inference_steps": P0_INFERENCE_STEPS,
        "run_count": 1,
        "candidate_selection": False,
        "science_denominator": 0,
    }


def package_p0d1_artifacts(
    output_directory: Path,
    *,
    exact: str,
    result: P0D1DiagnosticResult,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError("P0D1 output directory already exists")
    if result.status not in {P0D1_STATUS_STOPPED, P0D1_STATUS_COMPLETE}:
        raise ValueError("P0D1 status differs")
    if result.failure_point not in PUBLIC_FAILURE_POINTS:
        raise ValueError("P0D1 failure point differs")
    if tuple(result.counters) != COUNTER_NAMES:
        raise ValueError("P0D1 counter roster differs")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result.counters.values()
    ):
        raise ValueError("P0D1 counters must be nonnegative integers")
    output_directory.mkdir(parents=True, exist_ok=False)
    run_id = f"geometry-v3-qk-p0d1-{exact[:12]}"
    plan_digest = _digest(_json_bytes(public_plan()))
    receipt = {
        "run_id": run_id,
        "protocol": P0D1_PROTOCOL_ID,
        "execution_exact": exact,
        "model_id": P0_MODEL_ID,
        "config_id": P0D1_CONFIG.config_id,
        "plan_digest": plan_digest,
        "status": result.status,
        "artifact_status": "complete",
        "failure_point": result.failure_point,
        "error_class": result.error_class,
        "counters": result.counters,
        "candidate_selection": False,
        "science_denominator": P0D1_SCIENCE_DENOMINATOR,
    }
    terminal = {
        "run_id": run_id,
        "status": result.status,
        "artifact_status": "complete",
        "failure_point": result.failure_point,
        "error_class": result.error_class,
        "counters": result.counters,
        "science_denominator": 0,
    }
    payloads = {
        "receipt.json": _json_bytes(receipt),
        "terminal.json": _json_bytes(terminal),
    }
    manifest = {
        "run_id": run_id,
        "protocol": P0D1_PROTOCOL_ID,
        "execution_exact": exact,
        "plan_digest": plan_digest,
        "files": [
            {"name": name, "bytes": len(data), "sha256": _digest(data)}
            for name, data in sorted(payloads.items())
        ],
        "total_payload_bytes": sum(len(data) for data in payloads.values()),
        "artifact_status": "complete",
        "science_denominator": 0,
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    if sum(len(data) for data in payloads.values()) >= P0D1_ARTIFACT_MAX_BYTES:
        raise RuntimeError("P0D1 artifact exceeds its bound")
    for name, data in payloads.items():
        with (output_directory / name).open("xb") as stream:
            stream.write(data)
    return {
        "run_id": run_id,
        "p0d1_status": result.status,
        "artifact_status": "complete",
        "failure_point": result.failure_point,
        "error_class": result.error_class,
        "counters": result.counters,
        "science_denominator": 0,
    }


def execute_plan(
    plan: Mapping[str, Any],
    *,
    geometry_key: str,
    hf_token: str,
    preloader: Callable[[str, str], Any] = load_real_pipeline,
) -> dict[str, Any]:
    if set(plan) != {"expected_exact", "execution_exact", "output_directory"}:
        raise ValueError("P0D1 plan fields differ")
    expected, execution = plan["expected_exact"], plan["execution_exact"]
    if not isinstance(expected, str) or expected != execution or len(expected) != 40:
        raise ValueError("P0D1 execution identity differs")
    output = plan["output_directory"]
    if not isinstance(output, str) or not output.startswith(P0D1_DRIVE_ROOT):
        raise ValueError("P0D1 output must use its create-only Drive namespace")
    if not geometry_key.strip() or not hf_token.strip():
        raise ValueError("P0D1 runtime credentials are required")
    result = run_p0d1(
        geometry_key=geometry_key,
        hf_token=hf_token,
        preloader=preloader,
    )
    return package_p0d1_artifacts(Path(output), exact=execution, result=result)


def _read_plan(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_PLAN_BYTES:
        raise ValueError("P0D1 plan size differs")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError("P0D1 plan read is incomplete")
    value = json.loads(raw)
    if not isinstance(value, dict) or len(_json_bytes(value)) > MAX_PLAN_BYTES:
        raise ValueError("P0D1 canonical plan differs")
    return value


def _git_exact(expected: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    exact = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if exact != expected or status:
        raise RuntimeError("P0D1 checkout identity or cleanliness differs")
    return exact


def _emit(fd: int, value: dict[str, Any]) -> None:
    payload = _json_bytes(value)
    if len(payload) > MAX_CONTROL_BYTES:
        raise RuntimeError("P0D1 control receipt exceeds its bound")
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
            raise ValueError("P0D1 expected exact is missing")
        _git_exact(expected)
        stage = "execution"
        try:
            control = execute_plan(
                plan,
                geometry_key=key,
                hf_token=token,
                preloader=preloader,
            )
        finally:
            token = ""
            key = ""
        stage = "control_channel"
        _emit(args.control_fd, {"status": "success", **control})
        return 0
    except Exception as error:  # noqa: BLE001 - only a finite public receipt crosses
        token = ""
        key = ""
        if stage == "control_channel":
            return 1
        try:
            _emit(
                args.control_fd,
                {
                    "status": "failure",
                    "failure_point": stage,
                    "error_class": _public_error(error),
                    "science_denominator": 0,
                },
            )
        except Exception:  # noqa: BLE001 - never fall back to stdout or stderr
            pass
        return 1
    finally:
        try:
            os.close(args.control_fd)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(_main())

