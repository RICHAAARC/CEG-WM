"""One fixed synthetic RGB input through the real diagnostic runtime.

No formal/diagnostic image roster is read; no image generation is invoked.
Running this entry requires separate real-runtime authorization.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from diagnostic import BASELINE, CONDITIONS, REPO, TAU, FrozenRuntime, diagnose_image, write_new

SEED = 20260906


def synthetic_input():
    y,x = np.mgrid[:512,:512]
    rng = np.random.default_rng(SEED)
    base = np.stack((x/2, y/2, (x+y)/4), axis=-1)
    return Image.fromarray(np.clip(base+rng.integers(-16,17,(512,512,3)),0,255).astype("uint8"))


def assess(row):
    """Only operational validity; no positive-rate or geometry-quality gate."""
    failures = []
    for field in ("reference_score", "pre_score", "oracle_post_score"):
        if row.get(field) is None or not np.isfinite(row[field]):
            failures.append(field+" unavailable")
    if not (row.get("production") or {}).get("method_complete"):
        failures.append("production incomplete")
    geometry = row.get("geometry_record") or {}
    if geometry.get("status") not in ("RELIABLE", "UNRELIABLE", "UNSUPPORTED"):
        failures.append("geometry runtime not observed complete")
    if geometry.get("legal") is True and row.get("predicted_H") is None:
        failures.append("legal geometry matrix unavailable")
    if row.get("predicted_H") is not None and row.get("syncseal_post_score") is None:
        failures.append("legal-H post score unavailable")
    return failures


def execute(output, backend_factory):
    output = Path(output)
    if output.resolve().is_relative_to(REPO):
        raise ValueError("preflight output must remain outside Git worktree")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    header = {"science_denominator":0, "diagnostic_pair_count":0,
              "synthetic_image_count":1, "planned_condition_rows":2,
              "synthetic_seed":SEED, "baseline_main":BASELINE, "threshold":TAU,
              "started_utc":datetime.now(timezone.utc).isoformat(),
              "claim_ceiling":"runtime preflight only; no watermark effectiveness evidence"}
    write_new(output/"started.json", {**header,"status":"PREFLIGHT_STARTED"})
    rows = []
    try:
        backend = backend_factory()
        execution_kind = "FROZEN_REAL_RUNTIME" if type(backend) is FrozenRuntime else "INJECTED_TEST_BACKEND"
        image = synthetic_input()
        for condition in CONDITIONS:
            try:
                row = diagnose_image(image, condition, backend)
                row["preflight_failures"] = assess(row)
            except Exception as error:
                row = {"preflight_failures":[f"{type(error).__name__}: {error}"]}
            row.update(condition=condition, science_denominator=0, execution_kind=execution_kind)
            write_new(output/(condition+".json"),row)
            rows.append(row)
        passed = all(not row["preflight_failures"] for row in rows)
        result = {**header,"status":"PREFLIGHT_PASSED" if passed else "PREFLIGHT_FAILED",
                  "execution_kind":execution_kind,"completed_condition_rows":len(rows),
                  "elapsed_seconds":time.monotonic()-start}
    except Exception as error:
        result = {**header,"status":"PREFLIGHT_FAILED", "completed_condition_rows":len(rows),
                  "error":f"{type(error).__name__}: {error}", "elapsed_seconds":time.monotonic()-start}
    write_new(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-authorized-preflight", action="store_true", required=True)
    args = parser.parse_args()
    if args.runtime_root.resolve().is_relative_to(REPO):
        raise ValueError("runtime assets must remain outside Git worktree")
    result = execute(args.output, lambda: FrozenRuntime(args.runtime_root))
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "PREFLIGHT_PASSED" else 1)


if __name__ == "__main__":
    main()
