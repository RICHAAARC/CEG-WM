"""Fixed checkpoint-first continuation of attempt01, with exactly one repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cegwm.runtime.stage_protection_resume import RepairStore, RepairStopped, REPAIR, COMPAT_REPAIR, TARGET
from cegwm.runtime.stage_protection_store import atomic_json
from cegwm.runtime.stage_protection_v2 import StageProtectionProvider
from experiments.run_stage_protection_v2 import Canary, describe_plan, load_config


def execute(config, output, *, provider_factory=StageProtectionProvider, compatibility=False):
    import fcntl
    output = Path(output)
    # Opening the existing root is intentional: a typo/missing Drive never
    # creates an empty new experiment under the continuation entry point.
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = RepairStore(output, config, compatibility=compatibility)  # Before any provider/model.
        summary_path = store.directory / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            provider = provider_factory(config, counters=store.counters)
            try:
                canary = Canary(config, store, provider)
                summary = canary.evaluate(canary.freeze())
            finally:
                provider.close()
                store.persist_accounting()
        summary["continuation"] = store.accounting()
        summary["plan"]["maxima_scope"] = "Original uninterrupted P2 only; not a cumulative recovery cap"
        summary["plan"]["resume_policy"] = "One fixed target repair; reuse RETURNED; no second retry after STARTED/FAILED"
        if compatibility:
            summary["plan"]["resume_policy"] = "One compatibility continuation only for the exact import-only failure; retain both prior attempts; no automatic retry of a new STARTED/FAILED"
        atomic_json(summary_path, summary)
        return summary


def main(argv=None, *, compatibility=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(
        "/content/drive/MyDrive/CEG-WM_Result/stage-protection-v2-20261109/attempt01"))
    args = parser.parse_args(argv)
    config = load_config()
    directory = COMPAT_REPAIR if compatibility else REPAIR
    if not args.execute:
        print(json.dumps(dict(original_plan=describe_plan(config), repair_target=TARGET,
            new_attempts=1, compatibility_failure_only=compatibility,
            results_subdirectory=str(directory), cumulative_original_cap_claim=False), indent=2))
        return 0
    try:
        summary = execute(config, args.output, compatibility=compatibility)
    except RepairStopped as error:
        print(str(error), flush=True)
        return 1
    print(json.dumps(dict(output=str(args.output / directory),
        engineering_complete=summary["engineering_complete"],
        verdict=summary["verdict"], continuation=summary["continuation"]), indent=2))
    return 0 if summary["engineering_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
