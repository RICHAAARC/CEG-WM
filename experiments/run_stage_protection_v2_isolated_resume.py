"""Fixed early2 RGB-VJP recovery: model subprocesses and exact local disk saves."""
import argparse
import json
from pathlib import Path

from cegwm.runtime.stage_protection_isolated_resume import IsolatedStore, ISOLATED_REPAIR
from cegwm.runtime.stage_protection_process import ProcessProvider
from cegwm.runtime.stage_protection_resume import RepairStopped
from cegwm.runtime.stage_protection_store import atomic_json
from experiments.run_stage_protection_v2 import Canary, load_config


class IsolatedCanary(Canary):
    def op(self, key, function):
        self.provider.operation_key = key
        return super().op(key, function)


def execute(config, output, *, provider_factory=ProcessProvider, scratch_root="/content/ceg_stage_protection_scratch"):
    import fcntl
    output = Path(output)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = IsolatedStore(output, config)
        summary_path = store.directory / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        provider = provider_factory(config, counters=store.counters, directory=store.directory, scratch_root=scratch_root)
        try:
            canary = IsolatedCanary(config, store, provider)
            result = canary.evaluate(canary.freeze())
            result["continuation"] = store.accounting()
            result["plan"]["maxima_scope"] = "Original computation schedule only; cumulative recovery and process reload costs are separate"
            atomic_json(summary_path, result)
            return result
        finally:
            provider.close()
            store.persist_accounting()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/content/drive/MyDrive/CEG-WM_Result/stage-protection-v2-20261109/attempt01"))
    args = parser.parse_args(argv)
    if not args.execute:
        print(__doc__)
        return 0
    try:
        summary = execute(load_config(), args.output)
    except RepairStopped as error:
        print(str(error), flush=True)
        return 1
    print(json.dumps(dict(output=str(args.output / ISOLATED_REPAIR), engineering_complete=summary["engineering_complete"],
                         continuation=summary["continuation"]), indent=2))
    return 0 if summary["engineering_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
