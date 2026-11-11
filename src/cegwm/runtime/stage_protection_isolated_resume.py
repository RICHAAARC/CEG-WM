"""Layered checkpoint recovery of the observed early2 RGB-VJP interruption."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .stage_protection_resume import (
    REPAIR, COMPAT_REPAIR, REQUIRED, SOURCE, AGGREGATES, PreflightBlocked, RepairStopped,
    _counts, _json, difference, load_returned, preflight, validate_payload,
)
from .stage_protection_store import OperationStore, atomic_json, Unavailable
from .stage_protection_v2 import physical_counts

ISOLATED_TARGET = SOURCE + "/bit1/early2/rgb_vjp"
ISOLATED_REPAIR = Path("repairs/source06_bit1_early2_rgb_vjp_isolated_attempt01")
EARLY1 = tuple(SOURCE + "/bit1/early1/" + name for name in
               ("latent_vjp", "action", "render", "clean", "design_image", "design_read"))
DEPENDENCY_REPAIRS = EARLY1[-2:]


def _load_payload(path, key):
    value = torch.load(path, map_location="cpu", weights_only=False)
    validate_payload(key, value)
    return value


def scan_checkpoints(output):
    """Inspect every required cache before any reservation or model load.

    Recovery candidates must belong to the exact operation. A partial is usable
    only if it is itself a complete, validated result; never infer an image/read
    from a different basepoint or from a partial's incidental fields.
    """
    old = [output, output / REPAIR, output / COMPAT_REPAIR]
    own = output / ISOLATED_REPAIR
    report = dict(checked_operations=0, blocking_gaps=[], dependency_results={},
                  dependency_recompute_required=[], rejected_candidates=[])
    required = {(output, key) for key in REQUIRED}
    required.update((old[-1], key) for key in EARLY1[:-2])
    for directory in (output, old[-1], own):
        for path in (directory / "operations").rglob("status.json"):
            key = str(path.parent.relative_to(directory / "operations"))
            try:
                status = _json(path)
                if key in DEPENDENCY_REPAIRS:
                    continue
                if status.get("status") == "RETURNED" or (directory == output and key != EARLY1[0]):
                    required.add((directory, key))
                elif directory == own and key == ISOLATED_TARGET:
                    report["blocking_gaps"].append(key + ": isolated attempt already spent; no automatic retry")
            except Exception as error:
                report["blocking_gaps"].append(str(path.relative_to(output)) + ": " + str(error))
    for directory, key in sorted(required):
        report["checked_operations"] += 1
        try:
            value = load_returned(directory / "operations", key)
            del value
        except Exception as error:
            report["blocking_gaps"].append(str(directory.relative_to(output)) + "/" + key + ": " + str(error))
    for key in DEPENDENCY_REPAIRS:
        report["checked_operations"] += 1
        report["dependency_results"][key] = None
        folder = own / "operations" / key
        # A new attempt consumes its allowance even if only STARTED was saved.
        if folder.exists() and any(folder.iterdir()):
            try:
                value = load_returned(own / "operations", key)
                del value
                report["dependency_results"][key] = str((folder / "result.pt").relative_to(output))
            except Exception as error:
                report["blocking_gaps"].append(key + ": isolated dependency attempt already spent or invalid: " + str(error))
            continue
        for directory in reversed(old):
            for name in ("result.pt", "result.pt.tmp", "partial.pt"):
                path = directory / "operations" / key / name
                if not path.exists():
                    continue
                try:
                    value = _load_payload(path, key)
                    del value
                    report["dependency_results"][key] = str(path.relative_to(output))
                    break
                except Exception as error:
                    report["rejected_candidates"].append(dict(path=str(path.relative_to(output)), error=str(error)))
            if report["dependency_results"][key] is not None:
                break
        if report["dependency_results"][key] is None:
            report["dependency_recompute_required"].append(key)
    print("ISOLATED_CHECKPOINT_SCAN " + json.dumps(report), flush=True)
    if report["blocking_gaps"]:
        raise ValueError("all required cache gaps: " + "; ".join(report["blocking_gaps"]) +
                         "; fixed dependency gaps: " + ", ".join(report["dependency_recompute_required"]))
    return report


def isolated_preflight(output, config):
    output = Path(output)
    try:
        scan = scan_checkpoints(output)
        def checked_payload(operations, key):
            # Only these two dependency payloads were resolved by the full scan;
            # absence is recorded for the isolated, once-only recomputation.
            if key not in DEPENDENCY_REPAIRS:
                return load_returned(operations, key)
            path = scan["dependency_results"][key]
            return _load_payload(output / path, key) if path is not None else None
        checked = preflight(output, config, compatibility=True, compatibility_payload_loader=checked_payload)
        previous = output / COMPAT_REPAIR
        if any((previous / name).exists() for name in AGGREGATES):
            raise ValueError("previous P2 already has aggregate results; inspect rather than restart decisions")
        latest = _counts(_json(previous / "physical_counts.json"))
        # clean is the last unconditionally persisted prerequisite. Later design
        # work may have consumed counters without syncing either result.
        last_completed = _counts(_json(previous / "operations" / EARLY1[3] / "status.json")["counts_after"])
        if any(value < 0 for value in difference(latest, last_completed).values()):
            raise ValueError("latest durable counters predate completed early1")
        target = previous / "operations" / ISOLATED_TARGET
        if not target.is_dir():
            raise ValueError("observed early2/rgb_vjp directory missing; never regenerate early1")
        status = _json(target / "status.json") if (target / "status.json").exists() else {"status": "MISSING_UNSYNCED", "key": ISOLATED_TARGET}
        if status.get("key") != ISOLATED_TARGET or status.get("status") not in ("RETURNED", "STARTED", "FAILED", "MISSING_UNSYNCED"):
            raise ValueError("unexpected interrupted operation state")
        if status["status"] == "RETURNED":
            value = load_returned(previous / "operations", ISOLATED_TARGET)
            del value
        elif (target / "result.pt").exists():
            # A result exists: never overwrite it or silently recompute it.
            value = torch.load(target / "result.pt", map_location="cpu", weights_only=False)
            validate_payload(ISOLATED_TARGET, value)
            del value
            status = {**status, "usable_result_without_returned_status": True}
        elif status["status"] == "MISSING_UNSYNCED" and not any(difference(latest, last_completed).values()):
            raise ValueError("unsynced status requires durable evidence of work after early1")
        directory = output / ISOLATED_REPAIR
        if (directory / "reservation.json").exists():
            reserved = _json(directory / "reservation.json")
            if reserved["target"] != ISOLATED_TARGET or reserved["starting_counts"] != latest:
                raise ValueError("previous durable cost changed after isolated reservation")
            current = _counts(_json(directory / "physical_counts.json"))
            if any(value < 0 for value in difference(current, latest).values()):
                raise ValueError("isolated cumulative counters lost previous work")
            for path in (directory / "operations").rglob("status.json"):
                key = str(path.parent.relative_to(directory / "operations"))
                state = _json(path)
                if state["status"] == "RETURNED":
                    value = load_returned(directory / "operations", key)
                    del value
                elif key == ISOLATED_TARGET or key in DEPENDENCY_REPAIRS:
                    raise ValueError("isolated early2 attempt already spent; no automatic retry")
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError("isolated output exists without reservation")
        lag = None
        report = previous / "continuation_counts.json"
        if report.exists():
            try:
                lag = difference(latest, _counts(_json(report)["cumulative_raw_counts"]))
            except (ValueError, KeyError, TypeError):
                pass  # Derived report is diagnostic, never authority over durable counts.
        return dict(starting_counts=latest, previous_target_status=status,
                    previous_derived_report_lag=lag, original_counts=checked["original_counts"], cache_scan=scan)
    except Exception as error:
        raise PreflightBlocked(f"ISOLATED_CHECKPOINT_PREFLIGHT_BLOCKED: {error}") from error


class IsolatedStore(OperationStore):
    def __init__(self, output, config):
        self.original = Path(output)
        checked = isolated_preflight(output, config)
        self.old = [self.original, self.original / REPAIR, self.original / COMPAT_REPAIR]
        self.initial = checked["starting_counts"]
        self.checked = checked
        directory = self.original / ISOLATED_REPAIR
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "reservation.json").exists():
            atomic_json(directory / "reservation.json", dict(target=ISOLATED_TARGET, maximum_new_attempts=1,
                dependency_maximum_new_attempts={key: 1 for key in DEPENDENCY_REPAIRS}, **checked))
            atomic_json(directory / "physical_counts.json", self.initial)
        super().__init__(directory, config)
        atomic_json(directory / "checkpoint_scan.json", checked["cache_scan"])
        self.persist_accounting()

    def call(self, key, function):
        if key in DEPENDENCY_REPAIRS:
            # Own status wins, including a consumed failure; never overwrite it.
            if (self.operations / key / "status.json").exists():
                return self._new_call(key, function)
            path = self.checked["cache_scan"]["dependency_results"][key]
            if path is not None:
                try:
                    result = _load_payload(self.original / path, key)
                except Exception as error:
                    raise RepairStopped("validated dependency disappeared: " + key) from error
                self.cache_reuses.append(path)
                return result
            return self._new_call(key, function)
        for directory in reversed(self.old):
            folder = directory / "operations" / key
            status = _json(folder / "status.json") if (folder / "status.json").exists() else None
            if status and status["status"] == "RETURNED":
                result = load_returned(directory / "operations", key)
                self.cache_reuses.append(str(directory.relative_to(self.original)) + "/" + key)
                return result
            if key == ISOLATED_TARGET and (folder / "result.pt").exists():
                result = torch.load(folder / "result.pt", map_location="cpu", weights_only=False)
                validate_payload(key, result)
                self.cache_reuses.append("saved_result_without_returned_status/" + key)
                return result
            if status and key not in (ISOLATED_TARGET, SOURCE + "/bit1/early1/latent_vjp"):
                raise Unavailable(f"{key}: prior {status['status']}; no retry")
        if key in REQUIRED or key in EARLY1:
            raise PreflightBlocked("required completed checkpoint disappeared: " + key)
        return self._new_call(key, function)

    def _new_call(self, key, function):
        def attempt():
            if key == ISOLATED_TARGET:
                atomic_json(self.directory / "retry_counts_before.json", dict(self.counters))
            elif key in DEPENDENCY_REPAIRS:
                atomic_json(self.operations / key / "dependency_counts_before.json", dict(self.counters))
            result = function()
            if key in DEPENDENCY_REPAIRS:
                validate_payload(key, result)
            return result
        try:
            return super().call(key, attempt)
        except Exception as error:
            if key == ISOLATED_TARGET or key in DEPENDENCY_REPAIRS:
                raise RepairStopped(f"isolated recovery attempt stopped for {key}: {error}; no automatic retry") from error
            raise
        finally:
            self.persist_accounting()

    def accounting(self):
        current = dict(self.counters)
        status_path = self.operations / ISOLATED_TARGET / "status.json"
        status = _json(status_path) if status_path.exists() else {"status": "NOT_STARTED_OR_PRIOR_CACHE_REUSED"}
        before = self.directory / "retry_counts_before.json"
        retry = difference(status.get("counts_after", current), _json(before)) if before.exists() else {}
        dependencies, dependency_total = {}, {}
        for key in DEPENDENCY_REPAIRS:
            folder = self.operations / key
            state = _json(folder / "status.json") if (folder / "status.json").exists() else {"status": "NOT_STARTED_OR_PRIOR_CACHE_REUSED"}
            baseline = folder / "dependency_counts_before.json"
            delta = difference(state.get("counts_after", current), _json(baseline)) if baseline.exists() else {}
            dependencies[key] = dict(status=state["status"], raw_counts=delta,
                prior_result=self.checked["cache_scan"]["dependency_results"][key])
            for name, value in delta.items():
                dependency_total[name] = dependency_total.get(name, 0) + value
        return dict(target=ISOLATED_TARGET, retry_status=status["status"],
            starting_raw_counts=self.initial, new_raw_counts=difference(current, self.initial),
            retry_raw_counts=retry, cumulative_raw_counts=current,
            dependency_recomputations=dependencies, dependency_recomputation_raw_counts=dependency_total,
            cumulative_physical_counts=physical_counts(current),
            prior_attempt_raw_counts=[_counts(_json(path / "physical_counts.json")) for path in self.old],
            previous_derived_report_lag=self.checked["previous_derived_report_lag"],
            original_maxima_apply_to_cumulative=False,
            cost_note="All prior spent calls retained; the two fixed missing design dependencies may each be recomputed once and are costed separately. Independent workers also add model loads and disk IO.")

    def persist_accounting(self):
        atomic_json(self.directory / "continuation_counts.json", self.accounting())

    def inventory(self):
        rows = []
        for label, directory in zip(("original", "import_failed_repair", "compatibility_continuation"), self.old):
            rows.extend(dict(_json(path), checkpoint_origin=label) for path in sorted((directory / "operations").rglob("status.json")))
        if self.checked["previous_target_status"]["status"] == "MISSING_UNSYNCED":
            rows.append(dict(self.checked["previous_target_status"], checkpoint_origin="compatibility_continuation"))
        return rows + [dict(row, checkpoint_origin="isolated_continuation") for row in super().inventory()]
