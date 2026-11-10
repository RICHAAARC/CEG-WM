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


def isolated_preflight(output, config):
    output = Path(output)
    try:
        checked = preflight(output, config, compatibility=True)
        previous = output / COMPAT_REPAIR
        if any((previous / name).exists() for name in AGGREGATES):
            raise ValueError("previous P2 already has aggregate results; inspect rather than restart decisions")
        for key in EARLY1:
            value = load_returned(previous / "operations", key)
            del value
        latest = _counts(_json(previous / "physical_counts.json"))
        last_completed = _json(previous / "operations" / EARLY1[-1] / "status.json")["counts_after"]
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
                elif key == ISOLATED_TARGET:
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
                    previous_derived_report_lag=lag, original_counts=checked["original_counts"])
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
            atomic_json(directory / "reservation.json", dict(target=ISOLATED_TARGET, maximum_new_attempts=1, **checked))
            atomic_json(directory / "physical_counts.json", self.initial)
        super().__init__(directory, config)
        self.persist_accounting()

    def call(self, key, function):
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
        def attempt():
            if key == ISOLATED_TARGET:
                atomic_json(self.directory / "retry_counts_before.json", dict(self.counters))
            return function()
        try:
            return super().call(key, attempt)
        except Exception as error:
            if key == ISOLATED_TARGET:
                raise RepairStopped(f"isolated early2 attempt stopped: {error}; no automatic retry") from error
            raise
        finally:
            self.persist_accounting()

    def accounting(self):
        current = dict(self.counters)
        status_path = self.operations / ISOLATED_TARGET / "status.json"
        status = _json(status_path) if status_path.exists() else {"status": "NOT_STARTED_OR_PRIOR_CACHE_REUSED"}
        before = self.directory / "retry_counts_before.json"
        retry = difference(status.get("counts_after", current), _json(before)) if before.exists() else {}
        return dict(target=ISOLATED_TARGET, retry_status=status["status"],
            starting_raw_counts=self.initial, new_raw_counts=difference(current, self.initial),
            retry_raw_counts=retry, cumulative_raw_counts=current,
            cumulative_physical_counts=physical_counts(current),
            prior_attempt_raw_counts=[_counts(_json(path / "physical_counts.json")) for path in self.old],
            previous_derived_report_lag=self.checked["previous_derived_report_lag"],
            original_maxima_apply_to_cumulative=False,
            cost_note="All prior spent calls retained; independent workers add model loads and disk IO, not extra model forward/VJP calls.")

    def persist_accounting(self):
        atomic_json(self.directory / "continuation_counts.json", self.accounting())

    def inventory(self):
        rows = []
        for label, directory in zip(("original", "import_failed_repair", "compatibility_continuation"), self.old):
            rows.extend(dict(_json(path), checkpoint_origin=label) for path in sorted((directory / "operations").rglob("status.json")))
        if self.checked["previous_target_status"]["status"] == "MISSING_UNSYNCED":
            rows.append(dict(self.checked["previous_target_status"], checkpoint_origin="compatibility_continuation"))
        return rows + [dict(row, checkpoint_origin="isolated_continuation") for row in super().inventory()]
