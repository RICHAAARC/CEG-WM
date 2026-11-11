"""One authorized repair overlay for the observed V2 attempt01 interruption.

Original artifacts remain read-only. No generic retry policy is changed.
Only our own trusted torch checkpoints are accepted (never uploaded pickles).
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from .channel_precompensation_sd35 import SourceCapture, RGBSensitivity
from .stage_protection_store import OperationStore, Unavailable, atomic_json
from .stage_protection_v2 import physical_counts

SOURCE = "source06_fur_knit"
TARGET = SOURCE + "/bit1/early1/latent_vjp"
REPAIR = Path("repairs/source06_bit1_early1_latent_vjp_attempt01")
COMPAT_REPAIR = Path("repairs/source06_bit1_early1_latent_vjp_compat_attempt01")
AGGREGATES = ("selection_frozen.pt", "selection_frozen.json", "evaluated.pt", "summary.json")
COMPAT_ERROR = (
    r"RuntimeStageError: source_suffix_vjp: ImportError: cannot import name "
    r"'is_non_overlapping_and_dense' from 'torch\._prims_common'(?: \([^\r\n]*\))?"
)
REQUIRED = tuple(SOURCE + "/" + key for key in (
    "source", "canonical_early_zero", "unmarked/clean", "unmarked/design_image", "unmarked/design_read",
    *["bit0/E/" + x for x in ("render", "clean", "design_image", "design_read")],
    *["bit0/late/" + x for x in ("clean_rgb_vjp", "design_rgb_vjp", "latent_vjp")],
    *["bit0/late/" + arm + "/" + x for arm in ("A", "B", "C")
      for x in ("solve", "action", "render", "clean", "design_image", "design_read")],
    "bit1/early1/rgb_vjp",
))


class PreflightBlocked(Unavailable):
    """A required saved artifact is unavailable; do not load any model."""


class RepairStopped(BaseException):
    """Stop the runner immediately, across Canary's per-trial Exception handling.

The sole retry is spent. Do not freeze an incomplete retry as a new selection,
or proceed to source07 after this interruption. CLI translates this into exit 1.
"""


def _json(path):
    return json.loads(Path(path).read_text())


def _counts(value):
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or type(v) is not int or v < 0 for k, v in value.items()
    ):
        raise ValueError("physical counts must be nonnegative integer counters")
    return value


def difference(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(after.keys() | before.keys())}


def _finite(value):
    # Inspect one payload at a time. Do not accumulate loaded checkpoints.
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite checkpoint tensor")
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("nonfinite checkpoint array")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite checkpoint number")
    elif isinstance(value, Image.Image):
        value.load()
        if value.mode != "RGB" or min(value.size) <= 0:
            raise ValueError("checkpoint image is not readable RGB")
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite(item)
    elif is_dataclass(value):
        for field in fields(value):
            _finite(getattr(value, field.name))


def _tensor(value):
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise ValueError("required tensor missing or empty")


def _image(value):
    if not isinstance(value, Image.Image):
        raise ValueError("required RGB image missing")


def _capture(value, step):
    if not isinstance(value, SourceCapture) or value.remaining_steps != tuple(range(step + 1, 20)):
        raise ValueError("source capture or fixed suffix steps missing")
    _image(value.image)
    for name in ("latents", "prompt_embeds", "pooled_prompt_embeds"):
        _tensor(value.before_state[name])
    for name in ("latent_device", "scheduler_class", "joint_attention_kwargs",
                 "do_classifier_free_guidance", "guidance_scale"):
        value.before_state[name]
    if value.before_state["next_step_index"] != step + 1:
        raise ValueError("capture scheduler index differs from fixed suffix")
    for name in ("timesteps", "sigmas"):
        _tensor(value.scheduler_state[name])


def validate_payload(key, value):
    """Readability and fields needed by the saved runtime; no content hash gate."""
    _finite(value)
    leaf = key.rsplit("/", 1)[-1]
    if leaf == "source":
        for step in (8, 18):
            _capture(value[step], step)
    elif leaf in ("render", "canonical_early_zero"):
        _image(value["image"])
        _tensor(value["after"])
        _tensor(value["actual_delta"])
        if value["after"].shape != value["actual_delta"].shape:
            raise ValueError("render latent shapes differ")
        if leaf == "canonical_early_zero" or "/early" in key:
            _capture(value["late"], 18)
    elif leaf in ("clean", "design_read", "read"):
        _tensor(value["feature"])
        if not math.isfinite(float(value["score"])):
            raise ValueError("nonfinite saved reader score")
    elif leaf in ("design_image", "image"):
        _image(value)
    elif leaf.endswith("rgb_vjp"):
        if not isinstance(value, RGBSensitivity):
            raise ValueError("RGBSensitivity missing")
        _image(value.image)
        _tensor(value.gradient_rgb)
        _tensor(value.feature)
    elif leaf == "latent_vjp":
        _tensor(value["gradients"])
        if value["gradients"].shape[0] != (1 if "/early" in key else 2):
            raise ValueError("saved VJP count differs from fixed operation")
    elif leaf == "solve":
        if not isinstance(value["status"], str) or "delta" not in value:
            raise ValueError("saved solve missing status/delta")
        if value["delta"] is not None:
            _tensor(value["delta"])
    elif leaf == "action":
        for name in ("after", "actual"):
            _tensor(value[name])
        for name in ("relative", "within_budget", "nonzero"):
            value[name]
    else:
        raise ValueError("unknown saved operation " + key)


def load_returned(operations, key):
    folder = operations / key
    status = _json(folder / "status.json")
    if status.get("status") != "RETURNED" or status.get("key") != key:
        raise ValueError(key + ": required RETURNED checkpoint unavailable")
    value = torch.load(folder / "result.pt", map_location="cpu", weights_only=False)
    validate_payload(key, value)
    return value


def _compatibility_predecessor(output, config, checked):
    """Accept only the observed import failure before any new model computation.

    The failed attempt's partial.pt is evidence, not a completed VJP result.
    Read its final counters/status; never overwrite or reuse its partial payload.
    """
    previous = output / REPAIR
    if _json(previous / "config.json") != config:
        raise ValueError("previous repair configuration differs")
    reservation = _json(previous / "reservation.json")
    if (reservation.get("target") != TARGET or reservation.get("maximum_new_attempts") != 1
            or reservation.get("original_counts") != checked["original_counts"]
            or reservation.get("original_target_status") != checked["original_target_status"]):
        raise ValueError("previous repair reservation differs from original evidence")
    if any((previous / (name + suffix)).exists() for name in AGGREGATES for suffix in ("", ".tmp")):
        raise ValueError("previous repair has frozen/evaluated results; not the import-only failure")
    operations = previous / "operations"
    target = operations / TARGET
    # Reject later operations, including payloads left without a status file.
    if any(path.is_file() and path.parent != target for path in operations.rglob("*")):
        raise ValueError("previous repair contains another operation")
    status = _json(target / "status.json")
    if (status.get("key") != TARGET or status.get("status") != "FAILED"
            or not isinstance(status.get("error"), str)
            or re.fullmatch(COMPAT_ERROR, status["error"]) is None):
        raise ValueError("previous repair is not the exact source_suffix_vjp private-symbol ImportError")
    if any((target / name).exists() for name in ("result.pt", "result.pt.tmp")):
        raise ValueError("failed import target has a completed result")
    before = _counts(_json(previous / "retry_counts_before.json"))
    if before != checked["original_counts"]:
        raise ValueError("previous repair did work before the fixed target")
    final = _counts(_json(previous / "physical_counts.json"))
    if _counts(status.get("counts_after")) != final:
        raise ValueError("previous repair final/status counters disagree")
    delta = difference(final, checked["original_counts"])
    allowed = dict(source_pipeline_load_attempted=1, source_pipeline_load_returned=1)
    if any(delta.get(key, 0) != allowed.get(key, 0) for key in delta.keys() | allowed.keys()):
        raise ValueError("previous repair is not load-only: exactly one loader attempt/return and zero computation required")
    return dict(starting_counts=final, previous_repair_status=status)


def preflight(output, config, *, compatibility=False, compatibility_payload_loader=load_returned):
    """Read-only preflight; never falls back to generation or creates a run."""
    output = Path(output)
    try:
        if _json(output / "config.json") != config:
            raise ValueError("saved method configuration differs")
        initial = _counts(_json(output / "physical_counts.json"))
        target = output / "operations" / TARGET
        original = _json(target / "status.json")
        if original.get("status") != "STARTED" or original.get("key") != TARGET:
            raise ValueError("this repair applies only to the observed STARTED target")
        if (target / "result.pt").exists() or (target / "partial.pt").exists():
            raise ValueError("original target now has result/partial; inspect before changing recovery")
        if any((output / name).exists() for name in AGGREGATES):
            raise ValueError("original run acquired aggregate results; inspect changed run before recovery")
        old_operations = output / "operations"
        keys = set(REQUIRED) | {str(p.parent.relative_to(old_operations))
                               for p in old_operations.rglob("status.json")}
        for key in sorted(keys - {TARGET}):
            value = load_returned(old_operations, key)
            del value
        checked = dict(original_counts=initial, original_target_status=original,
                       checked_original_operations=len(keys) - 1)
        if compatibility:
            checked.update(_compatibility_predecessor(output, config, checked))
        baseline = checked.get("starting_counts", initial)
        directory = output / (COMPAT_REPAIR if compatibility else REPAIR)
        reservation = directory / "reservation.json"
        if reservation.exists():
            saved = _json(reservation)
            if saved["target"] != TARGET or saved["original_counts"] != initial or saved["original_target_status"] != original:
                raise ValueError("original counters/status changed after repair reservation")
            if compatibility and any(saved.get(key) != checked[key] for key in ("starting_counts", "previous_repair_status")):
                raise ValueError("previous compatibility failure changed after reservation")
            counter_file = directory / "physical_counts.json"
            if counter_file.exists():
                current = _counts(_json(counter_file))
                if any(v < 0 for v in difference(current, baseline).values()):
                    raise ValueError("cumulative counters lost earlier work")
            elif any((directory / "operations").rglob("status.json")):
                raise ValueError("continuation counters missing after operations started")
            for status_path in sorted((directory / "operations").rglob("status.json")):
                key = str(status_path.parent.relative_to(directory / "operations"))
                status = _json(status_path)
                if status.get("status") == "RETURNED":
                    if key == TARGET:
                        _counts(_json(directory / "retry_counts_before.json"))
                    value = compatibility_payload_loader(directory / "operations", key)
                    del value
                elif key == TARGET:
                    raise ValueError("authorized retry already " + str(status.get("status")) + "; no second retry")
            for name in ("selection_frozen.pt", "evaluated.pt"):
                if (directory / name).exists():
                    value = torch.load(directory / name, map_location="cpu", weights_only=False)
                    if not isinstance(value, dict) or len(value["trials"]) != 4:
                        raise ValueError(name + ": invalid fixed denominator")
                    del value
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError("repair artifacts without durable reservation")
        return checked
    except Exception as error:
        raise PreflightBlocked(f"CHECKPOINT_PREFLIGHT_BLOCKED: {error}") from error


class RepairStore(OperationStore):
    """Reuse the original cache, write only into one fixed continuation overlay."""
    def __init__(self, output, config, *, compatibility=False):
        self.original = Path(output)
        self.compatibility = compatibility
        checked = preflight(output, config, compatibility=compatibility)
        directory = self.original / (COMPAT_REPAIR if compatibility else REPAIR)
        directory.mkdir(parents=True, exist_ok=True)
        reservation = directory / "reservation.json"
        if not reservation.exists():
            atomic_json(reservation, dict(target=TARGET, maximum_new_attempts=1, **checked))
        self.root_counts = checked["original_counts"]
        self.initial = checked.get("starting_counts", self.root_counts)
        if not (directory / "physical_counts.json").exists():
            atomic_json(directory / "physical_counts.json", self.initial)
        super().__init__(directory, config)
        self.persist_accounting()

    def call(self, key, function):
        original_status = self.original / "operations" / key / "status.json"
        if key != TARGET and original_status.exists():
            # Even if a file changes after preflight, never replace it by compute.
            value = load_returned(self.original / "operations", key)
            self.cache_reuses.append("original/" + key)
            return value
        if key in REQUIRED:
            raise PreflightBlocked("required original operation disappeared: " + key)
        def attempt():
            if key == TARGET:
                atomic_json(self.directory / "retry_counts_before.json", dict(self.counters))
            return function()
        try:
            return super().call(key, attempt)
        except Exception as error:
            if key == TARGET:
                raise RepairStopped(f"Authorized retry stopped: {error}; no second automatic retry") from error
            raise
        finally:
            self.persist_accounting()

    def accounting(self):
        cumulative = dict(self.counters)
        new = difference(cumulative, self.initial)
        status_path = self.operations / TARGET / "status.json"
        status = _json(status_path) if status_path.exists() else {"status": "NOT_STARTED"}
        before = self.directory / "retry_counts_before.json"
        retry = difference(status.get("counts_after", cumulative), _json(before)) if before.exists() else {}
        report = dict(target=TARGET, retry_status=status["status"], original_raw_counts=self.root_counts,
                    starting_raw_counts=self.initial,
                    new_raw_counts=new, retry_raw_counts=retry,
                    later_continuation_raw_counts=difference(new, retry), cumulative_raw_counts=cumulative,
                    original_physical_counts=physical_counts(self.root_counts),
                    new_physical_counts=physical_counts(new), retry_physical_counts=physical_counts(retry),
                    cumulative_physical_counts=physical_counts(cumulative),
                    original_maxima_apply_to_cumulative=False,
                    cost_note="Cumulative includes the original interrupted work and the one authorized retry; original 40/258 cap is not claimed.")
        if self.compatibility:
            previous_delta = difference(self.initial, self.root_counts)
            report.update(previous_failed_repair_raw_counts=previous_delta,
                          previous_failed_repair_physical_counts=physical_counts(previous_delta),
                          previous_repair_cumulative_raw_counts=self.initial,
                          cost_note="Cumulative retains original interrupted work, the failed import's loader cost, and this one compatibility attempt plus subsequent P2 work; original 40/258 cap is not claimed.")
        return report

    def persist_accounting(self):
        atomic_json(self.directory / "continuation_counts.json", self.accounting())

    def inventory(self):
        old = [dict(_json(path), checkpoint_origin="original")
               for path in sorted((self.original / "operations").rglob("status.json"))]
        if self.compatibility:
            old += [dict(_json(path), checkpoint_origin="previous_failed_repair")
                    for path in sorted((self.original / REPAIR / "operations").rglob("status.json"))]
        origin = "compatibility_continuation" if self.compatibility else "continuation"
        return old + [dict(row, checkpoint_origin=origin) for row in super().inventory()]
