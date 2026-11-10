"""Only the observed load-only import failure gets the fixed compatibility entry."""
import json

from PIL import Image
import pytest
import torch

from cegwm.runtime.channel_precompensation_sd35 import RuntimeStageError
from cegwm.runtime.stage_protection_resume import (
    COMPAT_REPAIR, REPAIR, TARGET, REQUIRED, RepairStore, RepairStopped, PreflightBlocked,
)
from cegwm.runtime.stage_protection_store import atomic_json, atomic_torch
from experiments import run_stage_protection_v2_resume as runner
from test_stage_protection_resume import checkpoint_run, retry_value, snapshot

pytestmark = pytest.mark.unit
IMPORT_MESSAGE = ("cannot import name 'is_non_overlapping_and_dense' from 'torch._prims_common' "
                  "(/usr/local/lib/python3.13/dist-packages/torch/_prims_common/__init__.py)")


@pytest.fixture
def import_failed_run(checkpoint_run):
    root, config = checkpoint_run
    store = RepairStore(root, config)
    def import_failure():
        store.counters["source_pipeline_load_attempted"] += 1
        store.counters["source_pipeline_load_returned"] += 1
        raise RuntimeStageError("source_suffix_vjp", ImportError(IMPORT_MESSAGE),
            {"before_state": {"latents": torch.ones(1, 3, 2, 2)}}, store.counters)
    with pytest.raises(RepairStopped):
        store.call(TARGET, import_failure)
    return root, config


@pytest.mark.parametrize("keep_partial", [True, False])
def test_compatibility_correct_start_retains_prior_cost_and_all_three_attempts(import_failed_run, keep_partial):
    root, config = import_failed_run
    partial = root / REPAIR / "operations" / TARGET / "partial.pt"
    assert partial.exists()
    if not keep_partial:
        partial.unlink()  # Fixture variant: partial evidence is optional, never a result.
    old = snapshot(root)
    store = RepairStore(root, config, compatibility=True)
    assert store.directory == root / COMPAT_REPAIR
    assert store.counters["source_pipeline_load_attempted"] == 7
    assert store.counters["sd35_transformer_attempted"] == 46
    for key in REQUIRED:
        store.call(key, lambda: pytest.fail("no completed checkpoint computation"))
    assert not (store.operations / TARGET / "status.json").exists()  # Lazy target only.
    def retry():
        assert json.loads((store.operations / TARGET / "status.json").read_text())["status"] == "STARTED"
        store.counters["source_pipeline_load_attempted"] += 1
        store.counters["source_pipeline_load_returned"] += 1
        store.counters["sd35_transformer_attempted"] += 11
        store.counters["sd35_transformer_returned"] += 11
        return retry_value()
    store.call(TARGET, retry)
    store = RepairStore(root, config, compatibility=True)
    store.call(TARGET, lambda: pytest.fail("new returned target must not compute again"))
    store.call("source07_coastal_scene/unmarked/design_image", lambda: Image.new("RGB", (2, 2)))
    cost = store.accounting()
    assert cost["original_raw_counts"]["source_pipeline_load_attempted"] == 6
    assert cost["starting_raw_counts"]["source_pipeline_load_attempted"] == 7
    assert cost["previous_failed_repair_raw_counts"]["source_pipeline_load_attempted"] == 1
    assert cost["new_raw_counts"]["source_pipeline_load_attempted"] == 1
    assert cost["cumulative_raw_counts"]["source_pipeline_load_attempted"] == 8
    assert cost["cumulative_physical_counts"]["sd3_denoising_steps_attempted"] == 57
    assert cost["cumulative_physical_counts"]["sd3_denoising_steps_returned"] == 56
    assert cost["original_maxima_apply_to_cumulative"] is False
    assert {name: (root / name).read_bytes() for name in old} == old
    rows = [row for row in store.inventory() if row["key"] == TARGET]
    assert [row["status"] for row in rows] == ["STARTED", "FAILED", "RETURNED"]
    assert [row["checkpoint_origin"] for row in rows] == ["original", "previous_failed_repair", "compatibility_continuation"]
    with pytest.raises(PreflightBlocked, match="no second retry"):
        RepairStore(root, config)  # Original entry is deliberately unchanged.


@pytest.mark.parametrize("damage", [
    "wrong_error", "wrong_stage", "wrong_symbol", "started", "returned", "counter_disagrees",
    "nonzero_transformer", "nonzero_vjp", "nonzero_unknown", "negative_delta", "no_load_return",
    "extra_load", "prior_start_changed", "frozen", "frozen_tmp", "extra_operation", "orphan_operation",
    "result_present", "result_tmp", "config_changed", "root_counter_changed", "missing_prerequisite",
])
def test_other_failures_or_computed_work_block_before_provider(import_failed_run, damage):
    root, config = import_failed_run
    previous = root / REPAIR
    status_path = previous / "operations" / TARGET / "status.json"
    status = json.loads(status_path.read_text())
    counts_path = previous / "physical_counts.json"
    counts = json.loads(counts_path.read_text())
    if damage in ("wrong_error", "wrong_stage", "wrong_symbol"):
        status["error"] = {"wrong_error": "RuntimeError: OOM",
            "wrong_stage": status["error"].replace("source_suffix_vjp", "other_stage"),
            "wrong_symbol": status["error"].replace("is_non_overlapping_and_dense", "another_symbol")}[damage]
    elif damage in ("started", "returned"):
        status["status"] = damage.upper()
    elif damage == "counter_disagrees":
        counts["source_pipeline_load_attempted"] += 1
    elif damage.startswith("nonzero_"):
        key = {"nonzero_transformer": "sd35_transformer_attempted", "nonzero_vjp": "sd35_suffix_vjp_calls",
               "nonzero_unknown": "unrecognized_computation_calls"}[damage]
        counts[key] = counts.get(key, 0) + 1
        status["counts_after"] = dict(counts)
    elif damage in ("no_load_return", "extra_load", "negative_delta"):
        key = {"no_load_return": "source_pipeline_load_returned", "extra_load": "source_pipeline_load_attempted",
               "negative_delta": "sd35_transformer_attempted"}[damage]
        counts[key] += 1 if damage == "extra_load" else -1
        status["counts_after"] = dict(counts)
    elif damage == "prior_start_changed":
        atomic_json(previous / "retry_counts_before.json", counts)
    elif damage in ("frozen", "frozen_tmp"):
        atomic_torch(previous / ("selection_frozen.pt.tmp" if damage == "frozen_tmp" else "selection_frozen.pt"), {"trials": []})
    elif damage in ("extra_operation", "orphan_operation"):
        folder = previous / "operations/source07_coastal_scene/source"
        folder.mkdir(parents=True)
        if damage == "extra_operation":
            atomic_json(folder / "status.json", dict(key="source07_coastal_scene/source", status="STARTED"))
        else:
            atomic_torch(folder / "partial.pt", {})
    elif damage in ("result_present", "result_tmp"):
        atomic_torch(status_path.parent / ("result.pt.tmp" if damage == "result_tmp" else "result.pt"), retry_value())
    elif damage == "config_changed":
        atomic_json(previous / "config.json", {})
    elif damage == "root_counter_changed":
        atomic_json(root / "physical_counts.json", counts)
    elif damage == "missing_prerequisite":
        (root / "operations" / REQUIRED[0] / "result.pt").unlink()
    atomic_json(counts_path, counts)
    atomic_json(status_path, status)
    with pytest.raises(PreflightBlocked):
        runner.execute(config, root, compatibility=True,
            provider_factory=lambda *a, **k: pytest.fail("all compatibility checks precede provider"))
    assert not (root / COMPAT_REPAIR).exists()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_new_compatibility_failure_or_interruption_cannot_loop(import_failed_run, failure):
    root, config = import_failed_run
    store = RepairStore(root, config, compatibility=True)
    def fail():
        store.counters["sd35_transformer_attempted"] += 1
        raise failure("new attempt failed")
    with pytest.raises(RepairStopped if failure is RuntimeError else KeyboardInterrupt):
        store.call(TARGET, fail)
    assert store.counters["source_pipeline_load_attempted"] == 7
    with pytest.raises(RepairStopped):
        store.call(TARGET, lambda: pytest.fail("same instance cannot retry"))
    with pytest.raises(PreflightBlocked, match="no second retry"):
        runner.execute(config, root, compatibility=True,
            provider_factory=lambda *a, **k: pytest.fail("no provider after spent attempt"))


def test_compatibility_requires_prior_failure_not_just_original_started(checkpoint_run):
    root, config = checkpoint_run
    with pytest.raises(PreflightBlocked):
        RepairStore(root, config, compatibility=True)
    assert not (root / COMPAT_REPAIR).exists()
