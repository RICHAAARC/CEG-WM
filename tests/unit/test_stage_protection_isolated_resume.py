import json

import pytest
import torch

from cegwm.runtime.stage_protection_isolated_resume import IsolatedStore, ISOLATED_REPAIR, ISOLATED_TARGET, EARLY1
from cegwm.runtime.stage_protection_resume import COMPAT_REPAIR, RepairStore, RepairStopped, PreflightBlocked
from cegwm.runtime.stage_protection_store import atomic_json, atomic_torch
from test_stage_protection_resume import checkpoint_run, retry_value, snapshot
from test_stage_protection_compat_resume import import_failed_run

pytestmark = pytest.mark.unit


@pytest.fixture
def early2_interrupted(import_failed_run):
    root, config = import_failed_run
    store = RepairStore(root, config, compatibility=True)
    final = dict(store.counters)
    final.update(source_pipeline_load_attempted=8, source_pipeline_load_returned=8,
        sd35_transformer_attempted=68, sd35_transformer_returned=67,
        sd35_suffix_forward_calls=9, sd35_suffix_forward_returned=8,
        sd35_suffix_vjp_calls=3, sd35_suffix_vjp_returned=3,
        sdxl_unet_forward_attempted=48, sdxl_unet_forward_returned=48,
        sdxl_input_vae_encode_attempted=8, sdxl_input_vae_encode_returned=8,
        sdxl_output_vae_decode_attempted=8, sdxl_output_vae_decode_returned=8,
        sdxl_reconstruction_forward_calls=2, sdxl_reconstruction_forward_returned=2)
    for key, value in final.items():
        store.counters[key] = value
    source = "source06_fur_knit"
    aliases = dict(action=source + "/bit0/late/A/action", render=source + "/canonical_early_zero",
                   clean=source + "/unmarked/clean", design_image=source + "/unmarked/design_image",
                   design_read=source + "/unmarked/design_read")
    for key in EARLY1:
        name = key.rsplit("/", 1)[-1]
        value = retry_value() if name == "latent_vjp" else torch.load(root / "operations" / aliases[name] / "result.pt", weights_only=False)
        store.call(key, lambda: value)
    folder = store.operations / ISOLATED_TARGET
    folder.mkdir(parents=True)
    for key, value in dict(sdxl_unet_forward_attempted=54, sdxl_unet_forward_returned=54,
        sdxl_input_vae_encode_attempted=9, sdxl_input_vae_encode_returned=9,
        sdxl_output_vae_decode_attempted=9, sdxl_reconstruction_forward_calls=3).items():
        store.counters[key] = value
    # No persist_accounting: the old derived report genuinely lags counters.
    return root, config


@pytest.mark.parametrize("status", [None, "STARTED", "FAILED"])
def test_layered_cache_and_latest_cost_with_missing_started_failed_status(early2_interrupted, status):
    root, config = early2_interrupted
    previous = root / COMPAT_REPAIR
    if status:
        atomic_json(previous / "operations" / ISOLATED_TARGET / "status.json", dict(key=ISOLATED_TARGET, status=status))
    old = snapshot(root)
    store = IsolatedStore(root, config)
    for key in EARLY1:
        store.call(key, lambda: pytest.fail("completed early1 must never compute"))
    assert store.counters["sdxl_unet_forward_attempted"] == 54
    assert store.counters["sdxl_output_vae_decode_attempted"] == 9
    assert store.counters["sdxl_output_vae_decode_returned"] == 8
    assert store.accounting()["previous_derived_report_lag"]["sdxl_unet_forward_attempted"] == 6
    saved = torch.load(root / "operations/source06_fur_knit/bit1/early1/rgb_vjp/result.pt", weights_only=False)
    def retry():
        store.counters["sdxl_unet_forward_attempted"] += 6
        store.counters["sdxl_unet_forward_returned"] += 6
        return saved
    store.call(ISOLATED_TARGET, retry)
    resumed = IsolatedStore(root, config)
    resumed.call(ISOLATED_TARGET, lambda: pytest.fail("returned new attempt must be cached"))
    assert resumed.counters["sdxl_unet_forward_attempted"] == 60
    assert resumed.counters["sd35_full_generation_calls"] == 1
    assert resumed.accounting()["retry_raw_counts"]["sdxl_unet_forward_attempted"] == 6
    assert {name: (root / name).read_bytes() for name in old} == old
    if status is None:
        assert any(row["status"] == "MISSING_UNSYNCED" for row in resumed.inventory())


def test_usable_result_without_synced_status_is_reused(early2_interrupted):
    root, config = early2_interrupted
    value = torch.load(root / "operations/source06_fur_knit/bit1/early1/rgb_vjp/result.pt", weights_only=False)
    atomic_torch(root / COMPAT_REPAIR / "operations" / ISOLATED_TARGET / "result.pt", value)
    store = IsolatedStore(root, config)
    store.call(ISOLATED_TARGET, lambda: pytest.fail("never recompute existing usable target result"))
    assert not (store.operations / ISOLATED_TARGET / "status.json").exists()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_new_attempt_failure_consumed(early2_interrupted, failure):
    root, config = early2_interrupted
    store = IsolatedStore(root, config)
    def fail():
        store.counters["sdxl_unet_forward_attempted"] += 1
        raise failure("interrupted worker")
    with pytest.raises(RepairStopped if failure is RuntimeError else KeyboardInterrupt):
        store.call(ISOLATED_TARGET, fail)
    with pytest.raises(PreflightBlocked, match="already spent"):
        IsolatedStore(root, config)
    assert json.loads((root / ISOLATED_REPAIR / "physical_counts.json").read_text())["sdxl_unet_forward_attempted"] == 55


def test_missing_completed_early1_blocks_before_provider(early2_interrupted):
    from experiments.run_stage_protection_v2_isolated_resume import execute
    root, config = early2_interrupted
    (root / COMPAT_REPAIR / "operations" / EARLY1[0] / "result.pt").unlink()
    with pytest.raises(PreflightBlocked):
        execute(config, root, provider_factory=lambda *a, **k: pytest.fail("missing cache must not load model"))
