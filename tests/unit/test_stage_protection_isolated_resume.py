import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from cegwm.runtime.stage_protection_isolated_resume import IsolatedStore, ISOLATED_REPAIR, ISOLATED_TARGET, EARLY1, DEPENDENCY_REPAIRS
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


@pytest.fixture
def missing_design(early2_interrupted):
    """Observed Drive state: STARTED image with no payload, empty read folder."""
    root, config = early2_interrupted
    previous = root / COMPAT_REPAIR / "operations"
    for key in DEPENDENCY_REPAIRS:
        (previous / key / "result.pt").unlink()
        (previous / key / "status.json").unlink()
    atomic_json(previous / DEPENDENCY_REPAIRS[0] / "status.json", dict(
        key=DEPENDENCY_REPAIRS[0], status="STARTED", counts_before=dict(
            sdxl_unet_forward_attempted=42, sdxl_unet_forward_returned=42,
            sdxl_ordinary_forward_calls=5, sdxl_ordinary_forward_returned=5)))
    return root, config


class DesignRecoveryWorker:
    """CPU-only stand-in behind the actual process/disk transport."""
    fail_at = None

    def __init__(self, config, *, counters, disk_storage, resource_observer):
        self.config, self.counters, self.disk = config, counters, disk_storage

    def count(self, name, amount=1):
        self.counters[name] = self.counters.get(name, 0) + amount

    def reconstruct(self, image, seed):
        assert image.getpixel((0, 0)) == (127, 128, 129)
        assert seed == self.config["reconstruction"]["design_seed"]
        self.count("sdxl_ordinary_forward_calls")
        self.count("sdxl_unet_forward_attempted", 6)
        if self.fail_at == "reconstruct":
            raise RuntimeError("synthetic design reconstruction failure")
        self.count("sdxl_unet_forward_returned", 6)
        self.count("sdxl_ordinary_forward_returned")
        return Image.new("RGB", image.size, (21, 22, 23))

    def read(self, image):
        # The clean cache must be reused; only the new design image is read.
        assert image.getpixel((0, 0)) == (21, 22, 23)
        self.count("reader_result_ordinary_calls")
        if self.fail_at == "read":
            raise RuntimeError("synthetic design read failure")
        self.count("reader_result_ordinary_returned")
        return dict(feature=torch.ones(3), score=0.2)

    def rgb_gradient(self, image, ordinary_read, *, design_image, design_read):
        from cegwm.runtime.channel_precompensation_sd35 import RGBSensitivity
        assert image.getpixel((0, 0)) == (127, 128, 129)
        assert design_image.getpixel((0, 0)) == (21, 22, 23)
        assert ordinary_read["score"] == 0.1 and design_read["score"] == 0.2
        self.count("sdxl_unet_forward_attempted", 6)
        if self.fail_at == "rgb_gradient":
            raise RuntimeError("synthetic target failure")
        self.count("sdxl_unet_forward_returned", 6)
        return RGBSensitivity(design_image, torch.ones(3), 0.2, torch.ones(1, 3, 2, 2))


class FailedDesignImageWorker(DesignRecoveryWorker):
    fail_at = "reconstruct"


class FailedDesignReadWorker(DesignRecoveryWorker):
    fail_at = "read"


class FailedTargetWorker(DesignRecoveryWorker):
    fail_at = "rgb_gradient"


@pytest.fixture
def recovery_execute(monkeypatch):
    from experiments import run_stage_protection_v2_isolated_resume as runner
    from cegwm.runtime.stage_protection_process import ProcessProvider
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(root / "src"), str(root), str(Path(__file__).parent)]))

    class BoundaryCanary(runner.IsolatedCanary):
        """Exercise the recovery boundary, not the rest of the scientific P2."""
        def freeze(self):
            def unexpected_compute():
                pytest.fail("persisted early1 VJP/action/render must not compute")
            try:
                for key in EARLY1[:2]:
                    self.op(key, unexpected_compute)
                rendered = self.op(EARLY1[2], unexpected_compute)
                observed = self.observe(EARLY1[2].rsplit("/", 1)[0], rendered["image"])
                self.design_gradient(ISOLATED_TARGET, observed)
            except Exception:
                pytest.fail("recovery error must escape the runner's ordinary Exception handler")
            atomic_json(self.store.directory / "test_frozen.json", {"denominator": 4})
            return {"trials": [None] * 4}

        def evaluate(self, frozen):
            assert len(frozen["trials"]) == 4
            return dict(plan={}, engineering_complete=True)

    monkeypatch.setattr(runner, "IsolatedCanary", BoundaryCanary)
    def execute(root, config, worker="DesignRecoveryWorker"):
        def factory(config, **kwargs):
            return ProcessProvider(config, **kwargs, factory="test_stage_protection_isolated_resume:" + worker)
        return runner.execute(config, root, provider_factory=factory, scratch_root=root / "scratch")
    return execute


def test_execute_recovers_only_missing_design_dependencies_once(missing_design, recovery_execute):
    root, config = missing_design
    old = snapshot(root)
    result = recovery_execute(root, config)
    cost = result["continuation"]
    assert cost["starting_raw_counts"]["sdxl_unet_forward_attempted"] == 54
    assert cost["dependency_recomputation_raw_counts"]["sdxl_unet_forward_attempted"] == 6
    assert cost["dependency_recomputation_raw_counts"]["reader_result_ordinary_calls"] == 1
    assert cost["retry_raw_counts"]["sdxl_unet_forward_attempted"] == 6
    assert cost["cumulative_raw_counts"]["sdxl_unet_forward_attempted"] == 66
    assert cost["cumulative_raw_counts"]["sd35_full_generation_calls"] == 1
    assert cost["cumulative_raw_counts"]["sd35_suffix_vjp_calls"] == 3
    assert all(cost["dependency_recomputations"][key]["status"] == "RETURNED" for key in DEPENDENCY_REPAIRS)
    directory = root / ISOLATED_REPAIR
    receipts = [json.loads(p.read_text()) for p in (directory / "workers").glob("*/parent_status.json")]
    assert len(receipts) == 3
    assert all(r["child_reaped"] and r["scratch_removed"] for r in receipts)
    # Reentry before an aggregate also exercises own RETURNED operation reuse.
    (directory / "summary.json").unlink()
    recovery_execute(root, config)
    assert len(list((directory / "workers").glob("*/parent_status.json"))) == 3
    assert {name: (root / name).read_bytes() for name in old} == old


@pytest.mark.parametrize("worker,failed_key", [
    ("FailedDesignImageWorker", DEPENDENCY_REPAIRS[0]),
    ("FailedDesignReadWorker", DEPENDENCY_REPAIRS[1]),
    ("FailedTargetWorker", ISOLATED_TARGET),
])
def test_execute_failure_stops_before_freeze_and_keeps_spent_cost(missing_design, recovery_execute, worker, failed_key):
    root, config = missing_design
    old = snapshot(root)
    with pytest.raises(RepairStopped, match="no automatic retry"):
        recovery_execute(root, config, worker)
    directory = root / ISOLATED_REPAIR
    assert not (directory / "test_frozen.json").exists()
    assert not (directory / "selection_frozen.pt").exists()
    assert not (directory / "summary.json").exists()
    assert json.loads((directory / "operations" / failed_key / "status.json").read_text())["status"] == "FAILED"
    cost = json.loads((directory / "continuation_counts.json").read_text())
    assert cost["cumulative_raw_counts"]["sdxl_unet_forward_attempted"] >= 60
    assert cost["dependency_recomputation_raw_counts"]["sdxl_unet_forward_attempted"] == 6
    before_retry = snapshot(directory)
    with pytest.raises(PreflightBlocked, match="already spent"):
        recovery_execute(root, config, worker)
    assert snapshot(directory) == before_retry
    assert {name: (root / name).read_bytes() for name in old} == old


@pytest.mark.parametrize("name", ["result.pt", "result.pt.tmp", "partial.pt"])
def test_saved_design_payload_with_stale_status_reused(missing_design, name):
    root, config = missing_design
    key = DEPENDENCY_REPAIRS[0]
    path = root / COMPAT_REPAIR / "operations" / key / name
    atomic_torch(path, Image.new("RGB", (2, 2)))
    old = snapshot(root)
    store = IsolatedStore(root, config)
    store.call(key, lambda: pytest.fail("usable payload must be recovered without compute"))
    assert not (store.operations / key / "status.json").exists()
    assert key not in store.checked["cache_scan"]["dependency_recompute_required"]
    assert {name: (root / name).read_bytes() for name in old} == old


def test_all_missing_caches_reported_before_reservation(missing_design, capsys):
    root, config = missing_design
    for key in EARLY1[:2]:
        (root / COMPAT_REPAIR / "operations" / key / "result.pt").unlink()
    with pytest.raises(PreflightBlocked) as error:
        IsolatedStore(root, config)
    assert all(key in str(error.value) for key in EARLY1[:2] + DEPENDENCY_REPAIRS)
    assert "ISOLATED_CHECKPOINT_SCAN" in capsys.readouterr().out
    assert not (root / ISOLATED_REPAIR).exists()
