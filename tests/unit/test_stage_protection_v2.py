import json
import math
from pathlib import Path

import pytest
import torch

from cegwm.stage_protection_v2 import (
    cast_action, common_accept, early_update, grouped_verdict, protection_radius, solve_intervals,
)
from cegwm.runtime.stage_protection_store import OperationStore, Unavailable, atomic_json

pytestmark = pytest.mark.unit


def test_fixed_slabs_minimum_norm_and_protection_inactive_or_stricter():
    # x repairs clean; y cancels damage to design. A=[1,0], B=[1,-.25], C=[1,-.375].
    g = torch.tensor([[1., 0.], [0.5, 1.]])
    a = solve_intervals(g[:1], [-1.25], [0], [.25], 2)
    b = solve_intervals(g, [-1.25, 0], [0, 0], [.25, .25], 2)
    c = solve_intervals(g, [-1.25, 0], [0, 0], [.25, .125], 2)
    torch.testing.assert_close(a["delta"], torch.tensor([1., 0.], dtype=torch.float64))
    torch.testing.assert_close(b["delta"], torch.tensor([1., -.25], dtype=torch.float64))
    torch.testing.assert_close(c["delta"], torch.tensor([1., -.375], dtype=torch.float64))
    assert a["norm"] < b["norm"] < c["norm"]
    assert solve_intervals(g, [-1.25, 0], [0, 0], [.25, .125], 1)["status"] == "INFEASIBLE_BUDGET"
    c_same = solve_intervals(g, [-1.25, 0], [0, 0], [.25, .25], 2)
    assert torch.equal(b["delta"], c_same["delta"])
    impossible = solve_intervals(torch.ones(2, 3), [0, 0], [0, 2], [.25, .25], 9)
    assert impossible["status"] == "INFEASIBLE"


def test_fixed_cell_acceptance_never_uses_C_private_interval():
    assert protection_radius(0, 0) == .125
    assert protection_radius(.25, 0) == .25
    reads = {"clean": {"fixed_inner": True}, "design": {"fixed_inner": True}}
    assert common_accept(reads, .005, .006)
    assert not common_accept(reads, .008, 0)
    # .20 is outside C's private .125 but valid for the shared .25 task.
    b = solve_intervals(torch.eye(2), [-1, .20], [0, 0], [.25, .25], 1)
    assert b["status"] == "SOLVED" and b["predicted"][1] == pytest.approx(.20)


def test_actual_cast_and_fixed_unmarked_denominator_and_cumulative_early():
    base = torch.tensor([1.], dtype=torch.float16)
    result = cast_action(base, torch.tensor([.0005]), torch.tensor([.1]), rho=.01)
    assert result["actual"].item() == 2 ** -10
    assert not result["within_budget"]  # Requested fits; actual cast does not.
    g = torch.tensor([2., 0.])
    first = early_update(torch.zeros(2), g, -1, 0, 2)
    second = early_update(first, torch.tensor([0., 1.]), -.3, 0, 2)
    torch.testing.assert_close(second, torch.tensor([.5, .3], dtype=torch.float64))
    clipped = early_update(second, g, -10, 0, .2)
    assert torch.linalg.vector_norm(clipped).item() == pytest.approx(.2)
    with pytest.raises(ValueError, match="zero early"):
        early_update(first, torch.zeros(2), -1, 0, 2)


def test_two_contents_strict_improvement_and_no_other_bit_regression():
    rows = [dict(source_id=s, bit=b, evaluation_complete=True, mechanism_case=b == 1,
                 C_success=b == 1, B_success=False) for s in ("a", "b") for b in (0, 1)]
    assert grouped_verdict(rows, ["a", "b"])["operational_P2_signal"]
    rows[2]["B_success"] = True
    assert not grouped_verdict(rows, ["a", "b"])["operational_P2_signal"]
    rows[2]["B_success"] = False
    rows[3]["mechanism_case"] = False
    assert not grouped_verdict(rows, ["a", "b"])["operational_P2_signal"]
    assert grouped_verdict(rows[:2], ["a", "b"])["bit_trial_denominator"] == 4


def test_completed_cache_failure_and_hard_interrupt_are_never_reexecuted(tmp_path):
    store = OperationStore(tmp_path, {"frozen": 1})
    count = []
    assert store.call("source/ok", lambda: count.append(1) or torch.tensor([4])) == 4
    assert store.call("source/ok", lambda: pytest.fail("rerun")) == 4
    def fail():
        store.counters["attempted"] = 1
        raise RuntimeError("OOM")
    with pytest.raises(Unavailable):
        store.call("source/fail", fail)
    with pytest.raises(Unavailable):
        store.call("source/fail", lambda: pytest.fail("retry"))
    started = tmp_path / "operations/source/interrupted"
    started.mkdir(parents=True)
    atomic_json(started / "status.json", {"status": "STARTED"})
    resumed = OperationStore(tmp_path, {"frozen": 1})
    with pytest.raises(Unavailable, match="STARTED"):
        resumed.call("source/interrupted", lambda: pytest.fail("retry"))
    assert resumed.counters["attempted"] == 1 and count == [1]
    with pytest.raises(ValueError, match="configuration differs"):
        OperationStore(tmp_path, {"frozen": 2})
