"""CPU fake-provider tests for the fixed one-source method scope."""
import json

import pytest

from experiments import run_channel_precompensation_single_source_v1 as single
from test_channel_precompensation_runner import FakeProvider, config

pytestmark = pytest.mark.unit


def test_one_source_two_bits_each_method_and_remaining_roster_retained(tmp_path):
    settings = config()
    provider = FakeProvider(settings)
    result = single.execute_single(settings, tmp_path / "new", resume_from=None, provider=provider)
    assert [row["source_id"] for row in result["sources"]] == [settings["sources"][0]["source_id"]]
    assert len(result["full_roster"]) == 4
    assert [row["scope_status"] for row in result["full_roster"]][1:] == ["not_measured_in_this_scope"] * 3
    assert len(provider.write_requests) == 4
    assert result["fixed_denominators"] == {"bit_trials_per_method": 2, "both_bits_per_source_per_method": 1}
    assert result["counts_terminal"]
    assert result["method_validation"]["marked_rgb_returned"] == 4
    assert not result["method_validation"]["four_source_experiment_complete"]
    for trials in result["sources"][0]["trials"].values():
        for trial in trials.values():
            assert trial["collection_complete"]
            assert not trial["quality"]["gate_enabled"]
            assert len(trial["response_assessment"]["channels"]) == 2
            for channel in trial["response_assessment"]["channels"]:
                assert abs(channel["residual"]) < 1e-12
    saved = json.loads((tmp_path / "new/summary.json").read_text())
    assert saved == result


def test_failed_preparation_consumes_no_write_and_keeps_two_bit_denominator(tmp_path):
    settings = config()
    provider = FakeProvider(settings, preparation_failure=settings["sources"][0]["source_id"])
    result = single.execute_single(settings, tmp_path / "new", resume_from=None, provider=provider)
    assert len(provider.write_requests) == 0
    for method in single.METHODS:
        assert result["method_summaries"][method]["expected_bit_trials"] == 2
        assert result["method_summaries"][method]["collected_bit_trials"] == 0
        for trial in result["sources"][0]["trials"][method].values():
            assert trial["response_assessment"]["mechanisms"] == ["ACQUISITION_OR_PREPARATION_FAILURE"]


def test_holdout_failure_is_distinguished_from_design_prediction(tmp_path):
    settings = config()
    result = single.execute_single(settings, tmp_path / "new", resume_from=None,
                                   provider=FakeProvider(settings, holdout_shift=1))
    trials = result["sources"][0]["trials"]
    assert any("HOLDOUT_RECONSTRUCTION_TARGET_MISS" in trial["response_assessment"]["mechanisms"]
               for arms in trials.values() for trial in arms.values())
    for arms in trials.values():
        for trial in arms.values():
            assert not trial["blind_all_conditions_correct"]
            assert not any(row["opposite_sign"] for row in trial["response_assessment"]["channels"])


def test_quantization_collapse_is_retained_without_new_write(tmp_path):
    settings = config()
    provider = FakeProvider(settings, collapsed=True)
    result = single.execute_single(settings, tmp_path / "new", resume_from=None, provider=provider)
    assert len(provider.write_requests) == 4
    for arms in result["sources"][0]["trials"].values():
        trial = arms["1"]
        assert "ACTUAL_DTYPE_WRITE_COLLAPSED" in trial["response_assessment"]["mechanisms"]
        assert not trial["eligible_joint_success"]


def test_existing_output_cli_reads_without_execution(tmp_path, monkeypatch, capsys):
    settings = tmp_path / "config.json"
    settings.write_text(json.dumps(config()))
    output = tmp_path / "existing"
    output.mkdir()
    (output / "summary.json").write_text('{"retained": true}')
    monkeypatch.setattr(single, "execute_single", lambda *a, **k: pytest.fail("unexpected repeat"))
    assert single.main(["--config", str(settings), "--resume-from", str(tmp_path / "old"),
                        "--output", str(output), "--execute"]) == 0
    assert json.loads(capsys.readouterr().out) == {"retained": True}


def test_interrupt_preserves_nonterminal_counts(tmp_path):
    class Interrupted(FakeProvider):
        def prepare_sources(self, sources):
            raise KeyboardInterrupt()
            yield
    settings = config()
    with pytest.raises(KeyboardInterrupt):
        single.execute_single(settings, tmp_path / "new", resume_from=None, provider=Interrupted(settings))
    summary = json.loads((tmp_path / "new/summary.json").read_text())
    assert summary["counts_terminal"] is False
    assert summary["new_counts_status"] == "interrupted"
    assert summary["stages"]["prepare"] == "started"
