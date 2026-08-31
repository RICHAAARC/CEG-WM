from pathlib import Path
import subprocess

import pytest

from cegwm.baselines.external_canary import OFFICIAL_EXACTS, require_official_source


def test_official_source_requires_exact_detached_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"; source.mkdir()
    (source / ".git").mkdir()
    class Result:
        def __init__(self, text: str, code: int=0): self.stdout=text; self.returncode=code
    values = iter((Result(OFFICIAL_EXACTS["tree_ring"]+"\n"), Result("",1), Result("")))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: next(values))
    require_official_source("tree_ring", source)


def test_clear_lock_is_available_before_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cegwm.baselines import external_canary
    monkeypatch.setattr("sys.argv", ["external_canary", "--method", "tree_ring", "--run-dir", str(tmp_path), "--project-exact", "a"*40, "--official-source", str(tmp_path), "--clear-stale-lock"])
    (tmp_path / ".run.lock").write_text('{"pid":1,"token":"stale"}')
    external_canary.main()


def test_fixed_run_id_and_carrier_commitment_are_required() -> None:
    source = Path("src/cegwm/baselines/external_canary.py").read_text()
    assert "run_id != RUN_ID_DEFAULTS[args.method]" in source
    assert '"carrier_digest":c.digest' in source
