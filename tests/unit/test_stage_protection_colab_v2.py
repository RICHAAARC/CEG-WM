"""Credential routing and durable output checks; no network/model execution."""
import json
from pathlib import Path
import sys

import pytest
import requests
import huggingface_hub
from huggingface_hub import constants
from huggingface_hub.utils import HfHubHTTPError

from cegwm.runtime.stage_protection_colab import ensure_model_access, record_setup_errors, run_logged
from experiments.run_stage_protection_v2 import load_config

pytestmark = pytest.mark.unit


class Access:
    def __init__(self, allowed, status=403):
        self.allowed, self.status, self.calls = allowed, status, []

    def auth_check(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["token"] not in self.allowed:
            response = requests.Response()
            response.status_code = self.status
            raise HfHubHTTPError("synthetic model access failure", response=response)


@pytest.fixture
def clean_auth(tmp_path, monkeypatch):
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    cache = tmp_path / "token"
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(cache))
    return cache


@pytest.mark.parametrize("source", ["environment", "existing_login", "colab_secret"])
def test_reuse_credentials_without_login(source, clean_auth, monkeypatch, capsys):
    token = "synthetic-private-credential"
    secret_calls = []
    if source == "environment":
        monkeypatch.setenv("HF_TOKEN", token)
    elif source == "existing_login":
        monkeypatch.setenv("HF_TOKEN", "stale-environment-credential")
        clean_auth.write_text(token)
    def secret(name):
        secret_calls.append(name)
        return token
    api = Access({token})
    monkeypatch.setattr(huggingface_hub, "login", lambda **_: pytest.fail("unnecessary login"))
    result = ensure_model_access(load_config(), api=api, secret_getter=secret,
                                 prompt=lambda _: pytest.fail("unnecessary prompt"))
    assert result["credential_source"] == source
    assert len(result["models_checked"]) == 2
    assert secret_calls == (["HF_TOKEN"] if source == "colab_secret" else [])
    assert token not in json.dumps(result) + capsys.readouterr().out


def test_login_prompt_only_after_real_access_denial(clean_auth, monkeypatch):
    token = "synthetic-new-login-credential"
    api, logins = Access({token}), []
    def prompt(_):
        assert api.calls and api.calls[-1]["token"] is False
        return token
    monkeypatch.setattr(huggingface_hub, "login", lambda **kwargs: logins.append(kwargs))
    result = ensure_model_access(load_config(), api=api, secret_getter=lambda _: None, prompt=prompt)
    assert result["credential_source"] == "necessary_login"
    assert logins == [{"token": token, "add_to_git_credential": False}]


def test_network_failure_is_saved_without_requesting_login(clean_auth, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "synthetic-existing-credential")
    with pytest.raises(HfHubHTTPError):
        with record_setup_errors(tmp_path):
            ensure_model_access(load_config(), api=Access(set(), status=503),
                secret_getter=lambda _: pytest.fail("network failure must not ask for a secret"),
                prompt=lambda _: pytest.fail("network failure must not ask for login"))
    log = (tmp_path / "setup.log").read_text()
    assert "Traceback" in log and "synthetic model access failure" in log
    assert "synthetic-existing-credential" not in log


def test_runner_stdout_stderr_traceback_and_prior_logs_survive(tmp_path):
    command = [sys.executable, "-u", "-c",
               "import sys; print('saved stdout'); print('saved stderr', file=sys.stderr); raise RuntimeError('saved traceback')"]
    assert run_logged(command, cwd=tmp_path, output=tmp_path) == 1
    first = (tmp_path / "runner.log").read_text()
    assert "saved stdout" in first and "saved stderr" in first and "Traceback" in first
    assert run_logged([sys.executable, "-c", "print('second invocation')"], cwd=tmp_path, output=tmp_path) == 0
    second = (tmp_path / "runner.log").read_text()
    assert second.startswith(first) and "second invocation" in second


def test_process_launch_failure_also_leaves_traceback(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_logged([str(tmp_path / "nonexistent-python")], cwd=tmp_path, output=tmp_path)
    assert "FileNotFoundError" in (tmp_path / "runner.log").read_text()
