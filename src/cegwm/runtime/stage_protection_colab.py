"""Colab access reuse and durable process output; no model imports or loading."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import subprocess
import time
import traceback


def ensure_model_access(config, *, api=None, secret_getter=None, prompt=None):
    """Reuse credentials first; a blocking login prompt requires a 401/403.

    Explicit tokens avoid get_token() implicitly consulting a Colab secret
    before an already usable environment/cached login. No token is returned,
    printed, placed in a command, or included in the source snapshot.
    """
    from huggingface_hub import HfApi, constants, login
    from huggingface_hub.utils import HfHubHTTPError

    api = api or HfApi()
    models = list(dict.fromkeys(config[name]["model_id"] for name in ("generation", "reconstruction")))
    tried = set()

    def accepted(token):
        token = token.strip() if token else False
        if token in tried:
            return False
        tried.add(token)
        try:
            for model_id in models:
                api.auth_check(repo_id=model_id, repo_type="model", token=token)
        except HfHubHTTPError as error:
            if getattr(error.response, "status_code", None) in (401, 403):
                return False
            raise  # Network/server/repository failures must not trigger a login prompt.
        if token:
            os.environ["HF_TOKEN"] = token  # Inherited by the runner; never embedded.
        else:
            # An invalid inherited token must not override successful anonymous access.
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        return True

    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(name)
        if token and accepted(token):
            return {"credential_source": "environment", "models_checked": models}
    cache = Path(constants.HF_TOKEN_PATH)
    if cache.is_file() and accepted(cache.read_text().strip()):
        return {"credential_source": "existing_login", "models_checked": models}
    if secret_getter is None:
        try:
            from google.colab import userdata
            secret_getter = userdata.get
        except ImportError:
            secret_getter = lambda name: None
    try:
        token = secret_getter("HF_TOKEN")
    except Exception:
        token = None  # Missing/inaccessible secret; do not print its exception or contents.
    if token and accepted(token):
        return {"credential_source": "colab_secret", "models_checked": models}
    if accepted(False):
        return {"credential_source": "anonymous_access", "models_checked": models}
    print("现有凭据无法访问固定模型。请先确认该账户已获模型访问许可：")
    for model_id in models:
        print("https://huggingface.co/" + model_id)
    token = (prompt or getpass.getpass)("Hugging Face read token（仅本次缺少访问权限时需要；输入隐藏）: ").strip()
    # A user may reenter a token after granting access on the model page.
    tried.discard(token)
    if not token or not accepted(token):
        raise RuntimeError("固定模型访问仍未通过；未启动 P2。请检查账户许可或 token 权限。")
    login(token=token, add_to_git_credential=False)
    return {"credential_source": "necessary_login", "models_checked": models}


@contextmanager
def record_setup_errors(output):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield
    except BaseException:
        with (directory / "setup.log").open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
            log.flush()
            os.fsync(log.fileno())
        raise


def run_logged(command, *, cwd, output, env=None, log_name="runner.log"):
    """Direct child stdout/stderr to an append-only file and tail it to the UI.

    The child writes to the file itself, not to a frontend/parent stdout pipe.
    Reopening the notebook is not needed to retain a traceback. A VM shutdown
    can still stop execution; the operation store handles incomplete attempts.
    """
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / log_name
    process = None
    with path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\nSTARTED {datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        offset = log.tell()
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            with path.open(encoding="utf-8", errors="replace") as tail:
                tail.seek(offset)
                while True:
                    output_text = tail.read()
                    if output_text:
                        print(output_text, end="", flush=True)
                    returncode = process.poll()
                    if returncode is not None:
                        print(tail.read(), end="", flush=True)
                        break
                    time.sleep(0.2)
            log.write(f"EXIT {returncode} {datetime.now(timezone.utc).isoformat()}\n")
            log.flush()
            os.fsync(log.fileno())
            return returncode
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            traceback.print_exc(file=log)
            log.flush()
            os.fsync(log.fileno())
            raise
