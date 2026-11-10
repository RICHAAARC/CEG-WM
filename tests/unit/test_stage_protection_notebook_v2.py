import ast
import base64
import gzip
import io
import re
from pathlib import Path
import tarfile
import types
import sys

import nbformat
import pytest
import torch

from cegwm.runtime import stage_protection_colab as colab_helpers

from scripts.build_stage_protection_v2_notebook import OUTPUT, ROOT, source_files

pytestmark = pytest.mark.unit
RESUME_OUTPUT = ROOT / "notebooks/stage_protection_v2_resume_colab.ipynb"
COMPAT_OUTPUT = ROOT / "notebooks/stage_protection_v2_compat_resume_colab.ipynb"
ISOLATED_OUTPUT = ROOT / "notebooks/stage_protection_v2_isolated_resume_colab.ipynb"


@pytest.mark.parametrize("notebook_path", [OUTPUT, RESUME_OUTPUT, COMPAT_OUTPUT, ISOLATED_OUTPUT])
def test_embedded_notebook_matches_current_source_and_first_cell_only_mounts(notebook_path):
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    assert notebook.cells[0].cell_type == "code"
    assert notebook.cells[0].source == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert notebook.metadata.accelerator == "GPU"
    assert not any("RUN_P2" in cell.source for cell in notebook.cells)
    bundle = None
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BUNDLE" for t in node.targets):
                bundle = ast.literal_eval(node.value)
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(bundle)))) as archive:
        assert set(archive.getnames()) == {str(p.relative_to(ROOT)) for p in source_files()}
        for path in source_files():
            data = archive.extractfile(str(path.relative_to(ROOT))).read()
            assert data == path.read_bytes()
            assert not re.search(rb"\bhf_[A-Za-z0-9]{20,}\b|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", data)
    assert not re.search(r"\bhf_[A-Za-z0-9]{20,}\b", notebook_path.read_text())
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in notebook.cells if cell.cell_type == "code")
    assert not any("governance" in str(p) for p in source_files())


@pytest.mark.parametrize("runner_exit", [0, 1])
@pytest.mark.parametrize("notebook_path", [OUTPUT, RESUME_OUTPUT, COMPAT_OUTPUT, ISOLATED_OUTPUT])
def test_default_run_all_enters_execute_and_logs_cpu_substitute(tmp_path, monkeypatch, runner_exit, capsys, notebook_path):
    # Execute the actual cells without a run switch. Stub external services and
    # CUDA admission; replace pip and model commands with small CPU subprocesses.
    # The real logging helper and notebook control flow still execute.
    import huggingface_hub
    mounts, commands, permission_checks = [], [], []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: mounts.append(path))
    colab.userdata = types.SimpleNamespace(get=lambda _: pytest.fail("existing auth must not request a secret"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setenv("CEG_STAGE_PROTECTION_WORKDIR", str(tmp_path / "source"))
    monkeypatch.setenv("HF_TOKEN", "synthetic-environment-credential")
    class FakeApi:
        def auth_check(self, **kwargs):
            permission_checks.append(kwargs["repo_id"])
            assert kwargs["token"] == "synthetic-environment-credential"
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "login", lambda **_: pytest.fail("must reuse existing auth"))
    monkeypatch.setattr(torch, "__version__", "2.6.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    actual_logged = colab_helpers.run_logged
    (tmp_path / "output").mkdir()
    (tmp_path / "output/runner.log").write_text("Original interrupted run EXIT -9\n")
    def cpu_substitute(command, **kwargs):
        commands.append(command)
        if "pip" in command:
            code = "print('CPU dependency-install substitute')"
        else:
            assert "--execute" in command
            expected = "experiments.run_stage_protection_v2_resume" if notebook_path == RESUME_OUTPUT else "experiments.run_stage_protection_v2"
            if notebook_path == COMPAT_OUTPUT:
                expected = "experiments.run_stage_protection_v2_compat_resume"
            if notebook_path == ISOLATED_OUTPUT:
                expected = "experiments.run_stage_protection_v2_isolated_resume"
            assert expected in command
            code = "import sys; print('CPU runner stdout', flush=True); print('CPU runner stderr', file=sys.stderr, flush=True)"
            if runner_exit:
                code += "; raise RuntimeError('CPU synthetic runner traceback')"
        return actual_logged([sys.executable, "-u", "-c", code], **kwargs)
    monkeypatch.setattr(colab_helpers, "run_logged", cpu_substitute)
    scope, old_path = {"__name__": "__notebook__"}, list(sys.path)
    try:
        for cell in nbformat.read(notebook_path, as_version=4).cells:
            if cell.cell_type == "code":
                exec(compile(cell.source, str(notebook_path), "exec"), scope)
                if cell.source.startswith("# 固定输出目录"):
                    assert scope["OUTPUT"] == '/content/drive/MyDrive/CEG-WM_Result/stage-protection-v2-20261109/attempt01'
                    scope["OUTPUT"] = str(tmp_path / "output")  # Redirect only filesystem destination.
    finally:
        sys.path[:] = old_path
    assert mounts == ["/content/drive"]
    if notebook_path == COMPAT_OUTPUT:
        assert scope["RESULTS"] == tmp_path / "output/repairs/source06_bit1_early1_latent_vjp_compat_attempt01"
    if notebook_path == ISOLATED_OUTPUT:
        assert scope["RESULTS"] == tmp_path / "output/repairs/source06_bit1_early2_rgb_vjp_isolated_attempt01"
    assert "RUN_P2" not in scope
    assert scope["returncode"] == runner_exit
    assert len(commands) == 2 and "--execute" in commands[-1]
    assert len(permission_checks) == 2
    log = (tmp_path / "output/runner.log").read_text()
    assert "CPU runner stdout" in log and "CPU runner stderr" in log
    assert log.startswith("Original interrupted run EXIT -9\n")
    if runner_exit:
        assert "Traceback" in log and "CPU synthetic runner traceback" in log
    assert (tmp_path / "output/setup.log").exists()
    assert "synthetic-environment-credential" not in log + capsys.readouterr().out
    assert (tmp_path / "source/configs/stage_protection_v2_colab.json").exists()
