import ast
import json
import pathlib
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path


class ContentV9StabilityNotebookTests(unittest.TestCase):
    def test_thin_handoff_binds_real_stability_runner(self):
        notebook = json.loads((Path(__file__).parents[2] / "notebooks/content_v9_stability_colab.ipynb").read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(cells[0]["source"], ["from google.colab import drive\n", "drive.mount('/content/drive')\n"])
        source = "".join(line for cell in cells for line in cell["source"])
        self.assertIn("d9cd6932c3e9532453511203c5a2f5fcbefe8428", source)
        self.assertIn("experiments.run_content_v9_stability", source)
        self.assertIn("ATTEMPT_NONCE = uuid.uuid4().hex[:12]", source)
        self.assertIn("Content-V9-d9cd693-{run_utc}", source)
        self.assertIn("Content-V9-d9cd693-local-{ATTEMPT_NONCE}", source)
        self.assertIn("cegwm-content-v9-source-{ATTEMPT_NONCE}", source)
        self.assertIn("%Y%m%dT%H%M%S%fZ", source)
        self.assertIn("detach_v9_execution_checkout(SOURCE, BRANCH, EXPECTED_EXACT)", source)
        self.assertIn("allocate_v9_attempt_paths(", source)
        self.assertEqual(source.count("subprocess.Popen("), 1)
        self.assertIn("for raw_line in iter(p.stdout.readline,b'')", source)
        self.assertIn("CEGWM_SUMMARY", source)
        self.assertIn("summary_count!=1", source)
        self.assertIn("CEGWM_CONTENT_V9_INCOMPLETE", source)
        self.assertIn("completeness':'complete", source)
        self.assertIn("scientific_status':'not_evaluable", source)
        self.assertIn("try:", source)
        self.assertIn("finally:", source)
        self.assertNotIn("p.stdout.read(", source)
        self.assertNotIn("force_remount", source)
        self.assertTrue(all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells))
        for cell in cells: ast.parse("".join(cell["source"]))

    def _notebook_helpers(self):
        notebook = json.loads((Path(__file__).parents[2] / "notebooks/content_v9_stability_colab.ipynb").read_text(encoding="utf-8"))
        tree = ast.parse("".join(line for cell in notebook["cells"] if cell["cell_type"] == "code" for line in cell["source"]))
        names = {"_v9_git", "detach_v9_execution_checkout", "allocate_v9_attempt_paths"}
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in helpers}, names)
        namespace = {"pathlib": pathlib, "subprocess": subprocess}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "v9-notebook-helpers", "exec"), namespace)
        return namespace

    def test_execution_ancestor_is_accepted_and_unrelated_exact_is_rejected(self):
        detach = self._notebook_helpers()["detach_v9_execution_checkout"]
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            def git(*args): return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            git("config", "user.email", "v9-test@example.invalid"); git("config", "user.name", "V9 Test"); git("checkout", "-q", "-b", "Content-V9")
            (repo / "fixture.txt").write_text("ancestor\n", encoding="utf-8"); git("add", "fixture.txt"); git("commit", "-qm", "ancestor")
            execution = git("rev-parse", "HEAD")
            (repo / "fixture.txt").write_text("handoff\n", encoding="utf-8"); git("commit", "-am", "handoff", "-q")
            git("checkout", "-q", "--orphan", "unrelated"); git("rm", "-q", "-rf", "."); (repo / "unrelated.txt").write_text("unrelated\n", encoding="utf-8"); git("add", "unrelated.txt"); git("commit", "-qm", "unrelated")
            unrelated = git("rev-parse", "HEAD"); git("checkout", "-q", "Content-V9")
            detach(repo, "Content-V9", execution)
            self.assertEqual(git("rev-parse", "HEAD"), execution); self.assertEqual(git("branch", "--show-current"), "")
            git("checkout", "-q", "Content-V9")
            with self.assertRaises(RuntimeError): detach(repo, "Content-V9", unrelated)

    def test_nonce_paths_and_drive_retry_are_create_only(self):
        allocate = self._notebook_helpers()["allocate_v9_attempt_paths"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content, drive = root / "content", root / "drive"; content.mkdir(); drive.mkdir()
            class Clock:
                def __init__(self, values): self.values = iter(values)
                def __call__(self):
                    value = next(self.values)
                    return type("Moment", (), {"strftime": lambda self, _format: value})()
            source_one, local_one, first, _ = allocate(content, drive, "nonce-one001", Clock(["20260828T010203123456Z"]))
            first.mkdir()
            source_two, local_two, target, _ = allocate(content, drive, "nonce-two002", Clock(["20260828T010203123456Z", "20260828T010203123457Z"]))
            self.assertNotEqual((source_one, local_one, first), (source_two, local_two, target)); self.assertTrue(first.exists()); self.assertFalse(target.exists())


if __name__ == "__main__": unittest.main()
