import ast
import json
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
        self.assertIn("Content-V9-d9cd693-{RUN_UTC}", source)
        self.assertIn("Content-V9-d9cd693-local-{ATTEMPT_NONCE}", source)
        self.assertIn("cegwm-content-v9-source-{ATTEMPT_NONCE}", source)
        self.assertIn("%Y%m%dT%H%M%S%fZ", source)
        self.assertIn("HANDOFF_HEAD=git('rev-parse','HEAD')", source)
        self.assertIn("merge-base','--is-ancestor',EXPECTED_EXACT,HANDOFF_HEAD", source)
        self.assertIn("git('checkout','--detach',EXPECTED_EXACT)", source)
        self.assertLess(source.index("git('checkout','--detach',EXPECTED_EXACT)"), source.index("if git('rev-parse','HEAD') != EXPECTED_EXACT"))
        self.assertIn("git('branch','--show-current') != BRANCH", source)
        self.assertIn("git('branch','--show-current') != ''", source)
        self.assertIn("while True:", source)
        self.assertIn("if not DRIVE_TARGET.exists(): break", source)
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

    def test_execution_ancestor_is_accepted_and_unrelated_exact_is_rejected(self):
        root = Path(__file__).parents[2]

        def is_ancestor(ancestor, descendant):
            return subprocess.run(
                ["git", "-c", f"safe.directory={root}", "merge-base", "--is-ancestor", ancestor, descendant],
                cwd=root,
                check=False,
            ).returncode == 0

        self.assertTrue(is_ancestor("d9cd6932c3e9532453511203c5a2f5fcbefe8428", "ecd7f8a80adf75136da8bac0f8bbeec4c382723f"))
        self.assertFalse(is_ancestor("7917a7da15fbeee79083b4938362d2bdf202a740", "ecd7f8a80adf75136da8bac0f8bbeec4c382723f"))

    def test_nonce_paths_and_drive_retry_are_create_only(self):
        nonce_one, nonce_two = uuid.uuid4().hex[:12], uuid.uuid4().hex[:12]
        self.assertNotEqual(nonce_one, nonce_two)
        self.assertNotEqual(f"/content/cegwm-content-v9-source-{nonce_one}", f"/content/cegwm-content-v9-source-{nonce_two}")
        self.assertNotEqual(f"/content/Content-V9-d9cd693-local-{nonce_one}", f"/content/Content-V9-d9cd693-local-{nonce_two}")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stamps = iter(("20260828T010203123456Z", "20260828T010203123457Z"))
            first = root / f"Content-V9-d9cd693-{next(stamps)}"
            first.mkdir()
            while True:
                target = root / f"Content-V9-d9cd693-{next(stamps)}"
                if not target.exists():
                    break
            self.assertNotEqual(first, target)
            self.assertTrue(first.exists())
            self.assertFalse(target.exists())


if __name__ == "__main__": unittest.main()
