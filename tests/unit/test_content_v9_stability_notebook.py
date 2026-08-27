import ast
import json
import unittest
from pathlib import Path


class ContentV9StabilityNotebookTests(unittest.TestCase):
    def test_thin_handoff_binds_real_stability_runner(self):
        notebook = json.loads((Path(__file__).parents[2] / "notebooks/content_v9_stability_colab.ipynb").read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(cells[0]["source"], ["from google.colab import drive\n", "drive.mount('/content/drive')\n"])
        source = "".join(line for cell in cells for line in cell["source"])
        self.assertIn("d9cd6932c3e9532453511203c5a2f5fcbefe8428", source)
        self.assertIn("experiments.run_content_v9_stability", source)
        self.assertIn("Content-V9-{EXPECTED_EXACT[:7]}-{RUN_UTC}", source)
        self.assertEqual(source.count("subprocess.Popen("), 1)
        self.assertIn("for raw_line in iter(p.stdout.readline,b'')", source)
        self.assertIn("CEGWM_SUMMARY", source)
        self.assertIn("summary_count!=1", source)
        self.assertIn("CEGWM_CONTENT_V9_INCOMPLETE", source)
        self.assertIn("completeness':'complete", source)
        self.assertIn("scientific_status':'not_evaluable", source)
        self.assertIn("try:", source)
        self.assertIn("finally:", source)
        self.assertNotIn("p.stdout.read", source)
        self.assertNotIn("force_remount", source)
        self.assertTrue(all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells))
        for cell in cells: ast.parse("".join(cell["source"]))


if __name__ == "__main__": unittest.main()
