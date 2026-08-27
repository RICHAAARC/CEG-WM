from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks/content_texture_stratification_v1_colab.ipynb"
METHOD_EXACT = "a0b75406f75585d567e3be2388a1a76d0cc8b2cd"


class ContentTextureNotebookTests(unittest.TestCase):
    def test_notebook_is_drive_first_and_bound_to_canonical_method(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["metadata"]["colab"]["name"], NOTEBOOK.name)
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code), 5)
        self.assertEqual(
            code[0]["source"],
            ["from google.colab import drive\n", "drive.mount('/content/drive')\n"],
        )
        for cell in code:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            ast.parse("".join(cell["source"]))
        joined = "\n".join("".join(cell["source"]) for cell in code)
        self.assertIn("TARGET_BRANCH = 'Content-Texture'", joined)
        self.assertIn(f"EXPECTED_EXACT = '{METHOD_EXACT}'", joined)
        self.assertEqual(joined.count("subprocess.Popen("), 1)
        self.assertEqual(joined.count("experiments.run_content_texture_stratification_v1"), 1)
        self.assertNotIn("force_remount", joined)
        self.assertNotIn("files.download", joined)
        self.assertNotIn("--resume", joined)


if __name__ == "__main__":
    unittest.main()
