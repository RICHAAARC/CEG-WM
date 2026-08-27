from __future__ import annotations
import ast,json
from pathlib import Path
N=Path(__file__).parents[2]/"notebooks"/"geometry_v1_qk_d2_independent_confirmation_colab.ipynb"
def test_d2_notebook_contract():
 n=json.loads(N.read_text());c=n["cells"];s="\n".join("".join(x["source"]) for x in c if x["cell_type"]=="code");ast.parse(s)
 assert n["nbformat"]==4 and c[0]["source"]==["from google.colab import drive\n","drive.mount('/content/drive')\n"]
 assert all(x.get("execution_count") is None and not x.get("outputs") for x in c if x["cell_type"]=="code")
 assert "D2_RUNNER_EXACT='4b276ca031a97988fac39a4430c9eaf8dbc0d0f2'" in s and "git','clone','--no-checkout" in s and "git','checkout','--detach',D2_RUNNER_EXACT" in s
 assert s.count("subprocess.Popen(")==1 and "env=runner_env" in s and "timeout=7200" in s and "/Geometry-V1/D2" in s
 assert "userdata.get('HF_TOKEN')" in s and "runner_env.pop('HF_TOKEN',None)" in s
 for x in ("force_remount","torch","diffusers","retry","fallback","zipfile","ZipFile"):assert x not in s
