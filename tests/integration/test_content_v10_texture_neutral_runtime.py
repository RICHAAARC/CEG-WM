import ast, unittest
from pathlib import Path

class ContentV10RuntimeTests(unittest.TestCase):
 def test_v10_routes_private_v6_seam_and_public_v6_stays_unselectable(self):
  root=Path(__file__).parents[2]; v10=(root/'src/cegwm/runtime/content_v10_texture_neutral_sd35.py').read_text(); v6=(root/'src/cegwm/runtime/content_iss_sd35_v6.py').read_text()
  self.assertIn('_run_content_v6_pass2',v10); self.assertIn('allocation_factory=allocator',v10); self.assertIn('allocate_texture_neutral',v10)
  tree=ast.parse(v6); public=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_content_v6_evaluation_pair'); self.assertNotIn('allocation_factory',[a.arg for a in public.args.args+public.args.kwonlyargs])
  callback=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ContentV6InjectionCallback'); init=next(n for n in callback.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'); self.assertNotIn('allocation_factory',[a.arg for a in init.args.args+init.args.kwonlyargs])
  self.assertIn('_ContentV6PrivateInjectionCallback',v6)
  self.assertIn('step-18 allocation did not occur exactly once',v10)
if __name__=="__main__": unittest.main()
