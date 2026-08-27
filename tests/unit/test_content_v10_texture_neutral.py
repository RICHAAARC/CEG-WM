import hashlib, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
from cegwm.protocol.content_chain_v10 import METHOD_ID, load_content_v10_contract

_PATH=Path(__file__).parents[2]/"src/cegwm/method/content_v10_texture_neutral.py"
_SPEC=importlib.util.spec_from_file_location("v10_method",_PATH); _MODULE=importlib.util.module_from_spec(_SPEC); sys.modules["v10_method"]=_MODULE; _SPEC.loader.exec_module(_MODULE)
load_independent_calibration_asset=_MODULE.load_independent_calibration_asset; weighted_joint_v10=_MODULE.weighted_joint_v10

class ContentV10Tests(unittest.TestCase):
 def test_contract_and_v10_only_asset(self):
  self.assertEqual(load_content_v10_contract(Path(__file__).parents[2]).config["base_method_id"],"content_v9_v6_calibrated_weighted_joint_v1")
  root=Path(__file__).parents[2]; v9=root/'configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json'
  with self.assertRaises(ValueError): load_independent_calibration_asset(v9,v9.with_name(v9.name+'.sha256'))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"asset.json"; value={"schema_version":1,"method_id":METHOD_ID,"asset_role_id":"content_v10_weighted_joint_calibration","lf_weight":.25,"hf_weight":.75,"lf_scorer_id":"content_v4_whitened_lf_dct_matched_cosine_v1","hf_scorer_id":"frozen_hf_final_rgb_public_vae_global_normalized_correlation","calibration_manifest_digest":"a"*64,"mu_lf":0.,"sigma_lf":1.,"mu_hf":0.,"sigma_hf":2.,"rho":0.}; raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode(); p.write_bytes(raw); p.with_name("asset.json.sha256").write_bytes((hashlib.sha256(raw).hexdigest()+"  asset.json\n").encode("ascii"))
   asset=load_independent_calibration_asset(p,p.with_name("asset.json.sha256")); self.assertAlmostEqual(weighted_joint_v10(1.,.5,asset),(.25+.75*.25)/(.25**2+.75**2)**.5)
   value["method_id"]="content_v9_calibrated_weighted_joint_v1"; raw=json.dumps(value).encode(); p.write_bytes(raw); p.with_name("asset.json.sha256").write_bytes((hashlib.sha256(raw).hexdigest()+"  asset.json\n").encode("ascii"))
   with self.assertRaises(ValueError): load_independent_calibration_asset(p,p.with_name("asset.json.sha256"))
   for value in (True, float('nan'), 1.01):
    with self.assertRaises((TypeError,ValueError)): weighted_joint_v10(value,0.,asset)
if __name__=="__main__": unittest.main()
