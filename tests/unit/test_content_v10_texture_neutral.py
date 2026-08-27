import hashlib, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
from cegwm.protocol.content_chain_v10 import CALIBRATION_MANIFEST_DIGEST, METHOD_ID, TEXTURE_N96_MANIFEST_DIGEST, load_content_v10_contract

_PATH=Path(__file__).parents[2]/"src/cegwm/method/content_v10_texture_neutral.py"
_SPEC=importlib.util.spec_from_file_location("v10_method",_PATH); _MODULE=importlib.util.module_from_spec(_SPEC); sys.modules["v10_method"]=_MODULE; _SPEC.loader.exec_module(_MODULE)
load_independent_calibration_asset=_MODULE.load_independent_calibration_asset; weighted_joint_v10=_MODULE.weighted_joint_v10

class ContentV10Tests(unittest.TestCase):
 def test_calibration_roster_is_frozen_and_disjoint_from_texture_n96(self):
  root=Path(__file__).parents[2]
  calibration=root/'configs/content_chain/content_v10_calibration_v1.jsonl'
  texture=Path('D:/Projects/Image-WM/CEG-WM/worktrees/content-texture-normalization/configs/content_chain/content_texture_n96_evaluation_v1.jsonl')
  self.assertEqual(hashlib.sha256(texture.read_bytes()).hexdigest(),'73cdb9d6b840490567dd2a40dbf1bd10140e52ae46a43d00fdc01b24a9bc1fb8')
  rows=[json.loads(line) for line in calibration.read_text(encoding='utf-8').splitlines()]
  self.assertEqual(len(rows),32)
  families=('indoor_still_life','landscape','architecture','people','animals','food_material_closeup','abstract_geometry','low_light_weather')
  self.assertEqual([row['unit_id'] for row in rows],[f'content-v10-calibration-b{block:02d}-s{slot:02d}' for block in range(1,5) for slot in range(1,9)])
  self.assertEqual([row['source_id'] for row in rows],[f'content-v10-calibration-b{block:02d}-s{slot:02d}-source' for block in range(1,5) for slot in range(1,9)])
  self.assertEqual([row['seed'] for row in rows],list(range(2026110000,2026110032)))
  self.assertEqual([(row['block_id'],row['block_slot'],row['semantic_family'],row['height'],row['width']) for row in rows],[(f'b{block:02d}',slot,families[slot-1],512,512) for block in range(1,5) for slot in range(1,9)])
  for key in ('unit_id','source_id','seed'):
   self.assertEqual(len({row[key] for row in rows}),32)
  prompts=[row['prompt'].encode('utf-8') for row in rows]
  self.assertEqual(len(set(prompts)),32)
  texture_rows=[json.loads(line) for line in texture.read_text(encoding='utf-8').splitlines()]
  for ours,theirs in (({row['unit_id'] for row in rows},{row['unit_id'] for row in texture_rows}),({row['source_id'] for row in rows},{row['source_id'] for row in texture_rows}),(set(prompts),{row['prompt'].encode('utf-8') for row in texture_rows}),({row['seed'] for row in rows},{row['seed'] for row in texture_rows}),({(row['prompt'].encode('utf-8'),row['seed']) for row in rows},{(row['prompt'].encode('utf-8'),row['seed']) for row in texture_rows})):
   self.assertFalse(ours & theirs)
 def test_contract_and_v10_only_asset(self):
  root=Path(__file__).parents[2]; v9=root/'configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json'
  contract=load_content_v10_contract(root)
  self.assertEqual(contract.config["base_method_id"],"content_v9_v6_calibrated_weighted_joint_v1")
  self.assertEqual(contract.config["calibration_asset"]["calibration_manifest_digest"],CALIBRATION_MANIFEST_DIGEST)
  self.assertEqual(contract.config["texture_n96_provenance"]["manifest_digest"],TEXTURE_N96_MANIFEST_DIGEST)
  self.assertEqual(contract.config["calibration_protocol"],{"fixed_units":32,"ordered_pairs_per_unit":33,"required_pairs":1056,"pair_order":["candidate_wrong_00_to_15","primary_null_registered","primary_null_wrong_00_to_15"],"key_domain":"stage-a/content-v10-texture-neutral-weighted-joint-calibration-key/v1","wrong_key_domain":"stage-a/content-adaptive-v2-external-wrong-key/v1","wrong_key_count":16,"fit":{"mean":"binary64_fsum","sample_sd_ddof":1,"pearson_rho":"paired"},"terminal_failure":"all_or_none_rc2_no_asset","claim_ceiling":"v10_calibration_asset_generation_only_no_efficacy_claim"})
  with self.assertRaises(ValueError): load_independent_calibration_asset(v9,v9.with_name(v9.name+'.sha256'))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"asset.json"; value={"schema_version":1,"method_id":METHOD_ID,"asset_role_id":"content_v10_weighted_joint_calibration","lf_weight":.25,"hf_weight":.75,"lf_scorer_id":"content_v4_whitened_lf_dct_matched_cosine_v1","hf_scorer_id":"frozen_hf_final_rgb_public_vae_global_normalized_correlation","calibration_manifest_digest":CALIBRATION_MANIFEST_DIGEST,"producer_execution_exact":"b"*40,"protocol_digest":"c"*64,"calibration_public_key_digest":"d"*64,"mu_lf":0.,"sigma_lf":1.,"mu_hf":0.,"sigma_hf":2.,"rho":0.}; raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode(); p.write_bytes(raw); p.with_name("asset.json.sha256").write_bytes((hashlib.sha256(raw).hexdigest()+"  asset.json\n").encode("ascii"))
   asset=load_independent_calibration_asset(p,p.with_name("asset.json.sha256")); self.assertAlmostEqual(weighted_joint_v10(1.,.5,asset),(.25+.75*.25)/(.25**2+.75**2)**.5)
   for field,bad in (("calibration_manifest_digest","a"*64),("producer_execution_exact","bad"),("protocol_digest","bad"),("calibration_public_key_digest","bad")):
    altered=dict(value); altered[field]=bad; raw=json.dumps(altered,sort_keys=True,separators=(",",":")).encode(); p.write_bytes(raw); p.with_name("asset.json.sha256").write_bytes((hashlib.sha256(raw).hexdigest()+"  asset.json\n").encode("ascii"))
    with self.assertRaises(ValueError): load_independent_calibration_asset(p,p.with_name("asset.json.sha256"))
   value["method_id"]="content_v9_calibrated_weighted_joint_v1"; raw=json.dumps(value).encode(); p.write_bytes(raw); p.with_name("asset.json.sha256").write_bytes((hashlib.sha256(raw).hexdigest()+"  asset.json\n").encode("ascii"))
   with self.assertRaises(ValueError): load_independent_calibration_asset(p,p.with_name("asset.json.sha256"))
   for value in (True, float('nan'), 1.01):
    with self.assertRaises((TypeError,ValueError)): weighted_joint_v10(value,0.,asset)
if __name__=="__main__": unittest.main()
