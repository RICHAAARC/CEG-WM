from pathlib import Path
import unittest
from cegwm.protocol.content_chain_v10 import METHOD_ID, load_content_v10_contract

class ContentV10Tests(unittest.TestCase):
    def test_contract_is_independent_and_weight_frozen(self):
        contract = load_content_v10_contract(Path(__file__).parents[2])
        self.assertEqual(contract.config["method_id"], METHOD_ID)
        self.assertEqual(contract.config["joint_weights"], {"lf": .25, "hf": .75})
        self.assertEqual(contract.config["calibration_asset"]["status"], "independent_asset_required")
if __name__ == "__main__": unittest.main()
