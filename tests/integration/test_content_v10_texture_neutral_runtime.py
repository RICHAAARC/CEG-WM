import unittest
from cegwm.runtime.content_v10_texture_neutral_sd35 import require_v10_calibration_asset

class ContentV10RuntimeTests(unittest.TestCase):
    def test_missing_calibration_asset_fails_closed(self):
        with self.assertRaises(ValueError): require_v10_calibration_asset(None)
if __name__ == "__main__": unittest.main()
