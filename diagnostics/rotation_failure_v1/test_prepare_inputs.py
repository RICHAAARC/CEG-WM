import json
from pathlib import Path
import unittest
from prepare_inputs import extract_source
from diagnostic import CONDITIONS,PRODUCER,TAU


class SourceTests(unittest.TestCase):
    def setUp(self):
        entries=json.loads((Path(__file__).parent/'input_reference.json').read_text())['entries']
        self.manifest={'entries':entries,'source':{'result_file_id':'source'}}
        self.formal={'identity':{'expected_exact':PRODUCER,'stage':'evaluation_detection'},
                     'records':[{'physical_unit_id':e['sample_id'],'condition':c,'truth_role':t,'threshold':TAU}
                     for e in entries for c in CONDITIONS for t in ('negative','positive')]}

    def test_fixed_400_source_rows(self):
        self.assertEqual(len(extract_source(self.manifest,self.formal)['records']),400)

    def test_duplicate_row_cannot_replace_missing_negative(self):
        self.formal['records'][-1]=self.formal['records'][0]
        with self.assertRaises(ValueError):extract_source(self.manifest,self.formal)

    def test_changed_threshold_rejected(self):
        self.formal['records'][0]['threshold']=0.
        with self.assertRaises(ValueError):extract_source(self.manifest,self.formal)

if __name__=='__main__':unittest.main()
