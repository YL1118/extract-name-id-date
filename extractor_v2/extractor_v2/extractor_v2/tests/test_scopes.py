
import unittest
from extractor_v2.extractor import run_extraction

class TestScopes(unittest.TestCase):
    def test_nearest_batch_id(self):
        text = """申報基準日：113/07/01

A人：王大明  身分證字號：A123456789
本次調查名單檔：1111111111111

B人：李小華  身分證字號：B223456789
本次調查名單檔：2222222222222
"""
        res = run_extraction(text, None)
        self.assertEqual(res["document_level"]["global_inference"]["ref_date"]["scope"], "document")
        self.assertEqual(res["document_level"]["global_inference"]["batch_id"]["scope"], "nearest")
        # Each record should pick nearest batch id
        r1, r2 = res["records"][0], res["records"][1]
        self.assertEqual(r1["fields"]["batch_id"]["value"], "1111111111111")
        self.assertEqual(r2["fields"]["batch_id"]["value"], "2222222222222")

if __name__ == "__main__":
    unittest.main()
