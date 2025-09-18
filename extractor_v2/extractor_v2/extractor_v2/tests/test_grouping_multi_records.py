
import unittest
from extractor_v2.extractor import run_extraction

class TestGroupingMultiRecords(unittest.TestCase):
    def test_case_1_global_header(self):
        text = """本次調查名單檔：1234567890123
調查日：民國114年9月18日

調查人：王小明  身分證字號：A123456789
調查人：陳小美
身分證字號：B223456789
"""
        res = run_extraction(text, None)
        self.assertEqual(len(res["records"]), 2)
        r1, r2 = res["records"][0], res["records"][1]
        # both found name and id
        self.assertEqual(r1["fields"]["name"]["status"], "FOUND")
        self.assertEqual(r1["fields"]["id_no"]["status"], "FOUND")
        self.assertEqual(r2["fields"]["name"]["status"], "FOUND")
        self.assertEqual(r2["fields"]["id_no"]["status"], "FOUND")
        # document scope applied
        self.assertEqual(res["document_level"]["global_inference"]["ref_date"]["scope"], "document")
        self.assertEqual(res["document_level"]["global_inference"]["batch_id"]["scope"], "document")

    def test_case_3_only_id_anchor(self):
        text = """身分證字號：A123456789
身分證字號：B223456789
"""
        res = run_extraction(text, None)
        self.assertEqual(len(res["records"]), 2)
        for r in res["records"]:
            self.assertEqual(r["fields"]["id_no"]["status"], "FOUND")
            self.assertIn(r["fields"]["name"]["status"], ("KEYWORD_FOUND_NO_VALUE","KEYWORD_NOT_FOUND"))

if __name__ == "__main__":
    unittest.main()
