# extract-name-id-date
專案結構（重點模組）

extractor_v2/config.py：預設參數 +（可選）載入 JSON/YAML 設定；完整支援你在規格裡的 grouping/scopes/dedupe/weights/direction_prior。

extractor_v2/labels.py：同義標籤矩陣 + 模糊比對（difflib），含去重。

extractor_v2/patterns.py：

身分證抓取與台灣 ID checksum（字母對應碼 + 權重）

OCR 修正（如 O→0, I→1, L→1, S→5, B→8 …），若修正後校驗通過會在 explanation 註記

姓名（中日文統一碼 2–4 字）、日期（含民國年→ISO）、13 位 batch id。

extractor_v2/normalizer.py：全半形、標點正規化；日期解析（民國年 +1911 ⇒ YYYY-MM-DD）。

extractor_v2/scorer.py：距離分數、方向先驗、權重整合 final_score（完全對齊你給的公式）。

extractor_v2/clusterer.py：

Anchor 選擇（預設 ["id_no","name"]）

視窗聚類（window_lines_up/down, max_col_delta, max_span_lines_for_linking）

多對多配對：預設 greedy；可選 pairing_strategy="hungarian"（我用小規模暴力最佳化替代，n≤6 時跑完全配對，>6 自動退回 greedy——在你實際的版面一格通常遠小於 6，效果穩定）

半全域欄位套用：ref_date/batch_id 依 header_top_n 判斷 document|nearest，並在每筆 explanation 註記「由文件全域套用」或「就近分配」

batch_id 必須鄰近其標籤才可分配（孤立 13 碼會落到 orphans_unassigned）

去重：同一 id_no 合併，取較高信心（保留 merged_from 與 logs）

重要決策全數寫入 logs。

extractor_v2/extractor.py：高階 run_extraction(text, config_path)。

extractor_v2/cli.py：支援 --text-file / --text / --config / --out，執行完會在終端列印每筆摘要（status/value）並輸出 result.json。

extractor_v2/tests/：

test_grouping_multi_records.py：多筆 + 表頭全域；只有 ID 作 anchor；

test_scopes.py：ref_date 全域、batch_id 最近指派（每筆不同）。
