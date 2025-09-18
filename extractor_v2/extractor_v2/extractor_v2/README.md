
# Rule-based Multi-Record Extractor (v0.2.0)

Python 3.12, zero external deps (optional PyYAML for YAML config).

## Install & Run

```bash
# (optional) create venv
python3.12 -m venv .venv && source .venv/bin/activate

# run CLI
python -m extractor_v2.cli --text-file sample.txt --out result.json
# or
python -m extractor_v2.cli --text "調查人：王小明 身分證字號：A123456789" --out result.json
```

Config accepts JSON by default; YAML requires `PyYAML` if you prefer `.yaml` files.

## Config (JSON/YAML)

```json
{
  "grouping": {
    "anchor_order": ["id_no","name"],
    "window_lines_up": 1,
    "window_lines_down": 3,
    "max_col_delta": 30,
    "max_span_lines_for_linking": 3,
    "pairing_strategy": "greedy",
    "cluster_merge_if_overlap": true
  },
  "scopes": {
    "ref_date": {"header_top_n":8, "default":"auto"},
    "batch_id": {"header_top_n":8, "default":"auto"}
  },
  "dedupe": {"merge_same_id": true, "prefer_higher_confidence": true}
}
```

## Output JSON shape

See `tests/` and spec in the prompt. Includes:
- `records` (multi)
- `orphans_unassigned` with reasons
- `document_level.label_presence` and `.global_inference`
- `candidates_debug`, `logs`, `config_used`, `version`

## Notes

- Batch ID assignment **requires** proximity to a `batch_id` label; isolated 13-digit numbers are listed as orphans.
- Ref date converts ROC→AD (e.g., 民國114年=2025) to ISO `YYYY-MM-DD`.
- Optional pairing strategy `"hungarian"` implemented via brute-force search for small n (<=6); else falls back to greedy.

## Tests

```
python -m unittest extractor_v2.tests.test_grouping_multi_records
python -m unittest extractor_v2.tests.test_scopes
```
