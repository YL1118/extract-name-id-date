
from __future__ import annotations
import argparse, sys, json, os
from .extractor import run_extraction

def main():
    p = argparse.ArgumentParser(description="Rule-based multi-record extractor")
    p.add_argument("--text-file", type=str, help="Path to TXT file")
    p.add_argument("--text", type=str, help="Raw text content (overrides --text-file if both provided)")
    p.add_argument("--config", type=str, help="Path to config JSON/YAML", default=None)
    p.add_argument("--out", type=str, help="Output JSON path", default="result.json")
    args = p.parse_args()

    if args.text is not None:
        text = args.text
    elif args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("Must provide --text or --text-file", file=sys.stderr)
        sys.exit(2)

    result = run_extraction(text, args.config)

    # Print CLI summary
    print(f"Records: {len(result['records'])}")
    for r in result["records"]:
        rid = r["record_id"]
        fs = r["fields"]
        def fmt(f):
            s = fs[f]["status"]
            v = fs[f].get("value")
            return f"{f}:{s}" + (f"({v})" if v else "")
        summary = ", ".join([fmt("name"), fmt("id_no"), fmt("ref_date"), fmt("batch_id")])
        print(f" - {rid} -> {summary}")

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
