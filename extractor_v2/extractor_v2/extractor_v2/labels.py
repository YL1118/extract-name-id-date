
from __future__ import annotations
from typing import Dict, List, Tuple
import difflib
import re

# Label synonyms in Chinese (Traditional). Extend as needed.
LABEL_SYNONYMS: Dict[str, List[str]] = {
    "name": ["姓名", "調查人", "申報人", "被查人", "債務人", "納稅人", "查詢人", "當事人"],
    "id_no": ["身分證字號", "身分證統一編號", "身分證", "身分證統編", "國民身分證號", "身份證字號"],
    "ref_date": ["調查日", "申報基準日", "查詢基準日", "查調財產基準日", "基準日", "查詢日"],
    "batch_id": ["本次調查名單檔", "名單檔", "調查名單檔", "名單檔案"]
}

def find_label_hits(line: str, line_idx: int, ratio_threshold: float) -> List[dict]:
    """
    Return list of label nodes found in a line with fuzzy matching.
    Each node: {type, field, line, col, raw, norm, label_confidence}
    """
    hits = []
    # Pre-tokenize by potential separators
    # We'll scan entire line for approximate matches to synonyms.
    for field, words in LABEL_SYNONYMS.items():
        for w in words:
            for m in _find_approx_matches(line, w, ratio_threshold):
                hits.append({
                    "type": "label",
                    "field": field,
                    "line": line_idx,
                    "col": m[0],
                    "raw": line[m[0]:m[1]],
                    "norm": w,
                    "label_confidence": m[2]
                })
    return _dedup_overlaps(hits)

def _find_approx_matches(text: str, pattern: str, ratio_threshold: float):
    # Sliding compare windows around occurrences of first/last chars
    results = []
    # Try exact find first
    for m in re.finditer(re.escape(pattern), text):
        results.append((m.start(), m.end(), 1.0))
    # Fuzzy windows: check substrings around occurrences of first char
    indices = [i for i, ch in enumerate(text) if ch == pattern[0]]
    for i in indices:
        for L in range(max(1, len(pattern)-2), len(pattern)+3):
            j = i + L
            if j > len(text): break
            sub = text[i:j]
            r = difflib.SequenceMatcher(None, sub, pattern).ratio()
            if r >= ratio_threshold:
                results.append((i, j, r))
    # Try also small windows with +/- 1 char around any exact pattern prefix
    # (Heuristic to catch OCR one-char noise)
    return results

def _dedup_overlaps(nodes: List[dict]) -> List[dict]:
    nodes.sort(key=lambda x: (x["line"], x["col"], -x.get("label_confidence", 0)))
    deduped = []
    occupied = []
    for n in nodes:
        span = (n["line"], n["col"], n["col"]+len(n["raw"]))
        if any(_overlap(span, o) for o in occupied):
            continue
        occupied.append(span)
        deduped.append(n)
    return deduped

def _overlap(a, b):
    # match if same line and column ranges intersect
    return a[0]==b[0] and not (a[2] <= b[1] or a[1] >= b[2])
