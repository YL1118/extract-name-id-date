
from __future__ import annotations
from typing import Tuple, Optional
import re
from datetime import date

def normalize_text(s: str) -> str:
    s = to_half_width(s)
    s = normalize_punct(s)
    return s

def to_half_width(s: str) -> str:
    res = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            res.append(' ')
        elif 0xFF01 <= code <= 0xFF5E:
            res.append(chr(code - 0xFEE0))
        else:
            res.append(ch)
    return ''.join(res)

def normalize_punct(s: str) -> str:
    # unify common punctuation variants to ASCII
    table = str.maketrans({
        '：': ':', '，': ',', '。': '.', '；': ';', '、': ',', '（': '(', '）': ')',
        '／': '/', '－': '-', '～': '~', '　': ' '
    })
    s = s.translate(table)
    # collapse spaces
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip('\n')

def parse_date_to_iso(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse various date formats, including ROC year. Return (iso, reason).
    iso: 'YYYY-MM-DD' or None if failed.
    reason: textual explanation (e.g., 'ROC -> AD +1911').
    """
    s = raw.strip()
    # Extract digits
    import re
    m = re.search(r'(\d{2,4})\D+(\d{1,2})\D+(\d{1,2})', s)
    if not m:
        return None, "no_date_pattern"
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    reason = None
    # ROC detection: explicit '民國' / '中華民國' or small year <= 150
    if '民國' in s or '中華民國' in s or y <= 150:
        y = y + 1911
        reason = "ROC→AD +1911"
    try:
        from datetime import date
        dt = date(y, mo, d)
        return dt.isoformat(), reason
    except Exception:
        return None, "invalid_date_value"
