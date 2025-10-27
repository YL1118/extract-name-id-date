#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based multi-record extractor for TXT documents (Taiwanese admin-style docs)
Batch version with minimal output:
- For each requested field: only (value, confidence, context10)
- context10 = 10 chars before + matched span + 10 chars after (from normalized text)
- Batch processing: read all .txt files under a directory

Usage:
  # 單檔
  python extractor.py input.txt --surnames surnames.txt

  # 批次資料夾（只會讀 .txt）
  python extractor.py --input_dir ./folder --surnames surnames.txt -o out.json
"""
from __future__ import annotations

import re
import json
import math
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Iterable, Set
import os
import glob

# ==============================
# Configuration (tweak as needed)
# ==============================
LABELS: Dict[str, List[str]] = {
    "name": ["姓名", "調查人", "申報人"],
    "id_no": ["身分證字號", "身分證統一編號", "身分證", "身分證統編"],
    "ref_date": ["調查日", "申報基準日", "查詢基準日", "查調財產基準日"],
    "batch_id": ["本次調查名單檔"],
}

# Direction priors
DIRECTION_PRIOR = {"same_right": 1.2, "same_left": 0.9, "below": 0.6}

# Distance model
MAX_DOWN_LINES = 3
LINE_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3}
TAU_COL = 12.0

# Scoring weights
W_LABEL = 0.3
W_FORMAT = 0.9
W_DIST = 1.6
W_DIR = 0.4
W_CONTEXT = 0.2
W_PENALTY = 0.8

# Name rules
NAME_SEPARATORS = "·．• "
NAME_BLACKLIST_NEAR = {"公司","單位","科","處","部","電話","分機","地址","附件","銀行","分行","室","股","隊","路","段","號","樓","市","縣","鄉","鎮","村","里"}
HONORIFICS = {"先生","小姐","女士","太太","老師","主管","經理","博士"}
BIGRAM_BLACKLIST = {"應於","基準","查詢","調查","名單","身分","證號","統編","日期","時間","銀行","公司","單位","地址","電話"}

# 擴充規則
ENABLE_DYNAMIC_DOUBLE_SURNAME = True
ENABLE_IDLABEL_PROXIMITY = True
IDLABEL_BONUS_SCALE = 1.0

# Batch ID
RE_BATCH_13 = re.compile(r"\b\d{13}\b")

# ID patterns
RE_ID_TW = re.compile(r"^[A-Z][0-9]{9}$")
RE_ID_ARC = re.compile(r"^[A-Z]{2}[0-9]{8}$")

# Date patterns
DATE_PATTERNS = [
    re.compile(r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b"),
    re.compile(r"民國\s*\d{2,3}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"),
    re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"),
]

# Double surnames
DEFAULT_DOUBLE_SURNAMES = {
    "歐陽","司馬","諸葛","上官","東方","夏侯","司徒","司空","司寇","令狐",
    "公孫","公羊","公冶","慕容","端木","皇甫","長孫","尉遲","赫連","納蘭",
    "澹臺","南宮","拓跋","宇文","完顏","呼延","夏侯","聞人","司南","仲長",
}

CJK_RANGE = "\u4e00-\u9fff"
RE_CJK = re.compile(rf"^[{CJK_RANGE}]+$")

# ==============================
# Data structures
# ==============================
@dataclass
class LabelHit:
    field: str
    label_text: str
    matched: str
    distance: int
    line: int
    col: int

@dataclass
class Candidate:
    field: str
    value: str
    line: int
    col: int
    label_line: int
    label_col: int
    source_label: Optional[str]
    format_conf: float
    label_conf: float
    dir_prior: float
    dist_score: float
    context_bonus: float = 0.0
    penalty: float = 0.0

    def score(self) -> float:
        return (
            W_LABEL * self.label_conf
            + W_FORMAT * self.format_conf
            + W_DIST * self.dist_score
            + W_DIR * self.dir_prior
            + W_CONTEXT * self.context_bonus
            - W_PENALTY * self.penalty
        )

# ==============================
# Utilities
# ==============================
def to_halfwidth(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def normalize_text(s: str) -> List[str]:
    s = to_halfwidth(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    lines = [re.sub(r"[ \t　]+", " ", line) for line in lines]
    return lines

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost))
        prev = cur
    return prev[-1]

def find_label_hits(lines: List[str], labels: Dict[str, List[str]], max_edit: int = 1) -> List[LabelHit]:
    hits: List[LabelHit] = []
    for li, line in enumerate(lines):
        for field, labellist in labels.items():
            for lab in labellist:
                for m in re.finditer(re.escape(lab), line):
                    hits.append(LabelHit(field, lab, lab, 0, li, m.start()))
        tokens = re.finditer(r"[\w\u4e00-\u9fff]{2,6}", line)
        for t in tokens:
            text = t.group(0)
            for field, labellist in labels.items():
                for lab in labellist:
                    d = levenshtein(text, lab)
                    if 0 < d <= max_edit:
                        hits.append(LabelHit(field, lab, text, d, li, t.start()))
    uniq: Dict[Tuple[int,int,str], LabelHit] = {}
    for h in hits:
        key = (h.line, h.col, h.field)
        if key not in uniq or h.distance < uniq[key].distance:
            uniq[key] = h
    return list(uniq.values())

# ==============================
# Field validators & parsers
# ==============================
LETTER_MAP = {chr(ord('A')+i): 10+i for i in range(26)}
WEIGHTS_TW_ID = [1,9,8,7,6,5,4,3,2,1,1]

def tw_id_checksum_ok(code: str) -> bool:
    if not RE_ID_TW.fullmatch(code): return False
    n = LETTER_MAP.get(code[0])
    if n is None: return False
    a, b = divmod(n, 10)
    digits = [a, b] + [int(x) for x in code[1:]]
    return sum(d*w for d, w in zip(digits, WEIGHTS_TW_ID)) % 10 == 0

def arc_id_like(code: str) -> bool:
    return RE_ID_ARC.fullmatch(code) is not None

def parse_iso_date(txt: str) -> Optional[str]:
    txt = txt.strip()
    m = re.match(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$", txt)
    if m:
        y, mo, d = map(int, m.groups())
        try: return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError: return None
    m = re.match(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$", txt)
    if m:
        y, mo, d = map(int, m.groups())
        try: return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError: return None
    m = re.match(r"^民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$", txt)
    if m:
        roc, mo, d = map(int, m.groups())
        y = roc + 1911
        try: return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError: return None
    return None

def is_cjk(s: str) -> bool:
    return RE_CJK.fullmatch(s) is not None

def load_surnames_from_txt(path: str) -> Tuple[Set[str], Set[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception:
        content = ""
    singles: Set[str] = set()
    doubles: Set[str] = set(DEFAULT_DOUBLE_SURNAMES)
    if content:
        parts = [p.strip() for p in content.split(",") if p.strip()]
        for p in parts:
            if is_cjk(p):
                if len(p) == 1: singles.add(p)
                elif len(p) == 2: doubles.add(p)
    return singles, doubles

def distance_score(label_col: int, cand_col: int, line_delta: int, tau: float = TAU_COL) -> float:
    line_w = LINE_WEIGHTS.get(abs(line_delta), 0.0)
    col_w = math.exp(-abs(cand_col - label_col)/tau)
    return line_w * col_w

def name_candidates_from_text(line_text: str, surname_singles: Set[str], surname_doubles: Set[str]) -> List[Tuple[str, int]]:
    cands: List[Tuple[str,int]] = []
    text = line_text
    n = len(text)
    sep_set = set(NAME_SEPARATORS)

    def next_two_cjk_after(start: int) -> Tuple[Optional[str], Optional[int]]:
        j = start
        while j < n and text[j] in sep_set:
            j += 1
        given = []
        col = j
        while j < n and len(given) < 2:
            ch = text[j]
            if RE_CJK.fullmatch(ch):
                given.append(ch)
                j += 1
            else:
                break
        if len(given) == 2:
            return ("".join(given), col)
        return (None, None)

    doubles_sorted = sorted(surname_doubles, key=len, reverse=True)
    i = 0
    while i < n:
        matched = False
        # 已知雙姓
        for ds in doubles_sorted:
            L = len(ds)
            if i + L <= n and text[i:i+L] == ds:
                given, col = next_two_cjk_after(i + L)
                if given and given not in BIGRAM_BLACKLIST:
                    cands.append((ds + given, i))
                matched = True
                break
        if matched:
            i += 1
            continue

        ch = text[i]

        # 動態雙姓
        if ENABLE_DYNAMIC_DOUBLE_SURNAME and i + 1 < n:
            ch2 = text[i+1]
            if ch in surname_singles and ch2 in surname_singles:
                given, col = next_two_cjk_after(i + 2)
                if given and given not in BIGRAM_BLACKLIST:
                    cands.append((ch + ch2 + given, i))
                matched = True
        if matched:
            i += 1
            continue

        # 單姓
        if ch in surname_singles:
            given, col = next_two_cjk_after(i + 1)
            if given and given not in BIGRAM_BLACKLIST:
                cands.append((ch + given, i))

        i += 1

    return cands

def find_field_candidates_around_label(field: str, label: LabelHit, lines: List[str], surname_singles: Set[str], surname_doubles: Set[str]) -> List[Candidate]:
    results: List[Candidate] = []
    label_line_text = lines[label.line]

    def add_candidate(value: str, vcol: int, line: int, dir_key: str, fmt_conf: float) -> None:
        line_delta = line - label.line
        col_delta = abs(vcol - label.col)
        dist = distance_score(label.col, vcol, line_delta)

        if field == "name":
            if line_delta == 0 and col_delta > 14: return
            if line_delta != 0 and col_delta > 10: return
            if abs(line_delta) > 1: return
            if dist < 0.5: return
        else:
            if dist < 0.2: return

        dir_prior = DIRECTION_PRIOR.get(dir_key, 0.0)
        cand = Candidate(
            field=field,
            value=value,
            line=line,
            col=vcol,
            label_line=label.line,
            label_col=label.col,
            source_label=label.label_text,
            format_conf=fmt_conf,
            label_conf=1.0 - min(label.distance, 1) * 0.5,
            dir_prior=dir_prior,
            dist_score=dist,
        )
        results.append(cand)

    # same line: right
    right_seg = label_line_text[label.col:label.col+60]
    if field == "name":
        for name, c in name_candidates_from_text(right_seg, surname_singles, surname_doubles):
            add_candidate(name, label.col + c, label.line, "same_right", fmt_conf=0.8)
    elif field == "id_no":
        for m in re.finditer(r"[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8}", right_seg):
            code = m.group(0)
            fmt = 1.0 if tw_id_checksum_ok(code) or arc_id_like(code) else 0.5
            add_candidate(code, label.col + m.start(), label.line, "same_right", fmt)
    elif field == "ref_date":
        for pat in DATE_PATTERNS:
            for m in pat.finditer(right_seg):
                iso = parse_iso_date(m.group(0))
                if iso:
                    add_candidate(iso, label.col + m.start(), label.line, "same_right", 1.0)
    elif field == "batch_id":
        for m in RE_BATCH_13.finditer(right_seg):
            add_candidate(m.group(0), label.col + m.start(), label.line, "same_right", 0.9)

    # same line: left
    left_seg = label_line_text[max(0, label.col-60):label.col]
    if field == "name":
        for name, c in name_candidates_from_text(left_seg, surname_singles, surname_doubles):
            add_candidate(name, max(0, label.col-60) + c, label.line, "same_left", fmt_conf=0.8)
    elif field == "id_no":
        for m in re.finditer(r"[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8}", left_seg):
            code = m.group(0)
            fmt = 1.0 if tw_id_checksum_ok(code) or arc_id_like(code) else 0.5
            add_candidate(code, max(0, label.col-60) + m.start(), label.line, "same_left", fmt)
    elif field == "ref_date":
        for pat in DATE_PATTERNS:
            for m in pat.finditer(left_seg):
                iso = parse_iso_date(m.group(0))
                if iso:
                    add_candidate(iso, max(0, label.col-60) + m.start(), label.line, "same_left", 1.0)
    elif field == "batch_id":
        for m in RE_BATCH_13.finditer(left_seg):
            add_candidate(m.group(0), max(0, label.col-60) + m.start(), label.line, "same_left", 0.9)

    # below lines
    for dl in range(1, MAX_DOWN_LINES + 1):
        tgt_line_idx = label.line + dl
        if tgt_line_idx >= len(lines): break
        tgt = lines[tgt_line_idx]
        if field == "name":
            for name, c in name_candidates_from_text(tgt, surname_singles, surname_doubles):
                add_candidate(name, c, tgt_line_idx, "below", 0.8)
        elif field == "id_no":
            for m in re.finditer(r"[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8}", tgt):
                code = m.group(0)
                fmt = 1.0 if tw_id_checksum_ok(code) or arc_id_like(code) else 0.5
                add_candidate(code, m.start(), tgt_line_idx, "below", fmt)
        elif field == "ref_date":
            for pat in DATE_PATTERNS:
                for m in pat.finditer(tgt):
                    iso = parse_iso_date(m.group(0))
                    if iso:
                        add_candidate(iso, m.start(), tgt_line_idx, "below", 1.0)
        elif field == "batch_id":
            for m in RE_BATCH_13.finditer(tgt):
                add_candidate(m.group(0), m.start(), tgt_line_idx, "below", 0.9)

    return results

def pick_best_candidate(cands: List[Candidate]) -> Optional[Candidate]:
    if not cands:
        return None
    cands_sorted = sorted(cands, key=lambda c: c.score(), reverse=True)
    return cands_sorted[0]

# ==============================
# Context helpers
# ==============================
def get_context10(lines: List[str], line: int, col: int, span_len: int) -> str:
    """Return 10 chars before + span + 10 chars after from normalized text lines."""
    if line < 0 or line >= len(lines): return ""
    s = lines[line]
    start = max(0, col - 10)
    end = min(len(s), col + span_len + 10)
    return s[start:end]

# ==============================
# Record grouping / anchor logic
# ==============================
def group_records(all_cands: Dict[str, List[Candidate]]) -> List[Tuple[Optional[Candidate], Optional[Candidate], Optional[Candidate], Optional[Candidate], List[Candidate]]]:
    """Return list of tuples: (name_c, id_c, date_c, batch_c, name_top5) using ID as anchor if present."""
    records: List[Tuple[Optional[Candidate], Optional[Candidate], Optional[Candidate], Optional[Candidate], List[Candidate]]] = []

    def nearest(field: str, anchor: Candidate) -> Optional[Candidate]:
        best = None
        best_s = -1.0
        for c in all_cands.get(field, []):
            line_delta = abs(c.line - anchor.line)
            if line_delta > MAX_DOWN_LINES:
                continue
            dist = distance_score(anchor.col, c.col, c.line - anchor.line)
            s = (dist * 3.0) + (0.3 * c.format_conf) + (0.2 * c.dir_prior) - (0.3 * c.penalty)
            if s > best_s:
                best_s, best = s, c
        return best

    def nearest_k(field: str, anchor: Candidate, k: int = 5) -> List[Candidate]:
        scored: List[Tuple[float, Candidate]] = []
        for c in all_cands.get(field, []):
            line_delta = abs(c.line - anchor.line)
            if line_delta > MAX_DOWN_LINES:
                continue
            dist = distance_score(anchor.col, c.col, c.line - anchor.line)
            s = (dist * 3.0) + (0.3 * c.format_conf) + (0.2 * c.dir_prior) - (0.3 * c.penalty)
            scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    id_anchors = sorted(all_cands.get("id_no", []), key=lambda c: (c.line, c.col))

    if id_anchors:
        for a in id_anchors:
            name_topk_list = nearest_k("name", a, k=5)
            name_c = name_topk_list[0] if name_topk_list else None
            date_c = nearest("ref_date", a)
            batch_c = nearest("batch_id", a)
            records.append((name_c, a, date_c, batch_c, name_topk_list))
    else:
        name_anchors = sorted(all_cands.get("name", []), key=lambda c: (c.line, c.col))
        for a in name_anchors:
            name_topk_list = nearest_k("name", a, k=5)
            id_c = nearest("id_no", a)
            date_c = nearest("ref_date", a)
            batch_c = nearest("batch_id", a)
            records.append((a, id_c, date_c, batch_c, name_topk_list))

    if not records:
        records.append((None, None, None, None, []))

    return records

# ==============================
# Main extraction pipeline (returns minimal triples)
# ==============================
def extract_minimal_from_text(text: str, surname_txt_path: Optional[str] = None) -> Dict:
    lines = normalize_text(text)
    surname_singles, surname_doubles = load_surnames_from_txt(surname_txt_path) if surname_txt_path else (set(), set(DEFAULT_DOUBLE_SURNAMES))

    # 1) 找標籤
    label_hits = find_label_hits(lines, LABELS, max_edit=1)
    per_field_label_presence = {f: False for f in LABELS}
    for h in label_hits:
        per_field_label_presence[h.field] = True

    # 2) 依標籤產生候選
    all_cands: Dict[str, List[Candidate]] = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
    for h in label_hits:
        cands = find_field_candidates_around_label(h.field, h, lines, surname_singles, surname_doubles)
        all_cands[h.field].extend(cands)

    # 2.5) 若沒姓名候選但有ID → 用ID附近補抓姓名
    if (not per_field_label_presence["name"] or not all_cands["name"]) and all_cands["id_no"]:
        for idc in all_cands["id_no"]:
            for dl in range(0, MAX_DOWN_LINES + 1):
                li = idc.line + dl
                if li >= len(lines): break
                for name, col in name_candidates_from_text(lines[li], surname_singles, surname_doubles):
                    dist = distance_score(idc.col, col, li - idc.line)
                    all_cands["name"].append(Candidate(
                        field="name", value=name, line=li, col=col,
                        label_line=idc.label_line, label_col=idc.label_col,
                        source_label=idc.source_label or "(ID-anchored)",
                        format_conf=0.7, label_conf=0.4, dir_prior=0.6, dist_score=dist,
                        context_bonus=0.2
                    ))

    # 2.7) 姓名距離身分證標籤加分
    if ENABLE_IDLABEL_PROXIMITY:
        id_label_positions: List[Tuple[int,int]] = [(h.line, h.col) for h in label_hits if h.field == "id_no"]
        if id_label_positions:
            for c in all_cands.get("name", []):
                best = 0.0
                for li, lc in id_label_positions:
                    dscore = distance_score(lc, c.col, c.line - li)
                    if dscore > best:
                        best = dscore
                c.context_bonus += IDLABEL_BONUS_SCALE * best

    # 3) 分組
    grouped = group_records(all_cands)

    # 4) 只回傳三項：value, confidence, context10
    def pack_field(c: Optional[Candidate]) -> Dict[str, Optional[str]]:
        if c is None:
            return {"value": None, "confidence": 0.0, "context10": None}
        conf = max(0.0, min(1.0, c.score()/3.0))
        ctx = get_context10(lines, c.line, c.col, len(c.value))
        return {"value": c.value, "confidence": round(conf, 4), "context10": ctx}

    minimal_records = []
    for name_c, id_c, date_c, batch_c, _name_top5 in grouped:
        minimal_records.append({
            "name": pack_field(name_c),
            "id_no": pack_field(id_c),
            "ref_date": pack_field(date_c),
            "batch_id": pack_field(batch_c),
        })

    # 5) 報告（可選）：標籤是否存在（保留，方便除錯；你要最精簡也可拿掉）
    report: Dict[str, List[str]] = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
    for field in ["name", "id_no", "ref_date", "batch_id"]:
        if not per_field_label_presence[field]:
            report[field].append("文件中未找到該欄位標籤（含模糊）。")
        else:
            if not all_cands[field]:
                report[field].append("找到標籤，但附近沒有合格候選。")
            else:
                report[field].append(f"找到標籤與候選 {len(all_cands[field])} 條。")

    return {
        "records": minimal_records,
        "report": report  # 如不需要可移除
    }

def extract_minimal_from_file(txt_path: str, surname_txt_path: Optional[str]) -> Dict:
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return extract_minimal_from_text(text, surname_txt_path)

# ==============================
# CLI (single file or batch folder)
# ==============================
def main(argv: List[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Rule-based TXT extractor (minimal output: value, confidence, context10) with batch folder support")
    ap.add_argument("txt", nargs="?", help="Input .txt file path (omit if using --input_dir)")
    ap.add_argument("--input_dir", help="Directory containing .txt files to process", default=None)
    ap.add_argument("--surnames", help="Path to comma-separated surnames txt (no newline)", default=None)
    ap.add_argument("--output", "-o", help="Output JSON path (default: stdout)", default=None)
    args = ap.parse_args(argv)

    results = {}
    if args.input_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
        for p in files:
            try:
                results[os.path.basename(p)] = extract_minimal_from_file(p, args.surnames)
            except Exception as e:
                results[os.path.basename(p)] = {"error": str(e)}
    elif args.txt:
        results[os.path.basename(args.txt)] = extract_minimal_from_file(args.txt, args.surnames)
    else:
        ap.error("Please provide a file path or --input_dir")

    js = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(js)
    else:
        print(js)

if __name__ == "__main__":
    main(sys.argv[1:])