#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based multi-record extractor for TXT documents (Taiwanese admin-style docs)
Unified scoring + Batch + Minimal output

Minimal output per field:
- value
- confidence (0~1, rounded to 4 decimals)
- context10 (前後各10字 from normalized text)

CLI:
  # 單檔
  python extractor.py input.txt --surnames surnames.txt

  # 批次資料夾（只讀 .txt）
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
from typing import List, Dict, Tuple, Optional, Set
import os
import glob

# ==============================
# Configuration
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

# Scoring weights (base internal score)
W_LABEL = 0.3
W_FORMAT = 0.9
W_DIST = 1.6
W_DIR = 0.4
W_CONTEXT = 0.2
W_PENALTY = 0.8

# Name rules
NAME_SEPARATORS = "·．• "
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

# ===== Unified selection parameters =====
ANCHOR_PROX_WEIGHT = 1.0  # 影響「靠近 anchor」的加分（可調 0~2）
DIR_RANK = {"same_right": 3, "same_left": 2, "below": 1}  # 決勝方向優先序

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
        # 核心分數：單一來源
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

# ==============================
# Name candidates
# ==============================
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
        else:
            # 單姓
            if ch in surname_singles:
                given, col = next_two_cjk_after(i + 1)
                if given and given not in BIGRAM_BLACKLIST:
                    cands.append((ch + given, i))
            i += 1

    return cands

# ==============================
# Candidate discovery
# ==============================
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
        results.append(Candidate(
            field=field, value=value, line=line, col=vcol,
            label_line=label.line, label_col=label.col,
            source_label=label.label_text,
            format_conf=fmt_conf,
            label_conf=1.0 - min(label.distance, 1) * 0.5,
            dir_prior=dir_prior, dist_score=dist,
        ))

    # same line: right
    right_seg = label_line_text[label.col:label.col+60]
    if field == "name":
        for name, c in name_candidates_from_text(right_seg, surname_singles, surname_doubles):
            add_candidate(name, label.col + c, label.line, "same_right", 0.8)
    elif field == "id_no":
        for m in re.finditer(r"[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8}", right_seg):
            code = m.group(0)
            fmt = 1.0 if tw_id_checksum_ok(code) or arc_id_like(code) else 0.5
            add_candidate(code, label.col + m.start(), label.line, "same_right", fmt)
    elif field == "ref_date":
        for pat in DATE_PATTERNS:
            for m in pat.finditer(right_seg):
                iso = parse_iso_date(m.group(0))
                if iso: add_candidate(iso, label.col + m.start(), label.line, "same_right", 1.0)
    elif field == "batch_id":
        for m in RE_BATCH_13.finditer(right_seg):
            add_candidate(m.group(0), label.col + m.start(), label.line, "same_right", 0.9)

    # same line: left
    left_start = max(0, label.col-60)
    left_seg = label_line_text[left_start:label.col]
    if field == "name":
        for name, c in name_candidates_from_text(left_seg, surname_singles, surname_doubles):
            add_candidate(name, left_start + c, label.line, "same_left", 0.8)
    elif field == "id_no":
        for m in re.finditer(r"[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8}", left_seg):
            code = m.group(0)
            fmt = 1.0 if tw_id_checksum_ok(code) or arc_id_like(code) else 0.5
            add_candidate(code, left_start + m.start(), label.line, "same_left", fmt)
    elif field == "ref_date":
        for pat in DATE_PATTERNS:
            for m in pat.finditer(left_seg):
                iso = parse_iso_date(m.group(0))
                if iso: add_candidate(iso, left_start + m.start(), label.line, "same_left", 1.0)
    elif field == "batch_id":
        for m in RE_BATCH_13.finditer(left_seg):
            add_candidate(m.group(0), left_start + m.start(), label.line, "same_left", 0.9)

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
                    if iso: add_candidate(iso, m.start(), tgt_line_idx, "below", 1.0)
        elif field == "batch_id":
            for m in RE_BATCH_13.finditer(tgt):
                add_candidate(m.group(0), m.start(), tgt_line_idx, "below", 0.9)

    return results

# ==============================
# Unified selection logic
# ==============================
def _dir_of(c: Candidate, anchor: Candidate) -> str:
    if c.line == anchor.line:
        return "same_right" if c.col >= anchor.col else "same_left"
    return "below" if c.line > anchor.line else "above"  # "above" 會在 DIR_RANK 中當 0

def anchor_proximity_score(anchor_col: int, cand_col: int, line_delta: int) -> float:
    # 與距離函式一致，輸出 0~1
    return distance_score(anchor_col, cand_col, line_delta)

def final_score(c: Candidate, anchor: Optional[Candidate] = None) -> float:
    base = c.score()
    if anchor is None:
        return base
    prox = anchor_proximity_score(anchor.col, c.col, c.line - anchor.line)  # 0~1
    return base + ANCHOR_PROX_WEIGHT * prox

def tie_break_key(c: Candidate, anchor: Optional[Candidate] = None) -> Tuple:
    if anchor is None:
        # 無 anchor：偏向上/左、格式佳，以穩定排序
        return (10**6, 0, 10**6, -c.format_conf, c.line, c.col)
    dline = abs(c.line - anchor.line)
    dcol = abs(c.col - anchor.col)
    dir_score = DIR_RANK.get(_dir_of(c, anchor), 0)
    return (dline, -dir_score, dcol, -c.format_conf, c.line, c.col)

def pick_best_candidate(cands: List[Candidate], anchor: Optional[Candidate] = None) -> Optional[Candidate]:
    if not cands:
        return None
    cands_sorted = sorted(cands, key=lambda x: (-final_score(x, anchor), tie_break_key(x, anchor)))
    return cands_sorted[0]

# ==============================
# Context helper
# ==============================
def get_context10(lines: List[str], line: int, col: int, span_len: int) -> str:
    if line < 0 or line >= len(lines): return ""
    s = lines[line]
    start = max(0, col - 10)
    end = min(len(s), col + span_len + 10)
    return s[start:end]

# ==============================
# Grouping (uses unified selection)
# ==============================
def group_records(all_cands: Dict[str, List[Candidate]]) -> List[Tuple[Optional[Candidate], Optional[Candidate], Optional[Candidate], Optional[Candidate], List[Candidate]]]:
    records: List[Tuple[Optional[Candidate], Optional[Candidate], Optional[Candidate], Optional[Candidate], List[Candidate]]] = []

    def nearest(field: str, anchor: Candidate) -> Optional[Candidate]:
        pool = [c for c in all_cands.get(field, []) if abs(c.line - anchor.line) <= MAX_DOWN_LINES]
        if not pool: return None
        pool_sorted = sorted(pool, key=lambda c: (-final_score(c, anchor), tie_break_key(c, anchor)))
        return pool_sorted[0]

    def nearest_k(field: str, anchor: Candidate, k: int = 5) -> List[Candidate]:
        pool = [c for c in all_cands.get(field, []) if abs(c.line - anchor.line) <= MAX_DOWN_LINES]
        pool_sorted = sorted(pool, key=lambda c: (-final_score(c, anchor), tie_break_key(c, anchor)))
        return pool_sorted[:k]

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
# Extraction (minimal triples)
# ==============================
def extract_minimal_from_text(text: str, surname_txt_path: Optional[str] = None) -> Dict:
    lines = normalize_text(text)
    surname_singles, surname_doubles = load_surnames_from_txt(surname_txt_path) if surname_txt_path else (set(), set(DEFAULT_DOUBLE_SURNAMES))

    # 1) 標籤
    label_hits = find_label_hits(lines, LABELS, max_edit=1)
    per_field_label_presence = {f: False for f in LABELS}
    for h in label_hits:
        per_field_label_presence[h.field] = True

    # 2) 產生候選
    all_cands: Dict[str, List[Candidate]] = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
    for h in label_hits:
        all_cands[h.field].extend(find_field_candidates_around_label(h.field, h, lines, surname_singles, surname_doubles))

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

    # 2.7) 姓名距 ID 標籤加分（寫進 context_bonus，仍屬 base score 一部分）
    if ENABLE_IDLABEL_PROXIMITY:
        id_label_positions: List[Tuple[int,int]] = [(h.line, h.col) for h in label_hits if h.field == "id_no"]
        if id_label_positions:
            for c in all_cands.get("name", []):
                best = 0.0
                for li, lc in id_label_positions:
                    dscore = distance_score(lc, c.col, c.line - li)
                    best = max(best, dscore)
                c.context_bonus += IDLABEL_BONUS_SCALE * best

    # 3) 分組（統一評分）
    grouped = group_records(all_cands)

    # 4) 最小輸出
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

    # 可保留簡短報告（若不需要可刪）
    report: Dict[str, List[str]] = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
    for field in ["name", "id_no", "ref_date", "batch_id"]:
        if not per_field_label_presence[field]:
            report[field].append("未找到標籤")
        else:
            if not all_cands[field]:
                report[field].append("有標籤但無候選")
            else:
                report[field].append(f"候選 {len(all_cands[field])} 條")

    return {"records": minimal_records, "report": report}

def extract_minimal_from_file(txt_path: str, surname_txt_path: Optional[str]) -> Dict:
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return extract_minimal_from_text(text, surname_txt_path)

# ==============================
# CLI (single or batch)
# ==============================
def main(argv: List[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="TXT extractor (unified scoring, minimal output) with batch support")
    ap.add_argument("txt", nargs="?", help="Input .txt file (omit if using --input_dir)")
    ap.add_argument("--input_dir", help="Directory containing .txt files", default=None)
    ap.add_argument("--surnames", help="Comma-separated surnames txt", default=None)
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