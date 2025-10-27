#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TXT extractor with spaCy PERSON priority + rule fallback
CSV output format (value, confidence, ±10-char context) & batch processing.

Examples
--------
# 單檔 -> CSV（印到 stdout）
python extractor.py input.txt --csv

# 單檔 -> CSV 檔
python extractor.py input.txt --csv -o out.csv

# 批次：讀 in_dir 內所有 .txt，逐檔輸出同名 .csv 到 out_dir
python extractor.py --input-dir ./in_txt --out-dir ./out_csv --csv
"""
from __future__ import annotations

import os
import re
import csv
import json
import math
import sys
import unicodedata
import string
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Iterable, Set

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

# Scoring weights（全域總分；分組與輸出皆用這套）
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
ENABLE_IDLABEL_PROXIMITY    = True
IDLABEL_BONUS_SCALE         = 1.0

# Batch ID
RE_BATCH_13 = re.compile(r"\b\d{13}\b")

# ID patterns
RE_ID_TW  = re.compile(r"^[A-Z][0-9]{9}$")
RE_ID_ARC = re.compile(r"^[A-Z]{2}[0-9]{8}$")

# Date patterns
DATE_PATTERNS = [
    re.compile(r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b"),
    re.compile(r"民國\s*\d{2,3}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"),
    re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"),
]

# Double surnames (seed; 可再由檔案擴充)
DEFAULT_DOUBLE_SURNAMES = {
    "歐陽","司馬","諸葛","上官","東方","夏侯","司徒","司空","司寇","令狐",
    "公孫","公羊","公冶","慕容","端木","皇甫","長孫","尉遲","赫連","納蘭",
    "澹臺","南宮","拓跋","宇文","完顏","呼延","夏侯","聞人","司南","仲長",
}

CJK_RANGE = "\u4e00-\u9fff"
RE_CJK = re.compile(rf"^[{CJK_RANGE}]+$")

# ==============================
# spaCy PERSON integration (optional)
# ==============================
USE_SPACY_PERSON = True
SPACY_ZH_MODEL   = "zh_core_web_sm"  # 或 "zh_core_web_trf"

# ==============================
# 姓名過濾強化
# ==============================
REQUIRE_SURNAME_PREFIX_FOR_PERSON = True  # --person-loose 可放寬
ROLE_WORD_BLACKLIST = {
    "債務人","債權人","申報人","調查人","受文者","陳情人","本人","相對人",
    "被告","原告","承辦人","戶名","帳戶名","持有人","繳款人","申請人","受益人","收件人"
}
LATIN = set(string.ascii_letters)

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
        return (W_LABEL*self.label_conf + W_FORMAT*self.format_conf + W_DIST*self.dist_score
                + W_DIR*self.dir_prior + W_CONTEXT*self.context_bonus - W_PENALTY*self.penalty)

@dataclass
class FieldResult:
    value: Optional[str]
    confidence: float
    source: Optional[Dict]
    notes: List[str]

@dataclass
class Record:
    name: FieldResult
    id_no: FieldResult
    ref_date: FieldResult
    batch_id: FieldResult
    debug: Dict

# ==============================
# Utilities
# ==============================
def to_halfwidth(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def normalize_text(s: str) -> List[str]:
    s = to_halfwidth(s).replace("\r\n","\n").replace("\r","\n")
    lines = [re.sub(r"[ \t　]+", " ", x) for x in s.split("\n")]
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
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost))
        prev = cur
    return prev[-1]

def find_label_hits(lines: List[str], labels: Dict[str, List[str]], max_edit: int = 1) -> List[LabelHit]:
    hits: List[LabelHit] = []
    for li, line in enumerate(lines):
        for field, labellist in labels.items():
            for lab in labellist:
                for m in re.finditer(re.escape(lab), line):
                    hits.append(LabelHit(field, lab, lab, 0, li, m.start()))
        for t in re.finditer(r"[\w\u4e00-\u9fff]{2,6}", line):
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

LETTER_MAP = {chr(ord('A')+i): 10+i for i in range(26)}
WEIGHTS_TW_ID = [1,9,8,7,6,5,4,3,2,1,1]

def tw_id_checksum_ok(code: str) -> bool:
    if not RE_ID_TW.fullmatch(code): return False
    n = LETTER_MAP.get(code[0]);  a, b = divmod(n, 10) if n is not None else (None,None)
    if a is None: return False
    digits = [a, b] + [int(x) for x in code[1:]]
    return sum(d*w for d,w in zip(digits, WEIGHTS_TW_ID)) % 10 == 0

def arc_id_like(code: str) -> bool:
    return RE_ID_ARC.fullmatch(code) is not None

def parse_iso_date(txt: str) -> Optional[str]:
    txt = txt.strip()
    m = re.match(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$", txt)
    if m:
        y,mo,d = map(int, m.groups())
        try: return datetime(y,mo,d).strftime("%Y-%m-%d")
        except ValueError: return None
    m = re.match(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$", txt)
    if m:
        y,mo,d = map(int, m.groups())
        try: return datetime(y,mo,d).strftime("%Y-%m-%d")
        except ValueError: return None
    m = re.match(r"^民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$", txt)
    if m:
        roc,mo,d = map(int, m.groups()); y = roc + 1911
        try: return datetime(y,mo,d).strftime("%Y-%m-%d")
        except ValueError: return None
    return None

def is_cjk(s: str) -> bool:
    return RE_CJK.fullmatch(s) is not None

def load_surnames_from_txt(path: str) -> Tuple[Set[str], Set[str]]:
    try:
        content = open(path,"r",encoding="utf-8").read().strip()
    except Exception:
        content = ""
    singles: Set[str] = set()
    doubles: Set[str] = set(DEFAULT_DOUBLE_SURNAMES)
    if content:
        for p in [p.strip() for p in content.split(",") if p.strip()]:
            if is_cjk(p):
                (singles if len(p)==1 else doubles).add(p)
    return singles, doubles

def distance_score(label_col: int, cand_col: int, line_delta: int, tau: float = TAU_COL) -> float:
    line_w = LINE_WEIGHTS.get(abs(line_delta), 0.0)
    col_w = math.exp(-abs(cand_col - label_col)/tau)
    return line_w * col_w

# ------------------------------
# spaCy PERSON 索引建立
# ------------------------------
def build_spacy_person_index(lines: List[str]) -> Tuple[Dict[int, List[Tuple[str, int]]], Dict[str, str]]:
    meta = {"enabled": "false", "model": "", "error": ""}
    person_by_line: Dict[int, List[Tuple[str, int]]] = {}
    if not USE_SPACY_PERSON: return person_by_line, meta
    try:
        import spacy
        nlp = spacy.load(SPACY_ZH_MODEL)
        meta["enabled"], meta["model"] = "true", SPACY_ZH_MODEL
    except Exception as e:
        meta["error"] = f"spaCy load failed: {e}"
        return person_by_line, meta

    joined = "\n".join(lines)
    starts, cur = [], 0
    for i, line in enumerate(lines):
        starts.append(cur); cur += len(line) + (1 if i < len(lines)-1 else 0)
    doc = nlp(joined)
    for ent in doc.ents:
        if getattr(ent, "label_", "") != "PERSON": continue
        g0 = ent.start_char
        line_idx = 0
        for i in range(len(starts)-1, -1, -1):
            if g0 >= starts[i]:
                line_idx = i; break
        col = g0 - starts[line_idx]
        if 0 <= line_idx < len(lines) and 0 <= col <= len(lines[line_idx]):
            person_by_line.setdefault(line_idx, []).append((ent.text, col))
    for li in person_by_line: person_by_line[li].sort(key=lambda x: x[1])
    return person_by_line, meta

# ==============================
# Name candidates (rule-based)
# ==============================
def name_candidates_from_text(line_text: str, surname_singles: Set[str], surname_doubles: Set[str]) -> List[Tuple[str, int]]:
    cands: List[Tuple[str,int]] = []
    text = line_text
    n = len(text)
    sep_set = set(NAME_SEPARATORS)
    def next_two_cjk_after(start: int):
        j = start
        while j < n and text[j] in sep_set: j += 1
        given, col = [], j
        while j < n and len(given) < 2:
            ch = text[j]
            if RE_CJK.fullmatch(ch): given.append(ch); j += 1
            else: break
        return ("".join(given), col) if len(given)==2 else (None, None)
    doubles_sorted = sorted(surname_doubles, key=len, reverse=True)
    i = 0
    while i < n:
        matched = False
        for ds in doubles_sorted:
            L = len(ds)
            if i+L <= n and text[i:i+L] == ds:
                given, col = next_two_cjk_after(i+L)
                if given and given not in BIGRAM_BLACKLIST: cands.append((ds+given, i))
                matched = True; break
        if matched: i += 1; continue
        ch = text[i]
        if ENABLE_DYNAMIC_DOUBLE_SURNAME and i+1 < n:
            ch2 = text[i+1]
            if ch in surname_singles and ch2 in surname_singles:
                given, col = next_two_cjk_after(i+2)
                if given and given not in BIGRAM_BLACKLIST: cands.append((ch+ch2+given, i)); matched = True
        if matched: i += 1; continue
        if ch in surname_singles:
            given, col = next_two_cjk_after(i+1)
            if given and given not in BIGRAM_BLACKLIST: cands.append((ch+given, i))
        i += 1
    return cands

# ==============================
# 最後一道姓名門檻
# ==============================
def passes_name_gate(txt: str, surname_singles: Set[str], surname_doubles: Set[str], is_spacy_person: bool) -> bool:
    t = txt.strip()
    if t in ROLE_WORD_BLACKLIST: return False
    if not (2 <= len(t) <= 6): return False
    has_foreign = any(ch in t for ch in "·．• ") or any(ch in LATIN for ch in t)
    starts_double = any(t.startswith(ds) for ds in surname_doubles)
    starts_single = (len(t) >= 1 and (t[0] in surname_singles))
    if is_spacy_person and REQUIRE_SURNAME_PREFIX_FOR_PERSON:
        if not (starts_double or starts_single or has_foreign): return False
    else:
        if not (starts_double or starts_single or has_foreign): return False
    return True

# ==============================
# Candidate search around labels
# ==============================
@dataclass
class _AddCtx:  # 調試用
    used_any_person: bool = False

def find_field_candidates_around_label(field: str, label: LabelHit, lines: List[str],
                                       surname_singles: Set[str], surname_doubles: Set[str],
                                       person_index: Optional[Dict[int, List[Tuple[str, int]]]] = None) -> List[Candidate]:
    results: List[Candidate] = []
    label_line_text = lines[label.line]

    def add_candidate(value: str, vcol: int, line: int, dir_key: str, fmt_conf: float) -> None:
        line_delta = line - label.line
        col_delta = abs(vcol - label.col)
        dist = distance_score(label.col, vcol, line_delta)
        if field == "name":
            if (line_delta == 0 and col_delta > 14) or (line_delta != 0 and col_delta > 10) or abs(line_delta) > 1 or dist < 0.5:
                return
        else:
            if dist < 0.2: return
        dir_prior = DIRECTION_PRIOR.get(dir_key, 0.0)
        results.append(Candidate(
            field=field, value=value, line=line, col=vcol,
            label_line=label.line, label_col=label.col, source_label=label.label_text,
            format_conf=fmt_conf, label_conf=1.0 - min(label.distance,1)*0.5,
            dir_prior=dir_prior, dist_score=dist
        ))

    # same line: right
    right_seg = label_line_text[label.col:label.col+60]
    if field == "name":
        used_any = False
        if person_index is not None and label.line in person_index:
            for txt, c in person_index[label.line]:
                if label.col <= c < label.col+60 and passes_name_gate(txt, surname_singles, surname_doubles, True):
                    add_candidate(txt, c, label.line, "same_right", 0.9)
                    used_any = True
        if not used_any:
            for name, c in name_candidates_from_text(right_seg, surname_singles, surname_doubles):
                if passes_name_gate(name, surname_singles, surname_doubles, False):
                    add_candidate(name, label.col + c, label.line, "same_right", 0.8)
    elif field == "id_no":
        for m in re.finditer(r"[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8}", right_seg):
            code = m.group(0)
            fmt = 1.0 if tw_id_checksum_ok(code) or arc_id_like(code) else 0.5
            add_candidate(code, label.col+m.start(), label.line, "same_right", fmt)
    elif field == "ref_date":
        for pat in DATE_PATTERNS:
            for m in pat.finditer(right_seg):
                iso = parse_iso_date(m.group(0))
                if iso: add_candidate(iso, label.col+m.start(), label.line, "same_right", 1.0)
    elif field == "batch_id":
        for m in RE_BATCH_13.finditer(right_seg):
            add_candidate(m.group(0), label.col+m.start(), label.line, "same_right", 0.9)

    # same line: left
    left_seg = label_line_text[max(0, label.col-60):label.col]
    if field == "name":
        used_any = False
        if person_index is not None and label.line in person_index:
            for txt, c in person_index[label.line]:
                if (label.col-60) <= c < label.col and passes_name_gate(txt, surname_singles, surname_doubles, True):
                    add_candidate(txt, c, label.line, "same_left", 0.9)
                    used_any = True
        if not used_any:
            for name, c in name_candidates_from_text(left_seg, surname_singles, surname_doubles):
                if passes_name_gate(name, surname_singles, surname_doubles, False):
                    add_candidate(name, c, label.line, "same_left", 0.8)
    elif field == "id_no":
        for m in re.finditer(r"[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8}", left_seg):
            code = m.group(0)
            fmt = 1.0 if tw_id_checksum_ok(code) or arc_id_like(code) else 0.5
            add_candidate(code, m.start(), label.line, "same_left", fmt)
    elif field == "ref_date":
        for pat in DATE_PATTERNS:
            for m in pat.finditer(left_seg):
                iso = parse_iso_date(m.group(0))
                if iso: add_candidate(iso, m.start(), label.line, "same_left", 1.0)
    elif field == "batch_id":
        for m in RE_BATCH_13.finditer(left_seg):
            add_candidate(m.group(0), m.start(), label.line, "same_left", 0.9)

    # below lines
    for dl in range(1, MAX_DOWN_LINES+1):
        li = label.line + dl
        if li >= len(lines): break
        tgt = lines[li]
        if field == "name":
            used_any = False
            if person_index is not None and li in person_index:
                for txt, c in person_index[li]:
                    if passes_name_gate(txt, surname_singles, surname_doubles, True):
                        add_candidate(txt, c, li, "below", 0.9)
                        used_any = True
            if not used_any:
                for name, c in name_candidates_from_text(tgt, surname_singles, surname_doubles):
                    if passes_name_gate(name, surname_singles, surname_doubles, False):
                        add_candidate(name, c, li, "below", 0.8)
        elif field == "id_no":
            for m in re.finditer(r"[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8}", tgt):
                code = m.group(0)
                fmt = 1.0 if tw_id_checksum_ok(code) or arc_id_like(code) else 0.5
                add_candidate(code, m.start(), li, "below", fmt)
        elif field == "ref_date":
            for pat in DATE_PATTERNS:
                for m in pat.finditer(tgt):
                    iso = parse_iso_date(m.group(0))
                    if iso: add_candidate(iso, m.start(), li, "below", 1.0)
        elif field == "batch_id":
            for m in RE_BATCH_13.finditer(tgt):
                add_candidate(m.group(0), m.start(), li, "below", 0.9)
    return results

def pick_best_candidate(cands: List[Candidate]) -> Optional[Candidate]:
    return sorted(cands, key=lambda c: c.score(), reverse=True)[0] if cands else None

# ==============================
# Record grouping / anchor logic
# ==============================
def group_records(all_cands: Dict[str, List[Candidate]]) -> List[Record]:
    records: List[Record] = []
    def nearest(field: str, anchor: Candidate) -> Optional[Candidate]:
        best, best_val = None, -1.0
        for c in all_cands.get(field, []):
            if abs(c.line - anchor.line) > MAX_DOWN_LINES: continue
            if distance_score(anchor.col, c.col, c.line - anchor.line) <= 0.0: continue
            val = c.score()
            if val > best_val: best_val, best = val, c
        return best
    def nearest_k(field: str, anchor: Candidate, k: int = 5) -> List[Candidate]:
        pool: List[Candidate] = []
        for c in all_cands.get(field, []):
            if abs(c.line - anchor.line) > MAX_DOWN_LINES: continue
            if distance_score(anchor.col, c.col, c.line - anchor.line) <= 0.0: continue
            pool.append(c)
        pool.sort(key=lambda x: x.score(), reverse=True)
        return pool[:k]

    id_anchors = sorted(all_cands.get("id_no", []), key=lambda c: (c.line, c.col))
    if id_anchors:
        for a in id_anchors:
            name_topk = nearest_k("name", a, 5)
            rec = assemble_record(name_topk[0] if name_topk else None,
                                  a, nearest("ref_date", a), nearest("batch_id", a),
                                  all_cands, name_topk=name_topk)
            records.append(rec)
    else:
        name_anchors = sorted(all_cands.get("name", []), key=lambda c: (c.line, c.col))
        for a in name_anchors:
            name_topk = nearest_k("name", a, 5)
            rec = assemble_record(a, nearest("id_no", a), nearest("ref_date", a),
                                  nearest("batch_id", a), all_cands, name_topk=name_topk)
            records.append(rec)

    if not records:
        records.append(Record(
            name=FieldResult(None,0.0,None,["未偵測到任何姓名候選或身分證號錨點"]),
            id_no=FieldResult(None,0.0,None,["未偵測到任何身分證號候選"]),
            ref_date=FieldResult(None,0.0,None,["未偵測到任何日期候選"]),
            batch_id=FieldResult(None,0.0,None,["未偵測到任何13位名單檔候選"]),
            debug={"name_top5":[]}
        ))
    return records

def assemble_record(name_c: Optional[Candidate], id_c: Optional[Candidate],
                    date_c: Optional[Candidate], batch_c: Optional[Candidate],
                    all_cands: Dict[str, List[Candidate]],
                    name_topk: Optional[List[Candidate]] = None) -> Record:
    def field_result_from_cand(c: Optional[Candidate], fallback_notes: List[str]) -> FieldResult:
        if c is None: return FieldResult(None, 0.0, None, fallback_notes)
        return FieldResult(
            value=c.value, confidence=max(0.0, min(1.0, c.score()/3.0)),
            source={"line": c.line,"col": c.col,"label": c.source_label,"label_line": c.label_line,"label_col": c.label_col,
                    "score_breakdown":{"label_conf": c.label_conf,"format_conf": c.format_conf,"dist_score": c.dist_score,
                                       "dir_prior": c.dir_prior,"context_bonus": c.context_bonus,"penalty": c.penalty,
                                       "total": c.score()}},
            notes=[]
        )
    def pack_topk(cands: List[Candidate]) -> List[Dict]:
        out = []
        for c in (cands or [])[:5]:
            out.append({"value": c.value, "approx_confidence": max(0.0, min(1.0, c.score()/3.0)),
                        "line": c.line, "col": c.col, "label": c.source_label,
                        "label_line": c.label_line, "label_col": c.label_col,
                        "score_breakdown":{"label_conf": c.label_conf,"format_conf": c.format_conf,"dist_score": c.dist_score,
                                           "dir_prior": c.dir_prior,"context_bonus": c.context_bonus,"penalty": c.penalty,
                                           "total": c.score()}})
        return out
    rec = Record(
        name=field_result_from_cand(name_c, ["找不到姓名候選。"]),
        id_no=field_result_from_cand(id_c, ["找不到身分證號候選。"]),
        ref_date=field_result_from_cand(date_c, ["找不到日期候選。"]),
        batch_id=field_result_from_cand(batch_c, ["找不到13位名單檔候選。"]),
        debug={"name_top5": pack_topk(name_topk or []),
               "all_candidates_counts": {k: len(v) for k, v in all_cands.items()}}
    )
    return rec

# ==============================
# Extraction core
# ==============================
def extract_from_text(text: str, surname_txt_path: Optional[str] = None) -> Dict:
    lines = normalize_text(text)
    surname_singles, surname_doubles = (load_surnames_from_txt(surname_txt_path)
                                        if surname_txt_path else (set(), set(DEFAULT_DOUBLE_SURNAMES)))
    spacy_person_index, spacy_meta = {}, {"enabled":"false","model":"","error":""}
    try:
        spacy_person_index, spacy_meta = build_spacy_person_index(lines)
    except Exception as e:
        spacy_meta = {"enabled":"false","model":"","error":f"build failed: {e}"}

    label_hits = find_label_hits(lines, LABELS, max_edit=1)
    per_field_label_presence = {f: False for f in LABELS}
    for h in label_hits: per_field_label_presence[h.field] = True

    all_cands: Dict[str, List[Candidate]] = {k: [] for k in ["name","id_no","ref_date","batch_id"]}
    for h in label_hits:
        all_cands[h.field].extend(
            find_field_candidates_around_label(h.field, h, lines, surname_singles, surname_doubles, spacy_person_index)
        )

    if (not per_field_label_presence["name"] or not all_cands["name"]) and all_cands["id_no"]:
        for idc in all_cands["id_no"]:
            for dl in range(0, MAX_DOWN_LINES+1):
                li = idc.line + dl
                if li >= len(lines): break
                used = False
                if spacy_person_index.get(li):
                    for txt, col in spacy_person_index[li]:
                        if not passes_name_gate(txt, surname_singles, surname_doubles, True): continue
                        dist = distance_score(idc.col, col, li - idc.line)
                        all_cands["name"].append(Candidate("name", txt, li, col,
                            idc.label_line, idc.label_col, idc.source_label or "(ID-anchored)",
                            0.85, 0.5, 0.6, dist, 0.25))
                        used = True
                if not used:
                    for name, col in name_candidates_from_text(lines[li], surname_singles, surname_doubles):
                        if not passes_name_gate(name, surname_singles, surname_doubles, False): continue
                        dist = distance_score(idc.col, col, li - idc.line)
                        all_cands["name"].append(Candidate("name", name, li, col,
                            idc.label_line, idc.label_col, idc.source_label or "(ID-anchored)",
                            0.7, 0.4, 0.6, dist, 0.2))

    if ENABLE_IDLABEL_PROXIMITY:
        id_label_positions = [(h.line, h.col) for h in label_hits if h.field == "id_no"]
        if id_label_positions:
            for c in all_cands.get("name", []):
                best = 0.0
                for li, lc in id_label_positions:
                    dscore = distance_score(lc, c.col, c.line - li)
                    if dscore > best: best = dscore
                c.context_bonus += IDLABEL_BONUS_SCALE * best

    records = group_records(all_cands)

    report: Dict[str, List[str]] = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
    for field in ["name","id_no","ref_date","batch_id"]:
        if not per_field_label_presence[field]:
            report[field].append("文件中未找到任何該欄位標籤（含模糊匹配）。")
        else:
            report[field].append("找到了標籤與候選（共 {} 條），已打分。".format(len(all_cands[field])) if all_cands[field]
                                 else "找到了標籤，但附近未找到符合格式/校驗的候選值。")

    return {
        "records": [
            {"name": asdict(r.name), "id_no": asdict(r.id_no), "ref_date": asdict(r.ref_date),
             "batch_id": asdict(r.batch_id), "debug": r.debug}
            for r in records
        ],
        "report": report,
        "meta": {"lines": len(lines), "spacy_person": spacy_meta, "text_lines": lines}
    }

# ==============================
# CSV minimalization
# ==============================
def _context_around_line(line: str, col: Optional[int], value: Optional[str], pad: int = 10) -> str:
    if value is None or col is None: return ""
    start = max(0, col - pad)
    end = min(len(line), col + len(value) + pad)
    return line[start:end]

def _field_to_triplet(lines: List[str], fr: Dict) -> Tuple[str, float, str]:
    val = fr.get("value")
    conf = fr.get("confidence", 0.0) or 0.0
    src = fr.get("source") or {}
    line_idx, col = src.get("line"), src.get("col")
    ctx = _context_around_line(lines[line_idx], col, val, 10) if isinstance(line_idx, int) and 0 <= line_idx < len(lines) else ""
    return val or "", round(conf, 6), ctx

def to_csv_rows(filename: str, full_result: Dict) -> Tuple[List[str], List[List[str]]]:
    """回傳 (header, rows)。每筆 record 一列；只輸出 value/confidence/context 三項/欄位。"""
    lines = full_result.get("meta", {}).get("text_lines") or []
    header = [
        "file","record_idx",
        "name_value","name_confidence","name_context",
        "id_no_value","id_no_confidence","id_no_context",
        "ref_date_value","ref_date_confidence","ref_date_context",
        "batch_id_value","batch_id_confidence","batch_id_context",
    ]
    rows: List[List[str]] = []
    for idx, rec in enumerate(full_result["records"]):
        name_v, name_c, name_ctx = _field_to_triplet(lines, rec["name"])
        id_v, id_c, id_ctx       = _field_to_triplet(lines, rec["id_no"])
        date_v, date_c, date_ctx = _field_to_triplet(lines, rec["ref_date"])
        bid_v, bid_c, bid_ctx    = _field_to_triplet(lines, rec["batch_id"])
        rows.append([
            filename, str(idx),
            name_v, f"{name_c:.6f}", name_ctx,
            id_v,   f"{id_c:.6f}",   id_ctx,
            date_v, f"{date_c:.6f}", date_ctx,
            bid_v,  f"{bid_c:.6f}",  bid_ctx,
        ])
    return header, rows

# ==============================
# I/O helpers
# ==============================
def extract_from_file(txt_path: str, surname_txt_path: Optional[str]) -> Dict:
    text = open(txt_path,"r",encoding="utf-8").read()
    return extract_from_text(text, surname_txt_path)

def write_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

# ==============================
# CLI
# ==============================
def main(argv: List[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="TXT extractor (spaCy PERSON + rule fallback) with CSV output & batch mode")
    ap.add_argument("txt", nargs="?", help="Input .txt path (omit when using --input-dir)")
    ap.add_argument("--surnames", help="Comma-separated surnames txt path", default=None)
    ap.add_argument("--output","-o", help="Output file path (single-file mode). For --csv, use .csv", default=None)
    ap.add_argument("--csv", action="store_true", help="Output CSV with value/confidence/context triplets")
    ap.add_argument("--no-spacy", action="store_true", help="Disable spaCy PERSON")
    ap.add_argument("--spacy-model", default=None, help="spaCy zh model (zh_core_web_sm / zh_core_web_trf)")
    ap.add_argument("--person-loose", action="store_true", help="Allow PERSON not starting with Chinese surname")
    # batch
    ap.add_argument("--input-dir", help="Directory containing .txt files for batch run")
    ap.add_argument("--out-dir", help="Output directory for batch CSV/JSON (will be created if missing)")
    args = ap.parse_args(argv)

    # 覆寫設定
    global USE_SPACY_PERSON, SPACY_ZH_MODEL, REQUIRE_SURNAME_PREFIX_FOR_PERSON
    if args.no_spacy: USE_SPACY_PERSON = False
    if args.spacy_model: SPACY_ZH_MODEL = args.spacy_model
    if args.person_loose: REQUIRE_SURNAME_PREFIX_FOR_PERSON = False

    # 批次模式
    if args.input_dir:
        in_dir = args.input_dir
        out_dir = args.out_dir or "./_out_csv"
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(".txt")]
        for fname in sorted(files):
            fpath = os.path.join(in_dir, fname)
            try:
                full_res = extract_from_file(fpath, args.surnames)
                if args.csv:
                    header, rows = to_csv_rows(fname, full_res)
                    out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".csv")
                    write_csv(out_path, header, rows)
                else:
                    out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(full_res, f, ensure_ascii=False, indent=2)
            except Exception as e:
                err_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".error.txt")
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(str(e))
        return

    # 單檔模式
    if not args.txt:
        print("Error: provide a TXT path or use --input-dir for batch.", file=sys.stderr)
        sys.exit(2)

    full_res = extract_from_file(args.txt, args.surnames)
    if args.csv:
        header, rows = to_csv_rows(os.path.basename(args.txt), full_res)
        if args.output:
            write_csv(args.output, header, rows)
        else:
            # 印到 stdout
            w = csv.writer(sys.stdout)
            w.writerow(header)
            for r in rows: w.writerow(r)
    else:
        js = json.dumps(full_res, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f: f.write(js)
        else:
            print(js)

if __name__ == "__main__":
    main(sys.argv[1:])