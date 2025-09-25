#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based multi-record extractor for TXT documents (Taiwanese admin-style docs)

新增功能
- 針對每個錨點（優先以身分證號為錨），列出「姓名 Top-5 候選」：輸出於 records[i].name_top5
- 仍保留自動挑選的單一最佳姓名於 records[i].name.value
- 其他欄位與行為向下相容

原有功能（節選）
- 標籤驅動的鄰近搜尋（同列 + 下方多列），距離打分
- 嚴格驗證：姓名（姓氏表、雙姓、middle-dot）、身分證（本國/外來，含 checksum）、日期（含民國轉 ISO）、13 碼批號
- 模糊標籤（編輯距離 ≤ 1）抗 OCR 噪音
- 距離模型：行距權重 × 指數型列距衰減 + 方向先驗
- 以 ID 為錨點的分組（無 ID 則改用姓名）
- 清晰的報告：標籤缺失、找到標籤但無有效值等
- 模組化、可測、常數可調

Python 3.12; standard library only.
"""
from __future__ import annotations

import re
import json
import math
import sys
import unicodedata
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

# Direction priors (you may customize per field if desired)
DIRECTION_PRIOR = {
    "same_right": 1.2,
    "same_left": 0.9,
    "below": 0.6,
}

# Distance model
MAX_DOWN_LINES = 3  # search up to N lines below the label
LINE_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3}  # line distance → base weight
TAU_COL = 12.0      # column distance scale (in characters)

# Scoring weights
W_LABEL = 0.3
W_FORMAT = 0.9
W_DIST = 1.6
W_DIR = 0.4
W_CONTEXT = 0.2
W_PENALTY = 0.8

# Name rules
NAME_GIVEN_MIN = 1
NAME_GIVEN_MAX = 3
NAME_SEPARATORS = "·．• "  # middle-dot variants
NAME_BLACKLIST_NEAR = {"公司", "單位", "科", "處", "部", "電話", "分機", "地址", "附件", "銀行", "分行", "室", "股", "隊", "路", "段", "號", "樓", "市", "縣", "鄉", "鎮", "村", "里"}
HONORIFICS = {"先生","小姐","女士","太太","老師","主管","經理","博士"}
# common non-name bigrams that often appear right after labels; exclude as given-name
BIGRAM_BLACKLIST = {"應於","基準","查詢","調查","名單","身分","證號","統編","日期","時間","銀行","公司","單位","地址","電話"}

# Batch ID strict binding to label required
RE_BATCH_13 = re.compile(r"\b\d{13}\b")

# ID patterns
RE_ID_TW = re.compile(r"^[A-Z][0-9]{9}$")
RE_ID_ARC = re.compile(r"^[A-Z]{2}[0-9]{8}$")

# Date patterns (strings to search in text; parsing handled separately)
DATE_PATTERNS = [
    re.compile(r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b"),
    re.compile(r"民國\s*\d{2,3}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"),
    re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"),
]

# Double surnames (expandable). You will load the full set from your txt file.
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
    label_text: str  # canonical label matched to
    matched: str     # surface string in text
    distance: int    # edit distance
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
    """Normalize and split into lines, preserving line breaks.
    - Halfwidth normalization
    - Normalize newlines to \n
    - Collapse spaces per line (not across lines)
    """
    s = to_halfwidth(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    # collapse only spaces/tabs/ideographic spaces within each line
    lines = [re.sub(r"[ \t　]+", " ", line) for line in lines]
    return lines

# Simple Levenshtein distance (edit distance)
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,      # deletion
                cur[j-1] + 1,     # insertion
                prev[j-1] + cost  # substitution
            ))
        prev = cur
    return prev[-1]

def find_label_hits(lines: List[str], labels: Dict[str, List[str]], max_edit: int = 1) -> List[LabelHit]:
    hits: List[LabelHit] = []
    for li, line in enumerate(lines):
        # exact substring matches
        for field, labellist in labels.items():
            for lab in labellist:
                for m in re.finditer(re.escape(lab), line):
                    hits.append(LabelHit(field, lab, lab, 0, li, m.start()))
        # fuzzy matching by sliding window of up to 6 chars
        tokens = re.finditer(r"[\w\u4e00-\u9fff]{2,6}", line)
        for t in tokens:
            text = t.group(0)
            for field, labellist in labels.items():
                for lab in labellist:
                    d = levenshtein(text, lab)
                    if 0 < d <= max_edit:
                        hits.append(LabelHit(field, lab, text, d, li, t.start()))
    # de-duplicate: prefer exact matches over fuzzy at same (line,col)
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
    if not RE_ID_TW.fullmatch(code):
        return False
    n = LETTER_MAP.get(code[0])
    if n is None:
        return False
    a, b = divmod(n, 10)
    digits = [a, b] + [int(x) for x in code[1:]]
    return sum(d*w for d, w in zip(digits, WEIGHTS_TW_ID)) % 10 == 0

def arc_id_like(code: str) -> bool:
    return RE_ID_ARC.fullmatch(code) is not None

def parse_iso_date(txt: str) -> Optional[str]:
    txt = txt.strip()
    # yyyy-mm-dd, yyyy/mm/dd, yyyy.mm.dd
    m = re.match(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$", txt)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            return None
    # yyyy年mm月dd日
    m = re.match(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$", txt)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            return None
    # 民國YYY年mm月dd日 → +1911
    m = re.match(r"^民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$", txt)
    if m:
        roc, mo, d = map(int, m.groups())
        y = roc + 1911
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None

def is_cjk(s: str) -> bool:
    return RE_CJK.fullmatch(s) is not None

def load_surnames_from_txt(path: str) -> Tuple[Set[str], Set[str]]:
    """Load surnames from a comma-separated txt with no newline.
    Returns (single_surnames, double_surnames)
    - Any 1-char CJK entries → single
    - Any 2-char CJK entries → double
    - Ignores others
    """
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
                if len(p) == 1:
                    singles.add(p)
                elif len(p) == 2:
                    doubles.add(p)
    return singles, doubles

def iter_tokens_with_pos(text: str) -> Iterable[Tuple[str, int]]:
    for m in re.finditer(r"\S+", text):
        yield m.group(0), m.start()

def distance_score(label_col: int, cand_col: int, line_delta: int, tau: float = TAU_COL) -> float:
    line_w = LINE_WEIGHTS.get(abs(line_delta), 0.0)
    col_w = math.exp(-abs(cand_col - label_col)/tau)
    return line_w * col_w

def name_candidates_from_text(line_text: str, surname_singles: Set[str], surname_doubles: Set[str]) -> List[Tuple[str, int]]:
    """Return list of (name, col) candidates using the policy:
    - Find the nearest surname (double first, then single)
    - Take the next **two** CJK characters as the given name, skipping separators (space/·/．/•)
    This avoids missing names when they are glued to other tokens; we no longer require total length 2–4.
    """
    cands: List[Tuple[str,int]] = []
    text = line_text
    n = len(text)
    sep_set = set(NAME_SEPARATORS)

    def next_two_cjk_after(start: int) -> Tuple[Optional[str], Optional[int]]:
        j = start
        # skip separators after surname
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

    # scan across the line; prefer double surnames
    doubles_sorted = sorted(surname_doubles, key=len, reverse=True)
    for i in range(n):
        # try double surname first
        matched = False
        for ds in doubles_sorted:
            L = len(ds)
            if i + L <= n and text[i:i+L] == ds:
                given, col = next_two_cjk_after(i + L)
                if given:
                    if given in BIGRAM_BLACKLIST:
                        continue
                    cands.append((ds + given, i))
                matched = True
                break
        if matched:
            continue
        # try single surname
        ch = text[i]
        if ch in surname_singles:
            given, col = next_two_cjk_after(i + 1)
            if given:
                if given in BIGRAM_BLACKLIST:
                    continue
                cands.append((ch + given, i))
    return cands

def find_field_candidates_around_label(field: str, label: LabelHit, lines: List[str], surname_singles: Set[str], surname_doubles: Set[str]) -> List[Candidate]:
    """Generate candidates for a field around a label occurrence."""
    results: List[Candidate] = []
    label_line_text = lines[label.line]

    def add_candidate(value: str, vcol: int, line: int, dir_key: str, fmt_conf: float) -> None:
        line_delta = line - label.line
        col_delta = abs(vcol - label.col)
        dist = distance_score(label.col, vcol, line_delta)

        if field == "name":
            # Hard distance window: enforce proximity
            if line_delta == 0 and col_delta > 14:
                return
            if line_delta != 0 and col_delta > 10:
                return
            if abs(line_delta) > 1:
                return
            if dist < 0.5:
                return
        else:
            if dist < 0.2:
                return

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

    # Same line: right side
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

    # Same line: left side
    left_seg = label_line_text[max(0, label.col-60):label.col]
    if field == "name":
        for name, c in name_candidates_from_text(left_seg, surname_singles, surname_doubles):
            add_candidate(name, c, label.line, "same_left", fmt_conf=0.8)
    elif field == "id_no":
        for m in re.finditer(r"[A-Z][0-9]{9}|[A-Z]{2}[0-9]{8}", left_seg):
            code = m.group(0)
            fmt = 1.0 if tw_id_checksum_ok(code) or arc_id_like(code) else 0.5
            add_candidate(code, m.start(), label.line, "same_left", fmt)
    elif field == "ref_date":
        for pat in DATE_PATTERNS:
            for m in pat.finditer(left_seg):
                iso = parse_iso_date(m.group(0))
                if iso:
                    add_candidate(iso, m.start(), label.line, "same_left", 1.0)
    elif field == "batch_id":
        for m in RE_BATCH_13.finditer(left_seg):
            add_candidate(m.group(0), m.start(), label.line, "same_left", 0.9)

    # Below lines
    for dl in range(1, MAX_DOWN_LINES + 1):
        tgt_line_idx = label.line + dl
        if tgt_line_idx >= len(lines):
            break
        tgt = lines[tgt_line_idx]
        if field == "name":
            cands = name_candidates_from_text(tgt, surname_singles, surname_doubles)
            for name, c in cands:
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
# Record grouping / anchor logic
# ==============================
def group_records(all_cands: Dict[str, List[Candidate]]) -> List[Record]:
    """Greedy grouping using ID as anchor. If no IDs, fall back to name anchors.
    Heuristic: for each anchor, take nearest other-field candidate within line window.
    這裡同時會為「姓名」取鄰近 Top-5 候選，用於輸出。
    """
    records: List[Record] = []

    def nearest(field: str, anchor: Candidate) -> Optional[Candidate]:
        best = None
        best_s = -1.0
        for c in all_cands.get(field, []):
            # proximity score centered on anchor
            line_delta = abs(c.line - anchor.line)
            if line_delta > MAX_DOWN_LINES:  # strict window for grouping
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
            rec = assemble_record(name_c, a, date_c, batch_c, all_cands, name_topk=name_topk_list)
            records.append(rec)
    else:
        # fallback: use names as anchors (may create more ambiguous records)
        name_anchors = sorted(all_cands.get("name", []), key=lambda c: (c.line, c.col))
        for a in name_anchors:
            name_topk_list = nearest_k("name", a, k=5)  # will include itself
            id_c = nearest("id_no", a)
            date_c = nearest("ref_date", a)
            batch_c = nearest("batch_id", a)
            rec = assemble_record(a, id_c, date_c, batch_c, all_cands, name_topk=name_topk_list)
            records.append(rec)

    if not records:
        # No anchors at all: produce a single empty record with notes
        empty = Record(
            name=FieldResult(None, 0.0, None, ["未偵測到任何姓名候選或身分證號錨點"]),
            id_no=FieldResult(None, 0.0, None, ["未偵測到任何身分證號候選"]),
            ref_date=FieldResult(None, 0.0, None, ["未偵測到任何日期候選"]),
            batch_id=FieldResult(None, 0.0, None, ["未偵測到任何13位名單檔候選"]),
            debug={}
        )
        # 空紀錄也補一個 name_top5 空列表
        empty.debug["name_top5"] = []
        records.append(empty)

    return records

def assemble_record(name_c: Optional[Candidate],
                    id_c: Optional[Candidate],
                    date_c: Optional[Candidate],
                    batch_c: Optional[Candidate],
                    all_cands: Dict[str, List[Candidate]],
                    name_topk: Optional[List[Candidate]] = None) -> Record:
    def field_result_from_cand(c: Optional[Candidate], fallback_notes: List[str]) -> FieldResult:
        if c is None:
            return FieldResult(None, 0.0, None, fallback_notes)
        return FieldResult(
            value=c.value,
            confidence=max(0.0, min(1.0, c.score()/3.0)),  # rough normalization for [0,1]
            source={
                "line": c.line,
                "col": c.col,
                "label": c.source_label,
                "label_line": c.label_line,
                "label_col": c.label_col,
                "score_breakdown": {
                    "label_conf": c.label_conf,
                    "format_conf": c.format_conf,
                    "dist_score": c.dist_score,
                    "dir_prior": c.dir_prior,
                    "context_bonus": c.context_bonus,
                    "penalty": c.penalty,
                    "total": c.score(),
                }
            },
            notes=[],
        )

    def pack_topk(cands: List[Candidate]) -> List[Dict]:
        out = []
        for c in cands[:5]:
            out.append({
                "value": c.value,
                "approx_confidence": max(0.0, min(1.0, c.score()/3.0)),
                "line": c.line,
                "col": c.col,
                "label": c.source_label,
                "label_line": c.label_line,
                "label_col": c.label_col,
                "score_breakdown": {
                    "label_conf": c.label_conf,
                    "format_conf": c.format_conf,
                    "dist_score": c.dist_score,
                    "dir_prior": c.dir_prior,
                    "context_bonus": c.context_bonus,
                    "penalty": c.penalty,
                    "total": c.score(),
                }
            })
        return out

    name_notes = [] if name_c else [
        "找不到與標籤鄰近且符合規則的姓名候選；若身分證存在，已嘗試以ID為錨點向附近搜尋姓名。"
    ]
    id_notes = [] if id_c else [
        "找不到與標籤鄰近且符合格式/校驗的身分證號候選（支援本國與外來格式）。"
    ]
    date_notes = [] if date_c else [
        "找不到與標籤鄰近且可解析為ISO日期的候選（含民國年轉換）。"
    ]
    batch_notes = [] if batch_c else [
        "找不到與標籤鄰近的13位名單檔候選（已避免遠端13位數字誤判）。"
    ]

    rec = Record(
        name=field_result_from_cand(name_c, name_notes),
        id_no=field_result_from_cand(id_c, id_notes),
        ref_date=field_result_from_cand(date_c, date_notes),
        batch_id=field_result_from_cand(batch_c, batch_notes),
        debug={
            "all_candidates_counts": {k: len(v) for k, v in all_cands.items()},
        }
    )

    # 重要：把 Top-5 放到 debug 裡，稍後序列化時提升到 records 第一層
    rec.debug["name_top5"] = pack_topk(name_topk or [])
    return rec

# ==============================
# Main extraction pipeline
# ==============================
def extract_from_text(text: str, surname_txt_path: Optional[str] = None) -> Dict:
    lines = normalize_text(text)
    surname_singles, surname_doubles = load_surnames_from_txt(surname_txt_path) if surname_txt_path else (set(), set(DEFAULT_DOUBLE_SURNAMES))

    # 1) Locate labels (with fuzzy matching)
    label_hits = find_label_hits(lines, LABELS, max_edit=1)

    # Reporting init
    per_field_label_presence = {f: False for f in LABELS}
    for h in label_hits:
        per_field_label_presence[h.field] = True

    # 2) Generate field candidates around labels
    all_cands: Dict[str, List[Candidate]] = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
    for h in label_hits:
        cands = find_field_candidates_around_label(h.field, h, lines, surname_singles, surname_doubles)
        all_cands[h.field].extend(cands)

    # Anchor assist: If we have IDs but no name label hits, try name search near each ID
    if not per_field_label_presence["name"] and all_cands["id_no"]:
        for idc in all_cands["id_no"]:
            # scan same line and below lines for names
            for dl in range(0, MAX_DOWN_LINES + 1):
                li = idc.line + dl
                if li >= len(lines):
                    break
                for name, col in name_candidates_from_text(lines[li], surname_singles, surname_doubles):
                    dist = distance_score(idc.col, col, li - idc.line)
                    all_cands["name"].append(Candidate(
                        field="name", value=name, line=li, col=col,
                        label_line=idc.label_line, label_col=idc.label_col,
                        source_label=idc.source_label or "(ID-anchored)",
                        format_conf=0.7, label_conf=0.4, dir_prior=0.6, dist_score=dist,
                        context_bonus=0.2
                    ))

    # 3) Group into records（同時計算每筆 name Top-5）
    records = group_records(all_cands)

    # 4) Build global report about labels vs values
    report: Dict[str, List[str]] = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
    for field in ["name", "id_no", "ref_date", "batch_id"]:
        if not per_field_label_presence[field]:
            report[field].append("文件中未找到任何該欄位標籤（含模糊匹配）。")
        else:
            if not all_cands[field]:
                report[field].append("找到了標籤，但其附近未找到符合格式/校驗的候選值。")
            else:
                report[field].append(f"找到了標籤與候選（共 {len(all_cands[field])} 條），已根據距離與校驗打分。")

    # 5) Serialize
    output = {
        "records": [
            {
                "name": asdict(r.name),
                "id_no": asdict(r.id_no),
                "ref_date": asdict(r.ref_date),
                "batch_id": asdict(r.batch_id),
                "name_top5": r.debug.get("name_top5", []),  # 每筆紀錄的姓名前五候選
                "debug": r.debug,
            }
            for r in records
        ],
        "report": report,
        "meta": {
            "lines": len(lines),
            "label_hits": [asdict(h) for h in label_hits],
            "config": {
                "max_down_lines": MAX_DOWN_LINES,
                "line_weights": LINE_WEIGHTS,
                "tau_col": TAU_COL,
                "direction_prior": DIRECTION_PRIOR,
                "weights": {
                    "W_LABEL": W_LABEL,
                    "W_FORMAT": W_FORMAT,
                    "W_DIST": W_DIST,
                    "W_DIR": W_DIR,
                    "W_CONTEXT": W_CONTEXT,
                    "W_PENALTY": W_PENALTY,
                }
            }
        }
    }
    return output

def extract_from_file(txt_path: str, surname_txt_path: Optional[str]) -> Dict:
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return extract_from_text(text, surname_txt_path)

def main(argv: List[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Rule-based TXT extractor (multi-record) with per-record name Top-5")
    ap.add_argument("txt", help="Input .txt file path")
    ap.add_argument("--surnames", help="Path to comma-separated surnames txt (no newline)", default=None)
    ap.add_argument("--output", "-o", help="Output JSON path (default: stdout)", default=None)
    args = ap.parse_args(argv)

    result = extract_from_file(args.txt, args.surnames)
    js = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(js)
    else:
        print(js)

if __name__ == "__main__":
    main(sys.argv[1:])
