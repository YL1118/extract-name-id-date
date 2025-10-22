#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based multi-record extractor for TXT documents (Taiwanese admin-style docs)

Features (2025-09-25)
- 每筆紀錄輸出「姓名 Top-5 候選」（records[i].name_top5）
- 動態雙姓：若偵測到「兩個姓氏字元相鄰」，視為雙姓，後取兩字為名 → 共四字
- 姓名評分納入「與身分證標籤的距離」：越近越加分（可用常數調整影響力）
- 新邏輯：即使「有姓名標籤」但附近抓不到姓名候選，也會用「身分證」作錨點補抓姓名
- 其他欄位與行為向下相容（Python 3.12，僅標準函式庫）
- 2025-10-22：整合 spaCy 中文 NER（只取 PERSON）優先作為姓名候選，失敗時回退規則法
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

# Direction priors
DIRECTION_PRIOR = {
    "same_right": 1.2,
    "same_left": 0.9,
    "below": 0.6,
}

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
ENABLE_DYNAMIC_DOUBLE_SURNAME = True   # 兩個單姓相鄰 → 視為雙姓
ENABLE_IDLABEL_PROXIMITY = True        # 姓名距離身分證標籤越近越加分
IDLABEL_BONUS_SCALE = 1.0              # 與 id 標籤距離分數的縮放（0~1 → 0~1*scale）

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

# Double surnames (expandable). You will load the full set from your txt file.
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
SPACY_ZH_MODEL = "zh_core_web_sm"  # 或 "zh_core_web_trf"（較準確但較慢）

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
    s = to_halfwidth(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")
    lines = [re.sub(r"[ \t　]+", " ", line) for line in lines]
    return lines

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
    m = re.match(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$", txt)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            return None
    m = re.match(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$", txt)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            return None
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

# ------------------------------
# spaCy PERSON 索引建立
# ------------------------------
def build_spacy_person_index(lines: List[str]) -> Tuple[Dict[int, List[Tuple[str, int]]], Dict[str, str]]:
    """
    回傳:
      - person_by_line: { line_idx: [(person_text, col), ...], ... }
      - spacy_meta: { "enabled": "true/false", "model": "...", "error": "..."/"" }
    """
    meta = {"enabled": "false", "model": "", "error": ""}
    person_by_line: Dict[int, List[Tuple[str, int]]] = {}

    if not USE_SPACY_PERSON:
        return person_by_line, meta

    try:
        import spacy
        nlp = spacy.load(SPACY_ZH_MODEL)
        meta["enabled"] = "true"
        meta["model"] = SPACY_ZH_MODEL
    except Exception as e:
        meta["error"] = f"spaCy load failed: {e}"
        return person_by_line, meta

    joined = "\n".join(lines)
    line_starts: List[int] = []
    cur = 0
    for i, line in enumerate(lines):
        line_starts.append(cur)
        cur += len(line) + (1 if i < len(lines) - 1 else 0)

    doc = nlp(joined)

    for ent in doc.ents:
        if getattr(ent, "label_", "") != "PERSON":
            continue
        g0 = ent.start_char
        line_idx = 0
        for i in range(len(line_starts)-1, -1, -1):
            if g0 >= line_starts[i]:
                line_idx = i
                break
        col = g0 - line_starts[line_idx]
        if 0 <= line_idx < len(lines) and 0 <= col <= len(lines[line_idx]):
            person_by_line.setdefault(line_idx, []).append((ent.text, col))

    for li in person_by_line:
        person_by_line[li].sort(key=lambda x: x[1])

    return person_by_line, meta

# ==============================
# Name candidates (rule-based)
# ==============================
def name_candidates_from_text(line_text: str, surname_singles: Set[str], surname_doubles: Set[str]) -> List[Tuple[str, int]]:
    """
    回傳 (name, col) 候選：
    1) 先嘗試「已知雙姓」；成功 → 後取兩字為名（共四字）
    2) 若 ENABLE_DYNAMIC_DOUBLE_SURNAME 且連續兩字皆在單姓表 → 視為雙姓；後取兩字為名（共四字）
    3) 否則單姓；後取兩字為名（共三字）
    """
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
        # 1) 已知雙姓
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

        # 2) 動態雙姓偵測
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

        # 3) 單姓
        if ch in surname_singles:
            given, col = next_two_cjk_after(i + 1)
            if given and given not in BIGRAM_BLACKLIST:
                cands.append((ch + given, i))

        i += 1

    return cands

# ==============================
# Candidate search around labels
# ==============================
def find_field_candidates_around_label(
    field: str,
    label: LabelHit,
    lines: List[str],
    surname_singles: Set[str],
    surname_doubles: Set[str],
    person_index: Optional[Dict[int, List[Tuple[str, int]]]] = None,
) -> List[Candidate]:
    results: List[Candidate] = []
    label_line_text = lines[label.line]

    def add_candidate(value: str, vcol: int, line: int, dir_key: str, fmt_conf: float) -> None:
        line_delta = line - label.line
        col_delta = abs(vcol - label.col)
        dist = distance_score(label.col, vcol, line_delta)

        if field == "name":
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

    # same line: right
    right_seg = label_line_text[label.col:label.col+60]
    if field == "name":
        if person_index is not None and label.line in person_index:
            for txt, c in person_index[label.line]:
                if label.col <= c < label.col + 60:
                    add_candidate(txt, c, label.line, "same_right", fmt_conf=0.9)
        else:
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
        if person_index is not None and label.line in person_index:
            for txt, c in person_index[label.line]:
                if (label.col - 60) <= c < label.col:
                    add_candidate(txt, c, label.line, "same_left", fmt_conf=0.9)
        else:
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

    # below lines
    for dl in range(1, MAX_DOWN_LINES + 1):
        tgt_line_idx = label.line + dl
        if tgt_line_idx >= len(lines):
            break
        tgt = lines[tgt_line_idx]
        if field == "name":
            if person_index is not None and tgt_line_idx in person_index:
                for txt, c in person_index[tgt_line_idx]:
                    add_candidate(txt, c, tgt_line_idx, "below", 0.9)
            else:
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
# Record grouping / anchor logic
# ==============================
def group_records(all_cands: Dict[str, List[Candidate]]) -> List[Record]:
    """Greedy grouping using ID as anchor. If no IDs, fall back to name anchors.
    這裡同時會為「姓名」取鄰近 Top-5 候選，用於輸出。
    """
    records: List[Record] = []

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
            rec = assemble_record(name_c, a, date_c, batch_c, all_cands, name_topk=name_topk_list)
            records.append(rec)
    else:
        name_anchors = sorted(all_cands.get("name", []), key=lambda c: (c.line, c.col))
        for a in name_anchors:
            name_topk_list = nearest_k("name", a, k=5)
            id_c = nearest("id_no", a)
            date_c = nearest("ref_date", a)
            batch_c = nearest("batch_id", a)
            rec = assemble_record(a, id_c, date_c, batch_c, all_cands, name_topk=name_topk_list)
            records.append(rec)

    if not records:
        empty = Record(
            name=FieldResult(None, 0.0, None, ["未偵測到任何姓名候選或身分證號錨點"]),
            id_no=FieldResult(None, 0.0, None, ["未偵測到任何身分證號候選"]),
            ref_date=FieldResult(None, 0.0, None, ["未偵測到任何日期候選"]),
            batch_id=FieldResult(None, 0.0, None, ["未偵測到任何13位名單檔候選"]),
            debug={}
        )
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
            confidence=max(0.0, min(1.0, c.score()/3.0)),
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
        debug={"all_candidates_counts": {k: len(v) for k, v in all_cands.items()}}
    )
    rec.debug["name_top5"] = pack_topk(name_topk or [])
    return rec

# ==============================
# Main extraction pipeline
# ==============================
def extract_from_text(text: str, surname_txt_path: Optional[str] = None) -> Dict:
    lines = normalize_text(text)
    surname_singles, surname_doubles = load_surnames_from_txt(surname_txt_path) if surname_txt_path else (set(), set(DEFAULT_DOUBLE_SURNAMES))

    # 0.5) 建立 spaCy PERSON 索引（若啟用）
    spacy_person_index: Dict[int, List[Tuple[str, int]]] = {}
    spacy_meta = {"enabled": "false", "model": "", "error": ""}
    try:
        spacy_person_index, spacy_meta = build_spacy_person_index(lines)
    except Exception as e:
        spacy_meta = {"enabled": "false", "model": "", "error": f"build failed: {e}"}

    # 1) 找標籤
    label_hits = find_label_hits(lines, LABELS, max_edit=1)

    per_field_label_presence = {f: False for f in LABELS}
    for h in label_hits:
        per_field_label_presence[h.field] = True

    # 2) 依標籤產生候選（姓名：優先 PERSON，無則回退規則）
    all_cands: Dict[str, List[Candidate]] = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
    for h in label_hits:
        cands = find_field_candidates_around_label(
            h.field, h, lines, surname_singles, surname_doubles, person_index=spacy_person_index
        )
        all_cands[h.field].extend(cands)

    # 2.5) 新邏輯：若「沒有姓名標籤」或「有姓名標籤但抓不到姓名候選」，且有身分證候選 → 用 ID 當錨點補抓姓名
    if (not per_field_label_presence["name"] or not all_cands["name"]) and all_cands["id_no"]:
        for idc in all_cands["id_no"]:
            for dl in range(0, MAX_DOWN_LINES + 1):
                li = idc.line + dl
                if li >= len(lines):
                    break
                # 先用 PERSON
                used_any = False
                if spacy_person_index.get(li):
                    for txt, col in spacy_person_index[li]:
                        dist = distance_score(idc.col, col, li - idc.line)
                        all_cands["name"].append(Candidate(
                            field="name", value=txt, line=li, col=col,
                            label_line=idc.label_line, label_col=idc.label_col,
                            source_label=idc.source_label or "(ID-anchored)",
                            format_conf=0.85, label_conf=0.5, dir_prior=0.6, dist_score=dist,
                            context_bonus=0.25
                        ))
                        used_any = True
                # 沒有 PERSON 再退回規則法
                if not used_any:
                    for name, col in name_candidates_from_text(lines[li], surname_singles, surname_doubles):
                        dist = distance_score(idc.col, col, li - idc.line)
                        all_cands["name"].append(Candidate(
                            field="name", value=name, line=li, col=col,
                            label_line=idc.label_line, label_col=idc.label_col,
                            source_label=idc.source_label or "(ID-anchored)",
                            format_conf=0.7, label_conf=0.4, dir_prior=0.6, dist_score=dist,
                            context_bonus=0.2
                        ))

    # 2.7) 姓名候選加分（距離「身分證標籤」越近越加分）
    if ENABLE_IDLABEL_PROXIMITY:
        id_label_positions: List[Tuple[int,int]] = [(h.line, h.col) for h in label_hits if h.field == "id_no"]
        if id_label_positions:
            for c in all_cands.get("name", []):
                best = 0.0
                for li, lc in id_label_positions:
                    line_delta = c.line - li
                    dscore = distance_score(lc, c.col, line_delta)  # 0~1
                    if dscore > best:
                        best = dscore
                c.context_bonus += IDLABEL_BONUS_SCALE * best  # 進入總分：W_CONTEXT * context_bonus

    # 3) 分組成紀錄（同時計算 name Top-5）
    records = group_records(all_cands)

    # 4) 報告
    report: Dict[str, List[str]] = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
    for field in ["name", "id_no", "ref_date", "batch_id"]:
        if not per_field_label_presence[field]:
            report[field].append("文件中未找到任何該欄位標籤（含模糊匹配）。")
        else:
            if not all_cands[field]:
                report[field].append("找到了標籤，但其附近未找到符合格式/校驗的候選值。")
            else:
                report[field].append(f"找到了標籤與候選（共 {len(all_cands[field])} 條），已根據距離與校驗打分。")

    # 5) 輸出
    output = {
        "records": [
            {
                "name": asdict(r.name),
                "id_no": asdict(r.id_no),
                "ref_date": asdict(r.ref_date),
                "batch_id": asdict(r.batch_id),
                "name_top5": r.debug.get("name_top5", []),
                "debug": r.debug,
            }
            for r in records
        ],
        "report": report,
        "meta": {
            "lines": len(lines),
            "label_hits": [asdict(h) for h in label_hits],
            "spacy_person": spacy_meta,
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
                },
                "enable_dynamic_double_surname": ENABLE_DYNAMIC_DOUBLE_SURNAME,
                "enable_idlabel_proximity": ENABLE_IDLABEL_PROXIMITY,
                "idlabel_bonus_scale": IDLABEL_BONUS_SCALE,
                "use_spacy_person": USE_SPACY_PERSON,
                "spacy_zh_model": SPACY_ZH_MODEL,
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
    ap = argparse.ArgumentParser(description="Rule-based TXT extractor (name Top-5, dynamic double-surname, ID-label proximity, spaCy PERSON priority, and ID-anchored fallback when name label fails)")
    ap.add_argument("txt", help="Input .txt file path")
    ap.add_argument("--surnames", help="Path to comma-separated surnames txt (no newline)", default=None)
    ap.add_argument("--output", "-o", help="Output JSON path (default: stdout)", default=None)
    ap.add_argument("--no-spacy", action="store_true", help="Disable spaCy PERSON and use pure rule-based name extraction")
    ap.add_argument("--spacy-model", default=None, help="Override spaCy zh model name (e.g., zh_core_web_sm / zh_core_web_trf)")
    args = ap.parse_args(argv)

    # 動態覆寫設定（不破壞全域預設）
    global USE_SPACY_PERSON, SPACY_ZH_MODEL
    if args.no_spacy:
        USE_SPACY_PERSON = False
    if args.spacy_model:
        SPACY_ZH_MODEL = args.spacy_model

    result = extract_from_file(args.txt, args.surnames)
    js = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(js)
    else:
        print(js)

if __name__ == "__main__":
    main(sys.argv[1:])