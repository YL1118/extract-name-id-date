
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import re
from dataclasses import dataclass

# Taiwan National ID checksum (1 letter + 9 digits). Reference mapping A=10, B=11, ... Z=35.
LETTER_CODE = {
    'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':34,'J':18,
    'K':19,'L':20,'M':21,'N':22,'O':35,'P':23,'Q':24,'R':25,'S':26,'T':27,
    'U':28,'V':29,'W':32,'X':30,'Y':31,'Z':33
}

ID_REGEX = re.compile(r'\b([A-Z])([0-9]{9})\b')

# Common OCR confusions for digits/letters
OCR_MAP_DIGITISH = {
    'O':'0','o':'0','D':'0','Q':'0',
    'I':'1','l':'1','L':'1','|':'1',
    'Z':'2',
    'S':'5',
    'B':'8','g':'9','q':'9'
}

@dataclass
class IdCheckResult:
    valid: bool
    corrected: bool
    corrected_value: Optional[str] = None
    reason: Optional[str] = None
    ocr_note: Optional[str] = None

def id_checksum_ok(s: str) -> IdCheckResult:
    """
    Validate Taiwan National ID. If checksum fails, attempt OCR corrections
    on the 9-digit tail. If corrected, return corrected value and note.
    """
    m = ID_REGEX.fullmatch(s)
    if not m:
        return IdCheckResult(False, False, reason="format_mismatch")
    letter, digits = m.group(1), m.group(2)
    if letter not in LETTER_CODE:
        return IdCheckResult(False, False, reason="invalid_letter")
    if _checksum(letter, digits):
        return IdCheckResult(True, False)
    # Try OCR fix on the 9-digit tail (positions that must be digits)
    fixed = list(digits)
    changed_positions = []
    for i, ch in enumerate(fixed):
        if ch.isdigit():
            continue
        rep = OCR_MAP_DIGITISH.get(ch)
        if rep:
            fixed[i] = rep
            changed_positions.append((i, ch, rep))
    # Additionally, even if all are digits, try swapping common misreads in digits
    for i, ch in enumerate(fixed):
        if ch in 'OIlLZSB':
            rep = OCR_MAP_DIGITISH.get(ch)
            if rep:
                fixed[i] = rep
    fixed_s = ''.join(fixed)
    if _checksum(letter, fixed_s):
        note = "OCR 修正 " + ','.join([f"{old}->{new}" for _, old, new in changed_positions]) if changed_positions else "OCR 修正"
        return IdCheckResult(True, True, corrected_value=letter+fixed_s, ocr_note=note)
    return IdCheckResult(False, False, reason="checksum_fail")

def _checksum(letter: str, digits: str) -> bool:
    code = LETTER_CODE[letter]
    x1 = code // 10
    x2 = code % 10
    nums = [int(d) for d in digits]
    weights = [1,9,8,7,6,5,4,3,2,1]
    total = x1*1 + x2*9 + sum(n*w for n,w in zip(nums, weights[2:] + [weights[-1]]))
    return total % 10 == 0

# Name candidates (Chinese 2-4 chars). Allow middle separator dot variants.
NAME_REGEX = re.compile(r'([\u4e00-\u9fff][\u4e00-\u9fff·・．．]{1,3})')

# Date patterns
# Accept forms: 民國114年9月18日, 114/09/18, 2025-09-18, 114.9.18, 2025年9月18日 等
DATE_TOKEN = r'(?P<y>\d{2,4})[年/\-\.](?P<m>\d{1,2})[月/\-\.](?P<d>\d{1,2})[日號]?'
ROC_ON_PREFIX = r'(?:民國|中華民國)?\s*' + DATE_TOKEN
DATE_REGEXES = [
    re.compile(ROC_ON_PREFIX),
    re.compile(DATE_TOKEN),
]

BATCH_ID_REGEX = re.compile(r'\b(\d{13})\b')

def iter_id_candidates(line: str):
    for m in ID_REGEX.finditer(line):
        yield m.start(), m.group(0)

def iter_name_candidates(line: str):
    for m in NAME_REGEX.finditer(line):
        yield m.start(), m.group(1)

def iter_date_candidates(line: str):
    for rx in DATE_REGEXES:
        for m in rx.finditer(line):
            yield m.start(), m.group(0)

def iter_batch_candidates(line: str):
    for m in BATCH_ID_REGEX.finditer(line):
        yield m.start(), m.group(1)
