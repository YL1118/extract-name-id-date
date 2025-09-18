# extractor_min.py
# 極簡規則抽取器（同排/下方最近），Python 3.12+
import re, json, argparse, unicodedata

# --- 可調參（保持簡單） ---
MAX_LOOKDOWN = 3          # 往下最多看幾行
MAX_LOOKLEFT_CHARS = 40   # 同一行向左回看最多幾個字元
CJK_RANGE = r"\u4e00-\u9fff"

NAME_LABELS  = ["姓名", "調查人", "申報人"]
ID_LABELS    = ["身分證字號", "身分證統一編號", "身分證", "身分證統編"]
DATE_LABELS  = ["調查日", "申報基準日", "查詢基準日", "查調財產基準日"]
BATCH_LABELS = ["本次調查名單檔"]

ID_TW   = re.compile(r"[A-Z]\d{9}")      # 本國：含校驗碼檢查（下方函式）
ID_ARC  = re.compile(r"[A-Z]{2}\d{8}")   # 外來人口
BATCH13 = re.compile(r"\b\d{13}\b")

DATE_GREG = re.compile(r"\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})\b")
DATE_ZH   = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
DATE_ROC  = re.compile(r"民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")

def to_halfwidth(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def load_surnames(path: str) -> list[str]:
    # 你的百家姓 TXT：以逗號分隔、無換行，例如：王,李,張,陳,林,歐陽,司馬
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    parts = [p.strip() for p in content.split(",") if p.strip()]
    parts.sort(key=len, reverse=True)  # 複姓優先
    return parts

def compile_name_regex(surnames: list[str]):
    if not surnames:
        return None
    surn_alt = "|".join(map(re.escape, surnames))
    # <複/單姓> + 可選中點 + 給名 1-3 字
    return re.compile(rf"(?:{surn_alt})[·．•]?[{CJK_RANGE}]{{1,3}}")

def tw_id_checksum_ok(code: str) -> bool:
    if not re.fullmatch(r"[A-Z]\d{9}", code): 
        return False
    trans = {chr(ord('A')+i): 10+i for i in range(26)}  # A->10, B->11...
    n = trans[code[0]]
    a, b = divmod(n, 10)
    digits = [a, b] + [int(x) for x in code[1:]]
    weights = [1,9,8,7,6,5,4,3,2,1,1]
    return sum(d*w for d, w in zip(digits, weights)) % 10 == 0

def parse_date(text: str) -> str|None:
    m = DATE_GREG.search(text)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m = DATE_ZH.search(text)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m = DATE_ROC.search(text)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{(y+1911):04d}-{mo:02d}-{d:02d}"
    return None

def find_label_positions(lines: list[str], label_list: list[str]):
    hits = []
    for li, line in enumerate(lines):
        for lab in label_list:
            idx = line.find(lab)
            if idx != -1:
                hits.append((li, idx, lab))
    return hits

def search_same_line_right(line: str, start_col: int, pattern: re.Pattern):
    m = pattern.search(line, pos=start_col)
    return (m.group(0), m.start()) if m else None

def search_same_line_left(line: str, start_col: int, pattern: re.Pattern, max_left: int):
    left_bound = max(0, start_col - max_left)
    last = None
    for m in pattern.finditer(line, pos=left_bound, endpos=start_col):
        last = m
    return (last.group(0), last.start()) if last else None

def search_down_lines(lines: list[str], start_line: int, label_col: int, pattern: re.Pattern, max_down: int):
    best, best_dist, best_off = None, 10**9, None
    for off in range(1, max_down+1):
        li = start_line + off
        if li >= len(lines): break
        for m in pattern.finditer(lines[li]):
            dist = abs(m.start() - label_col)
            if dist < best_dist:
                best, best_dist, best_off = (m.group(0), m.start()), dist, off
    return (best, best_off, best_dist) if best else None

def explain_not_found(label_hits):
    return ("label_found_but_no_value", "有找到關鍵字，但未找到符合格式的值") if label_hits else ("label_not_found", "未找到任何關鍵字")

def extract_field(lines, label_list, right_pat, left_pat=None, down_pat=None, value_checker=None):
    label_hits = find_label_positions(lines, label_list)
    if not right_pat:  # 例如未載入百家姓時
        status, msg = explain_not_found(label_hits)
        if status == "label_found_but_no_value": msg += "（未載入百家姓，姓名規則停用）"
        return {"value": None, "status": status, "message": msg}
    if left_pat is None: left_pat = right_pat
    if down_pat is None: down_pat = right_pat

    for (li, col, lab) in label_hits:
        line = lines[li]
        # 1) 同行右側
        hit = search_same_line_right(line, col, right_pat)
        if hit and (value_checker is None or value_checker(hit[0])):
            return {"value": hit[0], "status": "ok", "message": f"從「{lab}」同一行右側取得（line {li+1}, col {hit[1]}）"}
        # 2) 同行左側
        hit = search_same_line_left(line, col, left_pat, MAX_LOOKLEFT_CHARS)
        if hit and (value_checker is None or value_checker(hit[0])):
            return {"value": hit[0], "status": "ok", "message": f"從「{lab}」同一行左側取得（line {li+1}, col {hit[1]}）"}
        # 3) 下方最近
        down = search_down_lines(lines, li, col, down_pat, MAX_LOOKDOWN)
        if down:
            (val, cpos), off, dist = down
            if (value_checker is None or value_checker(val)):
                return {"value": val, "status": "ok", "message": f"從「{lab}」下方第{off}行取得（line {li+1+off}, col {cpos}，Δcol={dist}）"}

    status, msg = explain_not_found(label_hits)
    return {"value": None, "status": status, "message": msg}

def main():
    ap = argparse.ArgumentParser(description="極簡規則抽取器（同排/下方最近）")
    ap.add_argument("--text", required=True, help="輸入 TXT 檔案路徑")
    ap.add_argument("--surnames", required=False, help="百家姓 TXT，逗號分隔、無換行")
    args = ap.parse_args()

    with open(args.text, "r", encoding="utf-8") as f:
        lines = to_halfwidth(f.read()).splitlines()

    # 姓名規則（可載入你的百家姓清單）
    surnames = []
    if args.surnames:
        try:
            surnames = load_surnames(args.surnames)
        except Exception:
            surnames = []
    name_regex = compile_name_regex(surnames)

    # 身分證規則與檢核
    id_union = re.compile(rf"(?:{ID_TW.pattern}|{ID_ARC.pattern})")
    def id_ok(s: str) -> bool:
        return (ID_TW.fullmatch(s) and tw_id_checksum_ok(s)) or bool(ID_ARC.fullmatch(s))

    # 日期規則與正規化
    date_union = re.compile(rf"(?:{DATE_GREG.pattern}|{DATE_ZH.pattern}|{DATE_ROC.pattern})")
    def date_ok(s: str) -> bool:
        return parse_date(s) is not None

    # 抽取四欄
    name_res = extract_field(lines, NAME_LABELS, right_pat=name_regex)
    id_res   = extract_field(lines, ID_LABELS, right_pat=id_union, value_checker=id_ok)
    date_res = extract_field(lines, DATE_LABELS, right_pat=date_union, value_checker=date_ok)
    if date_res["status"] == "ok":  # 轉 ISO
        date_res["value"] = parse_date(date_res["value"])  # 一定可 parse

    batch_res = extract_field(lines, BATCH_LABELS, right_pat=BATCH13, value_checker=lambda s: bool(BATCH13.fullmatch(s)))

    notes = []
    if not surnames:
        notes.append("未載入百家姓列表，姓名抽取停用（請用 --surnames 指定 TXT，格式：逗號分隔、無換行）")

    out = {"name": name_res, "id_no": id_res, "ref_date": date_res, "batch_id": batch_res, "notes": notes}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
