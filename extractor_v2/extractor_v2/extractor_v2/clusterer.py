
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import itertools
import math

from .labels import find_label_hits, LABEL_SYNONYMS
from .patterns import (
    iter_id_candidates, iter_name_candidates, iter_date_candidates, iter_batch_candidates,
    id_checksum_ok
)
from .normalizer import normalize_text, parse_date_to_iso
from .scorer import distance_score, direction_prior, final_score

@dataclass
class Node:
    id: int
    type: str           # 'label' | 'candidate'
    field: str          # 'name' | 'id_no' | 'ref_date' | 'batch_id'
    line: int
    col: int
    raw: str
    norm: str
    label_confidence: float = 0.0
    format_confidence: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecordField:
    status: str                 # FOUND | KEYWORD_FOUND_NO_VALUE | KEYWORD_NOT_FOUND
    value: Optional[str] = None
    confidence: Optional[float] = None
    label: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    explanation: Optional[str] = None

@dataclass
class Record:
    record_id: str
    fields: Dict[str, RecordField]
    cluster_debug: Dict[str, Any] = field(default_factory=dict)
    merged_from: List[str] = field(default_factory=list)

class Extractor:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.logs: List[str] = []

    # ----------------------------- main -----------------------------
    def extract(self, text: str) -> Dict[str, Any]:
        self.logs.append("start: normalize_text")
        norm = normalize_text(text)
        lines = norm.splitlines()
        # keep per-line char positions (we'll use col = index within line)
        # 1) label and candidate nodes
        ratio = self.cfg["fuzzy"]["label_ratio_threshold"]
        nodes: List[Node] = []
        nid = 1
        for i, line in enumerate(lines, start=1):
            # labels
            for h in find_label_hits(line, i, ratio):
                nodes.append(Node(nid, "label", h["field"], i, h["col"], h["raw"], h["norm"],
                                  label_confidence=h["label_confidence"]))
                self.logs.append(f"label_hit:#{nid} field={h['field']} line={i} col={h['col']} raw={h['raw']} conf={h['label_confidence']:.2f}")
                nid += 1
            # candidates: id_no
            for col, val in iter_id_candidates(line):
                check = id_checksum_ok(val)
                fmt = 1.0 if check.valid else 0.4
                normv = check.corrected_value or val
                meta = {}
                if check.valid and check.corrected:
                    meta["ocr_note"] = check.ocr_note or "OCR 修正"
                    self.logs.append(f"ocr_fix:id_no line={i} col={col} note={meta['ocr_note']}")
                elif not check.valid and check.reason:
                    meta["invalid_reason"] = check.reason
                nodes.append(Node(nid, "candidate", "id_no", i, col, val, normv, format_confidence=fmt, meta=meta))
                self.logs.append(f"cand:id_no:#{nid} line={i} col={col} raw={val} fmt={fmt}")
                nid += 1
            # candidates: name (Chinese 2-4)
            for col, val in iter_name_candidates(line):
                # heuristic: names extracted anywhere; lower base format conf unless near a name-label later
                fmt = 0.6
                nodes.append(Node(nid, "candidate", "name", i, col, val, val, format_confidence=fmt))
                self.logs.append(f"cand:name:#{nid} line={i} col={col} raw={val} fmt={fmt}")
                nid += 1
            # candidates: date
            for col, val in iter_date_candidates(line):
                iso, reason = parse_date_to_iso(val)
                fmt = 1.0 if iso else 0.4
                meta = {"iso": iso, "normalize_reason": reason}
                nodes.append(Node(nid, "candidate", "ref_date", i, col, val, iso or val, format_confidence=fmt, meta=meta))
                self.logs.append(f"cand:ref_date:#{nid} line={i} col={col} raw={val} iso={iso}")
                nid += 1
            # candidates: batch_id
            for col, val in iter_batch_candidates(line):
                nodes.append(Node(nid, "candidate", "batch_id", i, col, val, val, format_confidence=1.0))
                self.logs.append(f"cand:batch_id:#{nid} line={i} col={col} raw={val} fmt=1.0")
                nid += 1

        # mark batch_id with nearby label presence (must be near to count)
        nodes = self._annotate_batch_near_label(nodes)

        # document-level presence
        doc_label_presence = {k: "FOUND" if any(n for n in nodes if n.type=="label" and n.field==k) else "NOT_FOUND"
                              for k in ("name","id_no","ref_date","batch_id")}

        # 2) anchors selection
        anchors = self._select_anchors(nodes)

        # 3) window clustering per anchor
        groups = []
        for a in anchors:
            g = self._collect_group_window(a, nodes)
            if g:
                groups.append((a, g))

        # optional: merge overlapping groups (if anchors very close)
        if self.cfg["grouping"].get("cluster_merge_if_overlap"):
            groups = self._merge_overlapping_groups(groups)

        # 4) within each group: pair name<->id_no into subgroups (records)
        subgroups = []
        for anchor, gnodes in groups:
            subs = self._pair_within_group(anchor, gnodes)
            subgroups.extend(subs)

        # 5) assign semi-global fields (ref_date, batch_id) to records
        records: List[Record] = self._build_records(subgroups, nodes, lines, doc_label_presence)

        # 6) dedupe by id_no
        records = self._dedupe(records)

        # 7) collect orphans (unassigned candidates that are relevant)
        orphans = self._collect_orphans(nodes, records)

        # build candidates_debug
        candidates_debug = self._collect_candidates_debug(nodes)

        # document level global inference
        doc_global = self._infer_document_scopes(nodes, lines)

        # assemble JSON
        out = {
            "records": [self._record_to_json(r) for r in records],
            "orphans_unassigned": orphans,
            "document_level": {
                "label_presence": doc_label_presence,
                "global_inference": doc_global
            },
            "candidates_debug": candidates_debug,
            "logs": self.logs,
            "config_used": self.cfg,
            "version": self.cfg.get("version", "0.2.0")
        }
        return out

    # -------------------------- helpers --------------------------
    def _annotate_batch_near_label(self, nodes: List[Node]) -> List[Node]:
        # For each batch_id candidate, check if there is a batch label within +-1 lines and col delta <= 30
        lines_up = self.cfg["grouping"]["window_lines_up"]
        lines_down = self.cfg["grouping"]["window_lines_down"]
        max_col = self.cfg["grouping"]["max_col_delta"]
        labels = [n for n in nodes if n.type=="label" and n.field=="batch_id"]
        for n in nodes:
            if n.type=="candidate" and n.field=="batch_id":
                ok = False
                for lb in labels:
                    if abs(lb.line - n.line) <= max(lines_up, lines_down) and abs(lb.col - n.col) <= max_col:
                        ok = True
                        break
                n.meta["near_label"] = ok
        return nodes

    def _select_anchors(self, nodes: List[Node]) -> List[Node]:
        anchors = []
        fields = self.cfg["grouping"]["anchor_order"]
        # Candidates only
        cands = [n for n in nodes if n.type=="candidate" and n.field in fields]
        # Sort by field priority then by (line,col)
        priority = {f:i for i,f in enumerate(fields)}
        cands.sort(key=lambda n: (priority[n.field], n.line, n.col))
        # Simple dedupe: avoid placing two anchors that are the same candidate
        seen = set()
        for n in cands:
            key = (n.field, n.line, n.col, n.norm)
            if key in seen: 
                continue
            anchors.append(n)
            seen.add(key)
            self.logs.append(f"anchor: field={n.field} line={n.line} col={n.col} norm={n.norm}")
        return anchors

    def _collect_group_window(self, anchor: Node, nodes: List[Node]) -> List[Node]:
        cfg = self.cfg["grouping"]
        U, D = cfg["window_lines_up"], cfg["window_lines_down"]
        max_col = cfg["max_col_delta"]
        max_span = cfg["max_span_lines_for_linking"]
        group = []
        for n in nodes:
            if n is anchor: 
                group.append(n)
                continue
            dl = n.line - anchor.line
            dc = n.col - anchor.col
            if -U <= dl <= D and abs(dc) <= max_col and abs(dl) <= max_span:
                group.append(n)
        self.logs.append(f"group_window: anchor@{anchor.line},{anchor.col} size={len(group)}")
        return group

    def _merge_overlapping_groups(self, groups):
        # Merge groups whose anchors are within the same window (very close)
        merged = []
        consumed = set()
        for i, (a1, g1) in enumerate(groups):
            if i in consumed: 
                continue
            merged_anchor = a1
            merged_nodes = set(g1)
            for j, (a2, g2) in enumerate(groups):
                if j <= i or j in consumed: 
                    continue
                if abs(a1.line - a2.line) <= 1 and abs(a1.col - a2.col) <= self.cfg["grouping"]["max_col_delta"]:
                    merged_nodes |= set(g2)
                    consumed.add(j)
            merged.append((merged_anchor, list(merged_nodes)))
        return merged

    def _pair_within_group(self, anchor: Node, gnodes: List[Node]) -> List[Dict[str, Any]]:
        # Build lists
        names = [n for n in gnodes if n.type=="candidate" and n.field=="name"]
        ids = [n for n in gnodes if n.type=="candidate" and n.field=="id_no"]
        # If anchor is id_no or name, ensure it is included
        # Pairing strategy
        strategy = self.cfg["grouping"]["pairing_strategy"]
        pairs = []
        if names and ids:
            if strategy == "hungarian":
                pairs = self._optimal_pair(names, ids, anchor)
            else:
                pairs = self._greedy_pair(names, ids, anchor)
        elif ids and not names:
            pairs = [({"name": None, "id_no": i},) for i in ids]
        elif names and not ids:
            pairs = [({"name": n, "id_no": None},) for n in names]
        else:
            pairs = [({"name": None, "id_no": None},)]

        subgroups = []
        for p in pairs:
            if isinstance(p, tuple):
                # single element
                p = p[0]
            subgroups.append({
                "anchor": anchor,
                "name_nodes": [p["name"]] if p.get("name") else [],
                "id_nodes": [p["id_no"]] if p.get("id_no") else [],
                "group_nodes": gnodes
            })
        return subgroups

    def _pair_cost(self, n, i, anchor) -> float:
        # Lower cost is better. Use 1 - normalized final score
        dl = n.line - i.line
        dc = n.col - i.col
        line_weights = {0:1.0,1:0.7,2:0.5,3:0.3}
        dist = distance_score(dl, dc, line_weights)
        dirp = direction_prior(dl, dc, self.cfg["direction_prior"])
        # label/context not available at this point; rely on format + distance + direction
        fmt = (n.format_confidence + i.format_confidence)/2.0
        score = 0.35*fmt + 0.45*dist + 0.20*dirp
        return 1.0 - score

    def _optimal_pair(self, names, ids, anchor):
        # brute-force search (safe for small n,m<=6). Returns list of dicts {"name":n,"id_no":i}
        import itertools, math
        res = []
        if not names or not ids:
            return res
        if len(names) <= 6 and len(ids) <= 6:
            # match min(len(names), len(ids)) pairs
            # we try all permutations of the smaller set mapped into larger
            if len(names) <= len(ids):
                best = None
                for perm in itertools.permutations(ids, len(names)):
                    cost = sum(self._pair_cost(n, i, anchor) for n,i in zip(names, perm))
                    if best is None or cost < best[0]:
                        best = (cost, list(zip(names, perm)))
                chosen = best[1] if best else []
                used_ids = set(i.id for _,i in chosen)
                leftover_ids = [i for i in ids if i.id not in used_ids]
                res = [{"name": n, "id_no": i} for n,i in chosen] + [{"name": None, "id_no": i} for i in leftover_ids]
            else:
                best = None
                for perm in itertools.permutations(names, len(ids)):
                    cost = sum(self._pair_cost(n, i, anchor) for n,i in zip(perm, ids))
                    if best is None or cost < best[0]:
                        best = (cost, list(zip(perm, ids)))
                chosen = best[1] if best else []
                used_names = set(n.id for n,_ in chosen)
                leftover_names = [n for n in names if n.id not in used_names]
                res = [{"name": n, "id_no": i} for n,i in chosen] + [{"name": n, "id_no": None} for n in leftover_names]
        else:
            res = self._greedy_pair(names, ids, anchor)
        return res

    def _greedy_pair(self, names, ids, anchor):
        pairs = []
        used_n = set()
        used_i = set()
        # Score all possible edges by negative cost
        edges = []
        for n in names:
            for i in ids:
                cost = self._pair_cost(n, i, anchor)
                edges.append((1.0 - cost, n, i))
        edges.sort(key=lambda x: -x[0])
        for score, n, i in edges:
            if n.id in used_n or i.id in used_i:
                continue
            pairs.append({"name": n, "id_no": i})
            used_n.add(n.id); used_i.add(i.id)
        # leftovers
        for n in names:
            if n.id not in used_n:
                pairs.append({"name": n, "id_no": None})
        for i in ids:
            if i.id not in used_i:
                pairs.append({"name": None, "id_no": i})
        return pairs

    def _build_records(self, subgroups, nodes, lines, doc_label_presence) -> List[Record]:
        records: List[Record] = []
        rec_idx = 1

        # Decide scopes for ref_date/batch_id
        global_infer = self._infer_document_scopes(nodes, lines)
        ref_scope = global_infer["ref_date"]["scope"]
        batch_scope = global_infer["batch_id"]["scope"]

        for sg in subgroups:
            rid = f"r{rec_idx:03d}"
            rec_idx += 1
            fields = {
                "name": RecordField(status="KEYWORD_NOT_FOUND"),
                "id_no": RecordField(status="KEYWORD_NOT_FOUND"),
                "ref_date": RecordField(status="KEYWORD_NOT_FOUND"),
                "batch_id": RecordField(status="KEYWORD_NOT_FOUND")
            }
            # Select best name/id in subgroup
            best_name = self._select_best_field(sg, "name")
            best_id = self._select_best_field(sg, "id_no")
            if best_name:
                fields["name"] = best_name
            else:
                # check if name label present in group scope
                if any(n for n in sg["group_nodes"] if n.type=="label" and n.field=="name"):
                    fields["name"] = RecordField(status="KEYWORD_FOUND_NO_VALUE",
                                                 explanation="命中姓名標籤，但候選皆格式不符或距離過遠")
            if best_id:
                fields["id_no"] = best_id
            else:
                if any(n for n in sg["group_nodes"] if n.type=="label" and n.field=="id_no"):
                    fields["id_no"] = RecordField(status="KEYWORD_FOUND_NO_VALUE",
                                                  explanation="命中身分證標籤，但候選皆格式/校驗失敗或距離過遠")

            # ref_date assign
            fields["ref_date"] = self._assign_semi_global("ref_date", sg, nodes, ref_scope)
            # batch_id assign (only near_label candidates can be used)
            fields["batch_id"] = self._assign_semi_global("batch_id", sg, nodes, batch_scope)

            # cluster_debug: choose anchor, and list candidate members near anchor
            anchor = sg["anchor"]
            cluster_dbg = {
                "anchor": {"type": anchor.field, "value": anchor.norm, "line": anchor.line, "col": anchor.col},
                "members": [
                    {"type": n.type, "field": n.field, "value": n.norm, "line": n.line, "col": n.col,
                     "fmt": n.format_confidence, "label_conf": n.label_confidence}
                    for n in sg["group_nodes"]
                ]
            }
            records.append(Record(record_id=rid, fields=fields, cluster_debug=cluster_dbg))

        return records

    def _select_best_field(self, sg, field) -> Optional[RecordField]:
        # Among sg["name_nodes"] / ["id_nodes"], pick the best based on final score against anchor
        nodes = sg["name_nodes"] if field=="name" else sg["id_nodes"]
        if not nodes:
            return None
        anchor = sg["anchor"]
        weights = self.cfg["weights"]
        line_weights = {0:1.0,1:0.7,2:0.5,3:0.3}
        candidates = []
        for n in nodes:
            dl = n.line - anchor.line
            dc = n.col - anchor.col
            dist = distance_score(dl, dc, line_weights)
            dirp = direction_prior(dl, dc, self.cfg["direction_prior"])
            # find nearest label of this field within group
            labels = [lb for lb in sg["group_nodes"] if lb.type=="label" and lb.field==field]
            if labels:
                # pick closest label for label_conf
                label_conf = max(lb.label_confidence for lb in labels)
                context_bonus = 0.1
                label_name = labels[0].norm
            else:
                label_conf = 0.3  # weak if no explicit label
                context_bonus = 0.0
                label_name = None
            collision_penalty = 0.0  # MVP no collision model
            fmt = n.format_confidence
            score = final_score(label_conf, fmt, dist, dirp, context_bonus, collision_penalty, weights)
            candidates.append((score, n, label_name, dist, dirp))
        candidates.sort(key=lambda x: -x[0])
        best_score, best_node, best_label, _, _ = candidates[0]
        # top-2 saved in candidates_debug will be done elsewhere
        explanation = f"與 anchor 距離/方向最佳，score={best_score:.2f}。"
        if best_node.field == "id_no" and best_node.meta.get("ocr_note"):
            explanation += f" {best_node.meta.get('ocr_note')}。"
        return RecordField(
            status="FOUND",
            value=best_node.norm if field!="ref_date" else best_node.meta.get("iso") or best_node.norm,
            confidence=round(float(best_score), 2),
            label=best_label,
            line=best_node.line,
            col=best_node.col,
            explanation=explanation
        )

    def _assign_semi_global(self, field: str, sg, nodes, scope: str) -> RecordField:
        if scope == "document":
            # find best candidate overall
            cand = self._nearest_for_field(field, sg["anchor"], nodes, require_batch_near_label=(field=="batch_id"))
            if cand:
                expl = f"由文件全域套用（唯一且於標頭），{self._dist_expl(cand, sg['anchor'])}"
                value = cand.meta.get("iso") if field=="ref_date" else cand.norm
                return RecordField(status="FOUND", value=value, confidence=0.9,
                                   label=self._nearest_label_name(field, cand, nodes), line=cand.line, col=cand.col,
                                   explanation=expl)
            else:
                return RecordField(status="KEYWORD_NOT_FOUND", explanation="未在文件中找到該欄位")
        # nearest
        cand = self._nearest_for_field(field, sg["anchor"], nodes, require_batch_near_label=(field=="batch_id"))
        if cand:
            value = cand.meta.get("iso") if field=="ref_date" else cand.norm
            expl = f"就近分配；{self._dist_expl(cand, sg['anchor'])}"
            return RecordField(status="FOUND", value=value, confidence=0.8, 
                               label=self._nearest_label_name(field, cand, nodes), line=cand.line, col=cand.col,
                               explanation=expl)
        # No label present near group?
        if any(n for n in sg["group_nodes"] if n.type=="label" and n.field==field):
            return RecordField(status="KEYWORD_FOUND_NO_VALUE", explanation="命中標籤，但無有效值或距離過遠")
        return RecordField(status="KEYWORD_NOT_FOUND")

    def _nearest_for_field(self, field: str, anchor: Node, nodes: List[Node], require_batch_near_label=False) -> Optional[Node]:
        # pick candidate of given field with max distance-based score to anchor
        cands = [n for n in nodes if n.type=="candidate" and n.field==field]
        if field=="batch_id" and require_batch_near_label:
            cands = [n for n in cands if n.meta.get("near_label")]
        if not cands:
            return None
        best = None
        line_weights = {0:1.0,1:0.7,2:0.5,3:0.3}
        for n in cands:
            dl = n.line - anchor.line
            dc = n.col - anchor.col
            if abs(dl) > self.cfg["grouping"]["max_span_lines_for_linking"]:
                continue
            if abs(dc) > self.cfg["grouping"]["max_col_delta"]:
                continue
            dist = distance_score(dl, dc, line_weights)
            if best is None or dist > best[0]:
                best = (dist, n)
        return best[1] if best else None

    def _nearest_label_name(self, field: str, n: Node, nodes: List[Node]) -> Optional[str]:
        labels = [lb for lb in nodes if lb.type=="label" and lb.field==field]
        if not labels:
            return None
        # choose closest by Manhattan distance
        best = None
        for lb in labels:
            d = abs(lb.line - n.line) + abs(lb.col - n.col)
            if best is None or d < best[0]:
                best = (d, lb.norm)
        return best[1] if best else None

    def _dist_expl(self, n: Node, anchor: Node) -> str:
        dl = n.line - anchor.line
        dc = n.col - anchor.col
        return f"相對 anchor Δline={dl}, Δcol={dc}"

    def _dedupe(self, records: List[Record]) -> List[Record]:
        by_id = {}
        result: List[Record] = []
        for r in records:
            idf = r.fields.get("id_no")
            key = idf.value if (idf and idf.status=="FOUND" and idf.value) else None
            if not key:
                result.append(r)
                continue
            if key not in by_id:
                by_id[key] = r
                result.append(r)
            else:
                # merge: prefer higher-confidence fields
                existing = by_id[key]
                for fld in ("name","ref_date","batch_id"):
                    a = existing.fields[fld]
                    b = r.fields[fld]
                    existing_score = a.confidence or 0.0
                    new_score = b.confidence or 0.0
                    if (b.status=="FOUND" and (a.status!="FOUND" or new_score > existing_score)):
                        existing.fields[fld] = b
                existing.merged_from.append(r.record_id)
                self.logs.append(f"dedupe: merged {r.record_id} into {existing.record_id} by id_no={key}")
        return result

    def _collect_orphans(self, nodes: List[Node], records: List[Record]):
        assigned = set()
        for r in records:
            for fld in ("name","id_no","ref_date","batch_id"):
                rf = r.fields[fld]
                if rf.status=="FOUND" and rf.line is not None and rf.col is not None:
                    assigned.add((fld, rf.line, rf.col))
        orphans = []
        for n in nodes:
            if n.type != "candidate":
                continue
            key = (n.field, n.line, n.col)
            if key in assigned:
                continue
            reason = None
            if n.field=="batch_id" and not n.meta.get("near_label"):
                reason = "13位數孤立且未鄰近標籤"
            else:
                reason = "距離任何 anchor 過遠或未被配對"
            orphans.append({
                "field": n.field,
                "value": n.norm,
                "line": n.line,
                "col": n.col,
                "reason": reason
            })
        return orphans

    def _collect_candidates_debug(self, nodes: List[Node]):
        out = {"name": [], "id_no": [], "ref_date": [], "batch_id": []}
        for fld in out.keys():
            for n in nodes:
                if n.type=="candidate" and n.field==fld:
                    out[fld].append({
                        "value": n.norm,
                        "line": n.line,
                        "col": n.col,
                        "format_confidence": n.format_confidence,
                        "meta": n.meta
                    })
        return out

    def _infer_document_scopes(self, nodes: List[Node], lines: List[str]):
        topn_ref = self.cfg["scopes"]["ref_date"]["header_top_n"]
        topn_batch = self.cfg["scopes"]["batch_id"]["header_top_n"]
        # candidates with labels existences
        ref_candidates = [n for n in nodes if n.type=="candidate" and n.field=="ref_date"]
        batch_candidates = [n for n in nodes if n.type=="candidate" and n.field=="batch_id" and n.meta.get("near_label")]
        # determine scope
        ref_scope = "nearest"
        batch_scope = "nearest"
        reason_ref = "多處出現；採就近"
        reason_batch = "多處出現；採就近"
        if len(ref_candidates) == 1 and ref_candidates[0].line <= topn_ref:
            ref_scope = "document"
            reason_ref = "僅出現一次且位於標頭，套用到全部"
        if len(batch_candidates) == 1 and batch_candidates[0].line <= topn_batch:
            batch_scope = "document"
            reason_batch = "僅出現一次且位於標頭，套用到全部"
        return {
            "ref_date": {"scope": ref_scope, "reason": reason_ref},
            "batch_id": {"scope": batch_scope, "reason": reason_batch}
        }

    def _record_to_json(self, r: Record) -> Dict[str, Any]:
        return {
            "record_id": r.record_id,
            "fields": {
                "name": r.fields["name"].__dict__,
                "id_no": r.fields["id_no"].__dict__,
                "ref_date": r.fields["ref_date"].__dict__,
                "batch_id": r.fields["batch_id"].__dict__
            },
            "cluster_debug": r.cluster_debug,
            **({"merged_from": r.merged_from} if r.merged_from else {})
        }
