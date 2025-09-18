
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "weights": {
        "w_label": 0.25,
        "w_format": 0.30,
        "w_dist": 0.25,
        "w_dir": 0.10,
        "w_context": 0.10,
        "w_penalty": 0.25
    },
    "grouping": {
        "anchor_order": ["id_no", "name"],
        "window_lines_up": 1,
        "window_lines_down": 3,
        "max_col_delta": 30,
        "max_span_lines_for_linking": 3,
        "pairing_strategy": "greedy",   # or "hungarian"
        "cluster_merge_if_overlap": True
    },
    "scopes": {
        "ref_date": {
            "header_top_n": 8,
            "default": "auto"   # "auto"|"document"|"nearest"
        },
        "batch_id": {
            "header_top_n": 8,
            "default": "auto"
        }
    },
    "dedupe": {
        "merge_same_id": True,
        "prefer_higher_confidence": True
    },
    "fuzzy": {
        "label_ratio_threshold": 0.80,
        "edit_distance_allow": 1
    },
    "direction_prior": {
        "same_line_right": 1.0,
        "same_line_left": 0.8,
        "down_1": 0.7,
        "down_2": 0.5,
        "down_3": 0.3
    },
    "version": "0.2.0"
}

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load config from a JSON/YAML file if provided; otherwise return DEFAULT_CONFIG.
    YAML support is optional: if PyYAML is not installed, only JSON is accepted.
    """
    if not config_path:
        return DEFAULT_CONFIG
    path = str(config_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Try JSON first
        try:
            cfg = json.loads(text)
            return _merge(DEFAULT_CONFIG, cfg)
        except json.JSONDecodeError:
            pass
        # Try YAML if available
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(text)
            if cfg is None:
                cfg = {}
            return _merge(DEFAULT_CONFIG, cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to parse config file '{path}'. "
                               f"Provide JSON or install PyYAML. Original error: {e}")
    except FileNotFoundError:
        raise

def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def snapshot_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-ish copy suitable for embedding in results."""
    import copy
    return copy.deepcopy(cfg)
