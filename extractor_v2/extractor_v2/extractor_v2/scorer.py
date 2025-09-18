
from __future__ import annotations
from typing import Dict, Tuple

def distance_score(delta_line: int, delta_col: int, line_weights: Dict[int, float]) -> float:
    """
    line_weights: e.g., {0:1.0, 1:0.7, 2:0.5, 3:0.3}. Use abs lines with cap at max key.
    """
    a = abs(delta_line)
    maxk = max(line_weights)
    w = line_weights.get(a, line_weights[maxk] if a >= maxk else 0.0)
    import math
    tau = 12.0
    return w * (2.718281828 ** (-abs(delta_col)/tau))

def direction_prior(delta_line: int, delta_col: int, priors: Dict[str, float]) -> float:
    if delta_line == 0 and delta_col >= 0:
        return priors.get("same_line_right", 1.0)
    if delta_line == 0 and delta_col < 0:
        return priors.get("same_line_left", 0.8)
    if delta_line == 1:
        return priors.get("down_1", 0.7)
    if delta_line == 2:
        return priors.get("down_2", 0.5)
    if delta_line >= 3:
        return priors.get("down_3", 0.3)
    # Upwards (rare) - small prior
    return 0.2

def final_score(label_conf: float, format_conf: float, dist_score: float, dir_prior: float,
                context_bonus: float, collision_penalty: float, weights: Dict[str, float]) -> float:
    return (
        weights["w_label"]*label_conf +
        weights["w_format"]*format_conf +
        weights["w_dist"]*dist_score +
        weights["w_dir"]*dir_prior +
        weights["w_context"]*context_bonus -
        weights["w_penalty"]*collision_penalty
    )
