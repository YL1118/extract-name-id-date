
from __future__ import annotations
from typing import Dict, Any, Optional
from .config import load_config, snapshot_config
from .clusterer import Extractor

def run_extraction(text: str, config_path: Optional[str]) -> Dict[str, Any]:
    cfg = load_config(config_path)
    engine = Extractor(cfg)
    result = engine.extract(text)
    result["config_used"] = snapshot_config(cfg)
    return result
