from __future__ import annotations
from pathlib import Path
import json, os, time, math, random
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
DATA.mkdir(exist_ok=True, parents=True)

def data_dir() -> Path: return DATA
def prompts_dir() -> Path: return PROMPTS

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def now_ts() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().isoformat() + "Z"
