from __future__ import annotations
from typing import Dict, Any
import re

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def derive_tension(coh: float, fit: float, pressure: float) -> float:
    # Approx of policies.measure_tension (kept local to avoid cross-imports)
    return clamp(0.25 + 0.50 * pressure + 0.25 * (1.0 - 0.5*(coh+fit)), 0.0, 1.0)

def map_params(state: Dict[str, Any]) -> Dict[str, float]:
    ly = state.get("lyra", {})
    epi = state.get("epistemic", {})
    E_p = epi.get("E_p", {}); E_d = epi.get("E_d", {}); E_m = epi.get("E_m", {})
    rho = float(ly.get("rho", 0.5))
    dr = float(ly.get("delta_r", 0.5))
    tc = float(ly.get("tau_c", 0.3))
    coh = float(E_p.get("coherence", 0.5))
    fit = float(E_d.get("fit", 0.5))
    press = float(E_m.get("pressure", 0.4))
    tens = derive_tension(coh, fit, press)

    temp = 0.7
    if dr > 0.75: temp += 0.20
    elif dr < 0.35: temp -= 0.20
    if tc > 0.60: temp += 0.10
    elif tc < 0.28: temp -= 0.15
    if tens > 0.62: temp -= 0.10  # cool down when tension high
    temp = clamp(temp, 0.2, 1.2)

    top_p = 0.95 if temp >= 1.0 else (0.9 if temp >= 0.7 else 0.8)

    # gentle regularizers
    presence_penalty  = clamp((tens - 0.5) * 1.2, 0.0, 0.6)   # encourage diversity if tension > 0.5
    frequency_penalty = clamp((0.55 - coh) * 1.0, 0.0, 0.8)   # discourage repetition when coherence low

    return {
        "temperature": float(f"{temp:.3f}"),
        "top_p": float(f"{top_p:.3f}"),
        "presence_penalty": float(f"{presence_penalty:.3f}"),
        "frequency_penalty": float(f"{frequency_penalty:.3f}"),
        "derived_tension": float(f"{tens:.3f}")
    }

def score_text(txt: str) -> float:
    if not txt: return -1e9
    penalty = 0.0
    if re.search(r"\bAs an AI language model\b", txt, re.I): penalty += 1.0
    toks = re.findall(r"\w+", txt.lower())
    uniq_ratio = (len(set(toks))/len(toks)) if toks else 0.0
    n_words = len(toks)
    length_score = 1.0 - abs(n_words - 220)/220.0  # sweet spot ~220 words
    length_score = max(0.0, length_score)
    score = 2.0*uniq_ratio + 1.5*length_score - penalty
    return score
