from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import deque

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

@dataclass
class PatternEngine:
    control_cfg: Dict[str, Any]
    window: int = 5
    anti_sur_nudge_tau: float = 0.95

    coh_hist: deque = field(default_factory=lambda: deque(maxlen=16))
    tens_hist: deque = field(default_factory=lambda: deque(maxlen=16))

    def _rolling(self, s: deque, k: int) -> float:
        if not s: return 0.0
        k = min(k, len(s))
        return sum(list(s)[-k:]) / k if k > 0 else 0.0

    def step(self, state, coh: float, fit: float, pressure: float, tension: float) -> Dict[str, Any]:
        self.coh_hist.append(float(coh))
        self.tens_hist.append(float(tension))
        actions = []
        reasons = []

        t_sp = float(self.control_cfg.get("tension_setpoint", 0.55))
        tau_lo, tau_hi = self.control_cfg.get("tau_c_limits", [0.22, 1.6])

        reg = "R1"
        if tension > t_sp + 0.10:
            reasons.append("tension_high_instant")
            reg = "R2"

        if reg == "R2" and any(r.startswith("tension_high") for r in reasons):
            before = state.lyra.tau_c
            state.lyra.tau_c = clamp(self.anti_sur_nudge_tau * state.lyra.tau_c, tau_lo, tau_hi)
            actions.append(f"anti_surr(tau_c:{before:.3f}->{state.lyra.tau_c:.3f})")

        return {"regime": reg, "regime_reason": reasons, "actions": actions}
