from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
from collections import deque
from math import tanh

# Simple measures + structures

@dataclass
class LyraParams:
    rho: float = 0.5
    delta_r: float = 0.5
    tau_c: float = 0.3

@dataclass
class Epistemic:
    E_p: Dict[str, float] = field(default_factory=lambda: {"coherence": 0.5})
    E_d: Dict[str, float] = field(default_factory=lambda: {"fit": 0.5})
    E_m: Dict[str, float] = field(default_factory=lambda: {"pressure": 0.4})

@dataclass
class Flags:
    phase_lambda_active: bool = False
    plateau_streak: int = 0

@dataclass
class State:
    lyra: LyraParams = field(default_factory=LyraParams)
    epistemic: Epistemic = field(default_factory=Epistemic)
    flags: Flags = field(default_factory=Flags)
    step: int = 0

# Measures (toy but monotone/smooth)
def measure_pressure(delta_r: float, tau_c: float) -> float:
    # synthetic: pressure rises with delta_r and tau_c (bounded)
    return max(0.0, min(1.0, 0.25 + 0.55 * delta_r + 0.30 * (tau_c - 0.22)))

def measure_tension(coh: float, fit: float, pressure: float) -> float:
    # synthetic: tension increases with pressure and (1-coh)
    return max(0.0, min(1.0, 0.25 + 0.50 * pressure + 0.25 * (1.0 - 0.5*(coh+fit))))

def measure_coherence(rho: float, delta_r: float, tau_c: float) -> float:
    # synthetic: coherence improves with balanced rho and moderate tau_c
    b = 1.0 - abs(rho - 0.5) * 2.0
    c = max(0.0, 1.0 - abs(tau_c - 0.35) * 1.5)
    d = 1.0 - abs(delta_r - 0.6) * 1.2
    return max(0.0, min(1.0, 0.2 + 0.45*b + 0.2*c + 0.15*d))

def measure_fit(coh: float, pressure: float) -> float:
    # synthetic: fit follows coherence and moderate pressure
    return max(0.0, min(1.0, 0.3 + 0.6*coh - 0.05*abs(pressure-0.45)))

# Lambda policy (SEUIL-like)
class PolicySEUIL:
    def __init__(self, threshold=0.90, attenuation=0.96, tau_gain=1.04, tau_bias=0.015, cooldown=5):
        self.threshold = threshold
        self.attenuation = attenuation
        self.tau_gain = tau_gain
        self.tau_bias = tau_bias
        self.cooldown = cooldown
        self._cool = 0

    def step(self, state: State, cfg: Dict[str, Any]) -> None:
        coh = state.epistemic.E_p.get("coherence", 0.0)
        if self._cool > 0:
            self._cool -= 1
            state.flags.phase_lambda_active = False
            return

        if coh >= self.threshold:
            state.flags.phase_lambda_active = True
            # boost tau_c slightly (bounded by controllers clamp)
            state.lyra.tau_c = state.lyra.tau_c * self.tau_gain + self.tau_bias
        else:
            if state.flags.phase_lambda_active:
                self._cool = self.cooldown
            state.flags.phase_lambda_active = False
