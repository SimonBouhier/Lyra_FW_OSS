from __future__ import annotations
import os, json, time, csv
from pathlib import Path
from typing import Dict, Any
from .common import data_dir, read_json, write_json, clamp, now_ts
from .policies import State, LyraParams, Epistemic, PolicySEUIL, measure_pressure, measure_tension, measure_coherence, measure_fit

def load_config() -> Dict[str, Any]:
    return json.loads((data_dir() / "config.json").read_text(encoding="utf-8"))

def controllers(state: State, cfg: Dict[str, Any]) -> None:
    lo, hi = cfg["control"]["tau_c_limits"]
    if state.lyra.tau_c < lo: state.lyra.tau_c = lo
    elif state.lyra.tau_c > hi: state.lyra.tau_c = hi

    coh = state.epistemic.E_p.get("coherence", 0.0)
    fit = state.epistemic.E_d.get("fit", 0.0)
    press = state.epistemic.E_m.get("pressure", 0.0)
    tens = measure_tension(coh, fit, press)

    band_t = float(cfg["control"].get("tension_band", 0.0))
    band_p = float(cfg["control"].get("pressure_band", 0.0))

    err_t = tens - cfg["control"]["tension_setpoint"]
    kp_t = cfg["control"]["kp_tension"] * float(os.getenv("LYRA_KP_TENSION_BOOST", "1.0"))
    if abs(err_t) > band_t:
        state.lyra.tau_c = max(lo, min(hi, state.lyra.tau_c - kp_t * err_t))

    p_set = cfg["control"]["pressure_setpoint"]
    err_p = p_set - press
    kp_p  = cfg["control"]["kp_pressure"]

    ki     = float(cfg["control"].get("ki_pressure", 0.015))
    i_leak = float(cfg["control"].get("pressure_i_leak", 0.02))
    i_max  = float(cfg["control"].get("pressure_i_max", 0.12))
    split  = float(cfg["control"].get("pressure_i_split_tau", 0.65))

    p_int = getattr(state, "pressure_i", 0.0)
    if abs(err_p) > band_p:
        p_int = p_int + ki * err_p
    p_int = max(-i_max, min(i_max, p_int * (1.0 - i_leak)))
    state.pressure_i = p_int

    delta_p = kp_p * err_p + (1.0 - split) * p_int
    state.lyra.delta_r = clamp(state.lyra.delta_r + delta_p)

    gate = float(cfg["control"].get("pressure_tau_share_delta_r_gate", 0.82))
    share_gain = float(cfg["control"].get("pressure_tau_share_gain", 0.10))
    if press < p_set - band_p and state.lyra.delta_r > gate:
        d_tau = max(0.0, split * p_int) + share_gain * (p_set - press)
        state.lyra.tau_c = min(hi, state.lyra.tau_c + d_tau)

    margin = cfg["control"].get("pressure_margin", 0.04)
    nudge  = cfg["control"].get("delta_r_nudge_down", 0.02)
    floor  = cfg["control"].get("delta_r_floor", 0.28)
    if press > p_set + margin:
        state.lyra.delta_r = max(floor, state.lyra.delta_r - nudge)

    soft_cap = float(cfg["control"].get("delta_r_soft_cap", 0.90))
    if state.lyra.delta_r > soft_cap:
        state.lyra.delta_r = max(soft_cap, state.lyra.delta_r - 0.02)

    if state.lyra.tau_c < lo: state.lyra.tau_c = lo
    elif state.lyra.tau_c > hi: state.lyra.tau_c = hi

def main():
    cfg = load_config()
    st = State()
    steps = min(int(cfg["loop"]["steps"]), 15)
    for _ in range(steps):
        st.step += 1
        coh = measure_coherence(st.lyra.rho, st.lyra.delta_r, st.lyra.tau_c)
        fit = measure_fit(coh, measure_pressure(st.lyra.delta_r, st.lyra.tau_c))
        press = measure_pressure(st.lyra.delta_r, st.lyra.tau_c)
        tens = measure_tension(coh, fit, press)
        st.epistemic.E_p["coherence"] = coh
        st.epistemic.E_d["fit"] = fit
        st.epistemic.E_m["pressure"] = press
        controllers(st, cfg)
    print("fast loop done.")

if __name__ == "__main__":
    main()
