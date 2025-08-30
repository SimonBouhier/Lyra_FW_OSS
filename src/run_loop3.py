# src/run_loop3.py
from __future__ import annotations
import os, sys, json, time, csv
from pathlib import Path
from typing import Dict, Any

from .common import data_dir, prompts_dir, read_json, write_json, clamp, now_ts
from .policies import (
    State, LyraParams, Epistemic, PolicySEUIL,
    measure_pressure, measure_tension, measure_coherence, measure_fit
)
from .patterns import PatternEngine
from .llm_client import LocalLLMClient
from .runlog import prepare_run


# ---------- helpers ----------
def load_config() -> Dict[str, Any]:
    return json.loads((data_dir() / "config.json").read_text(encoding="utf-8"))


def load_or_init_state() -> State:
    p = data_dir() / "last_state.json"
    if p.exists():
        js = read_json(p)
        ly = js.get("lyra", {})
        ep = js.get("epistemic", {})
        flags = js.get("flags", {})
        st = State(
            lyra=LyraParams(
                rho=ly.get("rho", 0.5),
                delta_r=ly.get("delta_r", 0.5),
                tau_c=ly.get("tau_c", 0.3),
            ),
            epistemic=Epistemic(
                E_p={"coherence": ep.get("E_p", {}).get("coherence", 0.5)},
                E_d={"fit": ep.get("E_d", {}).get("fit", 0.5)},
                E_m={"pressure": ep.get("E_m", {}).get("pressure", 0.4)},
            ),
        )
        st.flags.phase_lambda_active = bool(flags.get("phase_lambda_active", False))
        st.flags.plateau_streak = int(flags.get("plateau_streak", 0))
        st.step = int(js.get("step", 0))
        return st
    return State()


def state_to_dict(s: State) -> Dict[str, Any]:
    return {
        "step": s.step,
        "lyra": {"rho": s.lyra.rho, "delta_r": s.lyra.delta_r, "tau_c": s.lyra.tau_c},
        "epistemic": {"E_p": s.epistemic.E_p, "E_d": s.epistemic.E_d, "E_m": s.epistemic.E_m},
        "flags": {"phase_lambda_active": s.flags.phase_lambda_active, "plateau_streak": s.flags.plateau_streak},
        "ts": now_ts(),
    }


# ---------- controllers ----------
def controllers(state: State, cfg: Dict[str, Any]) -> None:
    lo, hi = cfg["control"]["tau_c_limits"]
    if state.lyra.tau_c < lo:
        state.lyra.tau_c = lo
    elif state.lyra.tau_c > hi:
        state.lyra.tau_c = hi

    coh = state.epistemic.E_p.get("coherence", 0.0)
    fit = state.epistemic.E_d.get("fit", 0.0)
    press = state.epistemic.E_m.get("pressure", 0.0)
    tens = measure_tension(coh, fit, press)

    band_t = float(cfg["control"].get("tension_band", 0.0))
    band_p = float(cfg["control"].get("pressure_band", 0.0))

    # TENSION → tau_c (P)
    err_t = tens - cfg["control"]["tension_setpoint"]
    kp_boost = float(os.getenv("LYRA_KP_TENSION_BOOST", "1.0"))
    kp_t = cfg["control"]["kp_tension"] * kp_boost
    if abs(err_t) > band_t:
        state.lyra.tau_c = max(lo, min(hi, state.lyra.tau_c - kp_t * err_t))

    # PRESSURE → delta_r & tau_c (P + leaky I)
    p_set = cfg["control"]["pressure_setpoint"]
    err_p = p_set - press
    kp_p = cfg["control"]["kp_pressure"]

    ki = float(cfg["control"].get("ki_pressure", 0.015))
    i_leak = float(cfg["control"].get("pressure_i_leak", 0.02))
    i_max = float(cfg["control"].get("pressure_i_max", 0.12))
    split = float(cfg["control"].get("pressure_i_split_tau", 0.65))

    p_int = getattr(state, "pressure_i", 0.0)
    if abs(err_p) > band_p:
        p_int = p_int + ki * err_p
    p_int = max(-i_max, min(i_max, p_int * (1.0 - i_leak)))
    state.pressure_i = p_int

    # delta_r update (P + (1-split)*I)
    delta_p = kp_p * err_p + (1.0 - split) * p_int
    state.lyra.delta_r = clamp(state.lyra.delta_r + delta_p)

    # partage vers tau_c si delta_r haut + pression insuffisante
    gate = float(cfg["control"].get("pressure_tau_share_delta_r_gate", 0.82))
    share_gain = float(cfg["control"].get("pressure_tau_share_gain", 0.10))
    if press < p_set - band_p and state.lyra.delta_r > gate:
        d_tau = max(0.0, split * p_int) + share_gain * (p_set - press)
        state.lyra.tau_c = min(hi, state.lyra.tau_c + d_tau)

    # purge douce asymétrique si pression trop haute (P2)
    margin = cfg["control"].get("pressure_margin", 0.04)
    nudge = cfg["control"].get("delta_r_nudge_down", 0.02)
    floor = cfg["control"].get("delta_r_floor", 0.28)
    nudge_high = float(cfg["control"].get("delta_r_nudge_high", 0.025))

    if press > p_set + margin:
        # purge un peu plus énergique au-dessus du seuil
        state.lyra.delta_r = max(floor, state.lyra.delta_r - nudge_high)
        # anti-windup: si l’intégral pousse dans le mauvais sens, on le réduit
        if getattr(state, "pressure_i", 0.0) > 0:
            state.pressure_i *= 0.5

    # soft cap sur delta_r
    soft_cap = float(cfg["control"].get("delta_r_soft_cap", 0.90))
    if state.lyra.delta_r > soft_cap:
        state.lyra.delta_r = max(soft_cap, state.lyra.delta_r - 0.02)

    # re-clamp tau_c
    if state.lyra.tau_c < lo:
        state.lyra.tau_c = lo
    elif state.lyra.tau_c > hi:
        state.lyra.tau_c = hi


# ---------- nemeton (PCA 2D enrichie) ----------
def build_nemeton_csv_png(csv_in: Path, out_csv: Path, out_png: Path) -> None:
    import numpy as _np, csv as _csv, json as _json
    # read metrics
    rows = []
    with open(csv_in, "r", encoding="utf-8", newline="") as f:
        r = _csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return

    steps, feats, pres = [], [], []
    for row in rows:
        try:
            steps.append(int(row["step"]))
            c = float(row["coherence"])
            f = float(row["fit"])
            p = float(row["pressure"])
            t = float(row["tension"])
            feats.append([c, f, p, t])
            pres.append(p)
        except Exception:
            pass
    if not feats:
        return
    X = _np.array(feats, dtype=float)

    # z-score standardization
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    # PCA 2D
    Xc = Xz - Xz.mean(0, keepdims=True)
    U, S, VT = _np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ VT[:2].T

    # write CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["step", "x", "y"])
        for s, (x, y) in zip(steps, Z):
            w.writerow([s, float(x), float(y)])

    # path metrics
    def path_metrics(Z):
        diffs = _np.diff(Z, axis=0)
        lens = _np.linalg.norm(diffs, axis=1)
        path_len = float(lens.sum()) if len(lens) else 0.0
        net = float(_np.linalg.norm(Z[-1] - Z[0])) if len(Z) > 1 else 0.0
        dir_ratio = (net / path_len) if path_len > 0 else 0.0
        angs = []
        for i in range(1, len(diffs)):
            a, b = diffs[i - 1], diffs[i]
            na = _np.linalg.norm(a)
            nb = _np.linalg.norm(b)
            if na > 0 and nb > 0:
                cos = float(_np.clip((a @ b) / (na * nb), -1.0, 1.0))
                angs.append(_np.arccos(cos))
        mean_turn = float(_np.mean(angs)) if angs else 0.0
        return {"path_len": path_len, "net_disp": net, "directionality": dir_ratio, "mean_turn": mean_turn}

    m = path_metrics(Z)
    with open(out_csv.with_name("nemeton_metrics.json"), "w", encoding="utf-8") as f:
        _json.dump(m, f, indent=2)

    # PNG
    try:
        import matplotlib.pyplot as _plt
        import matplotlib as _mpl
        _plt.figure(figsize=(6.4, 4.8))
        cm = _mpl.cm.get_cmap("viridis")
        _plt.plot(Z[:, 0], Z[:, 1], linewidth=1.0, alpha=0.4)
        sc = _plt.scatter(Z[:, 0], Z[:, 1], s=26, c=_np.array(pres), cmap=cm)
        _plt.colorbar(sc, label="pressure")
        _plt.scatter([Z[0, 0]], [Z[0, 1]], s=70, marker="^", label="start")
        _plt.scatter([Z[-1, 0]], [Z[-1, 1]], s=70, marker="s", label="end")
        _plt.title("Nemeton map (PCA 2D, z-score)")
        _plt.xlabel("PC1")
        _plt.ylabel("PC2")
        _plt.legend()
        _plt.tight_layout()
        _plt.savefig(out_png)
        _plt.close()
    except Exception:
        # matplotlib non installée : PNG ignoré
        pass


# ---------- main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemeton", action="store_true", help="generate nemeton_map.csv and nemeton_map.png at end")
    args = ap.parse_args()

    cfg = load_config()
    run_id = os.getenv("LYRA_RUN_ID", "")
    notes = os.getenv("LYRA_NOTES", "")

    info = prepare_run(run_id if run_id else None)
    if run_id:
        info["run_id"] = run_id
        info["dir"] = str((Path(info["dir"]).parent / run_id).resolve())
        Path(info["dir"]).mkdir(parents=True, exist_ok=True)
        info["csv_run"] = str(Path(info["dir"]) / "metrics_log.csv")
        info["jsonl_run"] = str(Path(info["dir"]) / "loop3_log.jsonl")
        info["state_run"] = str(Path(info["dir"]) / f"state_{run_id}.json")

    write_json(
        Path(info["dir"]) / "RUNINFO.json",
        {"run_id": info["run_id"], "notes": notes, "ts": now_ts(), "cfg": cfg},
    )

    st = load_or_init_state()
    pat = PatternEngine(cfg["control"])

    sys_prompt = (prompts_dir() / "system_lyra.txt").read_text(encoding="utf-8")
    client = LocalLLMClient()

    steps = int(cfg["loop"]["steps"])
    sleep_sec = float(cfg["loop"].get("sleep_sec", 0))
    offline = bool(cfg["loop"].get("offline_fallback", True))

    csv_run = Path(info["csv_run"])
    jsonl_run = Path(info["jsonl_run"])
    state_run = Path(info["state_run"])

    need_header = not csv_run.exists()
    f_csv = open(csv_run, "a", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    if need_header:
        writer.writerow(
            ["run_id", "step", "coherence", "fit", "pressure", "tension", "rho", "delta_r", "tau_c", "lambda", "plateau"]
        )

    for _ in range(steps):
        st.step += 1

        # before measures
        coh0 = measure_coherence(st.lyra.rho, st.lyra.delta_r, st.lyra.tau_c)
        press0 = measure_pressure(st.lyra.delta_r, st.lyra.tau_c)
        fit0 = measure_fit(coh0, press0)
        tens0 = measure_tension(coh0, fit0, press0)
        st.epistemic.E_p["coherence"] = coh0
        st.epistemic.E_d["fit"] = fit0
        st.epistemic.E_m["pressure"] = press0

        # lambda policy
        lam = PolicySEUIL(
            threshold=float(os.getenv("LYRA_LAMBDA_THRESHOLD", "0.90")),
            attenuation=float(os.getenv("LYRA_LAMBDA_ATTENUATION", "0.96")),
            tau_gain=float(os.getenv("LYRA_LAMBDA_TAU_GAIN", "1.04")),
            tau_bias=float(os.getenv("LYRA_LAMBDA_TAU_BIAS", "0.015")),
            cooldown=int(os.getenv("LYRA_LAMBDA_COOLDOWN", "5")),
        )
        lam.step(st, cfg)

        # patterns
        pat_info = pat.step(st, coh0, fit0, press0, tens0)

        # LLM suggestion
        llm_ok = True
        user_payload = {
            "step": st.step,
            "state": state_to_dict(st),
            "targets": {
                "tension": cfg["control"]["tension_setpoint"],
                "pressure": cfg["control"]["pressure_setpoint"],
            },
            "hints": {"delta_r_soft_cap": cfg["control"]["delta_r_soft_cap"]},
        }
        try:
            resp = client.propose_action(sys_prompt, user_payload)
        except Exception:
            resp = {}
        if not resp:
            if offline:
                llm_ok = False
                if press0 < cfg["control"]["pressure_setpoint"] - cfg["control"]["pressure_band"]:
                    st.lyra.delta_r = clamp(st.lyra.delta_r + 0.01)
                else:
                    st.lyra.delta_r = clamp(st.lyra.delta_r - 0.005)
            else:
                raise RuntimeError("LLM returned empty response and offline_fallback is False")

        if "delta_r" in resp:
            st.lyra.delta_r = clamp(float(resp["delta_r"]))
        if "rho" in resp:
            st.lyra.rho = clamp(float(resp["rho"]))
        if "tau_c" in resp:
            st.lyra.tau_c = float(resp["tau_c"])

        # controllers
        controllers(st, cfg)

        # after measures
        coh1 = measure_coherence(st.lyra.rho, st.lyra.delta_r, st.lyra.tau_c)
        press1 = measure_pressure(st.lyra.delta_r, st.lyra.tau_c)
        fit1 = measure_fit(coh1, press1)
        tens1 = measure_tension(coh1, fit1, press1)
        st.epistemic.E_p["coherence"] = coh1
        st.epistemic.E_d["fit"] = fit1
        st.epistemic.E_m["pressure"] = press1

        # CSV (run_id included)
        writer.writerow(
            [
                info["run_id"],
                st.step,
                coh1,
                fit1,
                press1,
                tens1,
                st.lyra.rho,
                st.lyra.delta_r,
                st.lyra.tau_c,
                int(st.flags.phase_lambda_active),
                st.flags.plateau_streak,
            ]
        )
        f_csv.flush()

        # JSONL event
        ev = {
            "step": st.step,
            "patterns": pat_info,
            "before": {"coherence": coh0, "fit": fit0, "pressure": press0, "tension": tens0},
            "after": {"coherence": coh1, "fit": fit1, "pressure": press1, "tension": tens1},
            "state": state_to_dict(st),
            "llm_ok": llm_ok,
        }
        with open(jsonl_run, "a", encoding="utf-8") as fj:
            fj.write(json.dumps(ev, ensure_ascii=False) + "\n")

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    f_csv.close()

    # save final state (per-run + latest)
    write_json(state_run, state_to_dict(st))
    write_json(data_dir() / "last_state.json", state_to_dict(st))

    # nemeton (CSV + PNG) at end if requested
    if args.nemeton:
        try:
            build_nemeton_csv_png(
                csv_in=Path(info["csv_run"]),
                out_csv=Path(info["dir"]) / "nemeton_map.csv",
                out_png=Path(info["dir"]) / "nemeton_map.png",
            )
        except Exception:
            pass

    print(f"Loop3 done → {info['dir']}")


if __name__ == "__main__":
    main()
