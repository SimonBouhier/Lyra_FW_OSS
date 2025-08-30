# tools/apply_p1p2_patch.py
"""
Met à jour data/config.json pour P1+P2 (stabilité pression).
Modifs appliquées :
- kp_pressure := 0.07
- pressure_i_leak := 0.03
- delta_r_nudge_high := 0.025 (ajout si absent)
"""
from __future__ import annotations
import json
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    cfg_path = data_dir / "config.json"
    if not cfg_path.exists():
        print("ERREUR: data/config.json introuvable.")
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    ctrl = cfg.setdefault("control", {})
    ctrl["kp_pressure"] = 0.07
    ctrl["pressure_i_leak"] = 0.03
    if "delta_r_nudge_high" not in ctrl:
        ctrl["delta_r_nudge_high"] = 0.025

    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("OK: config.json mis à jour pour P1+P2")

if __name__ == "__main__":
    main()
