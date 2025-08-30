from __future__ import annotations
from .common import write_json, data_dir
from .policies import State

def main():
    s = State()
    write_json(data_dir() / "last_state.json", {
        "step": 0,
        "lyra": {"rho": s.lyra.rho, "delta_r": s.lyra.delta_r, "tau_c": s.lyra.tau_c},
        "epistemic": s.epistemic.__dict__,
        "flags": {"phase_lambda_active": False, "plateau_streak": 0}
    })
    print("Initialized data/last_state.json")

if __name__ == "__main__":
    main()
