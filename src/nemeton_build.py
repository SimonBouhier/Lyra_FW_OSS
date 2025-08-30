from __future__ import annotations
import json, argparse
from pathlib import Path
import pandas as pd
import numpy as np

def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ VT[:2].T
    return Z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=False, help="path to loop3_log.jsonl")
    ap.add_argument("--metrics", type=str, required=False, help="path to metrics_log.csv")
    args = ap.parse_args()

    base = Path(args.jsonl).parent if args.jsonl else Path(".")
    if args.metrics and Path(args.metrics).exists():
        m = pd.read_csv(args.metrics)
    else:
        m = pd.read_csv(base / "metrics_log.csv")
    cols = [c for c in ["coherence","fit","pressure","tension"] if c in m.columns]
    X = m[cols].to_numpy(dtype=float)
    Z = pca_2d(X)
    out = pd.DataFrame({"step": m["step"].values, "x": Z[:,0], "y": Z[:,1]})
    out.to_csv(base / "nemeton_map.csv", index=False, encoding="utf-8")
    print("wrote:", str(base / "nemeton_map.csv"))

if __name__ == "__main__":
    main()
