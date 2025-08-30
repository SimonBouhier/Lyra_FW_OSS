from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json, time, os
from .common import data_dir, write_json

def prepare_run(run_id: str | None) -> Dict[str, Any]:
    d = data_dir()
    runs = d / "runs"
    runs.mkdir(exist_ok=True, parents=True)
    rid = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    out = runs / rid
    out.mkdir(exist_ok=True, parents=True)

    latest_csv = d / "metrics_log.csv"
    latest_jsonl = d / "loop3_log.jsonl"
    latest_state = d / "last_state.json"
    run_csv = out / "metrics_log.csv"
    run_jsonl = out / "loop3_log.jsonl"
    run_state = out / f"state_{rid}.json"

    info = {
        "run_id": rid,
        "dir": str(out),
        "csv_run": str(run_csv),
        "jsonl_run": str(run_jsonl),
        "state_run": str(run_state),
        "csv_latest": str(latest_csv),
        "jsonl_latest": str(latest_jsonl),
        "state_latest": str(latest_state),
    }
    write_json(out / "RUNINFO.json", info)
    return info
