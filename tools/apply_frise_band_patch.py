import json, pathlib
p = pathlib.Path("data/config.json")
cfg = json.loads(p.read_text(encoding="utf-8"))
c = cfg.setdefault("control", {})
c["pressure_band"] = 0.07
p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("Frise patch #1: pressure_band=0.07")
