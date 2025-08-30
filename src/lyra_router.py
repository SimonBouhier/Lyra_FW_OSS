from __future__ import annotations
import os, sys, json, csv, time, requests, argparse
from pathlib import Path
from typing import Dict, Any, List
from .steering import map_params, score_text

def data_dir() -> Path:
    # assume running from project root (bundle root)
    return Path(__file__).resolve().parents[1] / "data"

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def call_openai_compatible(model: str, messages: List[Dict[str,str]], params: Dict[str, float], base: str, timeout: int = 120) -> str:
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','ollama')}"}
    payload = {"model": model, "messages": messages, "stream": False}
    # steering params
    for k in ["temperature","top_p","presence_penalty","frequency_penalty"]:
        if k in params: payload[k] = params[k]
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if "choices" in j and j["choices"]:
        return j["choices"][0]["message"]["content"]
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, help="User prompt (text)")
    ap.add_argument("--k", type=int, default=3, help="Number of candidates (>=1)")
    ap.add_argument("--system", type=str, default="", help="Optional system instruction (string)")
    args = ap.parse_args()

    dat = data_dir()
    last = dat / "last_state.json"
    if not last.exists():
        print("ERROR: data/last_state.json not found. Initialize state first.")
        sys.exit(2)
    state = read_json(last)
    params = map_params(state)

    model = os.getenv("LYRA_MODEL", "gpt-oss:20b")
    base = os.getenv("OLLAMA_OPENAI_BASE", "http://localhost:11434/v1")
    system_txt = args.system or "You are a helpful, concise assistant."

    # load system file if path given and exists
    if os.path.isfile(system_txt):
        system_txt = Path(system_txt).read_text(encoding="utf-8")

    messages = [{"role":"system","content": system_txt}]
    if args.prompt:
        messages.append({"role":"user","content": args.prompt})
    else:
        print("Enter prompt (Ctrl+Z then Enter on Windows):")
        user_prompt = sys.stdin.read()
        messages.append({"role":"user","content": user_prompt})

    k = max(1, int(args.k))
    cand = []
    for i in range(k):
        txt = call_openai_compatible(model, messages, params, base)
        cand.append((i, txt, score_text(txt)))
        time.sleep(0.03)

    cand.sort(key=lambda t: t[2], reverse=True)
    best = cand[0] if cand else (-1,"", -1e9)
    print("=== LYRA ROUTER RESULT ===")
    print(best[1])
    print("\n--- meta ---")
    print(json.dumps({"params": params, "k": k, "scores": [c[2] for c in cand]}, indent=2))

    logp = dat / "router_logs.csv"
    newfile = not logp.exists()
    with open(logp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["ts","model","k","temperature","top_p","presence_penalty","frequency_penalty","score_best","len_best"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), model, k,
                    params.get("temperature"), params.get("top_p"),
                    params.get("presence_penalty"), params.get("frequency_penalty"),
                    best[2], len(best[1])])

if __name__ == "__main__":
    main()
