# Lyra Framework — Local (Ollama + GPT-OSS 20B)

Ce bundle est un **socle propre** pour (re)lancer Lyra en local sous Windows 11 + Ollama.

## Prérequis
- Python 3.10+
- Ollama installé et modèle **gpt-oss:20b** importé : `ollama pull gpt-oss:20b`

## Structure
```
lyra_framework/
  data/
    config.json
  prompts/
    system_lyra.txt
  src/
    common.py
    policies.py
    patterns.py
    llm_client.py
    runlog.py
    run_loop3.py
    run_loop_fast.py
    nemeton_build.py
    init_heavy_state.py
  run_pi_leaky_A02.bat
  requirements.txt
  .gitignore
  README.md
```

## 1) (Optionnel) Initialiser un état
```
python -m src.init_heavy_state
```

## 2) Lancer un run (30 pas)
Double-cliquez : **run_pi_leaky_A02.bat**

Les sorties sont écrites dans `data/runs/<RUN_ID>/`.

## 3) Générer la carte Nemeton
```
python -m src.nemeton_build --metrics data\runs\<RUN_ID>\metrics_log.csv
```

## Contrôleurs (extrait config)
- `tau_c_limits: [0.22, 1.6]`
- `delta_r_soft_cap: 0.90`
- `pressure_tau_share_gain: 0.10`
- `pressure_tau_share_delta_r_gate: 0.82`
- `ki_pressure: 0.015`, `pressure_i_leak: 0.02`, `pressure_i_max: 0.12`, `pressure_i_split_tau: 0.65`

> Ajustez de préférence **un seul** paramètre à la fois.

Notes : le client LLM tente d'abord l'API OpenAI-compatible d’Ollama (`/v1/chat/completions`), puis retombe sur l'API native (`/api/chat`). Si aucun modèle n’est accessible, la loop fait un fallback offline.
