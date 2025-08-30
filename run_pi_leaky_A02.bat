@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d %~dp0
set "ROOT=%~dp0"
set "SRC=%ROOT%src"
set "DAT=%ROOT%data"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "LYRA_RUN_ID=pi_leaky_A02"
set "LYRA_NOTES=clamp:on;share=0.10;split=0.65;ki=0.015"
set "LYRA_MODEL=gpt-oss:20b"

set "LYRA_LAMBDA_THRESHOLD=0.90"
set "LYRA_LAMBDA_ATTENUATION=0.96"
set "LYRA_LAMBDA_TAU_GAIN=1.04"
set "LYRA_LAMBDA_TAU_BIAS=0.015"
set "LYRA_LAMBDA_COOLDOWN=5"

echo [Lyra] run_loop3 start (model=%LYRA_MODEL%)...
REM IMPORTANT: run as module so that relative imports (from .common ...) work
python -X utf8 -m src.run_loop3 --clean-plots || goto err

if exist "%DAT%\runs\%LYRA_RUN_ID%\metrics_log.csv" (
  python -X utf8 -m src.nemeton_build --metrics "%DAT%\runs\%LYRA_RUN_ID%\metrics_log.csv"
)

echo Done.
pause
exit /b 0

:err
echo An error occurred.
pause
exit /b 1
