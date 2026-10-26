@echo off
cd /d "%~dp0"
py -3.12 scripts\start_clef_ollama.py %*
if errorlevel 1 pause
