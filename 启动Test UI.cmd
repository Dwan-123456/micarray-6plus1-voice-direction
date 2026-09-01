@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0.venv-v1.4\Scripts\pythonw.exe" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('Install the project environment first by running the environment installer in the project root.','Runtime environment missing','OK','Warning') | Out-Null"
    exit /b 1
)

if /I "%~1"=="--check" (
    "%~dp0.venv-v1.4\Scripts\python.exe" -c "from pathlib import Path; from common.config import load_config; import gui.dev_test_ui.app; c=load_config(Path('config/config.yaml')); assert c.device.sample_rate == 48000; print('Test UI import and config check passed')"
    exit /b %ERRORLEVEL%
)

start "" powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0scripts\launch_dev_test_ui.ps1"
exit /b 0
