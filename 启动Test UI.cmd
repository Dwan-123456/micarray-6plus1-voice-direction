@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0.venv-v1.4\Scripts\pythonw.exe" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('Install the project environment first by running the environment installer in the project root.','Runtime environment missing','OK','Warning') | Out-Null"
    exit /b 1
)

if /I "%~1"=="--check" (
    "%~dp0.venv-v1.4\Scripts\python.exe" "%~dp0scripts\check_runtime_env.py"
    exit /b %ERRORLEVEL%
)

"%~dp0.venv-v1.4\Scripts\python.exe" "%~dp0scripts\check_runtime_env.py" >nul 2>&1
if errorlevel 1 (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('The project environment or Test UI configuration is invalid. Run the environment installer again and keep any error details it displays.','Test UI preflight failed','OK','Error') | Out-Null"
    exit /b 1
)

start "" powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0scripts\launch_dev_test_ui.ps1"
exit /b 0
