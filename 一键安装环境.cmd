@echo off
setlocal
cd /d "%~dp0"

set "NO_PAUSE="
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

echo ============================================================
echo  6+1 Microphone Array v1.4.4 - Environment Installer
echo ============================================================
echo.

py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>&1
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" goto setup_environment
    if exist "%ProgramFiles%\Python312\python.exe" goto setup_environment
    echo Python 3.12 was not found. Installing it for the current user with winget...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [ERROR] winget is unavailable, so Python 3.12 cannot be installed automatically.
        echo Install Python 3.12 from https://www.python.org/downloads/release/python-3120/
        echo and then run this installer again.
        goto failed
    )
    winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo [ERROR] Python 3.12 installation did not complete.
        goto failed
    )
)

:setup_environment
echo.
echo Creating the project environment and installing dependencies. Keep the network connected...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_vscode_env.ps1"
if errorlevel 1 goto failed

echo.
echo [OK] The environment is installed and verified.
echo You can now double-click the Test UI launcher in the project root.
if not defined NO_PAUSE pause
exit /b 0

:failed
echo.
echo Environment installation failed. Keep the error details shown above.
if not defined NO_PAUSE pause
exit /b 1
