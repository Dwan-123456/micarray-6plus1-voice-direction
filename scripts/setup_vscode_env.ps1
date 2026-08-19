param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ($Recreate -and (Test-Path -LiteralPath $venvPath)) {
    $resolvedVenv = (Resolve-Path -LiteralPath $venvPath).Path
    if ($resolvedVenv -ne (Join-Path $projectRoot ".venv")) {
        throw "Refusing to remove a non-project environment: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    & py -3.12 -m venv $venvPath
}

& $venvPython -m pip install --upgrade "pip==26.2.1" "setuptools==80.10.2" "wheel==0.48.0"
& $venvPython -m pip install --require-hashes -r (Join-Path $projectRoot "requirements.lock")
& $venvPython -m pip install --no-deps --editable $projectRoot
& $venvPython -m pip check
& $venvPython (Join-Path $projectRoot "scripts\check_runtime_env.py") --require-cuda

Write-Host ""
Write-Host "VS Code project environment is ready: $venvPython" -ForegroundColor Green
Write-Host "Reload the VS Code window; the interpreter is pinned to .venv."
