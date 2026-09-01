param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $projectRoot ".venv-v1.4"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

function Get-ProjectPython312 {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        & $launcher.Source -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @($launcher.Source, "-3.12")
        }
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:ProgramFiles "Python312\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            & $candidate -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @($candidate)
            }
        }
    }
    throw "Python 3.12 was not found. Run the environment installer from the project root."
}

if ($Recreate -and (Test-Path -LiteralPath $venvPath)) {
    $resolvedVenv = (Resolve-Path -LiteralPath $venvPath).Path
    if ($resolvedVenv -ne (Join-Path $projectRoot ".venv-v1.4")) {
        throw "Refusing to remove a non-project environment: $resolvedVenv"
    }
    Add-Type -AssemblyName Microsoft.VisualBasic
    [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory(
        $resolvedVenv,
        [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
        [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin
    )
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    $basePython = @(Get-ProjectPython312)
    $executable = $basePython[0]
    $arguments = @($basePython | Select-Object -Skip 1)
    & $executable @arguments -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "创建.venv-v1.4失败"
    }
}

& $venvPython -m pip install --upgrade "pip==26.2.1" "setuptools==80.10.2" "wheel==0.48.0"
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements-vscode.txt")
& $venvPython -m pip install --no-deps --editable $projectRoot
& $venvPython -m pip check
& $venvPython (Join-Path $projectRoot "scripts\check_runtime_env.py")

Write-Host ""
Write-Host "VS Code project environment is ready: $venvPython" -ForegroundColor Green
Write-Host "Reload the VS Code window; the interpreter is pinned to .venv-v1.4."
