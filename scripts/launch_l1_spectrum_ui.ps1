$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $projectRoot ".venv\Scripts\pythonw.exe"
$config = Join-Path $projectRoot "config\config.yaml"

if (-not (Test-Path -LiteralPath $pythonw)) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        "未找到项目专用 Python 环境：$pythonw",
        "L1 Spectrum UI 启动失败",
        "OK",
        "Error"
    ) | Out-Null
    exit 1
}

Set-Location -LiteralPath $projectRoot
$arguments = "-m gui.l1_spectrum_ui --config `"$config`""
Start-Process -FilePath $pythonw -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden
