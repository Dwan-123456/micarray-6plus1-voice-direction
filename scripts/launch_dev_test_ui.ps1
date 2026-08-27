$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $projectRoot ".venv-v1.4\Scripts\pythonw.exe"

if (-not (Test-Path -LiteralPath $pythonw)) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        "未找到项目专用 Python 环境：$pythonw",
        "Development Test UI 启动失败",
        "OK",
        "Error"
    ) | Out-Null
    exit 1
}

Set-Location -LiteralPath $projectRoot
& $pythonw -m gui.dev_test_ui.app
