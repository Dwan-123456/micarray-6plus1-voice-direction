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
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
& $pythonw -m gui.dev_test_ui.app
if ($LASTEXITCODE -ne 0) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        "Development Test UI exited during startup (exit code $LASTEXITCODE). Run the environment installer again and keep any error details it displays.",
        "Development Test UI startup failed",
        "OK",
        "Error"
    ) | Out-Null
    exit $LASTEXITCODE
}
