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
$process = Start-Process `
    -FilePath $pythonw `
    -ArgumentList @("-m", "gui.dev_test_ui.app") `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru
$exitedDuringStartup = $process.WaitForExit(1500)
if (-not $exitedDuringStartup) {
    $process.Dispose()
    exit 0
}
$exitCode = $process.ExitCode
$process.Dispose()
if ($exitCode -ne 0) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        "Development Test UI exited during startup (exit code $exitCode). Run the environment installer again and keep any error details it displays.",
        "Development Test UI startup failed",
        "OK",
        "Error"
    ) | Out-Null
    exit $exitCode
}
exit 0
