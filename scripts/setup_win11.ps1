param(
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Mode = 'auto',
    [string]$VenvPath = '.venv'
)

$ErrorActionPreference = 'Stop'

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python Launcher was not found. Install Python 3.11 x64 first.'
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or
    -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw 'ffmpeg/ffprobe were not found. Install FFmpeg and add it to PATH.'
}

if (-not (Test-Path "$VenvPath\Scripts\python.exe")) {
    py -3.11 -m venv $VenvPath
}
$Python = Resolve-Path "$VenvPath\Scripts\python.exe"

$SelectedMode = $Mode
if ($Mode -eq 'auto') {
    $SelectedMode = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { 'cuda' } else { 'cpu' }
}

Invoke-Python @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Python @('-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu', 'onnxruntime-directml')
if ($SelectedMode -eq 'cuda') {
    Invoke-Python @('-m', 'pip', 'install', '-r', 'requirements-win-gpu.txt')
} else {
    Invoke-Python @('-m', 'pip', 'install', '-r', 'requirements-cpu.txt')
}

$env:OCR_RUNTIME_PROFILE = 'win11'
$env:OCR_EXECUTION_MODE = $SelectedMode
Invoke-Python @('scripts\prewarm_win11.py')
Write-Host "Win11 OCR environment is ready. Selected mode: $SelectedMode"
