param(
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Mode = 'auto',
    [int]$Port = 8080,
    [string]$VenvPath = '.venv'
)

$ErrorActionPreference = 'Stop'
$Python = Resolve-Path "$VenvPath\Scripts\python.exe"
$env:OCR_RUNTIME_PROFILE = 'win11'
$env:OCR_EXECUTION_MODE = $Mode
& $Python -m uvicorn app.main:app --host 0.0.0.0 --port $Port
