Write-Host "=== Optimistic Smart Glasses: Windows Companion Setup ===" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

if (!(Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\playwright.exe install chromium

if (!(Test-Path "models\speech_models\vosk-model-small-en-us-0.15")) {
    New-Item -ItemType Directory -Force -Path "models\speech_models" | Out-Null
    Write-Host "Downloading Vosk small English model..."
    Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile "vosk.zip"
    Expand-Archive -Path "vosk.zip" -DestinationPath "models\speech_models" -Force
    Remove-Item "vosk.zip" -Force
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To run locally on Windows: .\.venv\Scripts\python.exe main.py"
Write-Host "To run as Windows companion for Raspberry Pi:"
Write-Host '  $env:OPTIMISTIC_AUTH_TOKEN="<shared-secret>"'
Write-Host "  .\.venv\Scripts\python.exe integration\windows_client.py"

