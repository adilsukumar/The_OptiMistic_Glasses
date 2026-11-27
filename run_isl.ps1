# The OptiMistic Glasses — Native PowerShell ISL Translator Runner
param (
    [switch]$Guide,
    [switch]$NoVoice,
    [int]$Camera = 0,
    [double]$Threshold = 0.55
)

$scriptPath = Join-Path $PSScriptRoot "isl_app.py"

$argsList = @()
if ($Guide) {
    $argsList += "--guide"
}
if ($NoVoice) {
    $argsList += "--no-voice"
}
$argsList += "--camera", $Camera
$argsList += "--threshold", $Threshold

Write-Host "Starting The OptiMistic Glasses ISL Live Translator on PowerShell..." -ForegroundColor Cyan
python $scriptPath @argsList
