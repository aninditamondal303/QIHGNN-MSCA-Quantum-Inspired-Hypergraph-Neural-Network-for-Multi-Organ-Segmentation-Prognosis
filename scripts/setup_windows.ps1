# One-time setup on Windows. Run from the repository root in a VS Code terminal:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Add -Cuda to install the NVIDIA build of PyTorch instead of the CPU build.
param([switch]$Cuda)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

Step "Checking Python and Node"
py -3 --version
node --version
npm --version

if (-not (Test-Path ".venv")) {
    Step "Creating virtual environment .venv"
    py -3 -m venv .venv
}
$py = ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Step "Installing PyTorch"
if ($Cuda) {
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
} else {
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}

Step "Installing Python requirements"
& $py -m pip install -r ml\requirements-dev.txt

Step "Installing website dependencies"
Push-Location web
npm install
npx playwright install chromium
Pop-Location

Write-Host "`nSetup complete. Next: powershell -ExecutionPolicy Bypass -File scripts\run_checks.ps1" -ForegroundColor Green
