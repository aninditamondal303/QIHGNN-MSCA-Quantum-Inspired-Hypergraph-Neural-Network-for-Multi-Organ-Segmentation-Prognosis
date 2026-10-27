# Runs everything that proves the project works, in order, and stops at the first failure:
#   powershell -ExecutionPolicy Bypass -File scripts\run_checks.ps1
# Steps: numerical verification, all Python tests (including the PyTorch model tests),
# a smoke training run on synthetic phantoms, evaluation of that checkpoint, site data export,
# website unit tests, typecheck, build and the browser end-to-end test.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run scripts\setup_windows.ps1 first." }

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

Step "1/8 Numerical verification"
Push-Location ml; & $py -m qihgnn.verify; Check "verification"; Pop-Location

Step "2/8 Python tests (verification, data, PyTorch model, API)"
& $py -m pytest; Check "pytest"

Step "3/8 Smoke training on synthetic phantoms (a few minutes on CPU)"
Push-Location ml; & $py -m qihgnn.train --config configs/smoke.json; Check "smoke training"; Pop-Location

Step "4/8 Evaluate the smoke checkpoint (pipeline check, not a performance result)"
$run = Get-Content (Join-Path $root "runs\LATEST")
$ckpt = Join-Path $root "runs\$run\last.pt"
Push-Location ml
& $py -m qihgnn.evaluate --checkpoint $ckpt --label smoke --out (Join-Path $root "results\smoke_metrics.json"); Check "evaluation"
Pop-Location

Step "5/8 Export data for the website"
Push-Location ml; & $py -m qihgnn.export_web; Check "export"; Pop-Location

Step "6/8 Website unit tests and typecheck"
Push-Location web
npm test; Check "web unit tests"
npm run typecheck; Check "typecheck"

Step "7/8 Build the website"
npm run build; Check "build"

Step "8/8 Browser end-to-end test"
npm run test:e2e; Check "e2e"
Pop-Location

Write-Host "`nAll checks passed." -ForegroundColor Green
Write-Host "Start the site:  .venv\Scripts\python -m uvicorn api.app:app --port 8000   then open http://localhost:8000"
Write-Host "With the smoke checkpoint present, 'Run network on this case' in the Workbench now works."
