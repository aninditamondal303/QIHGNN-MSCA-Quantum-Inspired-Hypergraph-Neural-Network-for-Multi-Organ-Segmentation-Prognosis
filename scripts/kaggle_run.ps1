# Run the pipeline on Kaggle from this laptop:
#   powershell -ExecutionPolicy Bypass -File scripts\kaggle_run.ps1 -Mode probe   # CPU, minutes: check the data loaders
#   powershell -ExecutionPolicy Bypass -File scripts\kaggle_run.ps1 -Mode train   # GPU, hours: train, evaluate, export
# Needs Kaggle credentials: %USERPROFILE%\.kaggle\access_token + username (new tokens), or kaggle.json (legacy).
# Uploads ml/ as a private Kaggle dataset, pushes a private kernel with the three datasets attached,
# waits for it to finish and downloads its output to kaggle_output\. Progress goes to kaggle.log.
param([ValidateSet("probe", "probe-train", "train")][string]$Mode = "probe",
      [ValidateSet("main", "screen", "organs")][string]$Job = "main",
      [switch]$WaitOnly)   # skip upload and push: keep waiting for the kernel already running, then download its output
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$suffix = $(if ($Job -eq "main") { "" } else { "_$Job" })
$log = Join-Path $root "kaggle$suffix.log"
function Log($m) { $line = "$(Get-Date -Format s) $m"; Write-Host $line; Add-Content -Path $log -Value $line }
function Run($exe, [string[]]$a) { $o = & cmd /c "`"$exe`" $($a -join ' ') 2>&1"; $o | ForEach-Object { Add-Content -Path $log -Value "    $_" }; return ($o -join "`n") }

# Run the Kaggle CLI as "python -m kaggle": Windows Smart App Control / Device Guard can block the kaggle.exe launcher.
$kaggle = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $kaggle)) { throw "Run scripts\setup_windows.ps1 first." }
$cfgDir = Join-Path $env:USERPROFILE ".kaggle"
$cfg = Join-Path $cfgDir "kaggle.json"
$tokenFile = Join-Path $cfgDir "access_token"
$userFile = Join-Path $cfgDir "username"
# Notepad often saves as access_token.txt / username.txt, or into %USERPROFILE%\Documents\.kaggle: fix that up.
foreach ($alt in @($cfgDir, (Join-Path $env:USERPROFILE "Documents\.kaggle"), (Join-Path $env:USERPROFILE "OneDrive\Documents\.kaggle"))) {
    foreach ($n in @("access_token", "username")) {
        $dst = Join-Path $cfgDir $n
        if (Test-Path $dst) { continue }
        foreach ($cand in @((Join-Path $alt $n), (Join-Path $alt "$n.txt"))) {
            if ((Test-Path $cand) -and ($cand -ne $dst)) {
                New-Item -ItemType Directory -Force $cfgDir | Out-Null
                Copy-Item $cand $dst
                break
            }
        }
    }
}
# Strip spaces, newlines and any byte-order mark that Notepad may have added (contents are never printed).
foreach ($f in @($tokenFile, $userFile)) {
    if (Test-Path $f) {
        $v = (Get-Content $f -Raw).Trim().Trim([char]0xFEFF).Trim()
        [System.IO.File]::WriteAllText($f, $v, [System.Text.Encoding]::ASCII)
    }
}
if (-not (Test-Path $cfg) -and -not (Test-Path $tokenFile)) {
    $dl = Get-ChildItem (Join-Path $env:USERPROFILE "Downloads") -Filter "kaggle*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($dl) { New-Item -ItemType Directory -Force $cfgDir | Out-Null; Copy-Item $dl.FullName $cfg }
}
if (Test-Path $cfg) {
    $user = (Get-Content $cfg -Raw | ConvertFrom-Json).username
} elseif (Test-Path $tokenFile) {
    # New-style Kaggle API token (KGAT_...): the CLI reads ~/.kaggle/access_token itself; the username is stored next to it.
    if (Test-Path $userFile) { $user = (Get-Content $userFile -Raw).Trim() } elseif ($env:KAGGLE_USERNAME) { $user = $env:KAGGLE_USERNAME }
    if (-not $user) { throw "Found access_token but no username. Save your Kaggle username to $userFile" }
} else {
    $seen = @()
    foreach ($d in @($cfgDir, (Join-Path $env:USERPROFILE "Documents\.kaggle"))) { if (Test-Path $d) { $seen += Get-ChildItem $d -Force | ForEach-Object { $_.FullName } } }
    Write-Host "Files found in .kaggle folders (names only): $($seen -join ', ')"
    throw "No Kaggle credentials. Save your API token to $tokenFile (and your username to $userFile), then run this again."
}
Log "==== job=$Job mode=$Mode kaggle user=$user"

if (-not $WaitOnly) {
# ---------------------------------------------------------------- code dataset
$build = Join-Path $root "kaggle_build$suffix"
if (Test-Path $build) { Remove-Item $build -Recurse -Force }
$code = Join-Path $build "code"
New-Item -ItemType Directory -Force $code | Out-Null
robocopy (Join-Path $root "ml") (Join-Path $code "ml") /E /XD __pycache__ tests .pytest_cache /NFL /NDL /NJH /NJS /NP | Out-Null
$dsId = "$user/qihgnn-msca-code"
@{ title = "qihgnn-msca-code"; id = $dsId; licenses = @(@{ name = "CC0-1.0" }) } | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $code "dataset-metadata.json") -Encoding ascii
$status = Run $kaggle @("-m", "kaggle", "datasets", "status", $dsId)
if ($status -match "ready|pending") {
    Log "Updating code dataset $dsId"
    Run $kaggle @("-m", "kaggle", "datasets", "version", "-p", "`"$code`"", "-m", "`"update $(Get-Date -Format s)`"", "--dir-mode", "zip") | Out-Null
} else {
    Log "Creating private code dataset $dsId"
    Run $kaggle @("-m", "kaggle", "datasets", "create", "-p", "`"$code`"", "--dir-mode", "zip") | Out-Null
}
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 15
    $s = Run $kaggle @("-m", "kaggle", "datasets", "status", $dsId)
    if ($s -match "ready") { break }
}
Log "Code dataset status: $s"
Start-Sleep -Seconds 20   # let the new version propagate before the kernel mounts it

# ---------------------------------------------------------------- kernel
$kern = Join-Path $build "kernel"
New-Item -ItemType Directory -Force $kern | Out-Null
$entry = $(if ($Job -eq "main") { "kaggle\run.py" } else { "kaggle\$Job.py" })
(Get-Content (Join-Path $root $entry) -Raw).Replace("__MODE__", $Mode) | Set-Content (Join-Path $kern "run.py") -Encoding ascii
$kid = $(if ($Job -eq "main") { "$user/qihgnn-msca-train" } else { "$user/qihgnn-msca-$Job" })
$sources = @($dsId, "awsaf49/brats2020-training-data", "harshwardhanbhangale/lits-dataset", "balajikartheek/cervical-cancer-segmentation")
if ($Job -eq "screen") {
    $sources += @("masoudnickparvar/brain-tumor-mri-dataset", "navoneel/brain-mri-images-for-brain-tumor-detection", "nikhilroxtomar/brain-tumor-segmentation")
}
if ($Job -eq "organs") {
    $sources += @("andrewmvd/liver-tumor-segmentation", "andrewmvd/liver-tumor-segmentation-part-2")  # original LiTS CT volumes (NIfTI, Hounsfield units)
}
$meta = [ordered]@{
    id                  = $kid
    title               = $kid.Split("/")[1]
    code_file           = "run.py"
    language            = "python"
    kernel_type         = "script"
    is_private          = $true
    enable_gpu          = ($Mode -eq "train")
    enable_internet     = $true
    dataset_sources     = $sources
    competition_sources = @()
    kernel_sources      = @()
}
if ($Mode -eq "train") { $meta["machine_shape"] = "NvidiaTeslaT4" }   # T4 x2 (16 GB each); the P100 is retired
$meta | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $kern "kernel-metadata.json") -Encoding ascii
Log "Pushing kernel $kid (gpu=$($Mode -eq 'train'))"
$push = Run $kaggle @("-m", "kaggle", "kernels", "push", "-p", "`"$kern`"")
Log $push
if ($push -match "error|Error|403|401") { throw "kernel push failed: $push" }

}
$kid = $(if ($Job -eq "main") { "$user/qihgnn-msca-train" } else { "$user/qihgnn-msca-$Job" })

# ---------------------------------------------------------------- wait
# Only Kaggle's own status values end the wait; network errors while polling are logged and retried.
$deadline = (Get-Date).AddHours(12.5)
do {
    Start-Sleep -Seconds 60
    $s = Run $kaggle @("-m", "kaggle", "kernels", "status", $kid)
    Log "status: $($s.Trim())"
} until ($s -match "KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL)" -or (Get-Date) -gt $deadline)

# ---------------------------------------------------------------- output
$outDir = Join-Path $root $(if ($Job -eq "main") { "kaggle_output\$Mode" } else { "kaggle_output\$Job-$Mode" })
if (Test-Path $outDir) { Remove-Item $outDir -Recurse -Force }
New-Item -ItemType Directory -Force $outDir | Out-Null
Run $kaggle @("-m", "kaggle", "kernels", "output", $kid, "-p", "`"$outDir`"") | Out-Null
Log "Output downloaded to $outDir"
Get-ChildItem $outDir -Recurse | Select-Object -First 40 | ForEach-Object { Log "  $($_.FullName.Substring($outDir.Length))" }
if ($s -match "KernelWorkerStatus\.COMPLETE") { Log "==== DONE ($Mode)" } else { Log "==== FAILED ($Mode): see the .log file in $outDir" }
