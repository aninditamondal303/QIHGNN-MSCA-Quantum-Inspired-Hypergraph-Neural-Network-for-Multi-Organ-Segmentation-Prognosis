# Turn on GitHub Pages (source: GitHub Actions) for this repository and start the site build.
#   powershell -ExecutionPolicy Bypass -File scripts\github_pages.ps1
# Uses the GitHub sign-in that Git Credential Manager already holds; the token is never printed or saved.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$out = "protocol=https`nhost=github.com`n`n" | git credential fill
$token = (($out | Where-Object { $_ -like "password=*" }) -replace "^password=", "")
if (-not $token) { throw "Git Credential Manager returned no GitHub credential. Run the 'Git: push to GitHub' task first." }
$h = @{ Authorization = "Bearer $token"; Accept = "application/vnd.github+json"; "User-Agent" = "qihgnn-pages"; "X-GitHub-Api-Version" = "2022-11-28" }

$remote = (git remote get-url origin) -replace "\.git$", ""
$slug = ($remote -split "github.com/")[1]
$repo = Invoke-RestMethod -Uri "https://api.github.com/repos/$slug" -Headers $h   # follows renames
$full = $repo.full_name
Write-Host "==> Repository: $full (private=$($repo.private))" -ForegroundColor Cyan
if ($repo.private) { throw "GitHub Pages on a free plan needs a public repository." }
if ($remote -ne "https://github.com/$full") { git remote set-url origin "https://github.com/$full.git"; Write-Host "Updated origin to https://github.com/$full.git" }

$pages = $null
try { $pages = Invoke-RestMethod -Uri "https://api.github.com/repos/$full/pages" -Headers $h } catch { }
if ($pages) {
    if ($pages.build_type -ne "workflow") {
        Invoke-RestMethod -Method Put -Uri "https://api.github.com/repos/$full/pages" -Headers $h -ContentType "application/json" -Body '{"build_type":"workflow"}' | Out-Null
        Write-Host "Switched Pages source to GitHub Actions."
    } else { Write-Host "Pages already uses GitHub Actions." }
} else {
    $pages = Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$full/pages" -Headers $h -ContentType "application/json" -Body '{"build_type":"workflow"}'
    Write-Host "Enabled GitHub Pages (source: GitHub Actions)."
}
$pages = Invoke-RestMethod -Uri "https://api.github.com/repos/$full/pages" -Headers $h

Write-Host "==> Starting the 'pages' workflow on main" -ForegroundColor Cyan
Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$full/actions/workflows/pages.yml/dispatches" -Headers $h -ContentType "application/json" -Body '{"ref":"main"}' | Out-Null
Start-Sleep -Seconds 8
for ($i = 0; $i -lt 40; $i++) {
    $run = (Invoke-RestMethod -Uri "https://api.github.com/repos/$full/actions/workflows/pages.yml/runs?per_page=1" -Headers $h).workflow_runs[0]
    Write-Host "  $($run.status) $($run.conclusion)"
    if ($run.status -eq "completed") { break }
    Start-Sleep -Seconds 15
}
if ($run.conclusion -eq "success") {
    Write-Host "`nLive at $($pages.html_url)" -ForegroundColor Green
} else {
    Write-Host "`nWorkflow finished with '$($run.conclusion)': $($run.html_url)" -ForegroundColor Yellow
}
Set-Content -Path "pages.log" -Value "$(Get-Date -Format s) $($run.conclusion) $($pages.html_url) $($run.html_url)"
