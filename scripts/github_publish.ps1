# Create the private GitHub repository (if missing) and push this project to it.
#   powershell -ExecutionPolicy Bypass -File scripts\github_publish.ps1
# Uses Git Credential Manager: if you are not signed in yet, it opens GitHub in your browser to approve.
# The token is only held in memory by this script and is never printed or written to disk.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$owner = "aninditamondal303"
$name = "qihgnn-msca"

Write-Host "==> Getting GitHub credentials (approve in the browser window if one opens)" -ForegroundColor Cyan
$out = "protocol=https`nhost=github.com`n`n" | git credential fill
$token = (($out | Where-Object { $_ -like "password=*" }) -replace "^password=", "")
if (-not $token) { throw "Git Credential Manager returned no GitHub credential." }
$headers = @{ Authorization = "Bearer $token"; Accept = "application/vnd.github+json"; "User-Agent" = "qihgnn-publish" }

Write-Host "==> Creating private repository $owner/$name (if it does not exist)" -ForegroundColor Cyan
$exists = $true
try { Invoke-RestMethod -Uri "https://api.github.com/repos/$owner/$name" -Headers $headers | Out-Null } catch { $exists = $false }
if ($exists) {
    Write-Host "Repository already exists."
} else {
    $body = @{
        name        = $name
        private     = $true
        description = "Quantum-inspired hypergraph network with multi-scale cross-attention for partially supervised multi-organ segmentation and prognosis."
    } | ConvertTo-Json
    Invoke-RestMethod -Method Post -Uri "https://api.github.com/user/repos" -Headers $headers -Body $body -ContentType "application/json" | Out-Null
    Write-Host "Created https://github.com/$owner/$name (private)."
}

Write-Host "==> Pushing" -ForegroundColor Cyan
$remotes = git remote
if ($remotes -notcontains "origin") { git remote add origin "https://github.com/$owner/$name.git" }
git branch -M main
git push -u origin main
if ($LASTEXITCODE -ne 0) { throw "git push failed" }
Write-Host "`nPushed to https://github.com/$owner/$name" -ForegroundColor Green
