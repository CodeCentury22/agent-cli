$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   🤖 Installing Agent CLI & Environment (Windows)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Verify/Install Python & Git
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Installing via winget..." -ForegroundColor Yellow
    winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git not found. Installing via winget..." -ForegroundColor Yellow
    winget install Git.Git --silent --accept-package-agreements --accept-source-agreements
}

# 2. Install 'uv'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv package manager..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path += ";$env:USERPROFILE\.cargo\bin;$env:USERPROFILE\.local\bin"
}

# 3. Install agent-cli
Write-Host "Installing agent-cli..." -ForegroundColor Green
uv tool install git+https://github.com/CodeCentury22/agent-cli.git --force

Write-Host "`n🎉 Installation complete! Restart your PowerShell session and run:" -ForegroundColor Green
Write-Host "    agent-cli" -ForegroundColor Cyan