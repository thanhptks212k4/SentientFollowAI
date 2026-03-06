# Quick setup script for Windows

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Person Detection System - Setup" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan

# Check Python version
Write-Host ""
Write-Host "[1/5] Checking Python version..." -ForegroundColor Yellow
python --version

# Create virtual environment
Write-Host ""
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host ""
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "[4/5] Upgrading pip..." -ForegroundColor Yellow
pip install --upgrade pip

# Install dependencies
Write-Host ""
Write-Host "[5/5] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "=======================================================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the detector:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python person_detector.py"
Write-Host ""
