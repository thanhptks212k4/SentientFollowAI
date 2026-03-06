#!/bin/bash

# Quick setup script for Ubuntu/Raspberry Pi

echo "======================================================================="
echo "Person Detection System - Setup"
echo "======================================================================="

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "[4/5] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[5/5] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================================================="
echo "Setup complete!"
echo "======================================================================="
echo ""
echo "To run the detector:"
echo "  source venv/bin/activate"
echo "  python person_detector.py"
echo ""
