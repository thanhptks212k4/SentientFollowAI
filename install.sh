#!/bin/bash

# SentientFollowAI Installation Script
# For Raspberry Pi OS (Debian-based systems)

set -e  # Exit on any error

echo "🤖 SentientFollowAI Installation Script"
echo "======================================"

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This script is optimized for Raspberry Pi"
    echo "   It may work on other systems but is not guaranteed"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    libusb-1.0-0-dev \
    libudev-dev

# Enable UART for ESP32 communication
echo "🔌 Configuring UART..."
if ! grep -q "enable_uart=1" /boot/config.txt; then
    echo "enable_uart=1" | sudo tee -a /boot/config.txt
    echo "✅ UART enabled in /boot/config.txt"
else
    echo "✅ UART already enabled"
fi

# Disable serial console (if enabled)
sudo systemctl disable serial-getty@ttyAMA0.service 2>/dev/null || true

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Check if model exists
if [ ! -f "models/yolov8n_person_224_int8.onnx" ]; then
    echo "🧠 YOLOv8n model not found. Downloading..."
    mkdir -p models
    
    # Try to export model
    if python export_yolo_224_int8.py; then
        echo "✅ Model exported successfully"
    else
        echo "❌ Model export failed. You may need to download it manually."
    fi
fi

# Set up permissions for camera and UART
echo "🔐 Setting up device permissions..."
sudo usermod -a -G dialout $USER
sudo usermod -a -G video $USER

# Create systemd service (optional)
read -p "📋 Create systemd service for auto-start? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > sentientfollow.service << EOF
[Unit]
Description=SentientFollowAI Person Following Robot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo mv sentientfollow.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable sentientfollow.service
    echo "✅ Systemd service created and enabled"
    echo "   Start with: sudo systemctl start sentientfollow"
    echo "   Stop with:  sudo systemctl stop sentientfollow"
    echo "   Logs with:  sudo journalctl -u sentientfollow -f"
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Reboot your Raspberry Pi: sudo reboot"
echo "   2. Connect your Orbbec Astra camera"
echo "   3. Connect ESP32 to /dev/ttyAMA0"
echo "   4. Run the system: cd src && python main.py"
echo ""
echo "🔧 Configuration:"
echo "   - Edit src/config.py to tune parameters"
echo "   - Check README.md for detailed documentation"
echo ""
echo "🆘 Troubleshooting:"
echo "   - Test camera: python -c 'from src.astra_camera import test_astra_camera; test_astra_camera()'"
echo "   - Test config: python src/config.py"
echo "   - Check UART: ls -la /dev/ttyAMA0"
echo ""

# Reminder about reboot
echo "⚠️  IMPORTANT: Reboot required for UART changes to take effect!"
read -p "Reboot now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi