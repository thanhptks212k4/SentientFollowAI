# SentientFollowAI - Person Detection System

Real-time person detection optimized for edge devices (Raspberry Pi, low-spec CPUs).

## 🚀 Quick Start

**📖 Hướng dẫn chi tiết:** Xem [HUONG_DAN_UBUNTU.md](HUONG_DAN_UBUNTU.md) hoặc [QUICK_START.md](QUICK_START.md)

### 1. Setup Virtual Environment

**⚠️ QUAN TRỌNG: Luôn dùng virtual environment để tránh conflict**

**Linux/Mac/Raspberry Pi:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Detection

**Basic Detection (no tracking):**
```bash
python src/person_detector.py
```

**With Person Tracking (ByteTrack):**
```bash
python src/person_detector_with_tracking.py
```

**Controls:**
- `q` - Quit
- `s` - Show statistics
- `l` - Lock target (tracking mode only)
- `u` - Unlock target (tracking mode only)

**Deactivate venv when done:**
```bash
deactivate
```

## 📁 Project Structure

```
SentientFollowAI/
├── src/
│   ├── person_detector.py              # Main detection script
│   ├── person_detector_with_tracking.py # Detection + ByteTrack tracking
│   └── bytetrack_tracker.py            # ByteTrack implementation
├── tools/
│   ├── quantize_model.py       # FP32 → INT8 conversion
│   ├── export_to_onnx.py       # PyTorch → ONNX export
│   ├── prune_yolo_classes.py   # 80 classes → 1 class
│   ├── optimized_nms.py        # Fast NMS implementations
│   └── test_pruned_model.py    # Model benchmarking
├── scripts/
│   ├── setup.sh                # Linux/Mac setup
│   └── setup.ps1               # Windows setup
├── models/
│   └── yolov5nu_int8.onnx      # INT8 quantized model
├── requirements.txt            # Python dependencies
├── HUONG_DAN_UBUNTU.md        # Ubuntu setup guide (Vietnamese)
├── QUICK_START.md             # Quick start guide
├── TRACKING_GUIDE.md          # Person tracking guide
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## ⚙️ Configuration

Edit `src/person_detector.py` to adjust settings:

```python
# Camera Settings
CAMERA_ID = 0           # Camera index (0, 1, 2...)
CAMERA_WIDTH = 320      # Resolution width
CAMERA_HEIGHT = 240     # Resolution height
CAMERA_FPS = 30         # Camera FPS

# Model Settings
INPUT_SIZE = 256        # Model input size (192, 256, 320, 416)
CONFIDENCE_THRESHOLD = 0.45  # Detection confidence threshold
IOU_THRESHOLD = 0.45    # NMS IoU threshold

# Performance Settings - CPU Stabilization
SKIP_FRAMES = 1         # Skip N frames (0=all, 1=every 2nd, 2=every 3rd)
TARGET_FPS = 15         # Target FPS (lower = more stable CPU)
ENABLE_CPU_STABILIZATION = True  # Keep CPU < 50%
```

### CPU Stabilization Explained

The detector uses **adaptive sleep** to keep CPU usage stable:

- **TARGET_FPS = 15**: Limits to 15 FPS, CPU sleeps between frames
- **Sleep time**: Automatically calculated to maintain target FPS
- **Result**: Stable CPU ~40-45% instead of fluctuating 50-80%

**How it works:**
```
Frame time = Inference (40ms) + Sleep (26ms) = 66ms = 15 FPS
CPU active: 40ms (60% of time)
CPU idle: 26ms (40% of time) → Stable CPU usage!
```

## 📊 Performance

| Platform | FPS | Latency | CPU Usage | Stability |
|----------|-----|---------|-----------|-----------|
| Desktop (Intel i5) | 15 | 40-45ms | 40-45% | ✅ Stable |
| Raspberry Pi 5 | 12-15 | 50-65ms | 45-50% | ✅ Stable |
| Raspberry Pi 4 | 8-10 | 80-100ms | 50-60% | ✅ Stable |

**Note:** With CPU stabilization enabled, CPU usage is stable and predictable.

## 🔧 Optimization Tips

### CPU Too High (> 50%)?
1. **Lower TARGET_FPS**: `TARGET_FPS = 12` (more sleep time)
2. **Increase frame skipping**: `SKIP_FRAMES = 2`
3. **Reduce resolution**: `CAMERA_WIDTH = 240, CAMERA_HEIGHT = 180`

### CPU Unstable (fluctuating)?
1. **Enable stabilization**: `ENABLE_CPU_STABILIZATION = True`
2. **Set lower target**: `TARGET_FPS = 12-15`
3. **Monitor sleep time**: Should be > 20ms

### Want Higher FPS?
1. **Increase TARGET_FPS**: `TARGET_FPS = 20`
2. **Disable stabilization**: `ENABLE_CPU_STABILIZATION = False`
3. **Accept higher CPU**: 60-70%

### Low FPS?
1. Reduce resolution: `CAMERA_WIDTH = 240, CAMERA_HEIGHT = 180`
2. Use smaller input: `INPUT_SIZE = 192`
3. Increase frame skipping: `SKIP_FRAMES = 2`

### Want Maximum Speed?
1. **Disable stabilization**: `ENABLE_CPU_STABILIZATION = False`
2. **Increase TARGET_FPS**: `TARGET_FPS = 20-25`
3. **Reduce frame skipping**: `SKIP_FRAMES = 0`
4. **Accept higher CPU**: 60-70%

## 🛠️ Advanced Tools

### 1. Model Quantization (FP32 → INT8)

```bash
python tools/quantize_model.py
```

**Benefit:** 2-3x faster inference, 75% smaller model

### 2. Class Pruning (80 classes → 1 class)

```bash
python tools/prune_yolo_classes.py --input yolov5nu.onnx --output yolov5nu_person.onnx
```

**Benefit:** 5-10% faster, smaller output tensor

### 3. Model Export (PyTorch → ONNX)

```bash
python tools/export_to_onnx.py
```

### 4. Benchmark Models

```bash
python tools/test_pruned_model.py
```

### 5. NMS Optimization Test

```bash
python tools/optimized_nms.py
```

## 🎯 Output Format

### Detection Output
```python
detections = [
    {
        'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
        'confidence': 0.85,         # Confidence score (0-1)
        'class_id': 0               # Person class ID
    }
]
```

### Tracking Output
```python
target = {
    'track_id': 5,              # Consistent ID across frames
    'bbox': [x1, y1, x2, y2],   # Bounding box
    'score': 0.85               # Confidence score
}

# Center point (for robot control)
x_center = (target['bbox'][0] + target['bbox'][2]) // 2
y_center = (target['bbox'][1] + target['bbox'][3]) // 2
```

## 🎯 Person Tracking (ByteTrack)

### Features

- **Consistent ID tracking**: Maintains same ID across frames
- **Robust to occlusions**: Continues tracking when person is temporarily hidden
- **Target locking**: Lock onto specific person and ignore others
- **Lightweight**: No ReID model needed (unlike DeepSORT)
- **Edge-friendly**: Only 2ms overhead on Raspberry Pi

### Quick Start

```python
from bytetrack_tracker import ByteTracker, TargetLocker

# Initialize tracker
tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
target_locker = TargetLocker(tracker)

# Lock onto first person (e.g., after voice command)
detections = detector.detect(frame)
target_locker.lock_target(detections)

# Track only locked target
while True:
    detections = detector.detect(frame)
    target = target_locker.update(detections)  # Returns only locked target
    
    if target:
        # Follow target
        x_c = (target['bbox'][0] + target['bbox'][2]) // 2
        y_c = (target['bbox'][1] + target['bbox'][3]) // 2
        robot.follow(x_c, y_c)
```

**See [TRACKING_GUIDE.md](TRACKING_GUIDE.md) for complete documentation.**

## 🤖 Robot Integration Example

### With Tracking (Recommended)

```python
import socket
import json
from bytetrack_tracker import ByteTracker, TargetLocker

# Initialize
tracker = ByteTracker()
target_locker = TargetLocker(tracker)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Lock target after voice command
if voice_command_received:
    detections = detector.detect(frame)
    target_locker.lock_target(detections)

# Track and follow
while True:
    detections = detector.detect(frame)
    target = target_locker.update(detections)
    
    if target:
        x_c = (target['bbox'][0] + target['bbox'][2]) // 2
        y_c = (target['bbox'][1] + target['bbox'][3]) // 2
        
        # Calculate error from center
        error_x = x_c - CAMERA_WIDTH // 2
        error_y = y_c - CAMERA_HEIGHT // 2
        
        # Send to robot controller
        data = json.dumps({
            'track_id': target['track_id'],
            'x': x_c,
            'y': y_c,
            'error_x': error_x,
            'error_y': error_y,
            'confidence': target['score']
        })
        sock.sendto(data.encode(), ('robot_ip', 5000))
```

### Without Tracking (Simple)

```python
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# In detection loop
if detections:
    person = detections[0]  # Track first person
    x_c = (person['bbox'][0] + person['bbox'][2]) // 2
    y_c = (person['bbox'][1] + person['bbox'][3]) // 2
    
    error_x = x_c - CAMERA_WIDTH // 2
    error_y = y_c - CAMERA_HEIGHT // 2
    
    data = json.dumps({
        'x': x_c,
        'y': y_c,
        'error_x': error_x,
        'error_y': error_y,
        'confidence': person['confidence']
    })
    sock.sendto(data.encode(), ('robot_ip', 5000))
```

## 🐛 Troubleshooting

### Camera Not Found

```python
# Try different camera IDs
CAMERA_ID = 1  # or 2, 3...

# Check available cameras (Linux)
ls /dev/video*

# Test camera
v4l2-ctl --list-devices
```

### Module Not Found

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Low Accuracy

```python
# Lower confidence threshold
CONFIDENCE_THRESHOLD = 0.35

# Process all frames
SKIP_FRAMES = 0
```

### Permission Denied (Linux)

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again
```

## 📦 Deployment on Raspberry Pi

### 1. System Setup

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install dependencies
sudo apt-get install python3-opencv libopencv-dev python3-pip

# Set CPU to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. Install Project

```bash
# Copy project to Pi
scp -r SentientFollowAI pi@raspberrypi:~/

# SSH to Pi
ssh pi@raspberrypi

# Navigate to project
cd ~/SentientFollowAI

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Detection

```bash
source venv/bin/activate
python src/person_detector.py
```

### 4. Auto-start on Boot (Optional)

```bash
# Create systemd service
sudo nano /etc/systemd/system/person-detector.service
```

Add:

```ini
[Unit]
Description=Person Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/SentientFollowAI
Environment="PATH=/home/pi/SentientFollowAI/venv/bin"
ExecStart=/home/pi/SentientFollowAI/venv/bin/python src/person_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable person-detector
sudo systemctl start person-detector

# Check status
sudo systemctl status person-detector

# View logs
sudo journalctl -u person-detector -f
```

## 📝 Development

### Code Structure

```python
# src/person_detector.py

class ThreadedCamera:
    """Zero-latency camera capture in separate thread"""
    
class PreProcessor:
    """Optimized image preprocessing for YOLO"""
    
class ONNXDetector:
    """ONNX Runtime inference with CPU optimization"""
    
def main():
    """Main detection loop with CPU stabilization"""
```

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Credits

- **YOLOv5**: Ultralytics
- **ONNX Runtime**: Microsoft
- **OpenCV**: OpenCV Foundation
- **NCNN**: Tencent

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review configuration settings
3. Test with different camera IDs
4. Monitor CPU/memory usage

---

**Ready to start?**

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 2. Run
python src/person_detector.py
```

**Happy detecting! 🎯**
