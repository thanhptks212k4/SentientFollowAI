# Person Following Robot

Ultra-optimized single-thread person detection and navigation system for Raspberry Pi 5.

## Performance
- **15 FPS** at **25% CPU per core** on Raspberry Pi 5
- **224x224 INT8** quantized YOLOv5 model
- **Single-thread architecture** with absolute thread locking
- **Anti-ghosting** and **ID switch prevention**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Export optimized model
python export_person_only_224_int8.py

# Run the system
python src/main.py
```

## Controls
- **'q'**: Quit
- **'s'**: Show statistics

## File Structure

### Core Files
- **`src/main.py`** - Main application entry point with single-thread pipeline
- **`src/config.py`** - System configuration parameters and constants
- **`src/decision_maker.py`** - Visual servoing navigation with P-controller
- **`src/astra_camera.py`** - Camera interface for USB and Orbbec Astra cameras
- **`src/bytetrack_tracker.py`** - ByteTrack algorithm for robust person tracking

### Models
- **`models/yolov5nu_person_224_int8.onnx`** - Optimized INT8 quantized detection model

### Utilities
- **`export_person_only_224_int8.py`** - Model export and optimization script
- **`requirements.txt`** - Python dependencies

## Key Features

### Thread Optimization
- Environment variables set before all imports to prevent CPU spikes
- OpenCV single-thread processing: `cv2.setNumThreads(1)`
- ONNX Runtime single-thread inference

### Detection & Tracking
- **Anti-ghosting**: `IOU_THRESH=0.25` for aggressive NMS
- **ID switch prevention**: `MATCH_THRESH=0.85` for tolerant matching
- **Target locking**: "Mặt dày" algorithm never loses locked target

### Navigation
- **Visual servoing**: P-controller keeps person centered
- **Differential drive**: Simultaneous forward/backward + turn control
- **High sensitivity**: `DEADZONE_X=15px` for responsive steering
- **Safe following**: Intelligent distance control with backward capability

## Configuration

Key parameters in `src/config.py`:

```python
# Detection
INPUT_SIZE = 224          # Model input resolution
CONF_THRESH = 0.45       # Detection confidence
IOU_THRESH = 0.25        # NMS threshold (anti-ghosting)

# Tracking  
TRACK_THRESH = 0.50      # Track creation threshold
MATCH_THRESH = 0.85      # Track matching (anti-ID switch)
TRACK_BUFFER = 60        # Track memory (4s at 15fps)

# Navigation
SAFE_DISTANCE_MM = 1500  # Following distance (1.5m)
DEADZONE_X = 15          # Steering sensitivity (±15px)
KP_ANGULAR = 0.015       # Angular P-controller gain
KP_LINEAR = 0.0008       # Linear P-controller gain
```

## Hardware Requirements
- **Raspberry Pi 5** (8GB recommended)
- **USB camera** or **Orbbec Astra** depth camera
- **4GB+ RAM**, **8GB+ storage**

## Troubleshooting

### High CPU Usage
Check thread environment variables are set before imports in `main.py`

### Low FPS
- Verify INT8 model is loaded
- Check camera resolution (320x240 recommended)
- Monitor memory usage

### Detection Issues
- **False positives**: Increase `CONF_THRESH`
- **Missed detections**: Decrease `CONF_THRESH`  
- **ID switching**: Increase `MATCH_THRESH`