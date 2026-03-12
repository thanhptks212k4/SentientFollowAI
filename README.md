# Person Detection with Astra Camera Distance Measurement

Real-time person detection and tracking system with accurate distance measurement using Orbbec Astra camera.

## Features

- **Real-time Person Detection**: YOLO-based person detection with ByteTrack tracking
- **Distance Measurement**: Accurate distance calculation using Astra camera depth data
- **High Performance**: Optimized C++ backend for 25+ FPS camera streaming
- **Auto Target Locking**: Automatically locks onto detected persons
- **Multi-person Tracking**: Tracks multiple people with unique IDs

## Quick Start

1. **Install Dependencies**:
   ```bash
   conda activate person_tracking
   pip install -r requirements.txt
   ```

2. **Build C++ Backend**:
   ```bash
   cd cam_depth/astra_cpp_driver/build
   make
   ```

3. **Run System**:
   ```bash
   python src/person_detector_with_tracking.py
   ```

## Controls

- **'q'**: Quit
- **'s'**: Show detailed statistics  
- **'d'**: Toggle depth window

## Performance

- **Camera FPS**: 25-30 FPS
- **AI FPS**: 15-20 FPS
- **Distance Accuracy**: ±2cm (0.5-5m range)
- **Inference Time**: ~45ms

## Output

Console displays real-time tracking info:
```
[LOCKED] Target ID:1 at ( 320, 240) | Dist: 1.38m | Tracks: 2 | Cam: 26.3 | AI: 16.4
```

- **Target ID**: Unique person identifier
- **Position**: (x, y) coordinates in frame
- **Distance**: Median distance from camera to person
- **Tracks**: Total number of people being tracked
- **Cam/AI FPS**: Performance metrics

## System Requirements

- **Camera**: Orbbec Astra Mini S or compatible
- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.9+
- **Dependencies**: OpenCV, ONNX Runtime, NumPy

## File Structure

```
├── src/
│   ├── person_detector_with_tracking.py  # Main detection system
│   ├── astra_camera.py                   # Astra camera wrapper
│   └── bytetrack_tracker.py              # Tracking algorithms
├── cam_depth/astra_cpp_driver/           # C++ backend for camera
└── models/yolov5nu_int8.onnx            # YOLO detection model
```