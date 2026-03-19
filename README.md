# 🤖 SentientFollowAI - Advanced Person Following Robot with Virtual Depth Radar

A sophisticated AI-powered person-following robot system built with **Hybrid Predictive Control**, **YOLOv8n object detection**, **Virtual Depth Radar**, and **real-time depth sensing**. Designed for Raspberry Pi with ESP32 motor control integration.

## 🎯 System Overview

This system creates an intelligent robot that can autonomously follow a person while maintaining safe distance, avoiding obstacles through multi-zone radar scanning, and handling complex scenarios like target loss and recovery. The robot operates at 20 FPS with ~80ms total latency from detection to motion.

### Key Features

- **🧠 Hybrid Predictive Controller**: Advanced control system with EMA filtering, full PID control, motion planning, and acceleration ramping
- **📡 Virtual Depth Radar**: Multi-zone obstacle detection with Left/Center/Right scanning regions for collision avoidance
- **👁️ YOLOv8n Object Detection**: Optimized INT8 quantized model (224x224) for real-time person detection
- **📏 Depth Sensing**: Orbbec Astra camera for accurate distance measurement and obstacle detection
- **🎯 Multi-Object Tracking**: ByteTrack algorithm for robust person tracking with ID locking
- **🔄 Target Recovery**: Intelligent inertial navigation when target is lost (0.5s search + emergency stop)
- **⚡ ESP32 Integration**: Dual ESP32 system (DevKit V1 → ESP-WROOM-32) via UART + ESP-NOW
- **🛡️ Advanced Safety**: Emergency stop, side avoidance, acceleration limiting, cornering speed reduction

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raspberry Pi  │    │     ESP32        │    │   ESP-WROOM-32  │
│                 │    │   DevKit V1      │    │                 │
│  ┌─────────────┐│    │                  │    │  ┌─────────────┐│
│  │ Main Script ││    │  UART Bridge     │    │  │Motor Control││
│  │             ││    │                  │    │  │             ││
│  │ YOLOv8n     ││    │  ESP-NOW         │    │  │ PWM Signals ││
│  │ Detection   ││    │  Communication   │    │  │             ││
│  │             ││    │                  │    │  └─────────────┘│
│  │ Hybrid PID  ││    │                  │    │                 │
│  │ Controller  ││    │                  │    │                 │
│  │             ││    │                  │    │                 │
│  │Virtual Radar││    │                  │    │                 │
│  │Multi-Zone   ││    │                  │    │                 │
│  │Obstacle Det.││    │                  │    │                 │
│  └─────────────┘│    │                  │    │                 │
│                 │    │                  │    │                 │
│  ┌─────────────┐│    │                  │    │                 │
│  │Astra Camera ││    │                  │    │                 │
│  │RGB + Depth  ││    │                  │    │                 │
│  └─────────────┘│    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        │                        │                        │
        ▼                        ▼                        ▼
   USB Camera              UART 115200              ESP-NOW Wireless
   RGB + Depth             "v,w\n" format           Motor Commands
   320x240 + Radar         3 decimal precision      PWM Control
```

## 🔄 System Workflow

### 1. **Initialization Phase**
```
Start → Camera Init → Model Loading → Controller Init → UART Setup → Radar Calibration → Ready
```

### 2. **Main Processing Loop (20 FPS)**
```
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN PROCESSING LOOP                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. Frame Capture & Synchronization                             │
│    ├── RGB Frame (320x240) from Astra Camera                   │
│    ├── Depth Frame (320x240) synchronized                      │
│    └── Frame Rate: 30 FPS camera → 20 FPS processing           │
│                                                                 │
│ 2. AI Detection Pipeline                                        │
│    ├── Preprocessing: Resize to 224x224, Normalize             │
│    ├── YOLOv8n Inference: Person detection (INT8 quantized)    │
│    ├── Post-processing: NMS, Confidence filtering (≥0.65)      │
│    └── Output: Bounding boxes with confidence scores           │
│                                                                 │
│ 3. Multi-Object Tracking                                       │
│    ├── ByteTrack: Associate detections with tracks             │
│    ├── Track Management: Create, update, delete tracks         │
│    ├── Target Selection: Lock onto highest confidence person   │
│    └── Output: Stable track ID with bbox coordinates           │
│                                                                 │
│ 4. Depth Processing & Distance Measurement                     │
│    ├── ROI Extraction: Get depth values in target bbox         │
│    ├── Depth Filtering: Remove invalid measurements (<100mm)   │
│    ├── Distance Calculation: Median depth in mm               │
│    └── Output: Target distance in millimeters                  │
│                                                                 │
│ 5. Virtual Depth Radar - Multi-Zone Obstacle Detection        │
│    ├── Zone Division: Left (0-33%), Center (34-66%), Right (67-100%) │
│    ├── Scan Region: 40%-60% height (torso level)              │
│    ├── Noise Filtering: Remove depths ≤100mm                  │
│    ├── EMA Smoothing: α=0.3 for stable readings              │
│    ├── Obstacle Detection:                                     │
│    │   ├── Left Zone: <500mm = Side obstacle                  │
│    │   ├── Center Zone: <400mm = Front collision risk         │
│    │   └── Right Zone: <500mm = Side obstacle                 │
│    └── Output: Obstacle flags + filtered distances            │
│                                                                 │
│ 6. Hybrid Predictive Control                                   │
│    ├── Signal Conditioning                                     │
│    │   ├── EMA Filtering: Noise reduction (α=0.2)             │
│    │   └── Target Prediction: Lead compensation (50ms ahead)  │
│    ├── Advanced PID Controller                                 │
│    │   ├── Error Calculation: Center offset & distance error  │
│    │   ├── Adaptive Gains: Distance-based Kp adjustment       │
│    │   ├── PID Computation: P + I + D terms with anti-windup  │
│    │   └── Output: Linear (v) and Angular (w) velocities      │
│    ├── Motion Profile & Safety                                 │
│    │   ├── Velocity Limiting: Max speed constraints            │
│    │   ├── Acceleration Ramping: 2.0 m/s² max acceleration    │
│    │   ├── Cornering Speed Reduction: Safety during turns     │
│    │   └── Deadzone Handling: Stop when close enough          │
│    └── Recovery Logic                                          │
│        ├── Target Lost Detection: No valid detections         │
│        ├── Inertial Navigation: Continue last motion (0.5s)   │
│        └── Emergency Stop: Stop after timeout                 │
│                                                                 │
│ 7. Virtual Radar Override Logic (Safety Priority System)      │
│    ├── Priority 1: Emergency Stop                             │
│    │   └── Front obstacle <400mm → v=0, w=0 immediately       │
│    ├── Priority 2: Side Avoidance                             │
│    │   ├── Turning right + right obstacle <500mm → w=0        │
│    │   └── Turning left + left obstacle <500mm → w=0          │
│    └── Priority 3: Normal Control                             │
│        └── No obstacles → Use PID commands                    │
│                                                                 │
│ 8. Motor Command Transmission                                   │
│    ├── UART Communication: Send "v,w\n" to ESP32              │
│    ├── ESP32 Bridge: Forward via ESP-NOW to WROOM-32          │
│    ├── Motor Control: Convert to PWM signals                   │
│    └── Robot Motion: Physical movement execution               │
│                                                                 │
│ 9. Visualization & Monitoring                                   │
│    ├── Bounding Box Rendering: Draw detection results          │
│    ├── Status Display: FPS, distance, control commands        │
│    ├── Target Highlighting: Show locked target in green       │
│    ├── Radar Status: L🟢/🔴 C🟢/🔴 R🟢/🔴 with distances    │
│    └── Real-time Display: OpenCV window output                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. **Control Flow States**

#### **Normal Tracking State**
```
Target Detected → Distance Measurement → Radar Scan → PID Control → 
Obstacle Override → Motor Commands → Motion
```

#### **Target Lost State**
```
No Detection → Recovery Mode → Inertial Navigation (0.5s) → Search Rotation → 
(Target Found: Resume Tracking) OR (Timeout: Emergency Stop)
```

#### **Obstacle Avoidance State**
```
Radar Detection → Override Logic → 
(Front: Emergency Stop) OR (Side: Cancel Turn) → Continue Straight
```

#### **Emergency State**
```
Critical Error → Emergency Stop → UART: "0.000,0.000" → Robot Stops
```

## 📁 Project Structure

```
SentientFollowAI/
├── 📁 src/                          # Core application code
│   ├── 🐍 main.py                   # Main application entry point
│   ├── 🐍 decision_maker.py         # Hybrid Predictive Controller + Virtual Depth Radar
│   ├── 🐍 astra_camera.py          # Orbbec Astra camera interface
│   ├── 🐍 bytetrack_tracker.py     # Multi-object tracking
│   └── 🐍 config.py                # System configuration
├── 📁 models/                       # AI models
│   └── 🧠 yolov8n_person_224_int8.onnx  # Optimized YOLOv8n model
├── 📁 cam_depth/                    # Camera depth processing
│   └── 📁 OrbbecSDK/               # Orbbec SDK and examples
├── 📁 backup_reactive_controller/   # Previous controller versions
├── 📁 Log/                         # System logs
├── 📄 export_yolo_224_int8.py      # Model export script
├── 📄 requirements.txt              # Python dependencies
├── 📄 install.sh                   # Installation script
├── 📄 README.md                     # This documentation
└── 📄 .env.example                  # Environment configuration template
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: Raspberry Pi 4/5, Orbbec Astra camera, ESP32 DevKit V1, ESP-WROOM-32
- **OS**: Raspberry Pi OS (64-bit recommended)
- **Python**: 3.8+ with pip

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/your-repo/SentientFollowAI.git
cd SentientFollowAI
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download Model** (if not included)
```bash
# YOLOv8n model should be in models/ directory
# If missing, run the export script:
python export_yolo_224_int8.py
```

4. **Hardware Setup**
```bash
# Enable UART on Raspberry Pi
sudo raspi-config
# Navigate to: Interface Options → Serial Port
# Enable serial interface, disable login shell
```

5. **Run System**
```bash
cd src
python main.py
```

### Controls

- **`q`**: Quit application
- **`s`**: Show detailed statistics (including radar data)
- **Mouse**: Click window to focus
- **ESC**: Emergency stop

### Statistics Display (`s` key)
```
Camera FPS: 29.8, AI FPS: 19.2, Tracks: 1
Commands: F=45, B=12, L=23, R=18
Radar: Scans=1250, Obstacles=89, Emergency=3
Overrides: Left=12, Right=15
```

## ⚙️ Configuration

### Core Parameters (`src/config.py`)

```python
# Camera Settings
CAMERA_WIDTH = 320          # RGB frame width
CAMERA_HEIGHT = 240         # RGB frame height  
CAMERA_FPS = 30            # Camera frame rate

# AI Model Settings
INPUT_SIZE = 224           # YOLOv8n input size
CONF_THRESH = 0.65         # Detection confidence threshold (increased for accuracy)
IOU_THRESH = 0.25          # NMS IoU threshold

# Control Parameters
SAFE_DISTANCE_MM: Final[int] = 1500    # Target following distance (mm)
MAX_LINEAR_SPEED = 0.449         # Maximum forward speed (m/s) - 449mm/s
MAX_ANGULAR_SPEED = 6.283        # Maximum rotation speed (rad/s) - 2π rad/s

# PID Controller Gains
KP_LINEAR = 0.0015         # Proportional gain for distance (adjusted for 449mm/s max)
KI_LINEAR = 0.0001         # Integral gain for distance
KD_LINEAR = 0.002          # Derivative gain for distance
KP_ANGULAR = 0.008         # Proportional gain for angle (adjusted for 2π rad/s max)
KI_ANGULAR = 0.0002        # Integral gain for angle
KD_ANGULAR = 0.005         # Derivative gain for angle

# Advanced Control
MAX_ACCEL = 1.0            # Maximum acceleration (m/s²) - adjusted for 449mm/s max
EMA_ALPHA = 0.2            # EMA filter coefficient

# Virtual Depth Radar Configuration
OBSTACLE_THRESHOLD_SIDE = 500    # Side obstacle threshold (mm)
OBSTACLE_THRESHOLD_FRONT = 400   # Front obstacle threshold (mm)
NOISE_THRESHOLD = 100            # Minimum valid depth (mm)
RADAR_SCAN_TOP = 0.40           # Top of scan region (40% of height)
RADAR_SCAN_BOTTOM = 0.60        # Bottom of scan region (60% of height)
RADAR_LEFT_BOUNDARY = 0.33      # Left zone boundary (33% of width)
RADAR_RIGHT_BOUNDARY = 0.67     # Right zone boundary (67% of width)
RADAR_EMA_ALPHA = 0.3           # EMA coefficient for depth smoothing
```

### UART Communication Format

```
Format: "v,w\n"
- v: Linear velocity (-0.449 to +0.449 m/s)
  * Positive: Forward (Tiến)
  * Negative: Backward (Lùi)
- w: Angular velocity (-6.283 to +6.283 rad/s)
  * Positive: Turn Left (Trái)  
  * Negative: Turn Right (Phải)
- Precision: 3 decimal places
- Baudrate: 115200
- Example: "0.225,-3.141\n" (Forward + Turn Right)
```

## 📡 Virtual Depth Radar System

The Virtual Depth Radar is a sophisticated obstacle detection system that provides multi-zone collision avoidance using the Astra camera's depth data.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VIRTUAL DEPTH RADAR                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Depth Frame (320x240)                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  ████████████████████████████████████████████████   │   │ ← 0-40% (Ceiling/Noise)
│  │  ████████████████████████████████████████████████   │   │
│  │  ┌─────────┬─────────────┬─────────────────────┐   │   │ ← 40% (Scan Start)
│  │  │  LEFT   │   CENTER    │       RIGHT         │   │   │
│  │  │ ZONE    │    ZONE     │       ZONE          │   │   │ ← Torso Level
│  │  │ 0-33%   │   34-66%    │      67-100%        │   │   │   Scanning
│  │  │         │             │                     │   │   │
│  │  └─────────┴─────────────┴─────────────────────┘   │   │ ← 60% (Scan End)
│  │  ████████████████████████████████████████████████   │   │
│  │  ████████████████████████████████████████████████   │   │ ← 60-100% (Floor/Noise)
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Processing Pipeline:                                       │
│  1. Zone Division → 2. Noise Filtering → 3. Min Distance   │
│  4. EMA Smoothing → 5. Obstacle Detection → 6. Override    │
└─────────────────────────────────────────────────────────────┘
```

### Zone Configuration

| Zone | Width Range | Purpose | Threshold |
|------|-------------|---------|-----------|
| **Left** | 0-33% (0-106px) | Left side obstacle detection | 500mm |
| **Center** | 34-66% (107-213px) | Front collision detection | 400mm |
| **Right** | 67-100% (214-320px) | Right side obstacle detection | 500mm |

### Scanning Parameters

```python
# Vertical Scanning Region (Torso Level)
RADAR_SCAN_TOP = 0.40      # 40% of frame height (96px at 240p)
RADAR_SCAN_BOTTOM = 0.60   # 60% of frame height (144px at 240p)

# Horizontal Zone Boundaries  
RADAR_LEFT_BOUNDARY = 0.33   # 33% of frame width (106px at 320p)
RADAR_RIGHT_BOUNDARY = 0.67  # 67% of frame width (213px at 320p)

# Filtering Parameters
NOISE_THRESHOLD = 100        # Remove depths ≤100mm (lens noise)
RADAR_EMA_ALPHA = 0.3       # EMA smoothing coefficient
```

### Processing Algorithm

#### 1. **Zone Division & ROI Extraction**
```python
def _scan_virtual_depth_radar(self, depth_frame):
    height, width = depth_frame.shape[:2]
    
    # Extract torso-level scanning region
    scan_top = int(height * 0.40)
    scan_bottom = int(height * 0.60)
    scan_region = depth_frame[scan_top:scan_bottom, :]
    
    # Divide into 3 zones
    left_boundary = int(width * 0.33)
    right_boundary = int(width * 0.67)
    
    left_zone = scan_region[:, :left_boundary]
    center_zone = scan_region[:, left_boundary:right_boundary]  
    right_zone = scan_region[:, right_boundary:]
```

#### 2. **Noise Filtering & Distance Calculation**
```python
    # Filter noise and find minimum distance per zone
    valid_left = left_zone[left_zone > NOISE_THRESHOLD]
    min_depth_left = np.min(valid_left) if len(valid_left) > 0 else ∞
    
    # Repeat for center and right zones...
```

#### 3. **EMA Smoothing**
```python
    # Apply exponential moving average for stable readings
    filtered_depth_left = (
        α × min_depth_left + (1-α) × prev_filtered_depth_left
    )
    # α = 0.3 (30% new, 70% previous)
```

#### 4. **Obstacle Detection**
```python
    # Set obstacle flags based on thresholds
    obstacle_left = filtered_depth_left < 500    # mm
    obstacle_center = filtered_depth_center < 400  # mm  
    obstacle_right = filtered_depth_right < 500   # mm
```

### Override Logic (Safety Priority System)

The radar system implements a 3-level priority system that can override the PID controller:

#### **Priority 1: Emergency Stop** 🚨
```python
if obstacle_center:  # Front obstacle < 400mm
    v, w = 0.0, 0.0  # Immediate stop
    print("[RADAR] 🚨 PHANH KHẨN CẤP - Vật cản phía trước")
```

#### **Priority 2: Side Avoidance** ⚠️
```python
elif control_w > 0 and obstacle_right:  # Turning right but blocked
    w = 0.0  # Cancel turn, keep straight motion
    print("[RADAR] ⚠️ Vướng sườn PHẢI - Đang đi thẳng vượt vật cản")
    
elif control_w < 0 and obstacle_left:   # Turning left but blocked  
    w = 0.0  # Cancel turn, keep straight motion
    print("[RADAR] ⚠️ Vướng sườn TRÁI - Đang đi thẳng vượt vật cản")
```

#### **Priority 3: Normal Control** ✅
```python
else:
    # No obstacles detected - use PID controller outputs
    final_v, final_w = pid_v, pid_w
```

### Real-Time Display

The radar status is displayed in real-time on the video feed:

```
Radar: L🟢450 C🟢∞ R🔴380mm
       │  │   │ │  │  │
       │  │   │ │  │  └── Distance in mm
       │  │   │ │  └────── Right zone status  
       │  │   │ └───────── Center zone distance
       │  │   └─────────── Center zone status
       │  └─────────────── Left zone distance
       └───────────────── Left zone status (🟢=safe, 🔴=obstacle)
```

### Performance Characteristics

| Metric | Value | Description |
|--------|-------|-------------|
| **Scan Rate** | 20 FPS | Synchronized with main loop |
| **Processing Time** | <0.06ms | Per radar scan |
| **CPU Usage** | <0.1% | Minimal overhead |
| **Memory Usage** | ~50KB | Radar state + buffers |
| **Response Time** | <50ms | 1 frame delay |
| **Detection Range** | 100mm - 5000mm | Valid depth range |
| **Zone Coverage** | 3 zones | Left/Center/Right |
| **Scan Region** | 20% height | Torso-level (40%-60%) |

### Tuning Parameters

#### **Obstacle Thresholds**
```python
OBSTACLE_THRESHOLD_FRONT = 400   # Decrease for earlier front detection
OBSTACLE_THRESHOLD_SIDE = 500    # Decrease for earlier side detection
```

#### **Scanning Region**
```python
RADAR_SCAN_TOP = 0.35           # Lower = scan higher objects
RADAR_SCAN_BOTTOM = 0.65        # Higher = scan lower objects
```

#### **EMA Smoothing**
```python
RADAR_EMA_ALPHA = 0.3           # Higher = more responsive
                                # Lower = more stable
```

#### **Zone Boundaries**
```python
RADAR_LEFT_BOUNDARY = 0.30      # Adjust left zone width
RADAR_RIGHT_BOUNDARY = 0.70     # Adjust right zone width
```

### Common Use Cases

#### **Narrow Doorway Navigation**
- Side zones detect door frames
- Robot goes straight through center
- Prevents collision during turns

#### **Furniture Avoidance**  
- Center zone detects tables/chairs
- Emergency stop prevents collision
- Robot waits for clear path

#### **Corner Navigation**
- Side zones detect walls during turns
- Override cancels turn command
- Robot continues straight until clear

#### **Dynamic Obstacle Avoidance**
- Moving objects trigger override
- Smooth transition back to normal control
- No jerky stop-start behavior

### Troubleshooting

#### **False Positives**
- Increase `NOISE_THRESHOLD` (100mm → 150mm)
- Adjust scanning region height
- Check camera mounting angle

#### **Missed Obstacles**
- Decrease obstacle thresholds
- Expand scanning region
- Reduce EMA smoothing

#### **Erratic Behavior**
- Increase EMA smoothing (0.3 → 0.5)
- Check depth frame quality
- Verify zone boundaries

## 🧠 Hybrid Predictive Controller Details

### 1. **Signal Conditioning & State Estimation**

#### EMA Filtering
```python
filtered_error = α × new_error + (1-α) × prev_filtered_error
# α = 0.2 (20% new, 80% previous)
```

#### Target Prediction (Lead Compensation)
```python
velocity = (current_position - previous_position) / dt
predicted_position = current_position + velocity × lead_time
# lead_time = 50ms (1 frame ahead)
```

### 2. **Advanced PID Controller**

#### PID Formula
```python
output = Kp×error + Ki×∫error×dt + Kd×(error-prev_error)/dt
```

#### Adaptive Gains
```python
if distance < decel_zone:
    Kp_adaptive = Kp_base × (distance / decel_zone)
    # Reduces gain when approaching target
```

#### Anti-Windup Protection
```python
integral = clamp(integral, -max_integral, +max_integral)
# Prevents integral term from growing too large
```

### 3. **Motion Profile & Safety**

#### Acceleration Limiting
```python
max_delta_v = MAX_ACCEL × dt  # 1.0 m/s² × 0.05s = 0.05 m/s per frame
if |target_v - current_v| > max_delta_v:
    limited_v = current_v + sign(delta_v) × max_delta_v
```

#### Cornering Speed Reduction
```python
if |angular_velocity| > threshold:
    speed_reduction = |angular_velocity| / 6.283 × 0.5  # Based on 2π max
    linear_velocity *= (1 - speed_reduction)
```

### 4. **Recovery & Lost Target Logic**

#### Inertial Navigation
```python
if frames_without_target <= 10:  # 0.5 seconds at 20 FPS
    recovery_w = last_w × 0.3    # Reduce rotation speed by 70%
    recovery_v = 0.0             # Stop forward motion
else:
    emergency_stop()             # Complete stop
```

### 5. **Virtual Depth Radar System**

#### Multi-Zone Scanning
```python
# Zone Division (320x240 depth frame)
left_zone = depth_frame[:, 0:106]           # 0-33% width
center_zone = depth_frame[:, 107:213]       # 34-66% width  
right_zone = depth_frame[:, 214:320]        # 67-100% width

# Scan Region (Torso Level)
scan_top = int(height × 0.40)     # 40% height
scan_bottom = int(height × 0.60)  # 60% height
scan_region = depth_frame[scan_top:scan_bottom, :]
```

#### Obstacle Detection Logic
```python
# Filter noise and find minimum distance per zone
valid_depths = zone[zone > NOISE_THRESHOLD]  # Remove ≤100mm
min_depth = np.min(valid_depths) if len(valid_depths) > 0 else ∞

# EMA Smoothing
filtered_depth = α × min_depth + (1-α) × prev_filtered_depth

# Obstacle Flags
obstacle_left = filtered_depth_left < 500mm
obstacle_center = filtered_depth_center < 400mm  
obstacle_right = filtered_depth_right < 500mm
```

#### Override Logic (Safety Priority System)
```python
# Priority 1: Emergency Stop
if obstacle_center:
    v, w = 0.0, 0.0  # Immediate stop
    
# Priority 2: Side Avoidance  
elif turning_right and obstacle_right:
    w = 0.0  # Cancel turn, keep straight motion
elif turning_left and obstacle_left:
    w = 0.0  # Cancel turn, keep straight motion
    
# Priority 3: Normal Control
else:
    # Use PID controller outputs
```

## 🔧 Hardware Integration

### ESP32 DevKit V1 (UART Bridge)
```cpp
// Receives from Pi via UART
Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2);

// Forwards to WROOM-32 via ESP-NOW
esp_now_send(peerMAC, (uint8_t*)command, strlen(command));
```

### ESP-WROOM-32 (Motor Controller)
```cpp
// Receives motor commands via ESP-NOW
void onDataRecv(const uint8_t *data, int len) {
    // Parse "v,w\n" format
    // Convert to PWM signals
    // Drive motors
}
```

### Motor Control Mapping

#### **Velocity Direction Convention**
| Parameter | Positive (+) | Negative (-) | Zero (0) |
|-----------|-------------|-------------|----------|
| **Linear Velocity (v)** | Tiến (Forward) | Lùi (Backward) | Dừng (Stop) |
| **Angular Velocity (w)** | Trái (Turn Left) | Phải (Turn Right) | Thẳng (Straight) |

#### **Speed Ranges**
```
Linear Velocity (v):
  +0.449 m/s → Forward (Tiến) - Maximum speed
  -0.449 m/s → Backward (Lùi) - Maximum speed
   0.0 m/s → Stop

Angular Velocity (w):
  +6.283 rad/s → Turn Left (Trái) - 2π rad/s maximum
  -6.283 rad/s → Turn Right (Phải) - 2π rad/s maximum
   0.0 rad/s → Straight
```

#### **Example Commands**
```
"0.225,0.000\n"    → Tiến với tốc độ 50% (Forward 50%)
"-0.225,0.000\n"   → Lùi với tốc độ 50% (Backward 50%)
"0.000,3.141\n"    → Xoay trái π rad/s (Turn Left π rad/s)
"0.000,-3.141\n"   → Xoay phải π rad/s (Turn Right π rad/s)
"0.225,1.571\n"    → Tiến + xoay trái (Forward + Turn Left)
"0.000,0.000\n"    → Dừng hoàn toàn (Complete Stop)
```

## 📊 Performance Characteristics

### System Performance
- **AI Processing**: ~23ms per frame (YOLOv8n INT8 224x224)
- **Control Loop**: 20 FPS (50ms cycle time)
- **UART Latency**: ~5ms (115200 baud)
- **Radar Scanning**: ~0.06ms per scan (<0.1% CPU)
- **Total Latency**: ~80ms (detection to motion)
- **Max Linear Speed**: 449mm/s (0.449 m/s)
- **Max Angular Speed**: 2π rad/s (6.283 rad/s)

### Accuracy Metrics
- **Detection Range**: 0.5m - 5.0m
- **Distance Accuracy**: ±50mm (depth sensor)
- **Angular Accuracy**: ±2° (camera resolution)
- **Tracking Stability**: >95% (ByteTrack)
- **Obstacle Detection**: 400mm front, 500mm sides

### Resource Usage
- **CPU**: ~60% (Raspberry Pi 4)
- **Memory**: ~500MB RAM
- **Storage**: ~2GB (with models)
- **Power**: ~15W total system

### Virtual Depth Radar Performance
- **Scan Rate**: 20 FPS (synchronized with main loop)
- **Processing Time**: <0.06ms per scan
- **Zone Coverage**: 3 zones × torso-level scanning
- **Noise Filtering**: Removes depths ≤100mm
- **EMA Smoothing**: α=0.3 for stable readings
- **Override Response**: <50ms (1 frame delay)

## 🛠️ Troubleshooting

### Common Issues

#### 1. **Camera Not Detected**
```bash
# Check camera connection
lsusb | grep -i orbbec

# Test camera access
python -c "from src.astra_camera import test_astra_camera; test_astra_camera()"
```

#### 2. **UART Connection Failed**
```bash
# Check UART status
ls -la /dev/ttyAMA0

# Enable UART
sudo raspi-config
# Interface Options → Serial Port → Enable
```

#### 3. **Model Loading Error**
```bash
# Verify model file
ls -la models/yolov8n_person_224_int8.onnx

# Test model loading
python -c "import onnxruntime; print('ONNX Runtime OK')"
```

#### 4. **Poor Detection Performance**
- Adjust `CONF_THRESH` in config.py (0.65 default, lower = more detections)
- Check lighting conditions (avoid backlighting)
- Ensure person is within 0.5-5m range
- Verify YOLOv8n model is properly loaded

#### 5. **Erratic Robot Movement**
- Tune PID gains in config.py
- Check `MAX_ACCEL` setting (lower = smoother)
- Verify UART communication integrity
- Check radar override logic

#### 6. **Radar False Positives**
- Adjust `OBSTACLE_THRESHOLD_SIDE` (500mm default)
- Adjust `OBSTACLE_THRESHOLD_FRONT` (400mm default)
- Increase `NOISE_THRESHOLD` if floor/ceiling interference
- Tune `RADAR_EMA_ALPHA` for more/less smoothing

#### 7. **Radar Not Working**
- Verify depth frame is not None
- Check Astra camera depth stream
- Ensure proper camera positioning (torso level)
- Test with `python -c "from src.decision_maker import test_visual_servoing; test_visual_servoing()"`

### Debug Mode

Enable detailed logging:
```python
# In src/config.py
DEBUG_MODE = True
VERBOSE_LOGGING = True
```

Test Virtual Depth Radar:
```bash
cd src
python -c "from decision_maker import test_visual_servoing; test_visual_servoing()"
```

Monitor UART communication:
```bash
# Monitor UART traffic
sudo cat /dev/ttyAMA0
```

## 🔬 Advanced Features

### Custom Model Training
```bash
# Train custom YOLOv8n model
pip install ultralytics
yolo train data=custom_dataset.yaml model=yolov8n.pt epochs=100

# Export to ONNX INT8
python export_yolo_224_int8.py --model=custom_model.pt
```

### Parameter Tuning
```python
# Real-time parameter adjustment
# Modify config.py and restart system
# Or implement dynamic parameter loading
```

### Multi-Camera Setup
```python
# Extend astra_camera.py for multiple cameras
# Implement camera fusion for better tracking
```

## 📈 Future Enhancements

### Planned Features
- [ ] **SLAM Integration**: Simultaneous localization and mapping
- [ ] **Advanced Obstacle Avoidance**: LiDAR-based path planning
- [ ] **Voice Commands**: Speech recognition integration
- [ ] **Mobile App**: Remote monitoring and control
- [ ] **Multi-Person Tracking**: Follow specific person by ID
- [ ] **Gesture Recognition**: Hand gesture commands
- [ ] **Auto-Charging**: Return to charging station
- [ ] **Enhanced Radar**: 5-zone scanning with height layers
- [ ] **Predictive Avoidance**: Obstacle trajectory prediction

### Performance Optimizations
- [ ] **TensorRT Acceleration**: GPU-accelerated inference
- [ ] **Model Quantization**: Further size reduction
- [ ] **Multi-Threading**: Parallel processing pipeline
- [ ] **Edge TPU Support**: Google Coral integration
- [ ] **Radar Optimization**: SIMD vectorization for depth processing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/SentientFollowAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/SentientFollowAI/discussions)
- **Email**: support@sentientfollow.ai

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 object detection framework
- **Orbbec**: Astra camera SDK and hardware
- **ByteTrack**: Multi-object tracking algorithm
- **OpenCV**: Computer vision library
- **ONNX Runtime**: AI model inference engine

---

**Built with ❤️ for autonomous robotics and AI research**