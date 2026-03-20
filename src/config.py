#!/usr/bin/env python3

import os
from typing import Final

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
MODEL_PATH: Final[str] = os.path.join(_project_root, 'models/yolov8n_person_224_int8.onnx')

CAMERA_ID: Final[int] = 0
CAMERA_WIDTH: Final[int] = 320
CAMERA_HEIGHT: Final[int] = 240
CAMERA_FPS: Final[int] = 30

FORCE_USB_CAMERA: Final[bool] = False

INPUT_SIZE: Final[int] = 224
CONF_THRESH: Final[float] = 0.65
IOU_THRESH: Final[float] = 0.25
PERSON_CLASS: Final[int] = 0

AI_FPS_TARGET: Final[int] = 20
WORK_MS: Final[int] = 25
GC_INTERVAL: Final[int] = 500

def get_wait_ms() -> int:
    return max(1, int(1000/AI_FPS_TARGET) - WORK_MS)

TRACK_THRESH: Final[float] = 0.50
TRACK_BUFFER: Final[int] = 60
MATCH_THRESH: Final[float] = 0.35

SAFE_DISTANCE_MM: Final[int] = 1500
BACKWARD_DISTANCE_THRESHOLD: Final[int] = 800

DEADZONE_X: Final[int] = 15
DEADZONE_Z: Final[int] = 100

MAX_LINEAR_SPEED: Final[float] = 0.449
MAX_ANGULAR_SPEED: Final[float] = 6.283
MIN_SPEED_THRESHOLD: Final[float] = 0.05

KP_LINEAR: Final[float] = 0.0015
KP_ANGULAR: Final[float] = 0.008

KI_LINEAR: Final[float] = 0.0001
KD_LINEAR: Final[float] = 0.002
KI_ANGULAR: Final[float] = 0.0002
KD_ANGULAR: Final[float] = 0.005

MAX_ACCEL: Final[float] = 1.0
EMA_ALPHA: Final[float] = 0.2

OBSTACLE_THRESHOLD_SIDE: Final[int] = 500
OBSTACLE_THRESHOLD_FRONT: Final[int] = 400
NOISE_THRESHOLD: Final[int] = 100

RADAR_SCAN_TOP: Final[float] = 0.40
RADAR_SCAN_BOTTOM: Final[float] = 0.60
RADAR_LEFT_BOUNDARY: Final[float] = 0.33
RADAR_RIGHT_BOUNDARY: Final[float] = 0.67

RADAR_EMA_ALPHA: Final[float] = 0.3

WINDOW_NAME: Final[str] = 'Person Tracking'

def validate_config() -> bool:
    errors = []
    
    if SAFE_DISTANCE_MM <= 0:
        errors.append("SAFE_DISTANCE_MM must be positive")
    if DEADZONE_X <= 0 or DEADZONE_Z <= 0:
        errors.append("Dead zones must be positive")
    if MAX_LINEAR_SPEED <= 0 or MAX_ANGULAR_SPEED <= 0:
        errors.append("Speed limits must be positive")
    if KP_LINEAR <= 0 or KP_ANGULAR <= 0:
        errors.append("P-Controller gains must be positive")
    if AI_FPS_TARGET <= 0:
        errors.append("AI_FPS_TARGET must be positive")
    if CAMERA_WIDTH <= 0 or CAMERA_HEIGHT <= 0:
        errors.append("Camera dimensions must be positive")
    if not (0.0 <= EMA_ALPHA <= 1.0):
        errors.append("EMA_ALPHA must be between 0.0 and 1.0")
    if MAX_ACCEL <= 0:
        errors.append("MAX_ACCEL must be positive")
    
    if OBSTACLE_THRESHOLD_SIDE <= 0 or OBSTACLE_THRESHOLD_FRONT <= 0:
        errors.append("Obstacle thresholds must be positive")
    if NOISE_THRESHOLD <= 0:
        errors.append("NOISE_THRESHOLD must be positive")
    if not (0.0 <= RADAR_SCAN_TOP < RADAR_SCAN_BOTTOM <= 1.0):
        errors.append("Radar scan region must be valid (0 <= top < bottom <= 1)")
    if not (0.0 <= RADAR_LEFT_BOUNDARY < RADAR_RIGHT_BOUNDARY <= 1.0):
        errors.append("Radar zone boundaries must be valid (0 <= left < right <= 1)")
    if not (0.0 <= RADAR_EMA_ALPHA <= 1.0):
        errors.append("RADAR_EMA_ALPHA must be between 0.0 and 1.0")
        
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    return True

def print_config_summary() -> None:
    print("=== SentientFollowAI Configuration ===")
    print(f"Target FPS: {AI_FPS_TARGET}")
    print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Safe Distance: {SAFE_DISTANCE_MM}mm")
    print(f"Max Speeds: {MAX_LINEAR_SPEED} m/s linear, {MAX_ANGULAR_SPEED} rad/s angular")
    print(f"PID Gains: Kp_lin={KP_LINEAR}, Kp_ang={KP_ANGULAR}")
    print("=====================================")

if __name__ == "__main__":
    if validate_config():
        print_config_summary()
        print("Configuration is valid")
    else:
        print("Configuration validation failed")
        exit(1)
else:
    if not validate_config():
        raise ValueError("Invalid configuration parameters")