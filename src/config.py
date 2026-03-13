#!/usr/bin/env python3

import os
from typing import Final

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
MODEL_PATH: Final[str] = os.path.join(_project_root, 'models/yolov5nu_person_224_int8.onnx')

CAMERA_ID: Final[int] = 0
CAMERA_WIDTH: Final[int] = 320
CAMERA_HEIGHT: Final[int] = 240
CAMERA_FPS: Final[int] = 30

INPUT_SIZE: Final[int] = 224
CONF_THRESH: Final[float] = 0.45
IOU_THRESH: Final[float] = 0.25
PERSON_CLASS: Final[int] = 0

AI_FPS_TARGET: Final[int] = 15
FORCE_USB_CAMERA: Final[bool] = False

TRACK_THRESH: Final[float] = 0.50
TRACK_BUFFER: Final[int] = 60
MATCH_THRESH: Final[float] = 0.85

GC_INTERVAL: Final[int] = 500

SAFE_DISTANCE_MM: Final[int] = 1500
DEADZONE_X: Final[int] = 15
DEADZONE_Z: Final[int] = 100

MAX_LINEAR_SPEED: Final[float] = 0.8
MAX_ANGULAR_SPEED: Final[float] = 1.0

KP_LINEAR: Final[float] = 0.0008
KP_ANGULAR: Final[float] = 0.015

MIN_SPEED_THRESHOLD: Final[float] = 0.1
BACKWARD_DISTANCE_THRESHOLD: Final[int] = 800

WORK_MS: Final[int] = 25

def get_wait_ms() -> int:
    return max(1, int(1000/AI_FPS_TARGET) - WORK_MS)

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
    if errors:
        for error in errors:
            print(f"   - {error}")
        return False
    return True

def print_config_summary() -> None:
    print(f"Target FPS: {AI_FPS_TARGET}")
    print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"Model: {MODEL_PATH}")
    print(f"Safe Distance: {SAFE_DISTANCE_MM}mm")

if __name__ == "__main__":
    if validate_config():
        print_config_summary()
else:
    if not validate_config():
        raise ValueError("Invalid configuration parameters")