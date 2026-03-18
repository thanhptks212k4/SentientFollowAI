#!/usr/bin/env python3
"""
System Configuration for SentientFollowAI

This file contains all configurable parameters for the person-following robot system.
Modify these values to tune system performance and behavior.
"""

import os
from typing import Final

# =============================================================================
# MODEL AND PATHS CONFIGURATION
# =============================================================================

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
MODEL_PATH: Final[str] = os.path.join(_project_root, 'models/yolov8n_person_224_int8.onnx')

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================

CAMERA_ID: Final[int] = 0
CAMERA_WIDTH: Final[int] = 320
CAMERA_HEIGHT: Final[int] = 240
CAMERA_FPS: Final[int] = 30

# Force USB camera instead of Astra (for testing)
FORCE_USB_CAMERA: Final[bool] = False

# =============================================================================
# AI MODEL CONFIGURATION
# =============================================================================

INPUT_SIZE: Final[int] = 224           # YOLOv8n input resolution
CONF_THRESH: Final[float] = 0.45       # Detection confidence threshold
IOU_THRESH: Final[float] = 0.25        # NMS IoU threshold
PERSON_CLASS: Final[int] = 0           # COCO person class ID

# =============================================================================
# SYSTEM PERFORMANCE CONFIGURATION
# =============================================================================

AI_FPS_TARGET: Final[int] = 20         # Target processing FPS
WORK_MS: Final[int] = 25               # Processing time budget per frame
GC_INTERVAL: Final[int] = 500          # Garbage collection interval (frames)

def get_wait_ms() -> int:
    """Calculate OpenCV waitKey delay for target FPS"""
    return max(1, int(1000/AI_FPS_TARGET) - WORK_MS)

# =============================================================================
# TRACKING CONFIGURATION
# =============================================================================

TRACK_THRESH: Final[float] = 0.50      # ByteTrack detection threshold
TRACK_BUFFER: Final[int] = 60          # Track buffer size (frames)
MATCH_THRESH: Final[float] = 0.35      # Track matching threshold

# =============================================================================
# ROBOT CONTROL CONFIGURATION
# =============================================================================

# Distance Control
SAFE_DISTANCE_MM: Final[int] = 1500    # Target following distance (mm)
BACKWARD_DISTANCE_THRESHOLD: Final[int] = 800  # Distance to trigger backward motion

# Dead Zones (prevent jittery motion)
DEADZONE_X: Final[int] = 15            # Horizontal deadzone (pixels)
DEADZONE_Z: Final[int] = 100           # Distance deadzone (mm)

# Speed Limits
MAX_LINEAR_SPEED: Final[float] = 0.8   # Maximum forward/backward speed (m/s)
MAX_ANGULAR_SPEED: Final[float] = 1.0  # Maximum rotation speed (rad/s)
MIN_SPEED_THRESHOLD: Final[float] = 0.1  # Minimum speed to send (m/s)

# =============================================================================
# PID CONTROLLER CONFIGURATION
# =============================================================================

# Basic PID Gains
KP_LINEAR: Final[float] = 0.0008       # Proportional gain for distance control
KP_ANGULAR: Final[float] = 0.015       # Proportional gain for angle control

# Advanced PID Gains (Hybrid Predictive Controller)
KI_LINEAR: Final[float] = 0.0001       # Integral gain for distance control
KD_LINEAR: Final[float] = 0.002        # Derivative gain for distance control
KI_ANGULAR: Final[float] = 0.0002      # Integral gain for angle control
KD_ANGULAR: Final[float] = 0.005       # Derivative gain for angle control

# Motion Profile Parameters
MAX_ACCEL: Final[float] = 2.0          # Maximum acceleration (m/s²)
EMA_ALPHA: Final[float] = 0.2          # EMA filter coefficient (0.0-1.0)

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

WINDOW_NAME: Final[str] = 'Person Tracking'

# =============================================================================
# VALIDATION AND UTILITIES
# =============================================================================

def validate_config() -> bool:
    """Validate configuration parameters"""
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
        
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    return True

def print_config_summary() -> None:
    """Print system configuration summary"""
    print("=== SentientFollowAI Configuration ===")
    print(f"Target FPS: {AI_FPS_TARGET}")
    print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Safe Distance: {SAFE_DISTANCE_MM}mm")
    print(f"Max Speeds: {MAX_LINEAR_SPEED} m/s linear, {MAX_ANGULAR_SPEED} rad/s angular")
    print(f"PID Gains: Kp_lin={KP_LINEAR}, Kp_ang={KP_ANGULAR}")
    print("=====================================")

# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Standalone execution - validate and show config
    if validate_config():
        print_config_summary()
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed")
        exit(1)
else:
    # Module import - validate configuration
    if not validate_config():
        raise ValueError("Invalid configuration parameters")
