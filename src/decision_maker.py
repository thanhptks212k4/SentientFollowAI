#!/usr/bin/env python3

import time
import math
from typing import List, Dict, Any, Optional
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    SAFE_DISTANCE_MM, DEADZONE_X, DEADZONE_Z,
    MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED,
    KP_LINEAR, KP_ANGULAR, MIN_SPEED_THRESHOLD,
    BACKWARD_DISTANCE_THRESHOLD
)

class VisualServoingDecisionMaker:
    def __init__(self):
        self.camera_center_x = CAMERA_WIDTH // 2
        self.camera_center_y = CAMERA_HEIGHT // 2
        self.safe_distance_mm = SAFE_DISTANCE_MM
        self.deadzone_x = DEADZONE_X
        self.deadzone_z = DEADZONE_Z
        self.backward_threshold = BACKWARD_DISTANCE_THRESHOLD
        self.max_linear_speed = MAX_LINEAR_SPEED
        self.max_angular_speed = MAX_ANGULAR_SPEED
        self.min_speed = MIN_SPEED_THRESHOLD
        self.kp_linear = KP_LINEAR
        self.kp_angular = KP_ANGULAR
        self.last_command = "STOP"
        self.decision_count = 0
        self.last_center_error = 0.0
        self.last_distance_error = 0.0
        self.error_history = []
        self.stats = {
            'total_decisions': 0,
            'move_forward_count': 0,
            'move_backward_count': 0,
            'turn_left_count': 0,
            'turn_right_count': 0,
            'stop_count': 0,
            'avg_center_error': 0.0,
            'avg_distance_error': 0.0
        }

    def move_forward(self, speed: float) -> None:
        speed = max(self.min_speed, min(self.max_linear_speed, speed))
        self.last_command = f"MOVE_FORWARD({speed:.3f})"
        self.stats['move_forward_count'] += 1

    def move_backward(self, speed: float) -> None:
        speed = max(self.min_speed, min(self.max_linear_speed, speed))
        self.last_command = f"MOVE_BACKWARD({speed:.3f})"
        self.stats['move_backward_count'] += 1

    def turn_left(self, speed: float) -> None:
        speed = max(self.min_speed, min(self.max_angular_speed, speed))
        self.last_command = f"TURN_LEFT({speed:.3f})"
        self.stats['turn_left_count'] += 1

    def turn_right(self, speed: float) -> None:
        speed = max(self.min_speed, min(self.max_angular_speed, speed))
        self.last_command = f"TURN_RIGHT({speed:.3f})"
        self.stats['turn_right_count'] += 1

    def stop(self) -> None:
        self.last_command = "STOP"
        self.stats['stop_count'] += 1

    def process_target(self, bbox: List[int], depth_mm: float) -> str:
        self.decision_count += 1
        self.stats['total_decisions'] += 1
        
        if len(bbox) != 4 or depth_mm <= 0:
            self.stop()
            return "INVALID_INPUT"
        
        x1, y1, x2, y2 = bbox
        target_center_x = (x1 + x2) // 2
        target_center_y = (y1 + y2) // 2
        
        center_error_x = target_center_x - self.camera_center_x
        distance_error_z = depth_mm - self.safe_distance_mm
        
        self.last_center_error = center_error_x
        self.last_distance_error = distance_error_z
        self._update_statistics(center_error_x, distance_error_z)
        
        v_speed = 0.0
        w_speed = 0.0
        action_description = []
        
        if abs(distance_error_z) > self.deadzone_z:
            if distance_error_z > 0:
                v_speed = min(self.max_linear_speed, abs(distance_error_z) * self.kp_linear)
                v_speed = max(self.min_speed, v_speed)
                action_description.append(f"Forward({v_speed:.3f})")
                self.stats['move_forward_count'] += 1
            else:
                if depth_mm < self.backward_threshold:
                    v_speed = -min(self.max_linear_speed, abs(distance_error_z) * self.kp_linear * 0.5)
                    v_speed = min(-self.min_speed, v_speed)
                    action_description.append(f"Backward({abs(v_speed):.3f})")
                    self.stats['move_backward_count'] += 1
                else:
                    v_speed = 0.0
                    action_description.append("Stop_Linear")
        
        if abs(center_error_x) > self.deadzone_x:
            w_speed = min(self.max_angular_speed, abs(center_error_x) * self.kp_angular)
            w_speed = max(self.min_speed, w_speed)
            
            if center_error_x > 0:
                action_description.append(f"Right({w_speed:.3f})")
                self.stats['turn_right_count'] += 1
                print(f"🚀 MOTOR: Turn Right at speed {w_speed:.3f}")
            else:
                w_speed = -w_speed
                action_description.append(f"Left({abs(w_speed):.3f})")
                self.stats['turn_left_count'] += 1
                print(f"🚀 MOTOR: Turn Left at speed {abs(w_speed):.3f}")
        
        if v_speed != 0.0 and w_speed != 0.0:
            self.last_command = f"COMBINED(v={v_speed:.3f}, w={w_speed:.3f})"
            print(f"🚀 MOTOR: Move Forward {v_speed:.3f} and Turn {w_speed:.3f}")
            return f"COMBINED_MOTION: {' + '.join(action_description)}"
        elif v_speed != 0.0:
            if v_speed > 0:
                self.last_command = f"MOVE_FORWARD({v_speed:.3f})"
            else:
                self.last_command = f"MOVE_BACKWARD({abs(v_speed):.3f})"
            return action_description[0]
        elif w_speed != 0.0:
            if w_speed > 0:
                self.last_command = f"TURN_RIGHT({w_speed:.3f})"
            else:
                self.last_command = f"TURN_LEFT({abs(w_speed):.3f})"
            return action_description[0]
        else:
            self.stop()
            return f"MAINTAIN_POSITION (centered, distance: {depth_mm:.0f}mm)"

    def _update_statistics(self, center_error: float, distance_error: float) -> None:
        alpha = 0.1
        self.stats['avg_center_error'] = (
            alpha * abs(center_error) + 
            (1 - alpha) * self.stats['avg_center_error']
        )
        self.stats['avg_distance_error'] = (
            alpha * abs(distance_error) + 
            (1 - alpha) * self.stats['avg_distance_error']
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'last_command': self.last_command,
            'decision_count': self.decision_count,
            'last_center_error': self.last_center_error,
            'last_distance_error': self.last_distance_error
        }

    def reset_statistics(self) -> None:
        self.stats = {
            'total_decisions': 0,
            'move_forward_count': 0,
            'move_backward_count': 0,
            'turn_left_count': 0,
            'turn_right_count': 0,
            'stop_count': 0,
            'avg_center_error': 0.0,
            'avg_distance_error': 0.0
        }
        self.decision_count = 0
        self.last_center_error = 0.0
        self.last_distance_error = 0.0

    def get_status_string(self) -> str:
        return f"Nav:{self.last_command} Decisions:{self.decision_count}"

    def emergency_stop(self) -> None:
        self.stop()

def test_visual_servoing():
    dm = VisualServoingDecisionMaker()
    test_cases = [
        ([300, 200, 400, 400], 2000, "Should move forward (target far)"),
        ([100, 200, 200, 400], 1500, "Should turn right (target left)"),
        ([450, 200, 550, 400], 1500, "Should turn left (target right)"),
        ([300, 200, 400, 400], 1500, "Should maintain position (centered)"),
        ([300, 200, 400, 400], 1200, "Should stop (too close)"),
        ([300, 200, 400, 400], 600, "Should move backward (very close)"),
        ([500, 200, 600, 400], 2500, "Should turn left while moving forward"),
    ]
    
    for i, (bbox, depth, expected) in enumerate(test_cases):
        action = dm.process_target(bbox, depth)
    
    stats = dm.get_statistics()

if __name__ == "__main__":
    test_visual_servoing()