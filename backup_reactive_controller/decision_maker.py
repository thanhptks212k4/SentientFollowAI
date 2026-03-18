#!/usr/bin/env python3

import time
import math
import serial
from typing import List, Dict, Any, Optional
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    SAFE_DISTANCE_MM, DEADZONE_X, DEADZONE_Z,
    MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED,
    KP_LINEAR, KP_ANGULAR, MIN_SPEED_THRESHOLD,
    BACKWARD_DISTANCE_THRESHOLD
)

class VisualServoingDecisionMaker:
    def __init__(self, uart_port='/dev/ttyAMA0', uart_baudrate=115200):
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

        # UART connection to ESP32
        self.uart_port = uart_port
        self.uart_baudrate = uart_baudrate
        self.serial_conn = self._init_uart()

    def _init_uart(self) -> Optional[serial.Serial]:
        """Khởi tạo kết nối UART an toàn, trả về None nếu thất bại."""
        try:
            conn = serial.Serial(
                port=self.uart_port,
                baudrate=self.uart_baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"[UART] Kết nối thành công: {self.uart_port} @ {self.uart_baudrate} baud")
            return conn
        except serial.SerialException as e:
            print(f"[UART] ⚠️ Không thể mở {self.uart_port}: {e}")
            return None

    def send_to_esp32(self, v_speed: float, w_speed: float) -> None:
        """Gửi v_speed, w_speed xuống ESP32 qua UART dưới dạng 'v,w\n'."""
        msg = f"{v_speed:.3f},{w_speed:.3f}\n"
        if self.serial_conn is not None and self.serial_conn.is_open:
            try:
                self.serial_conn.write(msg.encode('utf-8'))
                print(f"[UART TX] {msg.strip()}")
            except serial.SerialException as e:
                print(f"[UART] ❌ Lỗi gửi: {e}")
        else:
            print(f"[UART] ⚠️ Serial chưa kết nối — bỏ qua: {msg.strip()}")

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
        self.send_to_esp32(0.0, 0.0)

    def process_target(self, bbox: List[int], depth_mm: float, frame_width: int, frame_height: int) -> str:
        self.decision_count += 1
        self.stats['total_decisions'] += 1
        
        # Tính tâm camera tự động theo resolution thực tế
        self.camera_center_x = frame_width // 2
        self.camera_center_y = frame_height // 2
        
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
        
        # ✅ PRIORITY LOGIC: Distance first, then rotation
        # Ưu tiên đi thẳng đến khoảng cách an toàn trước, rồi mới rẽ
        
        distance_needs_adjustment = abs(distance_error_z) > self.deadzone_z
        center_needs_adjustment = abs(center_error_x) > self.deadzone_x
        
        if distance_needs_adjustment:
            # 🎯 PHASE 1: Adjust distance first (go straight to safe distance)
            if distance_error_z > 0:
                # Person is far - move forward
                v_speed = min(self.max_linear_speed, abs(distance_error_z) * self.kp_linear)
                v_speed = max(self.min_speed, v_speed)
                action_description.append(f"Forward({v_speed:.3f})")
                self.stats['move_forward_count'] += 1
                print(f"🚀 PHASE 1: Moving forward to reach safe distance")
            else:
                # Person is too close - move backward
                v_speed = -min(self.max_linear_speed, abs(distance_error_z) * self.kp_linear * 0.5)
                v_speed = min(-self.min_speed, v_speed)
                action_description.append(f"Backward({abs(v_speed):.3f})")
                self.stats['move_backward_count'] += 1
                print(f"🚀 PHASE 1: Moving backward to reach safe distance")
            
            # 🚫 Don't rotate while adjusting distance to avoid obstacles
            w_speed = 0.0
            
        elif center_needs_adjustment:
            # 🎯 PHASE 2: Distance is OK, now adjust rotation
            v_speed = 0.0  # Don't move forward/backward
            
            w_speed = min(self.max_angular_speed, abs(center_error_x) * self.kp_angular)
            w_speed = max(self.min_speed, w_speed)
            
            if center_error_x > 0:
                action_description.append(f"Right({w_speed:.3f})")
                self.stats['turn_right_count'] += 1
                print(f"🚀 PHASE 2: Turn Right at speed {w_speed:.3f}")
            else:
                w_speed = -w_speed
                action_description.append(f"Left({abs(w_speed):.3f})")
                self.stats['turn_left_count'] += 1
                print(f"🚀 PHASE 2: Turn Left at speed {abs(w_speed):.3f}")
        
        # If neither distance nor center needs adjustment, robot stays in position
        
        
        # Send commands to ESP32
        if v_speed != 0.0:
            # Moving forward or backward
            self.last_command = f"MOVE_{'FORWARD' if v_speed > 0 else 'BACKWARD'}({abs(v_speed):.3f})"
            self.send_to_esp32(v_speed, 0.0)
            return action_description[0] if action_description else f"MOVE_{'FORWARD' if v_speed > 0 else 'BACKWARD'}"
        elif w_speed != 0.0:
            # Rotating only
            self.last_command = f"TURN_{'RIGHT' if w_speed > 0 else 'LEFT'}({abs(w_speed):.3f})"
            self.send_to_esp32(0.0, w_speed)
            return action_description[0] if action_description else f"TURN_{'RIGHT' if w_speed > 0 else 'LEFT'}"
        else:
            # Maintain position
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

    def close(self) -> None:
        """Đóng kết nối UART an toàn."""
        self.send_to_esp32(0.0, 0.0)
        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[UART] Đã đóng kết nối serial.")

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
        action = dm.process_target(bbox, depth, 640, 480)
    
    stats = dm.get_statistics()

if __name__ == "__main__":
    test_visual_servoing()