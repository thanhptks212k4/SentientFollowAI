#!/usr/bin/env python3

import time
import math
import serial
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    SAFE_DISTANCE_MM, DEADZONE_X, DEADZONE_Z,
    MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED,
    KP_LINEAR, KP_ANGULAR, MIN_SPEED_THRESHOLD,
    BACKWARD_DISTANCE_THRESHOLD
)

try:
    from config import (
        KI_LINEAR, KD_LINEAR, KI_ANGULAR, KD_ANGULAR, MAX_ACCEL, EMA_ALPHA,
        OBSTACLE_THRESHOLD_SIDE, OBSTACLE_THRESHOLD_FRONT, NOISE_THRESHOLD,
        RADAR_SCAN_TOP, RADAR_SCAN_BOTTOM, RADAR_LEFT_BOUNDARY, RADAR_RIGHT_BOUNDARY,
        RADAR_EMA_ALPHA
    )
except ImportError:
    KI_LINEAR = 0.0001
    KD_LINEAR = 0.002
    KI_ANGULAR = 0.0002
    KD_ANGULAR = 0.005
    MAX_ACCEL = 2.0
    EMA_ALPHA = 0.2
    OBSTACLE_THRESHOLD_SIDE = 500
    OBSTACLE_THRESHOLD_FRONT = 400
    NOISE_THRESHOLD = 100
    RADAR_SCAN_TOP = 0.40
    RADAR_SCAN_BOTTOM = 0.60
    RADAR_LEFT_BOUNDARY = 0.33
    RADAR_RIGHT_BOUNDARY = 0.67
    RADAR_EMA_ALPHA = 0.3

@dataclass
class ControllerState:
    filtered_error_x: float = 0.0
    filtered_error_z: float = 0.0
    
    prev_error_x: float = 0.0
    prev_error_z: float = 0.0
    
    integral_x: float = 0.0
    integral_z: float = 0.0
    
    prev_v: float = 0.0
    prev_w: float = 0.0
    
    target_velocity_x: float = 0.0
    target_velocity_z: float = 0.0
    prev_target_x: float = 0.0
    prev_target_z: float = 0.0
    
    frames_without_target: int = 0
    last_w: float = 0.0
    
    last_update_time: float = 0.0


@dataclass
class RadarState:
    min_depth_left: float = float('inf')
    min_depth_center: float = float('inf')
    min_depth_right: float = float('inf')
    
    filtered_depth_left: float = float('inf')
    filtered_depth_center: float = float('inf')
    filtered_depth_right: float = float('inf')
    
    obstacle_left: bool = False
    obstacle_center: bool = False
    obstacle_right: bool = False
    
    override_active: bool = False
    override_reason: str = ""
    
    total_scans: int = 0
    obstacle_detections: int = 0


class VisualServoingDecisionMaker:
    def __init__(self, uart_port='/dev/ttyAMA0', uart_baudrate=115200):
        self.camera_center_x = CAMERA_WIDTH // 2
        self.camera_center_y = CAMERA_HEIGHT // 2
        
        self.safe_distance_mm = SAFE_DISTANCE_MM
        self.deadzone_x = DEADZONE_X
        self.deadzone_z = DEADZONE_Z
        
        self.kp_linear = KP_LINEAR
        self.ki_linear = KI_LINEAR
        self.kd_linear = KD_LINEAR
        
        self.kp_angular = KP_ANGULAR
        self.ki_angular = KI_ANGULAR
        self.kd_angular = KD_ANGULAR
        
        self.max_linear_speed = MAX_LINEAR_SPEED
        self.max_angular_speed = MAX_ANGULAR_SPEED
        self.min_speed = MIN_SPEED_THRESHOLD
        
        self.ema_alpha = EMA_ALPHA
        self.max_accel = MAX_ACCEL
        self.dt = 0.05
        self.max_integral = 0.5
        self.decel_zone_factor = 2.0
        
        self.max_frames_without_target = 10
        
        self.state = ControllerState()
        
        self.radar_state = RadarState()
        
        self.radar_ema_alpha = RADAR_EMA_ALPHA
        self.obstacle_threshold_side = OBSTACLE_THRESHOLD_SIDE
        self.obstacle_threshold_front = OBSTACLE_THRESHOLD_FRONT
        self.noise_threshold = NOISE_THRESHOLD
        
        self.radar_scan_top = int(CAMERA_HEIGHT * RADAR_SCAN_TOP)
        self.radar_scan_bottom = int(CAMERA_HEIGHT * RADAR_SCAN_BOTTOM)
        self.radar_left_boundary = int(CAMERA_WIDTH * RADAR_LEFT_BOUNDARY)
        self.radar_right_boundary = int(CAMERA_WIDTH * RADAR_RIGHT_BOUNDARY)
        
        self.last_command = "STOP"
        self.decision_count = 0
        self.last_center_error = 0.0
        self.last_distance_error = 0.0
        
        self.stats = {
            'total_decisions': 0,
            'move_forward_count': 0,
            'move_backward_count': 0,
            'turn_left_count': 0,
            'turn_right_count': 0,
            'stop_count': 0,
            'avg_center_error': 0.0,
            'avg_distance_error': 0.0,
            'pid_calls': 0,
            'target_lost_count': 0,
            'recovery_attempts': 0,
            'radar_scans': 0,
            'obstacle_detections': 0,
            'override_left': 0,
            'override_right': 0,
            'emergency_stops': 0
        }
        
        self.uart_port = uart_port
        self.uart_baudrate = uart_baudrate
        self.serial_conn = self._init_uart()
        
        print("Hybrid Predictive Controller with Virtual Depth Radar initialized")
    
    def _init_uart(self) -> Optional[serial.Serial]:
        try:
            conn = serial.Serial(
                port=self.uart_port,
                baudrate=self.uart_baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"Connected: {self.uart_port} @ {self.uart_baudrate} baud")
            return conn
        except serial.SerialException as e:
            print(f"Failed to connect {self.uart_port}: {e}")
            return None
    
    def _apply_ema_filter(self, new_error_x: float, new_error_z: float) -> Tuple[float, float]:
        self.state.filtered_error_x = (
            self.ema_alpha * new_error_x + 
            (1 - self.ema_alpha) * self.state.filtered_error_x
        )
        
        self.state.filtered_error_z = (
            self.ema_alpha * new_error_z + 
            (1 - self.ema_alpha) * self.state.filtered_error_z
        )
        
        return self.state.filtered_error_x, self.state.filtered_error_z
    
    def _predict_target_motion(self, current_target_x: float, current_target_z: float) -> Tuple[float, float]:
        current_time = time.time()
        
        if self.state.last_update_time > 0:
            dt = current_time - self.state.last_update_time
            
            if dt > 0:
                self.state.target_velocity_x = (current_target_x - self.state.prev_target_x) / dt
                self.state.target_velocity_z = (current_target_z - self.state.prev_target_z) / dt
                
                lead_time = self.dt
                predicted_x = current_target_x + self.state.target_velocity_x * lead_time
                predicted_z = current_target_z + self.state.target_velocity_z * lead_time
                
                predicted_x = np.clip(predicted_x, 0, CAMERA_WIDTH)
                predicted_z = max(predicted_z, 100)
                
            else:
                predicted_x, predicted_z = current_target_x, current_target_z
        else:
            predicted_x, predicted_z = current_target_x, current_target_z
        
        self.state.prev_target_x = current_target_x
        self.state.prev_target_z = current_target_z
        self.state.last_update_time = current_time
        
        return predicted_x, predicted_z
    
    def _calculate_adaptive_gains(self, error_z: float) -> Tuple[float, float]:
        decel_start_distance = self.deadzone_z * self.decel_zone_factor
        
        if abs(error_z) < decel_start_distance:
            reduction_factor = abs(error_z) / decel_start_distance
            reduction_factor = max(reduction_factor, 0.3)
            
            adaptive_kp_linear = self.kp_linear * reduction_factor
            adaptive_kp_angular = self.kp_angular * reduction_factor
        else:
            adaptive_kp_linear = self.kp_linear
            adaptive_kp_angular = self.kp_angular
        
        return adaptive_kp_linear, adaptive_kp_angular
    
    def _pid_controller(self, error: float, prev_error: float, integral: float, 
                       kp: float, ki: float, kd: float) -> Tuple[float, float]:
        p_term = kp * error
        
        integral += error * self.dt
        integral = np.clip(integral, -self.max_integral, self.max_integral)
        i_term = ki * integral
        
        d_error = (error - prev_error) / self.dt
        d_term = kd * d_error
        
        pid_output = p_term + i_term + d_term
        
        return pid_output, integral
    
    def _apply_acceleration_limiting(self, target_v: float, target_w: float) -> Tuple[float, float]:
        max_delta_v = self.max_accel * self.dt
        max_delta_w = (self.max_accel / 0.3) * self.dt
        
        delta_v = target_v - self.state.prev_v
        if abs(delta_v) > max_delta_v:
            limited_v = self.state.prev_v + np.sign(delta_v) * max_delta_v
        else:
            limited_v = target_v
        
        delta_w = target_w - self.state.prev_w
        if abs(delta_w) > max_delta_w:
            limited_w = self.state.prev_w + np.sign(delta_w) * max_delta_w
        else:
            limited_w = target_w
        
        return limited_v, limited_w
    
    def _apply_cornering_speed_reduction(self, v: float, w: float) -> float:
        if abs(w) > 0.1:
            angular_ratio = abs(w) / self.max_angular_speed
            reduction_factor = 0.5
            
            speed_reduction = angular_ratio * reduction_factor
            v_reduced = v * (1 - speed_reduction)
            
            return v_reduced
        
        return v
    
    def _handle_lost_target(self) -> Tuple[float, float]:
        self.state.frames_without_target += 1
        
        if self.state.frames_without_target <= self.max_frames_without_target:
            recovery_w = self.state.last_w * 0.3
            recovery_v = 0.0
            
            self.stats['recovery_attempts'] += 1
            return recovery_v, recovery_w
        else:
            if self.state.frames_without_target == self.max_frames_without_target + 1:
                self.stats['target_lost_count'] += 1
                return 0.0, 0.0
            else:
                return None, None
    
    def _scan_virtual_depth_radar(self, depth_frame: np.ndarray) -> None:
        if depth_frame is None or depth_frame.size == 0:
            return
        
        self.radar_state.total_scans += 1
        self.stats['radar_scans'] += 1
        
        height, width = depth_frame.shape[:2]
        
        scan_top = int(height * RADAR_SCAN_TOP)
        scan_bottom = int(height * RADAR_SCAN_BOTTOM)
        
        left_boundary = int(width * RADAR_LEFT_BOUNDARY)
        right_boundary = int(width * RADAR_RIGHT_BOUNDARY)
        
        scan_region = depth_frame[scan_top:scan_bottom, :]
        
        left_zone = scan_region[:, :left_boundary]
        valid_left = left_zone[left_zone > self.noise_threshold]
        self.radar_state.min_depth_left = np.min(valid_left) if len(valid_left) > 0 else float('inf')
        
        center_zone = scan_region[:, left_boundary:right_boundary]
        valid_center = center_zone[center_zone > self.noise_threshold]
        self.radar_state.min_depth_center = np.min(valid_center) if len(valid_center) > 0 else float('inf')
        
        right_zone = scan_region[:, right_boundary:]
        valid_right = right_zone[right_zone > self.noise_threshold]
        self.radar_state.min_depth_right = np.min(valid_right) if len(valid_right) > 0 else float('inf')
        
        self._apply_radar_ema_filtering()
        
        self.radar_state.obstacle_left = self.radar_state.filtered_depth_left < self.obstacle_threshold_side
        self.radar_state.obstacle_center = self.radar_state.filtered_depth_center < self.obstacle_threshold_front
        self.radar_state.obstacle_right = self.radar_state.filtered_depth_right < self.obstacle_threshold_side
        
        if any([self.radar_state.obstacle_left, self.radar_state.obstacle_center, self.radar_state.obstacle_right]):
            self.radar_state.obstacle_detections += 1
            self.stats['obstacle_detections'] += 1
    
    def _apply_radar_ema_filtering(self) -> None:
        alpha = self.radar_ema_alpha
        
        if self.radar_state.filtered_depth_left == float('inf'):
            self.radar_state.filtered_depth_left = self.radar_state.min_depth_left
            self.radar_state.filtered_depth_center = self.radar_state.min_depth_center
            self.radar_state.filtered_depth_right = self.radar_state.min_depth_right
        else:
            if self.radar_state.min_depth_left != float('inf'):
                self.radar_state.filtered_depth_left = (
                    alpha * self.radar_state.min_depth_left + 
                    (1 - alpha) * self.radar_state.filtered_depth_left
                )
            
            if self.radar_state.min_depth_center != float('inf'):
                self.radar_state.filtered_depth_center = (
                    alpha * self.radar_state.min_depth_center + 
                    (1 - alpha) * self.radar_state.filtered_depth_center
                )
            
            if self.radar_state.min_depth_right != float('inf'):
                self.radar_state.filtered_depth_right = (
                    alpha * self.radar_state.min_depth_right + 
                    (1 - alpha) * self.radar_state.filtered_depth_right
                )
    
    def _apply_obstacle_override_logic(self, control_v: float, control_w: float) -> Tuple[float, float]:
        override_v = control_v
        override_w = control_w
        override_reason = ""
        
        if self.radar_state.obstacle_center:
            override_v = 0.0
            override_w = 0.0
            override_reason = "EMERGENCY_STOP - Front obstacle detected"
            self.stats['emergency_stops'] += 1
            print(f"[RADAR] PHANH KHẨN CẤP - Vật cản phía trước: {self.radar_state.filtered_depth_center:.0f}mm")
        
        elif control_w > 0 and self.radar_state.obstacle_right:
            override_w = 0.0
            override_reason = "RIGHT_SIDE_BLOCKED - Going straight to clear obstacle"
            self.stats['override_right'] += 1
            print(f"[RADAR] Vướng sườn PHẢI - Đang đi thẳng vượt vật cản ({self.radar_state.filtered_depth_right:.0f}mm)")
        
        elif control_w < 0 and self.radar_state.obstacle_left:
            override_w = 0.0
            override_reason = "LEFT_SIDE_BLOCKED - Going straight to clear obstacle"
            self.stats['override_left'] += 1
            print(f"[RADAR] Vướng sườn TRÁI - Đang đi thẳng vượt vật cản ({self.radar_state.filtered_depth_left:.0f}mm)")
        
        self.radar_state.override_active = override_reason != ""
        self.radar_state.override_reason = override_reason
        
        return override_v, override_w
    
    def _get_radar_status_string(self) -> str:
        left_status = "BLOCKED" if self.radar_state.obstacle_left else "CLEAR"
        center_status = "BLOCKED" if self.radar_state.obstacle_center else "CLEAR"
        right_status = "BLOCKED" if self.radar_state.obstacle_right else "CLEAR"
        
        left_dist = f"{self.radar_state.filtered_depth_left:.0f}" if self.radar_state.filtered_depth_left != float('inf') else "∞"
        center_dist = f"{self.radar_state.filtered_depth_center:.0f}" if self.radar_state.filtered_depth_center != float('inf') else "∞"
        right_dist = f"{self.radar_state.filtered_depth_right:.0f}" if self.radar_state.filtered_depth_right != float('inf') else "∞"
        
        return f"L:{left_status}{left_dist} C:{center_status}{center_dist} R:{right_status}{right_dist}mm"
    
    def send_to_esp32(self, v_speed: float, w_speed: float) -> None:
        if v_speed is None or w_speed is None:
            return
            
        msg = f"{v_speed:.3f},{w_speed:.3f}\n"
        
        if self.serial_conn is not None and self.serial_conn.is_open:
            try:
                self.serial_conn.write(msg.encode('utf-8'))
                print(f"[UART TX] {msg.strip()}")
            except serial.SerialException as e:
                print(f"[UART] Send error: {e}")
        else:
            print(f"[UART] Not connected - Skip: {msg.strip()}")
    
    def process_target(self, bbox: List[int], depth_mm: float, 
                      frame_width: int, frame_height: int, depth_frame: Optional[np.ndarray] = None) -> str:
        self.stats['total_decisions'] += 1
        self.decision_count += 1
        
        self.camera_center_x = frame_width // 2
        self.camera_center_y = frame_height // 2
        
        if len(bbox) != 4 or depth_mm <= 0:
            recovery_v, recovery_w = self._handle_lost_target()
            
            if recovery_v is not None and recovery_w is not None:
                self.send_to_esp32(recovery_v, recovery_w)
                self.last_command = f"LOST_TARGET_RECOVERY: v={recovery_v:.3f}, w={recovery_w:.3f}"
            else:
                self.last_command = "TARGET_LOST_SILENT"
                
            return self.last_command
        
        self.state.frames_without_target = 0
        
        if depth_frame is not None:
            self._scan_virtual_depth_radar(depth_frame)
        
        x1, y1, x2, y2 = bbox
        current_target_x = (x1 + x2) // 2
        current_target_y = (y1 + y2) // 2
        
        predicted_x, predicted_z = self._predict_target_motion(current_target_x, depth_mm)
        
        raw_error_x = predicted_x - self.camera_center_x
        raw_error_z = predicted_z - self.safe_distance_mm
        
        filtered_error_x, filtered_error_z = self._apply_ema_filter(raw_error_x, raw_error_z)
        
        self.last_center_error = filtered_error_x
        self.last_distance_error = filtered_error_z
        
        self.stats['pid_calls'] += 1
        
        adaptive_kp_linear, adaptive_kp_angular = self._calculate_adaptive_gains(filtered_error_z)
        
        pid_v, self.state.integral_z = self._pid_controller(
            filtered_error_z, self.state.prev_error_z, self.state.integral_z,
            adaptive_kp_linear, self.ki_linear, self.kd_linear
        )
        
        pid_w, self.state.integral_x = self._pid_controller(
            filtered_error_x, self.state.prev_error_x, self.state.integral_x,
            adaptive_kp_angular, self.ki_angular, self.kd_angular
        )
        
        self.state.prev_error_x = filtered_error_x
        self.state.prev_error_z = filtered_error_z
        
        target_v = np.clip(pid_v, -self.max_linear_speed, self.max_linear_speed)
        target_w = np.clip(pid_w, -self.max_angular_speed, self.max_angular_speed)
        
        if abs(filtered_error_z) <= self.deadzone_z:
            target_v = 0.0
        if abs(filtered_error_x) <= self.deadzone_x:
            target_w = 0.0
        
        limited_v, limited_w = self._apply_acceleration_limiting(target_v, target_w)
        
        final_v = self._apply_cornering_speed_reduction(limited_v, limited_w)
        final_w = limited_w
        
        if abs(final_v) < self.min_speed:
            final_v = 0.0
        if abs(final_w) < self.min_speed:
            final_w = 0.0
        
        if depth_frame is not None:
            final_v, final_w = self._apply_obstacle_override_logic(final_v, final_w)
        
        self.state.prev_v = final_v
        self.state.prev_w = final_w
        self.state.last_w = final_w
        
        if final_v > 0:
            self.stats['move_forward_count'] += 1
        elif final_v < 0:
            self.stats['move_backward_count'] += 1
        
        if final_w > 0:
            self.stats['turn_right_count'] += 1
        elif final_w < 0:
            self.stats['turn_left_count'] += 1
        
        if final_v == 0 and final_w == 0:
            self.stats['stop_count'] += 1
        
        self.send_to_esp32(final_v, final_w)
        
        radar_status = self._get_radar_status_string() if depth_frame is not None else "RADAR:OFF"
        
        if self.radar_state.override_active:
            self.last_command = f"RADAR_OVERRIDE({self.radar_state.override_reason}) - {radar_status}"
        elif abs(final_v) > 0 and abs(final_w) > 0:
            self.last_command = f"HYBRID(v={final_v:.3f}, w={final_w:.3f}) - {radar_status}"
        elif abs(final_v) > 0:
            self.last_command = f"{'MOVE_FORWARD' if final_v > 0 else 'MOVE_BACKWARD'}({abs(final_v):.3f}) - {radar_status}"
        elif abs(final_w) > 0:
            self.last_command = f"{'TURN_RIGHT' if final_w > 0 else 'TURN_LEFT'}({abs(final_w):.3f}) - {radar_status}"
        else:
            self.last_command = f"MAINTAIN_POSITION - {radar_status}"
        
        return self.last_command
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'last_command': self.last_command,
            'decision_count': self.decision_count,
            'last_center_error': self.last_center_error,
            'last_distance_error': self.last_distance_error,
            'controller_state': {
                'filtered_error_x': self.state.filtered_error_x,
                'filtered_error_z': self.state.filtered_error_z,
                'integral_x': self.state.integral_x,
                'integral_z': self.state.integral_z,
                'target_velocity_x': self.state.target_velocity_x,
                'target_velocity_z': self.state.target_velocity_z,
                'prev_v': self.state.prev_v,
                'prev_w': self.state.prev_w,
                'frames_without_target': self.state.frames_without_target
            },
            'radar_state': {
                'min_depth_left': self.radar_state.min_depth_left,
                'min_depth_center': self.radar_state.min_depth_center,
                'min_depth_right': self.radar_state.min_depth_right,
                'filtered_depth_left': self.radar_state.filtered_depth_left,
                'filtered_depth_center': self.radar_state.filtered_depth_center,
                'filtered_depth_right': self.radar_state.filtered_depth_right,
                'obstacle_left': self.radar_state.obstacle_left,
                'obstacle_center': self.radar_state.obstacle_center,
                'obstacle_right': self.radar_state.obstacle_right,
                'override_active': self.radar_state.override_active,
                'override_reason': self.radar_state.override_reason,
                'total_scans': self.radar_state.total_scans,
                'obstacle_detections': self.radar_state.obstacle_detections
            }
        }
    
    def reset_statistics(self) -> None:
        self.state = ControllerState()
        self.radar_state = RadarState()
        self.stats = {
            'total_decisions': 0,
            'move_forward_count': 0,
            'move_backward_count': 0,
            'turn_left_count': 0,
            'turn_right_count': 0,
            'stop_count': 0,
            'avg_center_error': 0.0,
            'avg_distance_error': 0.0,
            'pid_calls': 0,
            'target_lost_count': 0,
            'recovery_attempts': 0,
            'radar_scans': 0,
            'obstacle_detections': 0,
            'override_left': 0,
            'override_right': 0,
            'emergency_stops': 0
        }
        self.decision_count = 0
        self.last_center_error = 0.0
        self.last_distance_error = 0.0
    
    def get_status_string(self) -> str:
        return f"Nav:{self.last_command} Decisions:{self.decision_count}"
    
    def emergency_stop(self) -> None:
        self.send_to_esp32(0.0, 0.0)
        self.last_command = "EMERGENCY_STOP"
    
    def stop(self) -> None:
        self.emergency_stop()
        self.state.frames_without_target = 0
        self.radar_state.override_active = False
        self.radar_state.override_reason = ""
    
    def close(self) -> None:
        self.emergency_stop()
        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[UART] Connection closed")
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


def test_visual_servoing():
    print("Testing Hybrid Predictive Controller with Virtual Depth Radar")
    
    controller = VisualServoingDecisionMaker(uart_port=None)
    
    mock_depth = np.full((240, 320), 2000, dtype=np.uint16)
    mock_depth[96:144, 0:106] = 450
    mock_depth[96:144, 107:213] = 350
    mock_depth[96:144, 214:320] = 600
    
    test_cases = [
        ([300, 200, 400, 400], 2500, mock_depth, "Person far and centered with obstacles"),
        ([450, 200, 550, 400], 2000, mock_depth, "Person far and right with obstacles"),
        ([100, 200, 200, 400], 1800, mock_depth, "Person far and left with obstacles"),
        ([320, 200, 420, 400], 1500, None, "Person at safe distance, no radar"),
        ([], 0, mock_depth, "Lost target with obstacles"),
    ]
    
    for i, (bbox, depth, depth_frame, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        action = controller.process_target(bbox, depth, 640, 480, depth_frame)
        print(f"Action: {action}")
        
        if depth_frame is not None:
            radar_status = controller._get_radar_status_string()
            print(f"Radar: {radar_status}")
        
        time.sleep(0.1)
    
    stats = controller.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"   Total decisions: {stats['total_decisions']}")
    print(f"   PID calls: {stats['pid_calls']}")
    print(f"   Radar scans: {stats['radar_scans']}")
    print(f"   Obstacle detections: {stats['obstacle_detections']}")
    print(f"   Emergency stops: {stats['emergency_stops']}")
    print(f"   Override left: {stats['override_left']}")
    print(f"   Override right: {stats['override_right']}")
    
    controller.close()


if __name__ == "__main__":
    test_visual_servoing()