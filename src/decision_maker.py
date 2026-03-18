#!/usr/bin/env python3
"""
Hybrid Predictive Visual Servoing Controller with Virtual Depth Radar

Advanced person-following robot controller with:
- Signal Conditioning & State Estimation (EMA Filter + Target Prediction)
- Advanced PID Controller (Anti-windup + Adaptive Gains)  
- Motion Profile & Safety (Smooth Pursuit + Acceleration Ramping)
- Recovery & Lost Target Logic (Inertial Navigation)
- Virtual Depth Radar (Multi-zone Obstacle Detection & Avoidance)
"""

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

# Import new radar parameters if available
try:
    from config import (
        KI_LINEAR, KD_LINEAR, KI_ANGULAR, KD_ANGULAR, MAX_ACCEL, EMA_ALPHA,
        OBSTACLE_THRESHOLD_SIDE, OBSTACLE_THRESHOLD_FRONT, NOISE_THRESHOLD,
        RADAR_SCAN_TOP, RADAR_SCAN_BOTTOM, RADAR_LEFT_BOUNDARY, RADAR_RIGHT_BOUNDARY,
        RADAR_EMA_ALPHA
    )
except ImportError:
    # Default values if not in config
    KI_LINEAR = 0.0001
    KD_LINEAR = 0.002
    KI_ANGULAR = 0.0002
    KD_ANGULAR = 0.005
    MAX_ACCEL = 2.0
    EMA_ALPHA = 0.2
    # Radar defaults
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
    """Controller state variables for PID and prediction algorithms"""
    # EMA Filtered errors
    filtered_error_x: float = 0.0
    filtered_error_z: float = 0.0
    
    # Previous errors for derivative calculation
    prev_error_x: float = 0.0
    prev_error_z: float = 0.0
    
    # Integral terms for PID
    integral_x: float = 0.0
    integral_z: float = 0.0
    
    # Previous velocities for acceleration limiting
    prev_v: float = 0.0
    prev_w: float = 0.0
    
    # Target prediction
    target_velocity_x: float = 0.0  # pixels/s
    target_velocity_z: float = 0.0  # mm/s
    prev_target_x: float = 0.0
    prev_target_z: float = 0.0
    
    # Lost target recovery
    frames_without_target: int = 0
    last_w: float = 0.0
    
    # Timestamps
    last_update_time: float = 0.0


@dataclass
class RadarState:
    """Virtual Depth Radar state for obstacle detection"""
    # Current depth readings (mm)
    min_depth_left: float = float('inf')
    min_depth_center: float = float('inf')
    min_depth_right: float = float('inf')
    
    # EMA filtered depths for smoothing
    filtered_depth_left: float = float('inf')
    filtered_depth_center: float = float('inf')
    filtered_depth_right: float = float('inf')
    
    # Obstacle detection flags
    obstacle_left: bool = False
    obstacle_center: bool = False
    obstacle_right: bool = False
    
    # Override states
    override_active: bool = False
    override_reason: str = ""
    
    # Statistics
    total_scans: int = 0
    obstacle_detections: int = 0


class VisualServoingDecisionMaker:
    """
    Hybrid Predictive Controller with Virtual Depth Radar for person-following robot
    
    Combines:
    1. Signal Conditioning: EMA filter + Target prediction
    2. Advanced PID: Full PID with anti-windup and adaptive gains
    3. Motion Profile: Smooth pursuit + acceleration ramping
    4. Recovery Logic: Inertial navigation when target is lost
    5. Virtual Depth Radar: Multi-zone obstacle detection and avoidance
    """
    
    def __init__(self, uart_port='/dev/ttyAMA0', uart_baudrate=115200):
        """
        Initialize the hybrid predictive controller
        
        Args:
            uart_port: UART port for ESP32 communication
            uart_baudrate: UART communication speed
        """
        # Camera parameters
        self.camera_center_x = CAMERA_WIDTH // 2
        self.camera_center_y = CAMERA_HEIGHT // 2
        
        # Control parameters
        self.safe_distance_mm = SAFE_DISTANCE_MM
        self.deadzone_x = DEADZONE_X
        self.deadzone_z = DEADZONE_Z
        
        # PID gains - sẽ được adaptive điều chỉnh
        self.kp_linear = KP_LINEAR
        self.ki_linear = KI_LINEAR
        self.kd_linear = KD_LINEAR
        
        self.kp_angular = KP_ANGULAR
        self.ki_angular = KI_ANGULAR
        self.kd_angular = KD_ANGULAR
        
        # Motion constraints
        self.max_linear_speed = MAX_LINEAR_SPEED
        self.max_angular_speed = MAX_ANGULAR_SPEED
        self.min_speed = MIN_SPEED_THRESHOLD
        
        # Advanced control parameters
        self.ema_alpha = EMA_ALPHA
        self.max_accel = MAX_ACCEL
        self.dt = 0.05               # Control loop time (20 FPS)
        self.max_integral = 0.5      # Anti-windup limit
        self.decel_zone_factor = 2.0 # Khoảng cách bắt đầu giảm tốc
        
        # Recovery parameters
        self.max_frames_without_target = 10  # 0.5s at 20FPS
        
        # Controller state
        self.state = ControllerState()
        
        # Virtual Depth Radar state
        self.radar_state = RadarState()
        
        # Radar configuration
        self.radar_ema_alpha = RADAR_EMA_ALPHA
        self.obstacle_threshold_side = OBSTACLE_THRESHOLD_SIDE
        self.obstacle_threshold_front = OBSTACLE_THRESHOLD_FRONT
        self.noise_threshold = NOISE_THRESHOLD
        
        # Radar scanning regions (calculated once)
        self.radar_scan_top = int(CAMERA_HEIGHT * RADAR_SCAN_TOP)
        self.radar_scan_bottom = int(CAMERA_HEIGHT * RADAR_SCAN_BOTTOM)
        self.radar_left_boundary = int(CAMERA_WIDTH * RADAR_LEFT_BOUNDARY)
        self.radar_right_boundary = int(CAMERA_WIDTH * RADAR_RIGHT_BOUNDARY)
        
        # Legacy compatibility
        self.last_command = "STOP"
        self.decision_count = 0
        self.last_center_error = 0.0
        self.last_distance_error = 0.0
        
        # Statistics
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
            # Radar statistics
            'radar_scans': 0,
            'obstacle_detections': 0,
            'override_left': 0,
            'override_right': 0,
            'emergency_stops': 0
        }
        
        # UART connection
        self.uart_port = uart_port
        self.uart_baudrate = uart_baudrate
        self.serial_conn = self._init_uart()
        
        print("Hybrid Predictive Controller with Virtual Depth Radar initialized")
    
    def _init_uart(self) -> Optional[serial.Serial]:
        """
        Initialize UART connection safely
        
        Returns:
            Serial connection or None if failed
        """
        try:
            conn = serial.Serial(
                port=self.uart_port,
                baudrate=self.uart_baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"[UART] ✅ Connected: {self.uart_port} @ {self.uart_baudrate} baud")
            return conn
        except serial.SerialException as e:
            print(f"[UART] ❌ Failed to connect {self.uart_port}: {e}")
            return None
    
    def _apply_ema_filter(self, new_error_x: float, new_error_z: float) -> Tuple[float, float]:
        """
        Áp dụng bộ lọc Exponential Moving Average để khử nhiễu
        
        Công thức EMA: filtered_value = α × new_value + (1-α) × prev_filtered_value
        - α = 0.2: Trọng số cho giá trị mới (20% mới, 80% cũ)
        - Giúp khử nhiễu từ camera và làm mượt tín hiệu điều khiển
        """
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
        """
        Dự báo chuyển động của mục tiêu (Lead Compensation)
        
        Tính vận tốc tức thời của target và dự báo vị trí ở frame tiếp theo
        để bù trừ độ trễ hệ thống (~50ms cho processing + communication)
        """
        current_time = time.time()
        
        if self.state.last_update_time > 0:
            dt = current_time - self.state.last_update_time
            
            if dt > 0:
                # Tính vận tốc tức thời
                self.state.target_velocity_x = (current_target_x - self.state.prev_target_x) / dt
                self.state.target_velocity_z = (current_target_z - self.state.prev_target_z) / dt
                
                # Lead compensation - dự báo vị trí sau 1 frame (dt)
                lead_time = self.dt  # Dự báo 1 frame ahead
                predicted_x = current_target_x + self.state.target_velocity_x * lead_time
                predicted_z = current_target_z + self.state.target_velocity_z * lead_time
                
                # Giới hạn prediction trong phạm vi hợp lý
                predicted_x = np.clip(predicted_x, 0, CAMERA_WIDTH)
                predicted_z = max(predicted_z, 100)  # Min 10cm
                
            else:
                predicted_x, predicted_z = current_target_x, current_target_z
        else:
            predicted_x, predicted_z = current_target_x, current_target_z
        
        # Cập nhật state
        self.state.prev_target_x = current_target_x
        self.state.prev_target_z = current_target_z
        self.state.last_update_time = current_time
        
        return predicted_x, predicted_z
    
    def _calculate_adaptive_gains(self, error_z: float) -> Tuple[float, float]:
        """
        Tính toán hệ số PID thích ứng dựa trên khoảng cách đến mục tiêu
        
        Khi robot gần đến vị trí mong muốn, giảm Kp để tránh overshoot
        """
        # Khoảng cách bắt đầu giảm tốc
        decel_start_distance = self.deadzone_z * self.decel_zone_factor
        
        if abs(error_z) < decel_start_distance:
            # Trong vùng giảm tốc - giảm Kp tuyến tính
            reduction_factor = abs(error_z) / decel_start_distance
            reduction_factor = max(reduction_factor, 0.3)  # Không giảm quá 70%
            
            adaptive_kp_linear = self.kp_linear * reduction_factor
            adaptive_kp_angular = self.kp_angular * reduction_factor
        else:
            # Ngoài vùng giảm tốc - sử dụng gain gốc
            adaptive_kp_linear = self.kp_linear
            adaptive_kp_angular = self.kp_angular
        
        return adaptive_kp_linear, adaptive_kp_angular
    
    def _pid_controller(self, error: float, prev_error: float, integral: float, 
                       kp: float, ki: float, kd: float) -> Tuple[float, float]:
        """
        Bộ điều khiển PID đầy đủ với anti-windup
        
        Công thức PID:
        - P (Proportional): Kp × error - Tỷ lệ với sai số hiện tại
        - I (Integral): Ki × ∫error dt - Tích lũy sai số theo thời gian  
        - D (Derivative): Kd × d(error)/dt - Tốc độ thay đổi sai số
        """
        # Proportional term
        p_term = kp * error
        
        # Integral term với anti-windup
        integral += error * self.dt
        integral = np.clip(integral, -self.max_integral, self.max_integral)
        i_term = ki * integral
        
        # Derivative term
        d_error = (error - prev_error) / self.dt
        d_term = kd * d_error
        
        # Tổng output PID
        pid_output = p_term + i_term + d_term
        
        return pid_output, integral
    
    def _apply_acceleration_limiting(self, target_v: float, target_w: float) -> Tuple[float, float]:
        """
        Giới hạn gia tốc để đảm bảo chuyển động mượt mà
        
        Nếu thay đổi vận tốc quá lớn giữa 2 frame, thực hiện tăng/giảm tốc tuyến tính
        """
        max_delta_v = self.max_accel * self.dt
        max_delta_w = (self.max_accel / 0.3) * self.dt  # Assuming 30cm wheelbase
        
        # Giới hạn thay đổi vận tốc tuyến tính
        delta_v = target_v - self.state.prev_v
        if abs(delta_v) > max_delta_v:
            limited_v = self.state.prev_v + np.sign(delta_v) * max_delta_v
        else:
            limited_v = target_v
        
        # Giới hạn thay đổi vận tốc góc
        delta_w = target_w - self.state.prev_w
        if abs(delta_w) > max_delta_w:
            limited_w = self.state.prev_w + np.sign(delta_w) * max_delta_w
        else:
            limited_w = target_w
        
        return limited_v, limited_w
    
    def _apply_cornering_speed_reduction(self, v: float, w: float) -> float:
        """
        Giảm tốc độ tuyến tính khi rẽ để đảm bảo an toàn
        
        Khi robot rẽ với tốc độ góc cao, tự động giảm tốc độ tiến
        để tránh quỹ đạo cong quá rộng và va chạm
        """
        if abs(w) > 0.1:  # Chỉ áp dụng khi có rẽ đáng kể
            # Hệ số giảm tốc dựa trên tốc độ rẽ
            angular_ratio = abs(w) / self.max_angular_speed
            reduction_factor = 0.5  # Giảm tối đa 50%
            
            speed_reduction = angular_ratio * reduction_factor
            v_reduced = v * (1 - speed_reduction)
            
            return v_reduced
        
        return v
    
    def _handle_lost_target(self) -> Tuple[float, float]:
        """
        Xử lý khi mất mục tiêu - Inertial Navigation
        
        Khi không detect được người:
        1. Tiếp tục xoay nhẹ theo hướng cũ trong 0.5s (10 frames)
        2. Nếu vẫn không thấy → Emergency stop và không gửi lệnh liên tục
        """
        self.state.frames_without_target += 1
        
        if self.state.frames_without_target <= self.max_frames_without_target:
            # Phase 1: Inertial navigation - tiếp tục xoay nhẹ
            recovery_w = self.state.last_w * 0.3  # Giảm 70% tốc độ xoay
            recovery_v = 0.0  # Không tiến khi đang tìm
            
            self.stats['recovery_attempts'] += 1
            return recovery_v, recovery_w
        else:
            # Phase 2: Emergency stop - chỉ gửi 1 lần
            if self.state.frames_without_target == self.max_frames_without_target + 1:
                self.stats['target_lost_count'] += 1
                return 0.0, 0.0
            else:
                # Đã gửi stop command rồi, không gửi nữa
                return None, None
    
    def _scan_virtual_depth_radar(self, depth_frame: np.ndarray) -> None:
        """
        Virtual Depth Radar - Multi-zone obstacle detection
        
        Scans depth frame in 3 vertical zones (Left, Center, Right) within
        the horizontal belt region (40%-60% height) to detect obstacles
        at robot's torso level, avoiding floor and ceiling noise.
        
        Args:
            depth_frame: Depth image array (height x width) in millimeters
        """
        if depth_frame is None or depth_frame.size == 0:
            return
        
        self.radar_state.total_scans += 1
        self.stats['radar_scans'] += 1
        
        # Get frame dimensions
        height, width = depth_frame.shape[:2]
        
        # Calculate scanning region (40%-60% of height for torso level)
        scan_top = int(height * RADAR_SCAN_TOP)
        scan_bottom = int(height * RADAR_SCAN_BOTTOM)
        
        # Calculate zone boundaries
        left_boundary = int(width * RADAR_LEFT_BOUNDARY)
        right_boundary = int(width * RADAR_RIGHT_BOUNDARY)
        
        # Extract scanning region
        scan_region = depth_frame[scan_top:scan_bottom, :]
        
        # Zone 1: Left (0% - 33%)
        left_zone = scan_region[:, :left_boundary]
        valid_left = left_zone[left_zone > self.noise_threshold]
        self.radar_state.min_depth_left = np.min(valid_left) if len(valid_left) > 0 else float('inf')
        
        # Zone 2: Center (34% - 66%)
        center_zone = scan_region[:, left_boundary:right_boundary]
        valid_center = center_zone[center_zone > self.noise_threshold]
        self.radar_state.min_depth_center = np.min(valid_center) if len(valid_center) > 0 else float('inf')
        
        # Zone 3: Right (67% - 100%)
        right_zone = scan_region[:, right_boundary:]
        valid_right = right_zone[right_zone > self.noise_threshold]
        self.radar_state.min_depth_right = np.min(valid_right) if len(valid_right) > 0 else float('inf')
        
        # Apply EMA filtering for smooth readings
        self._apply_radar_ema_filtering()
        
        # Update obstacle detection flags
        self.radar_state.obstacle_left = self.radar_state.filtered_depth_left < self.obstacle_threshold_side
        self.radar_state.obstacle_center = self.radar_state.filtered_depth_center < self.obstacle_threshold_front
        self.radar_state.obstacle_right = self.radar_state.filtered_depth_right < self.obstacle_threshold_side
        
        # Count obstacle detections
        if any([self.radar_state.obstacle_left, self.radar_state.obstacle_center, self.radar_state.obstacle_right]):
            self.radar_state.obstacle_detections += 1
            self.stats['obstacle_detections'] += 1
    
    def _apply_radar_ema_filtering(self) -> None:
        """
        Apply EMA filtering to radar depth readings for smooth obstacle detection
        
        Uses separate EMA coefficient for radar to balance responsiveness vs stability
        """
        alpha = self.radar_ema_alpha
        
        # Initialize filtered values on first reading
        if self.radar_state.filtered_depth_left == float('inf'):
            self.radar_state.filtered_depth_left = self.radar_state.min_depth_left
            self.radar_state.filtered_depth_center = self.radar_state.min_depth_center
            self.radar_state.filtered_depth_right = self.radar_state.min_depth_right
        else:
            # Apply EMA filtering
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
        """
        Apply obstacle avoidance override logic to control commands
        
        Priority system:
        1. Emergency Stop: Front obstacle < 400mm → Stop immediately
        2. Side Avoidance: Side obstacle < 500mm → Cancel turning, keep straight motion
        3. Normal Control: No obstacles → Use original PID commands
        
        Args:
            control_v: Original linear velocity from PID controller
            control_w: Original angular velocity from PID controller
            
        Returns:
            Tuple of (final_v, final_w) after obstacle override
        """
        override_v = control_v
        override_w = control_w
        override_reason = ""
        
        # Priority 1: Emergency Stop - Front collision imminent
        if self.radar_state.obstacle_center:
            override_v = 0.0
            override_w = 0.0
            override_reason = "EMERGENCY_STOP - Front obstacle detected"
            self.stats['emergency_stops'] += 1
            print(f"[RADAR] 🚨 PHANH KHẨN CẤP - Vật cản phía trước: {self.radar_state.filtered_depth_center:.0f}mm")
        
        # Priority 2: Side Obstacle Avoidance - Cancel turning to avoid collision
        elif control_w > 0 and self.radar_state.obstacle_right:
            # Turning right but right side blocked → Go straight
            override_w = 0.0
            override_reason = "RIGHT_SIDE_BLOCKED - Going straight to clear obstacle"
            self.stats['override_right'] += 1
            print(f"[RADAR] ⚠️ Vướng sườn PHẢI - Đang đi thẳng vượt vật cản ({self.radar_state.filtered_depth_right:.0f}mm)")
        
        elif control_w < 0 and self.radar_state.obstacle_left:
            # Turning left but left side blocked → Go straight  
            override_w = 0.0
            override_reason = "LEFT_SIDE_BLOCKED - Going straight to clear obstacle"
            self.stats['override_left'] += 1
            print(f"[RADAR] ⚠️ Vướng sườn TRÁI - Đang đi thẳng vượt vật cản ({self.radar_state.filtered_depth_left:.0f}mm)")
        
        # Update override state
        self.radar_state.override_active = override_reason != ""
        self.radar_state.override_reason = override_reason
        
        return override_v, override_w
    
    def _get_radar_status_string(self) -> str:
        """
        Get radar status string for display
        
        Returns:
            Formatted string with radar readings and obstacle status
        """
        left_status = "🔴" if self.radar_state.obstacle_left else "🟢"
        center_status = "🔴" if self.radar_state.obstacle_center else "🟢"
        right_status = "🔴" if self.radar_state.obstacle_right else "🟢"
        
        left_dist = f"{self.radar_state.filtered_depth_left:.0f}" if self.radar_state.filtered_depth_left != float('inf') else "∞"
        center_dist = f"{self.radar_state.filtered_depth_center:.0f}" if self.radar_state.filtered_depth_center != float('inf') else "∞"
        right_dist = f"{self.radar_state.filtered_depth_right:.0f}" if self.radar_state.filtered_depth_right != float('inf') else "∞"
        
        return f"L:{left_status}{left_dist} C:{center_status}{center_dist} R:{right_status}{right_dist}mm"
    
    def send_to_esp32(self, v_speed: float, w_speed: float) -> None:
        """
        Gửi lệnh điều khiển xuống ESP32 qua UART
        
        Format: "v,w\n" với 3 chữ số thập phân
        """
        # Skip sending if None (already sent stop command)
        if v_speed is None or w_speed is None:
            return
            
        msg = f"{v_speed:.3f},{w_speed:.3f}\n"
        
        if self.serial_conn is not None and self.serial_conn.is_open:
            try:
                self.serial_conn.write(msg.encode('utf-8'))
                print(f"[UART TX] {msg.strip()}")
            except serial.SerialException as e:
                print(f"[UART] ❌ Send error: {e}")
        else:
            print(f"[UART] ⚠️ Not connected - Skip: {msg.strip()}")
    
    def process_target(self, bbox: List[int], depth_mm: float, 
                      frame_width: int, frame_height: int, depth_frame: Optional[np.ndarray] = None) -> str:
        """
        Process target with Hybrid Predictive Control + Virtual Depth Radar
        
        Pipeline:
        1. Virtual Depth Radar: Multi-zone obstacle detection
        2. Signal Conditioning: EMA filter + Target prediction
        3. Advanced PID: Full PID with adaptive gains
        4. Motion Profile: Smooth pursuit + acceleration limiting
        5. Obstacle Override: Safety-first collision avoidance
        6. UART Communication: Send final commands to robot
        """
        self.stats['total_decisions'] += 1
        self.decision_count += 1
        
        # Cập nhật camera center theo resolution thực tế
        self.camera_center_x = frame_width // 2
        self.camera_center_y = frame_height // 2
        
        if len(bbox) != 4 or depth_mm <= 0:
            # Mất mục tiêu - chuyển sang recovery mode
            recovery_v, recovery_w = self._handle_lost_target()
            
            # Only send command if not None (avoid infinite stop commands)
            if recovery_v is not None and recovery_w is not None:
                self.send_to_esp32(recovery_v, recovery_w)
                self.last_command = f"LOST_TARGET_RECOVERY: v={recovery_v:.3f}, w={recovery_w:.3f}"
            else:
                self.last_command = "TARGET_LOST_SILENT"
                
            return self.last_command
        
        # Reset lost target counter
        self.state.frames_without_target = 0
        
        # 1. VIRTUAL DEPTH RADAR - Multi-zone obstacle detection
        if depth_frame is not None:
            self._scan_virtual_depth_radar(depth_frame)
        
        # 2. SIGNAL CONDITIONING & STATE ESTIMATION
        x1, y1, x2, y2 = bbox
        current_target_x = (x1 + x2) // 2
        current_target_y = (y1 + y2) // 2
        
        # Target prediction với lead compensation
        predicted_x, predicted_z = self._predict_target_motion(current_target_x, depth_mm)
        
        # Tính sai số dựa trên predicted position
        raw_error_x = predicted_x - self.camera_center_x
        raw_error_z = predicted_z - self.safe_distance_mm
        
        # EMA filtering để khử nhiễu
        filtered_error_x, filtered_error_z = self._apply_ema_filter(raw_error_x, raw_error_z)
        
        # Update legacy compatibility
        self.last_center_error = filtered_error_x
        self.last_distance_error = filtered_error_z
        
        # 3. ADVANCED PID CONTROLLER
        self.stats['pid_calls'] += 1
        
        # Adaptive gains dựa trên khoảng cách
        adaptive_kp_linear, adaptive_kp_angular = self._calculate_adaptive_gains(filtered_error_z)
        
        # PID cho trục Z (distance)
        pid_v, self.state.integral_z = self._pid_controller(
            filtered_error_z, self.state.prev_error_z, self.state.integral_z,
            adaptive_kp_linear, self.ki_linear, self.kd_linear
        )
        
        # PID cho trục X (angle)  
        pid_w, self.state.integral_x = self._pid_controller(
            filtered_error_x, self.state.prev_error_x, self.state.integral_x,
            adaptive_kp_angular, self.ki_angular, self.kd_angular
        )
        
        # Cập nhật previous errors
        self.state.prev_error_x = filtered_error_x
        self.state.prev_error_z = filtered_error_z
        
        # 4. MOTION PROFILE & SAFETY
        
        # Giới hạn vận tốc trong phạm vi cho phép
        target_v = np.clip(pid_v, -self.max_linear_speed, self.max_linear_speed)
        target_w = np.clip(pid_w, -self.max_angular_speed, self.max_angular_speed)
        
        # Áp dụng deadzone
        if abs(filtered_error_z) <= self.deadzone_z:
            target_v = 0.0
        if abs(filtered_error_x) <= self.deadzone_x:
            target_w = 0.0
        
        # Acceleration ramping
        limited_v, limited_w = self._apply_acceleration_limiting(target_v, target_w)
        
        # Cornering speed reduction
        final_v = self._apply_cornering_speed_reduction(limited_v, limited_w)
        final_w = limited_w
        
        # Áp dụng minimum speed threshold
        if abs(final_v) < self.min_speed:
            final_v = 0.0
        if abs(final_w) < self.min_speed:
            final_w = 0.0
        
        # 5. VIRTUAL DEPTH RADAR - Obstacle Override Logic
        if depth_frame is not None:
            final_v, final_w = self._apply_obstacle_override_logic(final_v, final_w)
        
        # Cập nhật state cho frame tiếp theo
        self.state.prev_v = final_v
        self.state.prev_w = final_w
        self.state.last_w = final_w  # Lưu cho recovery
        
        # Update legacy stats
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
        
        # 6. UART COMMUNICATION
        self.send_to_esp32(final_v, final_w)
        
        # Tạo mô tả hành động với radar status
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
        """Get controller operation statistics including radar data"""
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
        """Reset all statistics including radar data"""
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
            # Radar statistics
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
        """Legacy compatibility"""
        return f"Nav:{self.last_command} Decisions:{self.decision_count}"
    
    def emergency_stop(self) -> None:
        """Dừng khẩn cấp"""
        self.send_to_esp32(0.0, 0.0)
        self.last_command = "EMERGENCY_STOP"
    
    def stop(self) -> None:
        """Legacy compatibility - stop robot and reset state"""
        self.emergency_stop()
        # Reset lost target counter to prevent continuous recovery commands
        self.state.frames_without_target = 0
        # Reset radar override state
        self.radar_state.override_active = False
        self.radar_state.override_reason = ""
    
    def close(self) -> None:
        """Đóng kết nối và dọn dẹp tài nguyên"""
        self.emergency_stop()
        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[UART] 🔌 Connection closed")
    
    def __del__(self):
        """Destructor - đảm bảo đóng kết nối khi object bị hủy"""
        try:
            self.close()
        except:
            pass


# Legacy compatibility functions
def test_visual_servoing():
    """Test function for Hybrid Predictive Controller with Virtual Depth Radar"""
    print("🧪 Testing Hybrid Predictive Controller with Virtual Depth Radar")
    
    controller = VisualServoingDecisionMaker(uart_port=None)
    
    # Create mock depth frame for testing
    mock_depth = np.full((240, 320), 2000, dtype=np.uint16)  # 2m background
    # Add some obstacles
    mock_depth[96:144, 0:106] = 450    # Left obstacle (close)
    mock_depth[96:144, 107:213] = 350  # Center obstacle (very close)
    mock_depth[96:144, 214:320] = 600  # Right obstacle (medium)
    
    test_cases = [
        ([300, 200, 400, 400], 2500, mock_depth, "Person far and centered with obstacles"),
        ([450, 200, 550, 400], 2000, mock_depth, "Person far and right with obstacles"),
        ([100, 200, 200, 400], 1800, mock_depth, "Person far and left with obstacles"),
        ([320, 200, 420, 400], 1500, None, "Person at safe distance, no radar"),
        ([], 0, mock_depth, "Lost target with obstacles"),
    ]
    
    for i, (bbox, depth, depth_frame, description) in enumerate(test_cases, 1):
        print(f"\n🎯 Test {i}: {description}")
        action = controller.process_target(bbox, depth, 640, 480, depth_frame)
        print(f"Action: {action}")
        
        if depth_frame is not None:
            radar_status = controller._get_radar_status_string()
            print(f"Radar: {radar_status}")
        
        time.sleep(0.1)
    
    stats = controller.get_statistics()
    print(f"\n📊 Final Statistics:")
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