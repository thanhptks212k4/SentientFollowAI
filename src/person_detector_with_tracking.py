"""
Person Detection with ByteTrack Tracking
- YOLO detection
- ByteTrack for robust tracking
- Target locking system
- Voice command trigger simulation
"""

import cv2
import numpy as np
import time
import os
from threading import Thread, Lock
from collections import deque

# Import tracking module
from bytetrack_tracker import ByteTracker, TargetLocker
from astra_camera import AstraCamera, PYORBBECSDK_AVAILABLE

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Path
MODEL_PATH = 'models/yolov5nu_int8.onnx'

# Camera Settings
CAMERA_ID = 0
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# Model Settings
INPUT_SIZE = 320
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0

# Performance Settings
AI_INTERVAL = 0.1  # Run AI every 100ms (~10 FPS)
FORCE_USB_CAMERA = False  # Set to True to force USB camera

# Tracking Settings
TRACK_THRESH = 0.5  # High confidence threshold for tracking
TRACK_BUFFER = 90   # Keep lost tracks for 90 frames (3 seconds at 30 FPS)
MATCH_THRESH = 0.7  # Lower threshold for better matching when moving fast

# Depth Settings (for Astra camera)
SHOW_DEPTH_WINDOW = False  # Toggle depth visualization
MAX_DEPTH_RANGE = 5000  # Max depth in mm (5 meters)
DEPTH_COLORMAP = cv2.COLORMAP_JET  # Depth colormap

# ============================================================================
# THREADED CAMERA STREAM
# ============================================================================
class ThreadedCamera:
    """Threaded camera capture for zero-latency frame reading"""
    
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame = None
        self.lock = Lock()
        self.running = False
        
        self.frames_read = 0
        self.frames_dropped = 0
        
        # Camera FPS tracking
        self.camera_fps = 0
        self.fps_frame_count = 0
        self.fps_start_time = time.time()
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")
        
        print(f"[Camera] Initialized (ID: {src})")
    
    def start(self):
        if self.running:
            return self
        
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        time.sleep(0.5)
        print("[Camera] Thread started")
        return self
    
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            with self.lock:
                if self.frame is not None:
                    self.frames_dropped += 1
                self.frame = frame
                self.frames_read += 1
                self.fps_frame_count += 1
                
                # Calculate Camera FPS
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.camera_fps = self.fps_frame_count / elapsed
                    self.fps_frame_count = 0
                    self.fps_start_time = time.time()
    
    def read(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            cam_fps = self.camera_fps
        return frame, cam_fps
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()
        print(f"[Camera] Stopped")


# ============================================================================
# PREPROCESSING
# ============================================================================
class PreProcessor:
    """Optimized preprocessing for YOLO models"""
    
    def __init__(self, input_size=320):
        self.input_size = input_size
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                  cv2.BORDER_CONSTANT, value=color)
        
        return img, r, (dw, dh)
    
    def preprocess(self, frame):
        img, ratio, pad = self.letterbox(frame, new_shape=self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img, ratio, pad


# ============================================================================
# ONNX DETECTOR
# ============================================================================
class ONNXDetector:
    """ONNX Runtime inference engine"""
    
    def __init__(self, model_path, input_size=320, conf_thresh=0.4, iou_thresh=0.45):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install: pip install onnxruntime")
        
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 1  # Reduce to 1 thread for lower CPU
        sess_options.inter_op_num_threads = 1
        
        print(f"[ONNX] Loading model: {model_path}")
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"[ONNX] Model loaded")
    
    def infer(self, img):
        outputs = self.session.run(self.output_names, {self.input_name: img})
        return outputs[0]
    
    def postprocess(self, predictions, orig_shape, ratio, pad):
        predictions = predictions[0].T
        
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        
        class_ids = np.argmax(class_scores, axis=1)
        class_confs = np.max(class_scores, axis=1)
        
        mask = (class_confs > self.conf_thresh) & (class_ids == PERSON_CLASS_ID)
        boxes = boxes[mask]
        class_confs = class_confs[mask]
        
        if len(boxes) == 0:
            return []
        
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (x_center - width / 2 - pad[0]) / ratio
        y1 = (y_center - height / 2 - pad[1]) / ratio
        x2 = (x_center + width / 2 - pad[0]) / ratio
        y2 = (y_center + height / 2 - pad[1]) / ratio
        
        x1 = np.clip(x1, 0, orig_shape[1])
        y1 = np.clip(y1, 0, orig_shape[0])
        x2 = np.clip(x2, 0, orig_shape[1])
        y2 = np.clip(y2, 0, orig_shape[0])
        
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        indices = self.nms(boxes_xyxy, class_confs, self.iou_thresh)
        
        detections = []
        for idx in indices:
            detections.append({
                'bbox': boxes_xyxy[idx].astype(int),
                'confidence': float(class_confs[idx]),
                'class_id': PERSON_CLASS_ID
            })
        
        return detections
    
    @staticmethod
    def nms(boxes, scores, iou_threshold):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep


# ============================================================================
# MAIN DETECTION + TRACKING PIPELINE
# ============================================================================
def main():
    """Main detection and tracking loop"""
    
    global SHOW_DEPTH_WINDOW  # Declare global at the beginning
    
    print("\n" + "="*70)
    print("Person Detection with ByteTrack Tracking")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"AI Interval: {AI_INTERVAL*1000:.0f}ms (~{1/AI_INTERVAL:.1f} FPS)")
    print(f"Track Buffer: {TRACK_BUFFER} frames (~{TRACK_BUFFER/30:.1f}s)")
    print(f"Match Threshold: {MATCH_THRESH} (optimized for fast movement)")
    print("="*70 + "\n")
    
    # Initialize components
    print("[1/4] Initializing camera...")
    if PYORBBECSDK_AVAILABLE and not FORCE_USB_CAMERA:
        print("[Camera] Using Orbbec Astra camera (pyorbbecsdk)")
        try:
            camera = AstraCamera(fps=CAMERA_FPS).start()
            using_astra = True
        except Exception as e:
            print(f"[Camera Error] Failed to init AstraCamera: {e}")
            print("[Camera] Falling back to standard USB camera (cv2.VideoCapture)")
            camera = ThreadedCamera(src=CAMERA_ID, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS).start()
            using_astra = False
    else:
        if FORCE_USB_CAMERA:
            print("[Camera] Forced to use standard USB camera (cv2.VideoCapture)")
        else:
            print("[Camera] Using standard USB camera (cv2.VideoCapture)")
        camera = ThreadedCamera(
            src=CAMERA_ID,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS
        ).start()
        using_astra = False
    
    print("\n[2/4] Initializing preprocessor...")
    preprocessor = PreProcessor(input_size=INPUT_SIZE)
    
    print("\n[3/4] Loading detector...")
    detector = ONNXDetector(
        model_path=MODEL_PATH,
        input_size=INPUT_SIZE,
        conf_thresh=CONFIDENCE_THRESHOLD,
        iou_thresh=IOU_THRESHOLD
    )
    
    print("\n[4/4] Initializing tracker...")
    tracker = ByteTracker(
        track_thresh=TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH
    )
    target_locker = TargetLocker(tracker)
    print("[Tracker] ByteTrack initialized")
    
    print("\nStarting detection loop...")
    print("Optimizations:")
    print(f"  - AI runs every {AI_INTERVAL*1000:.0f}ms (~{1/AI_INTERVAL:.1f} FPS) to save CPU")
    print(f"  - Track buffer: {TRACK_BUFFER} frames for stable tracking")
    print(f"  - Single thread inference for lower CPU usage")
    print("\nFeatures:")
    print("  - ALWAYS LOCKED: System always locks a person when detected")
    print("  - Auto switch to new person if current target is lost")
    print("  - All persons are tracked with unique IDs")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Show stats")
    print("  'd' - Toggle depth window (Astra camera only)")
    print("="*70 + "\n")
    
    # Performance tracking
    fps = 0
    frame_count = 0
    start_time = time.time()
    inference_times = deque(maxlen=30)
    
    # AI FPS tracking
    ai_fps = 0
    ai_frame_count = 0
    ai_start_time = time.time()
    
    # Time-based AI execution
    last_ai_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # Cơ chế nhả CPU tránh Busy Waiting
            if current_time - last_ai_time < AI_INTERVAL:
                time.sleep(0.01)  # Ngủ 10ms để giảm tải CPU
                continue
            
            last_ai_time = current_time
            
            if using_astra:
                frame, depth_frame, camera_fps = camera.read()
            else:
                frame, camera_fps = camera.read()
                depth_frame = None
                
            if frame is None:
                continue
            
            frame_count += 1
            orig_shape = frame.shape[:2]
            
            # Preprocessing
            img, ratio, pad = preprocessor.preprocess(frame)
            
            # Inference
            t1 = time.time()
            predictions = detector.infer(img)
            inference_time = (time.time() - t1) * 1000
            inference_times.append(inference_time)
            
            # Post-processing
            detections = detector.postprocess(predictions, orig_shape, ratio, pad)
            
            # Update AI FPS (only when inference runs)
            ai_frame_count += 1
            ai_elapsed = time.time() - ai_start_time
            if ai_elapsed >= 1.0:
                ai_fps = ai_frame_count / ai_elapsed
                ai_frame_count = 0
                ai_start_time = time.time()
            
            # Tracking - Always update tracker to get all tracks
            all_tracks = tracker.update(detections)
            
            # Auto-lock: Always maintain lock when there are people
            if len(all_tracks) > 0:
                if not target_locker.is_locked:
                    # Lock the first tracked person
                    first_track = all_tracks[0]
                    first_detection = {
                        'bbox': first_track['bbox'],
                        'confidence': first_track['score'],
                        'class_id': PERSON_CLASS_ID
                    }
                    if target_locker.lock_target([first_detection]):
                        print(f"\n[AUTO-LOCK] Locked person - ID:{first_track['track_id']}")
                
                # Update locked target
                target = target_locker.update(detections)
                
                # If target is lost but there are still people, lock the first one
                if target is None and len(all_tracks) > 0:
                    target_locker.unlock_target()  # Force unlock
                    first_track = all_tracks[0]
                    first_detection = {
                        'bbox': first_track['bbox'],
                        'confidence': first_track['score'],
                        'class_id': PERSON_CLASS_ID
                    }
                    if target_locker.lock_target([first_detection]):
                        target = target_locker.update([])
                        print(f"\n[RE-LOCK] Switched to new person - ID:{first_track['track_id']}")
            else:
                # No people detected - unlock
                if target_locker.is_locked:
                    target_locker.unlock_target()
                target = None
            
            # Draw all tracked persons
            for track in all_tracks:
                tid = track['track_id']
                bbox = track['bbox']
                score = track['score']
                x1, y1, x2, y2 = bbox
                
                # Validate bbox coordinates
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue  # Skip invalid bboxes
                
                x_c = (x1 + x2) // 2
                y_c = (y1 + y2) // 2
                
                # Check if this is the locked target
                is_locked_target = (target is not None and tid == target['track_id'])
                
                if is_locked_target:
                    # Draw locked target (GREEN with thick border)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw center crosshair
                    cv2.circle(frame, (x_c, y_c), 8, (0, 0, 255), -1)
                    cv2.line(frame, (x_c - 15, y_c), (x_c + 15, y_c), (0, 0, 255), 2)
                    cv2.line(frame, (x_c, y_c - 15), (x_c, y_c + 15), (0, 0, 255), 2)
                    
                    # Calculate distance from depth frame
                    distance_str = ""
                    if depth_frame is not None:
                        try:
                            depth_h, depth_w = depth_frame.shape[:2]
                            
                            # Scale bounding box to depth frame size if needed
                            scale_x = depth_w / frame.shape[1]
                            scale_y = depth_h / frame.shape[0]
                            
                            depth_x1 = int(x1 * scale_x)
                            depth_y1 = int(y1 * scale_y)
                            depth_x2 = int(x2 * scale_x)
                            depth_y2 = int(y2 * scale_y)
                            
                            # Ensure coordinates are within depth frame bounds
                            depth_x1 = max(0, min(depth_x1, depth_w - 1))
                            depth_y1 = max(0, min(depth_y1, depth_h - 1))
                            depth_x2 = max(0, min(depth_x2, depth_w - 1))
                            depth_y2 = max(0, min(depth_y2, depth_h - 1))
                            
                            if depth_x2 > depth_x1 and depth_y2 > depth_y1:
                                # Extract depth values in the person's bounding box
                                person_depth_region = depth_frame[depth_y1:depth_y2, depth_x1:depth_x2]
                                
                                # Filter out zero/invalid depth values
                                valid_depths = person_depth_region[person_depth_region > 0]
                                
                                if len(valid_depths) > 0:
                                    # Use median for more robust distance estimation
                                    median_dist_mm = np.median(valid_depths)
                                    distance_str = f" - {median_dist_mm/1000.0:.2f}m"
                                    
                                    # Also show min/max for debugging
                                    min_dist = np.min(valid_depths) / 1000.0
                                    max_dist = np.max(valid_depths) / 1000.0
                                    distance_str += f" ({min_dist:.2f}-{max_dist:.2f}m)"
                        except Exception as e:
                            distance_str = " - N/A"  # Fallback if depth calculation fails
                    
                    # Draw label
                    label = f"TARGET ID:{tid} ({score:.2f}){distance_str}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw coordinates
                    coord_text = f"({x_c}, {y_c})"
                    cv2.putText(frame, coord_text, (x_c + 15, y_c - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Draw other tracked persons (CYAN)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    
                    # Calculate distance for other persons too
                    distance_str = ""
                    if depth_frame is not None:
                        try:
                            depth_h, depth_w = depth_frame.shape[:2]
                            scale_x = depth_w / frame.shape[1]
                            scale_y = depth_h / frame.shape[0]
                            
                            depth_x1 = int(x1 * scale_x)
                            depth_y1 = int(y1 * scale_y)
                            depth_x2 = int(x2 * scale_x)
                            depth_y2 = int(y2 * scale_y)
                            
                            depth_x1 = max(0, min(depth_x1, depth_w - 1))
                            depth_y1 = max(0, min(depth_y1, depth_h - 1))
                            depth_x2 = max(0, min(depth_x2, depth_w - 1))
                            depth_y2 = max(0, min(depth_y2, depth_h - 1))
                            
                            if depth_x2 > depth_x1 and depth_y2 > depth_y1:
                                person_depth_region = depth_frame[depth_y1:depth_y2, depth_x1:depth_x2]
                                valid_depths = person_depth_region[person_depth_region > 0]
                                
                                if len(valid_depths) > 0:
                                    median_dist_mm = np.median(valid_depths)
                                    distance_str = f" - {median_dist_mm/1000.0:.2f}m"
                        except Exception as e:
                            distance_str = " - N/A"
                    
                    # Draw label
                    label = f"ID:{tid} ({score:.2f}){distance_str}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Show depth visualization if enabled and available
            if SHOW_DEPTH_WINDOW and depth_frame is not None:
                # Create depth visualization
                depth_vis = depth_frame.copy()
                
                # Normalize depth for visualization (0-MAX_DEPTH_RANGE -> 0-255)
                depth_vis = np.clip(depth_vis, 0, MAX_DEPTH_RANGE)
                depth_vis = (depth_vis * 255.0 / MAX_DEPTH_RANGE).astype(np.uint8)
                
                # Apply colormap
                depth_colored = cv2.applyColorMap(depth_vis, DEPTH_COLORMAP)
                
                # Draw bounding boxes on depth image
                for track in all_tracks:
                    bbox = track['bbox']
                    tid = track['track_id']
                    x1, y1, x2, y2 = bbox
                    
                    # Scale to depth frame
                    depth_h, depth_w = depth_frame.shape[:2]
                    scale_x = depth_w / frame.shape[1]
                    scale_y = depth_h / frame.shape[0]
                    
                    depth_x1 = int(x1 * scale_x)
                    depth_y1 = int(y1 * scale_y)
                    depth_x2 = int(x2 * scale_x)
                    depth_y2 = int(y2 * scale_y)
                    
                    # Draw rectangle
                    is_locked = (target is not None and tid == target['track_id'])
                    color = (0, 255, 0) if is_locked else (255, 255, 0)
                    thickness = 3 if is_locked else 2
                    
                    cv2.rectangle(depth_colored, (depth_x1, depth_y1), (depth_x2, depth_y2), color, thickness)
                    cv2.putText(depth_colored, f"ID:{tid}", (depth_x1, depth_y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add depth scale info
                cv2.putText(depth_colored, f"Depth Range: 0-{MAX_DEPTH_RANGE/1000.0:.1f}m", 
                           (10, depth_colored.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.imshow('Depth View', depth_colored)
            
            # Draw status indicator
            if target_locker.is_locked and target is not None:
                cv2.putText(frame, f"LOCKED: ID {target['track_id']}", (10, orig_shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif len(all_tracks) > 0:
                cv2.putText(frame, f"TRACKING {len(all_tracks)} person(s)", 
                           (10, orig_shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Waiting for person...", 
                           (10, orig_shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Draw performance info
            y_offset = 30
            cv2.putText(frame, f"Camera FPS: {camera_fps:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            y_offset += 30
            cv2.putText(frame, f"AI FPS: {ai_fps:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            y_offset += 30
            avg_inference = np.mean(inference_times) if inference_times else 0
            cv2.putText(frame, f"Inference: {avg_inference:.1f}ms", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
            cv2.putText(frame, f"Tracks: {len(all_tracks)}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Person Tracking - ByteTrack', frame)
            
            # Console output (update every AI cycle to reduce overhead)
            if ai_frame_count % 5 == 0:  # Update every 5 AI cycles
                if target_locker.is_locked and target:
                    try:
                        x_c = (target['bbox'][0] + target['bbox'][2]) // 2
                        y_c = (target['bbox'][1] + target['bbox'][3]) // 2
                        
                        # Get distance info for console
                        distance_info = ""
                        if depth_frame is not None:
                            try:
                                depth_h, depth_w = depth_frame.shape[:2]
                                scale_x = depth_w / frame.shape[1]
                                scale_y = depth_h / frame.shape[0]
                                
                                depth_x1 = int(target['bbox'][0] * scale_x)
                                depth_y1 = int(target['bbox'][1] * scale_y)
                                depth_x2 = int(target['bbox'][2] * scale_x)
                                depth_y2 = int(target['bbox'][3] * scale_y)
                                
                                depth_x1 = max(0, min(depth_x1, depth_w - 1))
                                depth_y1 = max(0, min(depth_y1, depth_h - 1))
                                depth_x2 = max(0, min(depth_x2, depth_w - 1))
                                depth_y2 = max(0, min(depth_y2, depth_h - 1))
                                
                                if depth_x2 > depth_x1 and depth_y2 > depth_y1:
                                    person_depth_region = depth_frame[depth_y1:depth_y2, depth_x1:depth_x2]
                                    valid_depths = person_depth_region[person_depth_region > 0]
                                    
                                    if len(valid_depths) > 0:
                                        median_dist_mm = np.median(valid_depths)
                                        distance_info = f" | Dist: {median_dist_mm/1000.0:.2f}m"
                            except Exception as e:
                                distance_info = " | Dist: N/A"
                        
                        print(f"\r[LOCKED] Target ID:{target['track_id']} at ({x_c:4d}, {y_c:4d}){distance_info} | "
                              f"Tracks: {len(all_tracks)} | Cam: {camera_fps:5.1f} | AI: {ai_fps:5.1f} | "
                              f"Inf: {avg_inference:4.0f}ms", 
                              end='', flush=True)
                    except Exception as e:
                        print(f"\r[LOCKED] Target ID:{target['track_id']} (invalid coords) | "
                              f"Tracks: {len(all_tracks)} | Cam: {camera_fps:5.1f} | AI: {ai_fps:5.1f} | "
                              f"Inf: {avg_inference:4.0f}ms", 
                              end='', flush=True)
                elif len(all_tracks) > 0:
                    print(f"\r[TRACKING] {len(all_tracks)} person(s) | "
                          f"Cam: {camera_fps:5.1f} | AI: {ai_fps:5.1f} | Inf: {avg_inference:4.0f}ms", 
                          end='', flush=True)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Show stats
                status = target_locker.get_status()
                print(f"\n\n{'='*70}")
                print("Tracking Statistics:")
                print(f"{'='*70}")
                print(f"Camera FPS: {camera_fps:.2f}")
                print(f"AI FPS: {ai_fps:.2f}")
                print(f"Avg Inference: {np.mean(inference_times):.2f}ms")
                print(f"Target Locked: {status['is_locked']}")
                if status['is_locked']:
                    print(f"Target ID: {status['target_id']}")
                    print(f"Frames without target: {status['frames_without_target']}")
                print(f"Using Astra Camera: {using_astra}")
                if using_astra and depth_frame is not None:
                    print(f"Depth Frame Size: {depth_frame.shape}")
                    print(f"Depth Range: {np.min(depth_frame[depth_frame > 0])}-{np.max(depth_frame)}mm")
                print(f"{'='*70}\n")
            elif key == ord('d'):
                # Toggle depth window
                if using_astra:
                    SHOW_DEPTH_WINDOW = not SHOW_DEPTH_WINDOW
                    if not SHOW_DEPTH_WINDOW:
                        cv2.destroyWindow('Depth View')
                    print(f"\nDepth window: {'ON' if SHOW_DEPTH_WINDOW else 'OFF'}")
                else:
                    print("\nDepth view only available with Astra camera")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("\n" + "="*70)
        print("Detection stopped")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()