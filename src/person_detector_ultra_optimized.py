"""
Ultra-Optimized Person Detection with ByteTrack
- AI Inference: 4 FPS (250ms interval) - Time-based, not frame-based
- Display Stream: 30 FPS (smooth, independent)
- Clean UI: Only tracking info, no raw detection boxes
- Maximum CPU/GPU efficiency
"""

import cv2
import numpy as np
import time
from threading import Thread, Lock
from collections import deque

# Import ByteTrack
import sys
sys.path.append('src')
from bytetrack_tracker import ByteTracker, TargetLocker

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'models/yolov5nu_int8.onnx'

# Camera Settings
CAMERA_ID = 0
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# Model Settings
INPUT_SIZE = 256
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0

# AI Inference Settings (Time-based)
AI_INFERENCE_INTERVAL = 0.05  # 250ms = 4 FPS (4 times per second)

# Tracking Settings
TRACK_THRESH = 0.5
TRACK_BUFFER = 30
MATCH_THRESH = 0.8

# ============================================================================
# THREADED CAMERA
# ============================================================================

class ThreadedCamera:
    """Threaded camera for continuous frame capture"""
    
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
        
        # Camera FPS tracking
        self.camera_fps = 0
        self.camera_frame_count = 0
        self.camera_fps_start = time.time()
        
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
        """Continuously capture frames"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            with self.lock:
                self.frame = frame
                self.camera_frame_count += 1
            
            # Calculate camera FPS
            elapsed = time.time() - self.camera_fps_start
            if elapsed >= 1.0:
                with self.lock:
                    self.camera_fps = self.camera_frame_count / elapsed
                    self.camera_frame_count = 0
                    self.camera_fps_start = time.time()
    
    def read(self):
        """Get latest frame"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.camera_fps
            return None, 0
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()
        print("[Camera] Stopped")


# ============================================================================
# PREPROCESSING
# ============================================================================

class PreProcessor:
    """Optimized preprocessing for YOLO"""
    
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
        sess_options.intra_op_num_threads = 2
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
# CLEAN UI VISUALIZATION (No raw detection boxes)
# ============================================================================

def draw_clean_tracking(frame, tracks, locked_target_id=None):
    """
    Draw clean tracking visualization
    - Only show tracking boxes (no raw detections)
    - Highlight locked target
    - Minimal, clean interface
    """
    for track in tracks:
        track_id = track['track_id']
        bbox = track['bbox']
        score = track['score']
        
        x1, y1, x2, y2 = bbox
        x_c = (x1 + x2) // 2
        y_c = (y1 + y2) // 2
        
        # Color and style based on lock status
        if locked_target_id is not None and track_id == locked_target_id:
            # Locked target: GREEN, thick
            color = (0, 255, 0)
            thickness = 3
            label = f"TARGET ID:{track_id}"
            
            # Draw crosshair for locked target
            cv2.circle(frame, (x_c, y_c), 8, (0, 0, 255), -1)
            cv2.line(frame, (x_c - 15, y_c), (x_c + 15, y_c), (0, 0, 255), 2)
            cv2.line(frame, (x_c, y_c - 15), (x_c, y_c + 15), (0, 0, 255), 2)
        else:
            # Other tracks: CYAN, thin
            color = (255, 255, 0)
            thickness = 2
            label = f"ID:{track_id}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame


def draw_clean_info(frame, camera_fps, ai_fps, num_tracks, locked_target_id, time_since_ai):
    """
    Draw clean performance info
    - Minimal text overlay
    - Only essential information
    """
    h, w = frame.shape[:2]
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (250, 140), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    y_offset = 25
    
    # Camera FPS
    cv2.putText(frame, f"Camera: {camera_fps:.1f} FPS", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset += 25
    
    # AI FPS
    cv2.putText(frame, f"AI: {ai_fps:.1f} FPS", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_offset += 25
    
    # Time since last AI
    cv2.putText(frame, f"AI Update: {time_since_ai*1000:.0f}ms ago", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_offset += 25
    
    # Tracks
    cv2.putText(frame, f"Tracks: {num_tracks}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 25
    
    # Lock status
    if locked_target_id is not None:
        cv2.putText(frame, f"LOCKED: ID {locked_target_id}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "UNLOCKED", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return frame


# ============================================================================
# MAIN ULTRA-OPTIMIZED LOOP
# ============================================================================

def main():
    """
    Ultra-optimized main loop
    - AI runs every 250ms (4 FPS) - Time-based
    - Display runs at camera speed (30 FPS) - Frame-based
    - Clean UI with tracking only
    """
    
    print("\n" + "="*70)
    print("Ultra-Optimized Person Detection")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"AI Inference: Every {AI_INFERENCE_INTERVAL*1000:.0f}ms (4 FPS)")
    print(f"Display: Camera speed (~30 FPS)")
    print("="*70 + "\n")
    
    # Initialize components
    print("[1/4] Initializing camera...")
    camera = ThreadedCamera(
        src=CAMERA_ID,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS
    ).start()
    
    print("\n[2/4] Initializing preprocessor and detector...")
    preprocessor = PreProcessor(input_size=INPUT_SIZE)
    detector = ONNXDetector(
        model_path=MODEL_PATH,
        input_size=INPUT_SIZE,
        conf_thresh=CONFIDENCE_THRESHOLD,
        iou_thresh=IOU_THRESHOLD
    )
    
    print("\n[3/4] Initializing tracker...")
    tracker = ByteTracker(
        track_thresh=TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH
    )
    target_locker = TargetLocker(tracker)
    
    print("\n[4/4] Starting ultra-optimized loop...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'l' - Lock target")
    print("  'u' - Unlock target")
    print("  's' - Show stats")
    print("="*70 + "\n")
    
    # State variables
    last_inference_time = 0
    latest_tracks = []  # Store latest tracking results
    
    # AI FPS tracking
    ai_fps = 0
    ai_count = 0
    ai_fps_start = time.time()
    
    # Display FPS tracking
    display_fps = 0
    display_count = 0
    display_fps_start = time.time()
    
    # Performance stats
    inference_times = deque(maxlen=30)
    
    try:
        while True:
            current_time = time.time()
            
            # Get latest frame from camera (always)
            frame, camera_fps = camera.read()
            
            if frame is None:
                continue
            
            # ================================================================
            # TIME-BASED AI INFERENCE (Every 250ms = 4 FPS)
            # ================================================================
            time_since_last_ai = current_time - last_inference_time
            
            if time_since_last_ai >= AI_INFERENCE_INTERVAL:
                # Time to run AI inference
                last_inference_time = current_time
                
                orig_shape = frame.shape[:2]
                
                # Preprocess
                img, ratio, pad = preprocessor.preprocess(frame)
                
                # Inference
                t1 = time.time()
                predictions = detector.infer(img)
                inference_time = (time.time() - t1) * 1000
                inference_times.append(inference_time)
                
                # Postprocess
                detections = detector.postprocess(predictions, orig_shape, ratio, pad)
                
                # Update tracker
                if target_locker.is_locked:
                    # Update and get only locked target
                    target = target_locker.update(detections)
                    latest_tracks = [target] if target else []
                else:
                    # Update and get all tracks
                    latest_tracks = tracker.update(detections)
                
                # Update AI FPS
                ai_count += 1
                elapsed = time.time() - ai_fps_start
                if elapsed >= 1.0:
                    ai_fps = ai_count / elapsed
                    ai_count = 0
                    ai_fps_start = time.time()
            
            # ================================================================
            # SMOOTH DISPLAY (Every frame = ~30 FPS)
            # ================================================================
            # Use latest_tracks from last AI inference
            # This keeps display smooth even when AI is not running
            
            # Draw clean tracking (no raw detection boxes)
            display_frame = draw_clean_tracking(
                frame.copy(),
                latest_tracks,
                locked_target_id=target_locker.locked_target_id
            )
            
            # Draw clean info overlay
            display_frame = draw_clean_info(
                display_frame,
                camera_fps,
                ai_fps,
                len(latest_tracks),
                target_locker.locked_target_id,
                time_since_last_ai
            )
            
            # Show frame (smooth, independent of AI)
            cv2.imshow('Ultra-Optimized Detection', display_frame)
            
            # Update display FPS
            display_count += 1
            elapsed = time.time() - display_fps_start
            if elapsed >= 1.0:
                display_fps = display_count / elapsed
                display_count = 0
                display_fps_start = time.time()
            
            # Console output
            if latest_tracks:
                track = latest_tracks[0]
                x_c = (track['bbox'][0] + track['bbox'][2]) // 2
                y_c = (track['bbox'][1] + track['bbox'][3]) // 2
                status = "LOCKED" if target_locker.is_locked else "TRACKING"
                print(f"\r[{status}] Cam:{camera_fps:4.1f} | AI:{ai_fps:4.1f} | "
                      f"Disp:{display_fps:4.1f} | "
                      f"ID:{track['track_id']} ({x_c:3d},{y_c:3d}) | "
                      f"AI:{time_since_last_ai*1000:3.0f}ms ago", 
                      end='', flush=True)
            else:
                print(f"\rCam:{camera_fps:4.1f} | AI:{ai_fps:4.1f} | "
                      f"Disp:{display_fps:4.1f} | No tracks | "
                      f"AI:{time_since_last_ai*1000:3.0f}ms ago", 
                      end='', flush=True)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                # Lock target
                if not target_locker.is_locked and len(latest_tracks) > 0:
                    # Need to get detections for locking
                    frame_for_lock, _ = camera.read()
                    if frame_for_lock is not None:
                        img, ratio, pad = preprocessor.preprocess(frame_for_lock)
                        predictions = detector.infer(img)
                        detections = detector.postprocess(predictions, frame_for_lock.shape[:2], ratio, pad)
                        
                        if target_locker.lock_target(detections):
                            print("\n[LOCKED] Target locked!")
                else:
                    print("\n[INFO] No tracks to lock or already locked")
            elif key == ord('u'):
                # Unlock target
                if target_locker.is_locked:
                    target_locker.unlock_target()
                    print("\n[UNLOCKED] Target unlocked")
            elif key == ord('s'):
                # Show stats
                avg_inference = np.mean(inference_times) if inference_times else 0
                print(f"\n\n{'='*70}")
                print("Performance Statistics:")
                print(f"{'='*70}")
                print(f"Camera FPS: {camera_fps:.2f}")
                print(f"AI FPS: {ai_fps:.2f} (Target: 4 FPS)")
                print(f"Display FPS: {display_fps:.2f}")
                print(f"Avg Inference Time: {avg_inference:.2f}ms")
                print(f"AI Interval: {AI_INFERENCE_INTERVAL*1000:.0f}ms")
                print(f"Active Tracks: {len(latest_tracks)}")
                print(f"Locked: {target_locker.is_locked}")
                if target_locker.is_locked:
                    print(f"Target ID: {target_locker.locked_target_id}")
                print(f"{'='*70}\n")
    
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
