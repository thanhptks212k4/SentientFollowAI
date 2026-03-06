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
from threading import Thread, Lock
from collections import deque

# Import tracking module
from bytetrack_tracker import ByteTracker, TargetLocker

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
INPUT_SIZE = 256
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0

# Performance Settings
SKIP_FRAMES = 1
TARGET_FPS = 15
ENABLE_CPU_STABILIZATION = True

# Tracking Settings
TRACK_THRESH = 0.5  # High confidence threshold for tracking
TRACK_BUFFER = 30   # Frames to keep lost tracks
MATCH_THRESH = 0.8  # IoU threshold for matching

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
    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
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
# MAIN DETECTION + TRACKING PIPELINE
# ============================================================================

def main():
    """Main detection and tracking loop"""
    
    print("\n" + "="*70)
    print("Person Detection with ByteTrack Tracking")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"Target FPS: {TARGET_FPS}")
    print("="*70 + "\n")
    
    # Initialize components
    print("[1/4] Initializing camera...")
    camera = ThreadedCamera(
        src=CAMERA_ID,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS
    ).start()
    
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
    print("Controls:")
    print("  'q' - Quit")
    print("  'l' - Lock target (simulate voice command)")
    print("  'u' - Unlock target")
    print("  's' - Show stats")
    print("="*70 + "\n")
    
    # Performance tracking
    fps = 0
    frame_count = 0
    start_time = time.time()
    inference_times = deque(maxlen=30)
    sleep_times = deque(maxlen=30)
    
    target_frame_time = 1.0 / TARGET_FPS if ENABLE_CPU_STABILIZATION else 0
    
    try:
        while True:
            loop_start = time.time()
            
            frame = camera.read()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Skip frames if configured
            if SKIP_FRAMES > 0 and frame_count % (SKIP_FRAMES + 1) != 0:
                continue
            
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
            
            # Tracking
            if target_locker.is_locked:
                # Update tracker and get only locked target
                target = target_locker.update(detections)
                
                if target is not None:
                    # Draw locked target
                    bbox = target['bbox']
                    x1, y1, x2, y2 = bbox
                    x_c = (x1 + x2) // 2
                    y_c = (y1 + y2) // 2
                    
                    # Draw bounding box (GREEN for locked target)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw center crosshair
                    cv2.circle(frame, (x_c, y_c), 8, (0, 0, 255), -1)
                    cv2.line(frame, (x_c - 15, y_c), (x_c + 15, y_c), (0, 0, 255), 2)
                    cv2.line(frame, (x_c, y_c - 15), (x_c, y_c + 15), (0, 0, 255), 2)
                    
                    # Draw label
                    label = f"TARGET ID:{target['track_id']} ({target['score']:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw coordinates
                    coord_text = f"({x_c}, {y_c})"
                    cv2.putText(frame, coord_text, (x_c + 15, y_c - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Draw "LOCKED" indicator
                    cv2.putText(frame, "LOCKED", (10, orig_shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Target lost
                    status = target_locker.get_status()
                    lost_text = f"TARGET LOST ({status['frames_without_target']} frames)"
                    cv2.putText(frame, lost_text, (10, orig_shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Not locked - show all detections
                for det in detections:
                    bbox = det['bbox']
                    conf = det['confidence']
                    x1, y1, x2, y2 = bbox
                    
                    # Draw bounding box (YELLOW for unlocked)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    # Draw label
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw "UNLOCKED" indicator
                if len(detections) > 0:
                    cv2.putText(frame, "UNLOCKED - Press 'L' to lock", (10, orig_shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Draw performance info
            y_offset = 30
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            y_offset += 30
            avg_inference = np.mean(inference_times) if inference_times else 0
            cv2.putText(frame, f"Inference: {avg_inference:.1f}ms", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
            cv2.putText(frame, f"Detections: {len(detections)}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Person Tracking - ByteTrack', frame)
            
            # Console output
            if target_locker.is_locked:
                target = target_locker.update([])  # Get cached target
                if target:
                    x_c = (target['bbox'][0] + target['bbox'][2]) // 2
                    y_c = (target['bbox'][1] + target['bbox'][3]) // 2
                    print(f"\r[LOCKED] Target ID:{target['track_id']} at ({x_c:4d}, {y_c:4d}) | "
                          f"FPS: {fps:5.1f} | Inf: {avg_inference:4.0f}ms", 
                          end='', flush=True)
            
            # CPU Stabilization
            if ENABLE_CPU_STABILIZATION:
                loop_time = time.time() - loop_start
                sleep_time = target_frame_time - loop_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    sleep_times.append(sleep_time * 1000)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                # Lock target (simulate voice command)
                if not target_locker.is_locked and len(detections) > 0:
                    success = target_locker.lock_target(detections)
                    if success:
                        print("\n[Command] Target locked!")
                else:
                    print("\n[Command] No detections to lock or already locked")
            elif key == ord('u'):
                # Unlock target
                if target_locker.is_locked:
                    target_locker.unlock_target()
                    print("\n[Command] Target unlocked")
            elif key == ord('s'):
                # Show stats
                status = target_locker.get_status()
                print(f"\n\n{'='*70}")
                print("Tracking Statistics:")
                print(f"{'='*70}")
                print(f"FPS: {fps:.2f}")
                print(f"Avg Inference: {np.mean(inference_times):.2f}ms")
                print(f"Target Locked: {status['is_locked']}")
                if status['is_locked']:
                    print(f"Target ID: {status['target_id']}")
                    print(f"Frames without target: {status['frames_without_target']}")
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
