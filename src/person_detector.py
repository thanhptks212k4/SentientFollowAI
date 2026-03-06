"""
Optimized Person Detection for Edge AI
- Threaded camera stream (zero-latency)
- ONNX Runtime inference with INT8 quantization
- Optimized preprocessing
- Target: 15-20 FPS on low-spec CPU (4 cores)
"""

import cv2
import numpy as np
import time
from threading import Thread, Lock
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

# Inference Engine
INFERENCE_ENGINE = 'onnx'  # 'onnx' or 'openvino'

# Model Path
MODEL_PATH = 'models/yolov5nu_int8.onnx'  # INT8 quantized model

# Camera Settings
CAMERA_ID = 0
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# Model Settings
INPUT_SIZE = 256  # Model input size (must match exported model)
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0

# Performance Settings - CPU Stabilization
SKIP_FRAMES = 2  # Skip N frames between detections (0=no skip, 1=every 2nd frame)
TARGET_FPS = 10  # Target FPS for CPU stabilization (lower = more stable CPU)
ENABLE_CPU_STABILIZATION = True  # Enable adaptive sleep to keep CPU < 50%

# ============================================================================
# THREADED CAMERA STREAM
# ============================================================================

class ThreadedCamera:
    """Threaded camera capture for zero-latency frame reading"""
    
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Thread-safe frame storage
        self.frame = None
        self.lock = Lock()
        self.running = False
        
        # Stats
        self.frames_read = 0
        self.frames_dropped = 0
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")
        
        print(f"[Camera] Initialized (ID: {src})")
        print(f"[Camera] Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    def start(self):
        """Start the camera thread"""
        if self.running:
            return self
        
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()
        time.sleep(0.5)  # Wait for first frame
        print("[Camera] Thread started")
        return self
    
    def _update(self):
        """Continuously read frames in background"""
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
        """Read the latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop the camera thread"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()
        print(f"[Camera] Stopped (Read: {self.frames_read}, Dropped: {self.frames_dropped})")
    
    def get_stats(self):
        """Get camera statistics"""
        return {
            'frames_read': self.frames_read,
            'frames_dropped': self.frames_dropped,
            'drop_rate': self.frames_dropped / max(self.frames_read, 1) * 100
        }


# ============================================================================
# OPTIMIZED PREPROCESSING
# ============================================================================

class PreProcessor:
    """Optimized preprocessing for YOLO models"""
    
    def __init__(self, input_size=320):
        self.input_size = input_size
        print(f"[PreProcessor] Input size: {input_size}x{input_size}")
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize image with letterboxing (maintain aspect ratio)"""
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
        """Complete preprocessing pipeline"""
        img, ratio, pad = self.letterbox(frame, new_shape=self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img, ratio, pad


# ============================================================================
# ONNX RUNTIME INFERENCE ENGINE
# ============================================================================

class ONNXDetector:
    """ONNX Runtime inference engine (CPU optimized)"""
    
    def __init__(self, model_path, input_size=320, conf_thresh=0.4, iou_thresh=0.45):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install: pip install onnxruntime")
        
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # CPU optimized session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        print(f"[ONNX] Loading model: {model_path}")
        print(f"[ONNX] Thread config: intra_op=2, inter_op=1")
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"[ONNX] Model loaded successfully")
    
    def infer(self, img):
        """Run inference"""
        outputs = self.session.run(self.output_names, {self.input_name: img})
        return outputs[0]
    
    def postprocess(self, predictions, orig_shape, ratio, pad):
        """Post-process YOLO predictions"""
        predictions = predictions[0].T  # [84, 2100] -> [2100, 84]
        
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        
        class_ids = np.argmax(class_scores, axis=1)
        class_confs = np.max(class_scores, axis=1)
        
        # Filter by confidence and person class
        mask = (class_confs > self.conf_thresh) & (class_ids == PERSON_CLASS_ID)
        boxes = boxes[mask]
        class_confs = class_confs[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert xywh to xyxy
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (x_center - width / 2 - pad[0]) / ratio
        y1 = (y_center - height / 2 - pad[1]) / ratio
        x2 = (x_center + width / 2 - pad[0]) / ratio
        y2 = (y_center + height / 2 - pad[1]) / ratio
        
        # Clip to image boundaries
        x1 = np.clip(x1, 0, orig_shape[1])
        y1 = np.clip(y1, 0, orig_shape[0])
        x2 = np.clip(x2, 0, orig_shape[1])
        y2 = np.clip(y2, 0, orig_shape[0])
        
        # Apply NMS
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        indices = self.nms(boxes_xyxy, class_confs, self.iou_thresh)
        
        # Format results
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
        """Non-Maximum Suppression"""
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
# MAIN DETECTION PIPELINE
# ============================================================================

def main():
    """Main detection loop"""
    
    print("\n" + "="*70)
    print("Optimized Person Detection for Edge AI")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"Input Size: {INPUT_SIZE}")
    print(f"Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"Skip Frames: {SKIP_FRAMES}")
    # print(f"Target FPS: {TARGET_FPS}")
    print(f"CPU Stabilization: {'Enabled' if ENABLE_CPU_STABILIZATION else 'Disabled'}")
    print("="*70 + "\n")
    
    # Initialize components
    print("[1/3] Initializing camera...")
    camera = ThreadedCamera(
        src=CAMERA_ID,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS
    ).start()
    
    print("\n[2/3] Initializing preprocessor...")
    preprocessor = PreProcessor(input_size=INPUT_SIZE)
    
    print("\n[3/3] Loading inference engine...")
    detector = ONNXDetector(
        model_path=MODEL_PATH,
        input_size=INPUT_SIZE,
        conf_thresh=CONFIDENCE_THRESHOLD,
        iou_thresh=IOU_THRESHOLD
    )
    
    print("\nStarting detection loop...")
    print("Controls: 'q' - Quit | 's' - Show stats")
    print("="*70 + "\n")
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Performance stats
    inference_times = deque(maxlen=30)
    preprocess_times = deque(maxlen=30)
    sleep_times = deque(maxlen=30)
    
    # CPU stabilization
    target_frame_time = 1.0 / TARGET_FPS if ENABLE_CPU_STABILIZATION else 0
    
    try:
        while True:
            loop_start = time.time()
            
            # Read frame
            frame = camera.read()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Skip frames if configured
            if SKIP_FRAMES > 0 and frame_count % (SKIP_FRAMES + 1) != 0:
                continue
            
            orig_shape = frame.shape[:2]
            
            # Preprocessing
            t1 = time.time()
            img, ratio, pad = preprocessor.preprocess(frame)
            preprocess_time = (time.time() - t1) * 1000
            preprocess_times.append(preprocess_time)
            
            # Inference
            t2 = time.time()
            predictions = detector.infer(img)
            inference_time = (time.time() - t2) * 1000
            inference_times.append(inference_time)
            
            # Post-processing
            detections = detector.postprocess(predictions, orig_shape, ratio, pad)
            
            # Visualization
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                x1, y1, x2, y2 = bbox
                
                # Calculate center
                x_c = (x1 + x2) // 2
                y_c = (y1 + y2) // 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, (x_c, y_c), 5, (0, 0, 255), -1)
                cv2.line(frame, (x_c - 10, y_c), (x_c + 10, y_c), (0, 0, 255), 2)
                cv2.line(frame, (x_c, y_c - 10), (x_c, y_c + 10), (0, 0, 255), 2)
                
                # Draw label
                label = f"Person: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw center coordinates
                coord_text = f"({x_c}, {y_c})"
                cv2.putText(frame, coord_text, (x_c + 10, y_c - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
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
            avg_sleep = np.mean(sleep_times) if sleep_times else 0
            cv2.putText(frame, f"Sleep: {avg_sleep:.1f}ms (CPU saver)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            y_offset += 25
            cv2.putText(frame, f"Persons: {len(detections)}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Person Detection', frame)
            
            # Console output
            if detections:
                det = detections[0]
                x_c = (det['bbox'][0] + det['bbox'][2]) // 2
                y_c = (det['bbox'][1] + det['bbox'][3]) // 2
                print(f"\rTarget: ({x_c:4d}, {y_c:4d}) | FPS: {fps:5.1f} | "
                      f"Inf: {avg_inference:4.0f}ms | Sleep: {avg_sleep:4.0f}ms | P: {len(detections)}", 
                      end='', flush=True)
            
            # CPU Stabilization: Sleep to limit CPU usage
            if ENABLE_CPU_STABILIZATION:
                loop_time = time.time() - loop_start
                sleep_time = target_frame_time - loop_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    sleep_times.append(sleep_time * 1000)
                else:
                    sleep_times.append(0)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = camera.get_stats()
                avg_sleep = np.mean(sleep_times) if sleep_times else 0
                print(f"\n\n{'='*70}")
                print("Performance Statistics:")
                print(f"{'='*70}")
                print(f"FPS: {fps:.2f} (Target: {TARGET_FPS})")
                print(f"Avg Inference Time: {np.mean(inference_times):.2f}ms")
                print(f"Avg Preprocess Time: {np.mean(preprocess_times):.2f}ms")
                print(f"Avg Sleep Time: {avg_sleep:.2f}ms")
                print(f"CPU Stabilization: {'Enabled' if ENABLE_CPU_STABILIZATION else 'Disabled'}")
                print(f"Camera Frames Read: {stats['frames_read']}")
                print(f"Camera Frames Dropped: {stats['frames_dropped']} ({stats['drop_rate']:.1f}%)")
                print(f"{'='*70}\n")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("\n\n" + "="*70)
        print("Detection stopped")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
