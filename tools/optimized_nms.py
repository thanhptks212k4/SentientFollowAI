"""
Optimized Single-Class NMS for YOLOv5 Person Detection
Optimized for ARM NEON (Raspberry Pi 5)

Key optimizations:
1. Vectorized threshold filtering (NumPy)
2. Vectorized IoU calculation
3. Pre-allocated arrays (no dynamic memory allocation)
4. Single-class only (no class loop)
"""

import numpy as np
import time

# Try to import numba, but make it optional
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[INFO] Numba not available. Install with: pip install numba")
    print("[INFO] Running with NumPy only (still fast!)\n")


def nms_vectorized(boxes, scores, iou_threshold=0.45):
    """
    Optimized NMS for single class using NumPy vectorization
    
    Args:
        boxes: np.array of shape [N, 4] (x1, y1, x2, y2)
        scores: np.array of shape [N] (confidence scores)
        iou_threshold: float, IoU threshold for suppression
    
    Returns:
        indices: List of indices to keep
    """
    
    if len(boxes) == 0:
        return []
    
    # Pre-compute areas (vectorized)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by scores (descending)
    order = np.argsort(scores)[::-1]
    
    # Pre-allocate keep array (avoid dynamic allocation)
    keep = np.zeros(len(boxes), dtype=np.int32)
    keep_count = 0
    
    # Suppression mask
    suppressed = np.zeros(len(boxes), dtype=bool)
    
    for i in range(len(order)):
        idx = order[i]
        
        if suppressed[idx]:
            continue
        
        keep[keep_count] = idx
        keep_count += 1
        
        if i == len(order) - 1:
            break
        
        # Get remaining boxes
        remaining_indices = order[i+1:]
        remaining_mask = ~suppressed[remaining_indices]
        remaining = remaining_indices[remaining_mask]
        
        if len(remaining) == 0:
            break
        
        # Vectorized IoU calculation
        xx1 = np.maximum(x1[idx], x1[remaining])
        yy1 = np.maximum(y1[idx], y1[remaining])
        xx2 = np.minimum(x2[idx], x2[remaining])
        yy2 = np.minimum(y2[idx], y2[remaining])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[idx] + areas[remaining] - inter)
        
        # Mark boxes with high IoU as suppressed
        suppressed[remaining[iou > iou_threshold]] = True
    
    return keep[:keep_count].tolist()


if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def nms_numba(boxes, scores, iou_threshold=0.45):
        """
        Ultra-fast NMS using Numba JIT compilation
        Optimized for ARM NEON (parallel operations)
        
        Args:
            boxes: np.array of shape [N, 4] (x1, y1, x2, y2)
            scores: np.array of shape [N] (confidence scores)
            iou_threshold: float, IoU threshold for suppression
        
        Returns:
            keep: np.array of indices to keep
        """
        
        n = len(boxes)
        
        if n == 0:
            return np.empty(0, dtype=np.int32)
        
        # Pre-compute areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by scores
        order = np.argsort(scores)[::-1]
        
        # Pre-allocate arrays
        keep = np.zeros(n, dtype=np.int32)
        keep_count = 0
        suppressed = np.zeros(n, dtype=np.bool_)
        
        for i in range(n):
            idx = order[i]
            
            if suppressed[idx]:
                continue
            
            keep[keep_count] = idx
            keep_count += 1
            
            # Vectorized IoU calculation for remaining boxes
            for j in prange(i + 1, n):
                idx2 = order[j]
                
                if suppressed[idx2]:
                    continue
                
                # Calculate intersection
                xx1 = max(x1[idx], x1[idx2])
                yy1 = max(y1[idx], y1[idx2])
                xx2 = min(x2[idx], x2[idx2])
                yy2 = min(y2[idx], y2[idx2])
                
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                
                inter = w * h
                
                if inter > 0:
                    iou = inter / (areas[idx] + areas[idx2] - inter)
                    
                    if iou > iou_threshold:
                        suppressed[idx2] = True
        
        return keep[:keep_count]
else:
    def nms_numba(boxes, scores, iou_threshold=0.45):
        """Fallback to vectorized version if Numba not available"""
        return np.array(nms_vectorized(boxes, scores, iou_threshold))


def postprocess_yolo_single_class(predictions, conf_threshold=0.45, iou_threshold=0.45):
    """
    Complete post-processing for YOLOv5 single-class output
    
    Args:
        predictions: np.array of shape [1, num_anchors, 6]
                    Format: [x, y, w, h, objectness, class_score]
        conf_threshold: float, confidence threshold
        iou_threshold: float, NMS IoU threshold
    
    Returns:
        boxes: np.array of shape [N, 4] (x1, y1, x2, y2)
        scores: np.array of shape [N]
    """
    
    # Remove batch dimension
    predictions = predictions[0]  # [num_anchors, 6]
    
    # Extract components
    boxes_xywh = predictions[:, :4]  # [x, y, w, h]
    objectness = predictions[:, 4]
    class_scores = predictions[:, 5]
    
    # Calculate final confidence (vectorized)
    confidences = objectness * class_scores
    
    # Threshold filtering (vectorized - simulates NEON vmask)
    mask = confidences > conf_threshold
    
    if not np.any(mask):
        return np.empty((0, 4)), np.empty(0)
    
    # Filter boxes and scores
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    
    # Convert xywh to xyxy (vectorized)
    x_center = boxes_xywh[:, 0]
    y_center = boxes_xywh[:, 1]
    width = boxes_xywh[:, 2]
    height = boxes_xywh[:, 3]
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    # Apply NMS
    keep_indices = nms_vectorized(boxes_xyxy, confidences, iou_threshold)
    
    return boxes_xyxy[keep_indices], confidences[keep_indices]


def benchmark_nms():
    """Benchmark different NMS implementations"""
    
    print("="*70)
    print("NMS Performance Benchmark")
    print("="*70)
    
    # Generate test data
    num_boxes = 1000
    np.random.seed(42)
    
    boxes = np.random.rand(num_boxes, 4).astype(np.float32)
    boxes[:, 2:] += boxes[:, :2]  # Convert to x1,y1,x2,y2
    scores = np.random.rand(num_boxes).astype(np.float32)
    
    iou_threshold = 0.45
    
    # Test 1: Vectorized NumPy
    print("\n[1/2] Testing Vectorized NumPy NMS...")
    times = []
    for _ in range(100):
        start = time.time()
        keep1 = nms_vectorized(boxes, scores, iou_threshold)
        times.append((time.time() - start) * 1000)
    
    print(f"  Mean time: {np.mean(times):.3f} ms")
    print(f"  Std: {np.std(times):.3f} ms")
    print(f"  Kept boxes: {len(keep1)}")
    
    # Test 2: Numba JIT (if available)
    if NUMBA_AVAILABLE:
        print("\n[2/2] Testing Numba JIT NMS...")
        
        # Warmup
        for _ in range(10):
            nms_numba(boxes, scores, iou_threshold)
        
        times = []
        for _ in range(100):
            start = time.time()
            keep2 = nms_numba(boxes, scores, iou_threshold)
            times.append((time.time() - start) * 1000)
        
        print(f"  Mean time: {np.mean(times):.3f} ms")
        print(f"  Std: {np.std(times):.3f} ms")
        print(f"  Kept boxes: {len(keep2)}")
        
        print("\n" + "="*70)
        print("Recommendation: Use Numba JIT for best performance on Pi 5")
        print("="*70 + "\n")
    else:
        print("\n[2/2] Numba not available - skipping JIT test")
        print("\n" + "="*70)
        print("Recommendation: Install numba for 5-10x speedup")
        print("  pip install numba")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Run benchmark
    benchmark_nms()
    
    # Test with YOLO output
    print("\nTesting with simulated YOLO output...")
    
    # Simulate YOLO output: [1, 2100, 6]
    predictions = np.random.rand(1, 2100, 6).astype(np.float32)
    
    start = time.time()
    boxes, scores = postprocess_yolo_single_class(predictions, conf_threshold=0.45, iou_threshold=0.45)
    elapsed = (time.time() - start) * 1000
    
    print(f"Post-processing time: {elapsed:.2f} ms")
    print(f"Detections: {len(boxes)}")
    print(f"FPS (post-processing only): {1000/elapsed:.1f}")
