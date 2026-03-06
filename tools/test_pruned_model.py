"""
Test pruned ONNX model (1-class Person only)
Compare performance with original 80-class model
"""

import cv2
import numpy as np
import time
import onnxruntime as ort

def test_model(model_path, num_frames=100):
    """Test model inference speed"""
    
    print(f"\nTesting: {model_path}")
    print("="*70)
    
    # Load model
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 3
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    
    # Get input/output info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    
    print(f"Input: {input_name} {input_shape}")
    print(f"Output: {output_name} {output_shape}")
    
    # Create dummy input
    batch_size = 1
    channels = 3
    height = input_shape[2] if len(input_shape) > 2 else 256
    width = input_shape[3] if len(input_shape) > 3 else 256
    
    dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        session.run([output_name], {input_name: dummy_input})
    
    # Benchmark
    print(f"Running {num_frames} inferences...")
    times = []
    
    for i in range(num_frames):
        start = time.time()
        output = session.run([output_name], {input_name: dummy_input})
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_frames}")
    
    # Statistics
    times = np.array(times)
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"Mean inference time: {np.mean(times):.2f} ms")
    print(f"Std deviation: {np.std(times):.2f} ms")
    print(f"Min: {np.min(times):.2f} ms")
    print(f"Max: {np.max(times):.2f} ms")
    print(f"Median: {np.median(times):.2f} ms")
    print(f"FPS: {1000 / np.mean(times):.1f}")
    print("="*70)
    
    return np.mean(times)


def main():
    print("\n" + "="*70)
    print("ONNX Model Performance Comparison")
    print("="*70)
    
    models = [
        ("Original (80 classes)", "yolov5nu.onnx"),
        ("Pruned (1 class)", "yolov5nu_person_only.onnx")
    ]
    
    results = {}
    
    for name, path in models:
        try:
            avg_time = test_model(path, num_frames=100)
            results[name] = avg_time
        except FileNotFoundError:
            print(f"\nSkipping {name}: File not found ({path})")
        except Exception as e:
            print(f"\nError testing {name}: {e}")
    
    # Compare results
    if len(results) == 2:
        print("\n" + "="*70)
        print("Comparison")
        print("="*70)
        
        original_time = results["Original (80 classes)"]
        pruned_time = results["Pruned (1 class)"]
        
        speedup = original_time / pruned_time
        reduction = (1 - pruned_time / original_time) * 100
        
        print(f"Original: {original_time:.2f} ms ({1000/original_time:.1f} FPS)")
        print(f"Pruned: {pruned_time:.2f} ms ({1000/pruned_time:.1f} FPS)")
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Latency reduction: {reduction:.1f}%")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
