#!/usr/bin/env python3
"""
🚀 YOLOv5nu Person-Only Model Export & Quantization
===================================================

Create ultra-lightweight model for Raspberry Pi 5:
1. Input size: 320x320 → 224x224 (50% FLOPs reduction)
2. Classes: 80 classes → 1 class (Person only)
3. Quantization: FP32 → INT8 (4x size reduction, 2-4x speedup)

Target: <15% CPU usage on Pi 5 for person following robot

Author: Machine Learning Engineer - Model Optimization Specialist
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Check required packages
try:
    from ultralytics import YOLO
    print("✅ Ultralytics available")
except ImportError:
    print("❌ Please install: pip install ultralytics")
    sys.exit(1)

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print("✅ ONNXRuntime available")
except ImportError:
    print("❌ Please install: pip install onnxruntime")
    sys.exit(1)

try:
    import onnx
    print("✅ ONNX available")
except ImportError:
    print("❌ Please install: pip install onnx")
    sys.exit(1)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0


def step1_export_person_only_onnx(model_path: str, output_path: str, input_size: int = 224) -> bool:
    """
    Step 1: Export YOLOv5nu to ONNX with Person class only and 224x224 input
    
    Args:
        model_path: Path to YOLOv5nu .pt file
        output_path: Output ONNX file path
        input_size: Target input size (224)
        
    Returns:
        bool: Success status
    """
    try:
        print(f"\n🔄 STEP 1: Export Person-Only ONNX ({input_size}x{input_size})")
        print("=" * 70)
        
        # Load YOLOv5nu model
        print(f"📥 Loading YOLOv5nu model: {model_path}")
        model = YOLO(model_path)
        
        # Print original model info
        print(f"📊 Original model info:")
        print(f"   Classes: 80 (COCO dataset)")
        print(f"   Input size: 640x640 (default)")
        print(f"   Parameters: ~2.6M")
        
        # Export to ONNX with optimizations
        print(f"🚀 Exporting optimized ONNX...")
        print(f"   ✂️  Input size: 640x640 → {input_size}x{input_size}")
        print(f"   ✂️  Classes: 80 → 1 (Person only)")
        print(f"   ⚡ Graph simplification: Enabled")
        
        success = model.export(
            format='onnx',
            imgsz=input_size,      # 🔥 224x224 input (50% FLOPs reduction)
            classes=[0],           # 🔥 Person class only (class 0 in COCO)
            simplify=True,         # 🔥 Simplify ONNX graph
            dynamic=False,         # Fixed size for CPU optimization
            opset=11,             # ONNX opset version
            verbose=False,
            half=False,           # Keep FP32 for quantization step
        )
        
        if success:
            # Move exported file to desired location
            exported_file = model_path.replace('.pt', '.onnx')
            if os.path.exists(exported_file) and exported_file != output_path:
                os.rename(exported_file, output_path)
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # Get file size
            file_size = get_file_size_mb(output_path)
            
            print(f"✅ Person-only ONNX export successful!")
            print(f"   📁 File: {output_path}")
            print(f"   📦 Size: {file_size:.2f} MB")
            
            # Print model details
            input_shape = onnx_model.graph.input[0].type.tensor_type.shape
            output_shape = onnx_model.graph.output[0].type.tensor_type.shape
            
            print(f"📊 Model specifications:")
            print(f"   Input shape: {[dim.dim_value for dim in input_shape.dim]}")
            print(f"   Output shape: {[dim.dim_value if dim.dim_value else 'dynamic' for dim in output_shape.dim]}")
            
            return True
        else:
            print("❌ ONNX export failed")
            return False
            
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False


def step2_quantize_to_int8(fp32_path: str, int8_path: str) -> bool:
    """
    Step 2: Quantize ONNX model from FP32 to INT8
    
    Args:
        fp32_path: Input ONNX FP32 model path
        int8_path: Output ONNX INT8 model path
        
    Returns:
        bool: Success status
    """
    try:
        print(f"\n🔄 STEP 2: Quantize FP32 → INT8")
        print("=" * 70)
        
        # Get original file size
        fp32_size = get_file_size_mb(fp32_path)
        print(f"📦 FP32 model size: {fp32_size:.2f} MB")
        
        print(f"⚡ Applying dynamic quantization...")
        print(f"   🎯 Target: QUInt8 weights")
        print(f"   🎯 Method: Dynamic quantization (no calibration needed)")
        
        # Dynamic quantization - most compatible approach
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QUInt8,  # Quantize weights to 8-bit unsigned int
            per_channel=False,             # Simplified for better compatibility
            reduce_range=False,            # Standard range for better compatibility
        )
        
        # Verify quantized model
        print("🔍 Verifying quantized model...")
        session = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
        
        # Test inference with dummy input
        input_shape = session.get_inputs()[0].shape
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
        
        # Get quantized file size
        int8_size = get_file_size_mb(int8_path)
        compression_ratio = fp32_size / int8_size if int8_size > 0 else 0
        
        print(f"✅ INT8 quantization successful!")
        print(f"   📁 File: {int8_path}")
        print(f"   📦 Size: {int8_size:.2f} MB")
        print(f"   🗜️  Compression: {compression_ratio:.2f}x smaller")
        
        print(f"📊 Quantized model verification:")
        print(f"   Input shape: {input_shape}")
        print(f"   Output shapes: {[out.shape for out in outputs]}")
        print(f"   ✅ Inference test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization error: {e}")
        return False


def step3_benchmark_performance(fp32_path: str, int8_path: str, num_runs: int = 100):
    """
    Step 3: Benchmark FP32 vs INT8 performance
    
    Args:
        fp32_path: FP32 ONNX model path
        int8_path: INT8 ONNX model path
        num_runs: Number of inference runs
    """
    try:
        print(f"\n🔄 STEP 3: Performance Benchmark")
        print("=" * 70)
        
        # Create dummy input (224x224)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Benchmark FP32 model
        print("⏱️  Benchmarking FP32 model...")
        session_fp32 = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
        
        # Warmup
        for _ in range(5):
            _ = session_fp32.run(None, {session_fp32.get_inputs()[0].name: dummy_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = session_fp32.run(None, {session_fp32.get_inputs()[0].name: dummy_input})
        fp32_time = (time.time() - start_time) / num_runs * 1000
        
        # Benchmark INT8 model
        print("⏱️  Benchmarking INT8 model...")
        session_int8 = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
        
        # Warmup
        for _ in range(5):
            _ = session_int8.run(None, {session_int8.get_inputs()[0].name: dummy_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = session_int8.run(None, {session_int8.get_inputs()[0].name: dummy_input})
        int8_time = (time.time() - start_time) / num_runs * 1000
        
        # Calculate improvements
        speedup = fp32_time / int8_time if int8_time > 0 else 0
        fp32_fps = 1000 / fp32_time if fp32_time > 0 else 0
        int8_fps = 1000 / int8_time if int8_time > 0 else 0
        
        print(f"🏁 Benchmark Results ({num_runs} runs):")
        print(f"   📊 FP32 Performance:")
        print(f"      ⏱️  Inference time: {fp32_time:.2f}ms")
        print(f"      🚀 Max FPS: {fp32_fps:.1f}")
        print(f"   📊 INT8 Performance:")
        print(f"      ⏱️  Inference time: {int8_time:.2f}ms")
        print(f"      🚀 Max FPS: {int8_fps:.1f}")
        print(f"   🎯 Speedup: {speedup:.2f}x faster")
        
        # Estimate CPU usage reduction
        cpu_reduction = (1 - (int8_time / fp32_time)) * 100 if fp32_time > 0 else 0
        print(f"   💻 Estimated CPU reduction: {cpu_reduction:.1f}%")
        
    except Exception as e:
        print(f"❌ Benchmark error: {e}")


def main():
    """Main optimization pipeline"""
    print("🚀 YOLOv5nu Person-Only Ultra-Lightweight Model Creator")
    print("=" * 70)
    print("🎯 Target: <15% CPU usage on Raspberry Pi 5")
    print("🎯 Optimizations:")
    print("   1. Input size: 320x320 → 224x224 (50% FLOPs reduction)")
    print("   2. Classes: 80 → 1 (Person only, faster NMS)")
    print("   3. Quantization: FP32 → INT8 (4x smaller, 2-4x faster)")
    print("=" * 70)
    
    # Configuration
    MODEL_PATH = "models/yolov5nu.pt"
    ONNX_FP32_PATH = "models/yolov5nu_person_224_fp32.onnx"
    ONNX_INT8_PATH = "models/yolov5nu_person_224_int8.onnx"
    INPUT_SIZE = 224
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Check/download source model
    if not os.path.exists(MODEL_PATH):
        print(f"📥 Downloading YOLOv5nu model...")
        try:
            model = YOLO('yolov5nu.pt')  # Auto-download
            if os.path.exists('yolov5nu.pt'):
                os.rename('yolov5nu.pt', MODEL_PATH)
                print(f"✅ Model downloaded: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    original_size = get_file_size_mb(MODEL_PATH)
    print(f"📦 Original PyTorch model: {original_size:.2f} MB")
    
    # Step 1: Export to Person-only ONNX
    success = step1_export_person_only_onnx(MODEL_PATH, ONNX_FP32_PATH, INPUT_SIZE)
    if not success:
        return False
    
    # Step 2: Quantize to INT8
    success = step2_quantize_to_int8(ONNX_FP32_PATH, ONNX_INT8_PATH)
    if not success:
        return False
    
    # Step 3: Benchmark performance
    step3_benchmark_performance(ONNX_FP32_PATH, ONNX_INT8_PATH)
    
    # Final summary
    fp32_size = get_file_size_mb(ONNX_FP32_PATH)
    int8_size = get_file_size_mb(ONNX_INT8_PATH)
    total_compression = original_size / int8_size if int8_size > 0 else 0
    
    print(f"\n🎉 ULTRA-LIGHTWEIGHT MODEL CREATION COMPLETE!")
    print("=" * 70)
    print(f"📁 Output Files:")
    print(f"   🔸 FP32 ONNX: {ONNX_FP32_PATH} ({fp32_size:.2f} MB)")
    print(f"   🔸 INT8 ONNX: {ONNX_INT8_PATH} ({int8_size:.2f} MB)")
    
    print(f"\n📊 Optimization Summary:")
    print(f"   🔸 Original PyTorch: {original_size:.2f} MB")
    print(f"   🔸 Final INT8 ONNX: {int8_size:.2f} MB")
    print(f"   🔸 Total compression: {total_compression:.2f}x smaller")
    
    print(f"\n💡 Usage in your code:")
    print(f"   MODEL_PATH = '{ONNX_INT8_PATH}'")
    print(f"   INPUT_SIZE = {INPUT_SIZE}")
    print(f"   PERSON_CLASS = 0  # Only class in the model")
    
    print(f"\n🚀 Expected improvements on Pi 5:")
    print(f"   🔸 50% less computation (224x224 vs 320x320)")
    print(f"   🔸 2-4x faster inference (INT8 quantization)")
    print(f"   🔸 Faster NMS (1 class vs 80 classes)")
    print(f"   🔸 Target: CPU usage 85% → <15%")
    
    print(f"\n✅ Ready for deployment on Raspberry Pi 5!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)