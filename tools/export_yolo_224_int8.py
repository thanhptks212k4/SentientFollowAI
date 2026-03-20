#!/usr/bin/env python3
"""
🚀 YOLOv5nu Model Export & Quantization Tool
===========================================

Export YOLOv5nu from PyTorch to ONNX (224x224) and quantize to INT8
for optimal performance on Raspberry Pi 5.

Author: Machine Learning Engineer - Model Optimization Specialist
Target: Raspberry Pi 5 - 50% FLOPs reduction
"""

import os
import sys
import numpy as np
from pathlib import Path
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization import quantize_static, CalibrationDataReader
import cv2
import glob
from typing import List, Dict, Any

# Check required packages
try:
    from ultralytics import YOLO
    print("✅ Ultralytics available")
except ImportError:
    print("❌ Please install: pip install ultralytics")
    sys.exit(1)

try:
    import onnxruntime
    print("✅ ONNXRuntime available")
except ImportError:
    print("❌ Please install: pip install onnxruntime")
    sys.exit(1)


class CalibrationDataset(CalibrationDataReader):
    """
    Calibration dataset for static quantization
    Uses sample images to calibrate INT8 quantization
    """
    
    def __init__(self, calibration_images: List[str], input_size: int = 224):
        """
        Initialize calibration dataset
        
        Args:
            calibration_images: List of image paths for calibration
            input_size: Target input size (224x224)
        """
        self.calibration_images = calibration_images
        self.input_size = input_size
        self.current_index = 0
        
        print(f"📊 Calibration dataset: {len(calibration_images)} images")
    
    def get_next(self) -> Dict[str, np.ndarray]:
        """Get next calibration sample"""
        if self.current_index >= len(self.calibration_images):
            return None
        
        # Load and preprocess image
        img_path = self.calibration_images[self.current_index]
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Could not load: {img_path}")
            self.current_index += 1
            return self.get_next()
        
        # Preprocess like YOLO (letterbox + normalize)
        img_processed = self._preprocess_image(img)
        
        self.current_index += 1
        return {"images": img_processed}
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO input"""
        # Letterbox resize to 224x224
        h, w = img.shape[:2]
        ratio = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        # Resize
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_h, self.input_size - new_h - pad_h,
            pad_w, self.input_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW
        img_final = img_normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return img_final


def find_calibration_images(search_paths: List[str], max_images: int = 100) -> List[str]:
    """
    Find calibration images from various sources
    
    Args:
        search_paths: List of paths to search for images
        max_images: Maximum number of images to use
        
    Returns:
        List of image paths
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    calibration_images = []
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for ext in image_extensions:
                pattern = os.path.join(search_path, '**', ext)
                images = glob.glob(pattern, recursive=True)
                calibration_images.extend(images)
                
                if len(calibration_images) >= max_images:
                    break
            
            if len(calibration_images) >= max_images:
                break
    
    # Limit to max_images
    calibration_images = calibration_images[:max_images]
    
    print(f"📸 Found {len(calibration_images)} calibration images")
    return calibration_images


def export_pytorch_to_onnx(model_path: str, output_path: str, input_size: int = 224) -> bool:
    """
    Export PyTorch YOLOv5nu to ONNX with fixed 224x224 input
    
    Args:
        model_path: Path to YOLOv5nu .pt file
        output_path: Output ONNX file path
        input_size: Target input size (224)
        
    Returns:
        bool: Success status
    """
    try:
        print(f"\n🔄 Step 1: Export PyTorch → ONNX (Fixed {input_size}x{input_size})")
        print("=" * 60)
        
        # Load YOLOv5nu model
        print(f"📥 Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Export to ONNX with fixed size
        print(f"🚀 Exporting to ONNX: {output_path}")
        success = model.export(
            format='onnx',
            imgsz=input_size,      # Fixed size 224x224
            dynamic=False,         # No dynamic axes for CPU optimization
            simplify=True,         # Simplify ONNX graph
            opset=11,             # ONNX opset version
            verbose=False
        )
        
        if success:
            # Move exported file to desired location
            exported_file = model_path.replace('.pt', '.onnx')
            if os.path.exists(exported_file) and exported_file != output_path:
                os.rename(exported_file, output_path)
            
            print(f"✅ ONNX export successful: {output_path}")
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX model validation passed")
            
            # Print model info
            input_shape = onnx_model.graph.input[0].type.tensor_type.shape
            print(f"📊 Input shape: {[dim.dim_value for dim in input_shape.dim]}")
            
            return True
        else:
            print("❌ ONNX export failed")
            return False
            
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False


def quantize_onnx_to_int8(onnx_path: str, output_path: str, 
                         calibration_images: List[str] = None,
                         use_static: bool = True) -> bool:
    """
    Quantize ONNX model from FP32 to INT8
    
    Args:
        onnx_path: Input ONNX FP32 model path
        output_path: Output ONNX INT8 model path
        calibration_images: Images for static quantization
        use_static: Use static quantization (better accuracy) vs dynamic
        
    Returns:
        bool: Success status
    """
    try:
        print(f"\n🔄 Step 2: Quantize ONNX FP32 → INT8")
        print("=" * 60)
        
        if use_static and calibration_images and len(calibration_images) > 0:
            print("🎯 Using Static Quantization (better accuracy)")
            
            # Create calibration dataset
            calibration_dataset = CalibrationDataset(calibration_images, input_size=224)
            
            # Static quantization
            quantize_static(
                model_input=onnx_path,
                model_output=output_path,
                calibration_data_reader=calibration_dataset,
                quant_format=ort.quantization.QuantFormat.QUInt8,
                activation_type=ort.quantization.QuantType.QUInt8,
                weight_type=ort.quantization.QuantType.QUInt8,
                optimize_model=True,
                per_channel=True,
                reduce_range=True,  # Better for CPU
            )
            
            print("✅ Static quantization completed")
            
        else:
            print("⚡ Using Dynamic Quantization (faster, no calibration needed)")
            
            # Dynamic quantization
            quantize_dynamic(
                model_input=onnx_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8,
                per_channel=False,  # Simplified for compatibility
                reduce_range=False,  # Simplified for compatibility
            )
            
            print("✅ Dynamic quantization completed")
        
        # Verify quantized model
        print("🔍 Verifying quantized model...")
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        
        # Test inference
        input_shape = session.get_inputs()[0].shape
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
        
        print(f"✅ Quantized model verification passed")
        print(f"📊 Input shape: {input_shape}")
        print(f"📊 Output shapes: {[out.shape for out in outputs]}")
        
        # Compare file sizes
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        compression_ratio = original_size / quantized_size
        
        print(f"📦 Original size: {original_size:.2f} MB")
        print(f"📦 Quantized size: {quantized_size:.2f} MB")
        print(f"🗜️ Compression ratio: {compression_ratio:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization error: {e}")
        return False


def benchmark_models(fp32_path: str, int8_path: str, num_runs: int = 100):
    """
    Benchmark FP32 vs INT8 model performance
    
    Args:
        fp32_path: FP32 ONNX model path
        int8_path: INT8 ONNX model path
        num_runs: Number of inference runs for benchmarking
    """
    try:
        print(f"\n🏁 Step 3: Benchmark Performance")
        print("=" * 60)
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Benchmark FP32
        print("⏱️ Benchmarking FP32 model...")
        session_fp32 = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
        
        import time
        start_time = time.time()
        for _ in range(num_runs):
            _ = session_fp32.run(None, {session_fp32.get_inputs()[0].name: dummy_input})
        fp32_time = (time.time() - start_time) / num_runs * 1000
        
        # Benchmark INT8
        print("⏱️ Benchmarking INT8 model...")
        session_int8 = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = session_int8.run(None, {session_int8.get_inputs()[0].name: dummy_input})
        int8_time = (time.time() - start_time) / num_runs * 1000
        
        speedup = fp32_time / int8_time
        
        print(f"📊 Benchmark Results ({num_runs} runs):")
        print(f"   FP32: {fp32_time:.2f}ms per inference")
        print(f"   INT8: {int8_time:.2f}ms per inference")
        print(f"   🚀 Speedup: {speedup:.2f}x")
        print(f"   🎯 Max FPS: {1000/int8_time:.1f} FPS")
        
    except Exception as e:
        print(f"❌ Benchmark error: {e}")


def main():
    """Main export and quantization pipeline"""
    print("🚀 YOLOv5nu Model Export & Quantization Tool")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "models/yolov5nu.pt"  # Input PyTorch model
    ONNX_FP32_PATH = "models/yolov5nu_224_fp32.onnx"  # Intermediate FP32 ONNX
    ONNX_INT8_PATH = "models/yolov5nu_224_int8.onnx"  # Final INT8 ONNX
    INPUT_SIZE = 224  # Target input size
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Check if source model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("💡 Please download YOLOv5nu model:")
        print("   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt -O models/yolov5nu.pt")
        
        # Try to download automatically
        try:
            print("🔄 Attempting automatic download...")
            model = YOLO('yolov5nu.pt')  # This will download the model
            # Move to models directory
            if os.path.exists('yolov5nu.pt'):
                os.rename('yolov5nu.pt', MODEL_PATH)
                print(f"✅ Model downloaded: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Auto-download failed: {e}")
            return False
    
    # Step 1: Export PyTorch to ONNX
    success = export_pytorch_to_onnx(MODEL_PATH, ONNX_FP32_PATH, INPUT_SIZE)
    if not success:
        return False
    
    # Find calibration images for static quantization
    calibration_search_paths = [
        "/tmp",  # Temporary images from camera
        ".",     # Current directory
        "images", # Common image directory
        os.path.expanduser("~/Pictures"),  # User pictures
    ]
    
    calibration_images = find_calibration_images(calibration_search_paths, max_images=50)
    
    # Step 2: Quantize ONNX to INT8 (Use dynamic for reliability)
    use_static = False  # Use dynamic quantization for better compatibility
    success = quantize_onnx_to_int8(
        ONNX_FP32_PATH, 
        ONNX_INT8_PATH, 
        calibration_images, 
        use_static=use_static
    )
    
    if not success:
        return False
    
    # Step 3: Benchmark performance
    benchmark_models(ONNX_FP32_PATH, ONNX_INT8_PATH)
    
    print(f"\n🎉 Export & Quantization Complete!")
    print("=" * 60)
    print(f"📁 Output files:")
    print(f"   FP32 ONNX: {ONNX_FP32_PATH}")
    print(f"   INT8 ONNX: {ONNX_INT8_PATH}")
    print(f"\n💡 Usage in your code:")
    print(f"   MODEL_PATH = '{ONNX_INT8_PATH}'")
    print(f"   INPUT_SIZE = {INPUT_SIZE}")
    print(f"\n🚀 Expected improvements on Pi 5:")
    print(f"   - 50% reduction in FLOPs (320→224)")
    print(f"   - 2-4x speedup from INT8 quantization")
    print(f"   - 4x smaller model size")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)