"""
ONNX Model Quantization: FP32 → INT8
Convert YOLOv5nu from FP32 to INT8 for faster inference on CPU
"""

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input model (FP32)
INPUT_MODEL = 'yolov5nu.onnx'

# Output model (INT8)
OUTPUT_MODEL = 'yolov5nu_int8.onnx'

# Quantization type
QUANT_TYPE = QuantType.QUInt8  # Unsigned INT8

# ============================================================================
# QUANTIZATION
# ============================================================================

def quantize_model(input_path, output_path):
    """
    Convert ONNX model from FP32 to INT8 using Dynamic Quantization
    
    Dynamic Quantization:
    - Quantizes weights to INT8 (offline)
    - Quantizes activations to INT8 (runtime, dynamic)
    - No calibration data needed
    - Fast to implement
    """
    
    print("\n" + "="*70)
    print("ONNX Model Quantization: FP32 → INT8")
    print("="*70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Method: Dynamic Quantization")
    print(f"Type:   {QUANT_TYPE}")
    print("="*70 + "\n")
    
    # Check input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")
    
    # Get input model size
    input_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"[1/3] Input model size: {input_size:.2f} MB")
    
    # Quantize
    print(f"[2/3] Quantizing model...")
    print("      This may take 1-2 minutes...")
    
    try:
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QUANT_TYPE,
            per_channel=False,    # Per-tensor quantization (faster)
            reduce_range=False,   # Full INT8 range
            extra_options={
                'EnableSubgraph': True,  # Quantize subgraphs
                'ForceQuantizeNoInputCheck': False,
                'MatMulConstBOnly': True  # Only quantize constant weights
            }
        )
        print("      ✓ Quantization complete!")
    except Exception as e:
        print(f"      ✗ Quantization failed: {e}")
        return False
    
    # Get output model size
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = (1 - output_size / input_size) * 100
    
    print(f"[3/3] Output model size: {output_size:.2f} MB")
    print(f"      Compression: {compression_ratio:.1f}% smaller")
    
    # Verify model
    print("\n" + "="*70)
    print("Verification")
    print("="*70)
    
    try:
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("✓ Model is valid!")
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Original (FP32):  {input_size:.2f} MB")
    print(f"Quantized (INT8): {output_size:.2f} MB")
    print(f"Reduction:        {compression_ratio:.1f}%")
    print(f"Output file:      {output_path}")
    print("="*70)
    
    print("\n✅ Quantization successful!")
    print("\nNext steps:")
    print("  1. Update optimized_detector.py:")
    print("     MODEL_PATHS = {'onnx': 'yolov5nu_int8.onnx'}")
    print("  2. Run: python optimized_detector.py")
    print("  3. Compare FPS and accuracy")
    print("\nExpected improvements:")
    print("  - Inference time: 102ms → 40-60ms (2x faster)")
    print("  - Model size: ~10MB → ~3MB (70% smaller)")
    print("  - CPU usage: 80% → 50-60% (lower)")
    print("  - FPS: 8.6 → 15-25 (2-3x higher)")
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main quantization process"""
    
    # Check dependencies
    try:
        import onnxruntime
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
    except ImportError:
        print("Error: onnxruntime not installed")
        print("Install: pip install onnxruntime")
        return
    
    try:
        import onnx
        print(f"ONNX version: {onnx.__version__}")
    except ImportError:
        print("Error: onnx not installed")
        print("Install: pip install onnx")
        return
    
    print()
    
    # Quantize
    success = quantize_model(INPUT_MODEL, OUTPUT_MODEL)
    
    if success:
        print("\n" + "="*70)
        print("🎉 Done! Your INT8 model is ready to use.")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("❌ Quantization failed. Check errors above.")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
