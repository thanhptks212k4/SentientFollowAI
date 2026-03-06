"""
Export YOLOv5/YOLOv10 models to ONNX format for optimized inference
"""

from ultralytics import YOLO
import os

print("\n" + "="*70)
print("Model Export Script - ONNX & OpenVINO")
print("="*70 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Models to export
MODELS = {
    'yolov5nu': 'yolov5nu.pt',
    'yolov10n': 'yolov10n.pt'
}

# Export settings
INPUT_SIZE = 320  # Must match your detection script
FORMATS = ['onnx', 'openvino']  # Export formats

# ============================================================================
# EXPORT FUNCTION
# ============================================================================

def export_model(model_name, model_path, input_size=320):
    """Export model to ONNX and OpenVINO formats"""
    
    print(f"\n{'='*70}")
    print(f"Exporting: {model_name}")
    print(f"{'='*70}")
    
    if not os.path.exists(model_path):
        print(f"[Info] Model not found, downloading: {model_path}")
        model = YOLO(model_path)  # Will auto-download
    else:
        print(f"[Info] Loading model: {model_path}")
        model = YOLO(model_path)
    
    # Export to ONNX
    print(f"\n[1/2] Exporting to ONNX (input size: {input_size})...")
    try:
        onnx_path = model.export(
            format='onnx',
            imgsz=input_size,
            simplify=True,  # Simplify ONNX model
            opset=12,  # ONNX opset version
            dynamic=False  # Static shape for better performance
        )
        print(f"✓ ONNX export successful: {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
    
    # Export to OpenVINO
    print(f"\n[2/2] Exporting to OpenVINO (input size: {input_size})...")
    try:
        openvino_path = model.export(
            format='openvino',
            imgsz=input_size,
            half=False  # FP32 for CPU
        )
        print(f"✓ OpenVINO export successful: {openvino_path}")
    except Exception as e:
        print(f"✗ OpenVINO export failed: {e}")
    
    print(f"\n{'='*70}")
    print(f"✓ {model_name} export complete!")
    print(f"{'='*70}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Export all models"""
    
    print("\nThis script will export models to ONNX and OpenVINO formats.")
    print("Exported models will be optimized for CPU inference.\n")
    
    # Export each model
    for model_name, model_path in MODELS.items():
        export_model(model_name, model_path, INPUT_SIZE)
    
    # Summary
    print("\n" + "="*70)
    print("Export Summary")
    print("="*70)
    print("\nGenerated files:")
    print("  - yolov5nu.onnx")
    print("  - yolov5nu_openvino_model/")
    print("  - yolov10n.onnx")
    print("  - yolov10n_openvino_model/")
    print("\nUsage:")
    print("  1. Edit optimized_detector.py")
    print("  2. Set INFERENCE_ENGINE = 'onnx' or 'openvino'")
    print("  3. Update MODEL_PATHS to point to exported models")
    print("  4. Run: python optimized_detector.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
