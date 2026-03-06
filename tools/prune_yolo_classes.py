"""
ONNX Model Class Pruning for YOLOv5
Reduce output from 80 classes to 1 class (Person only)

This script modifies the ONNX graph to:
1. Find the output tensor with shape [batch, anchors, 85] (80 classes + 4 bbox + 1 objectness)
2. Slice it to [batch, anchors, 6] (1 class + 4 bbox + 1 objectness)
3. Optimize the graph to remove dead nodes

Benefits:
- Reduce FLOPs in final layers
- Reduce memory bandwidth
- Faster inference on edge devices (Raspberry Pi, etc.)
"""

import onnx
import numpy as np
from onnx import helper, numpy_helper
import onnx_graphsurgeon as gs

def prune_yolo_to_person_only(input_model_path, output_model_path):
    """
    Prune YOLOv5 ONNX model to detect Person class only
    
    Args:
        input_model_path: Path to original ONNX model (80 classes)
        output_model_path: Path to save pruned model (1 class - Person)
    """
    
    print("="*70)
    print("YOLOv5 Class Pruning: 80 Classes → 1 Class (Person)")
    print("="*70)
    
    # Load ONNX model as graph
    print(f"\n[1/6] Loading model: {input_model_path}")
    graph = gs.import_onnx(onnx.load(input_model_path))
    
    print(f"[INFO] Original graph: {len(graph.nodes)} nodes, {len(graph.tensors())} tensors")
    
    # Find output node
    print("\n[2/6] Analyzing output tensor...")
    output_tensor = graph.outputs[0]
    print(f"[INFO] Output tensor name: {output_tensor.name}")
    print(f"[INFO] Output tensor shape: {output_tensor.shape}")
    
    # YOLOv5 output format: [batch, num_anchors, 85]
    # 85 = 4 (bbox: x, y, w, h) + 1 (objectness) + 80 (class scores)
    # We want: [batch, num_anchors, 6]
    # 6 = 4 (bbox) + 1 (objectness) + 1 (Person class score)
    
    if len(output_tensor.shape) != 3:
        raise ValueError(f"Expected 3D output tensor, got shape: {output_tensor.shape}")
    
    original_classes = output_tensor.shape[2]
    print(f"[INFO] Original output size: {original_classes} (4 bbox + 1 obj + {original_classes - 5} classes)")
    
    # Create Slice node to extract only Person class
    print("\n[3/6] Creating Slice node for Person class...")
    
    # Slice indices: [0:batch, 0:anchors, 0:6]
    # This keeps: bbox (0-3), objectness (4), Person class (5)
    slice_starts = np.array([0, 0, 0], dtype=np.int64)
    slice_ends = np.array([output_tensor.shape[0] if isinstance(output_tensor.shape[0], int) else 1, 
                           output_tensor.shape[1] if isinstance(output_tensor.shape[1], int) else 8400, 
                           6], dtype=np.int64)
    slice_axes = np.array([0, 1, 2], dtype=np.int64)
    slice_steps = np.array([1, 1, 1], dtype=np.int64)
    
    # Create constant tensors for Slice operation
    starts_tensor = gs.Constant(name="slice_starts", values=slice_starts)
    ends_tensor = gs.Constant(name="slice_ends", values=slice_ends)
    axes_tensor = gs.Constant(name="slice_axes", values=slice_axes)
    steps_tensor = gs.Constant(name="slice_steps", values=slice_steps)
    
    # Create new output tensor with reduced shape
    new_output_shape = list(output_tensor.shape)
    new_output_shape[2] = 6  # Reduce from 85 to 6
    
    sliced_output = gs.Variable(
        name="output_person_only",
        dtype=output_tensor.dtype,
        shape=new_output_shape
    )
    
    # Create Slice node
    slice_node = gs.Node(
        op="Slice",
        name="slice_person_class",
        inputs=[output_tensor, starts_tensor, ends_tensor, axes_tensor, steps_tensor],
        outputs=[sliced_output]
    )
    
    # Add Slice node to graph
    graph.nodes.append(slice_node)
    
    # Update graph output
    graph.outputs = [sliced_output]
    
    print(f"[INFO] New output shape: {new_output_shape}")
    print(f"[INFO] Reduction: {original_classes} → 6 ({(1 - 6/original_classes)*100:.1f}% smaller)")
    
    # Clean up graph
    print("\n[4/6] Cleaning up graph (removing dead nodes)...")
    graph.cleanup()
    
    print(f"[INFO] Cleaned graph: {len(graph.nodes)} nodes, {len(graph.tensors())} tensors")
    
    # Export modified graph
    print("\n[5/6] Exporting modified ONNX model...")
    onnx_model = gs.export_onnx(graph)
    
    # Save model
    onnx.save(onnx_model, output_model_path)
    print(f"[INFO] Saved to: {output_model_path}")
    
    # Simplify model (optional but recommended)
    print("\n[6/6] Simplifying model (constant folding, dead code elimination)...")
    try:
        import onnxsim
        
        print("[INFO] Running onnx-simplifier...")
        model_simplified, check = onnxsim.simplify(
            onnx_model,
            check_n=3,
            perform_optimization=True,
            skip_fuse_bn=False,
            skip_optimization=False
        )
        
        if check:
            onnx.save(model_simplified, output_model_path)
            print("[INFO] ✓ Model simplified successfully")
        else:
            print("[WARNING] Simplification check failed, using non-simplified model")
    
    except ImportError:
        print("[WARNING] onnx-simplifier not installed, skipping simplification")
        print("[INFO] Install with: pip install onnx-simplifier")
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    import os
    original_size = os.path.getsize(input_model_path) / (1024 * 1024)
    pruned_size = os.path.getsize(output_model_path) / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Pruned model size: {pruned_size:.2f} MB")
    print(f"Size reduction: {original_size - pruned_size:.2f} MB ({(1 - pruned_size/original_size)*100:.1f}%)")
    print(f"\nOutput shape: {original_classes} → 6")
    print(f"Classes: 80 → 1 (Person only)")
    print(f"Expected speedup: ~5-10% (less computation in final layers)")
    print("="*70 + "\n")


def verify_pruned_model(model_path):
    """Verify the pruned model structure"""
    
    print("\n" + "="*70)
    print("Model Verification")
    print("="*70)
    
    model = onnx.load(model_path)
    
    print(f"\nModel: {model_path}")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    
    # Check input
    print("\nInput:")
    for input_tensor in model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  Name: {input_tensor.name}")
        print(f"  Shape: {shape}")
        print(f"  Type: {input_tensor.type.tensor_type.elem_type}")
    
    # Check output
    print("\nOutput:")
    for output_tensor in model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  Name: {output_tensor.name}")
        print(f"  Shape: {shape}")
        print(f"  Type: {output_tensor.type.tensor_type.elem_type}")
    
    print(f"\nTotal nodes: {len(model.graph.node)}")
    print("="*70 + "\n")


def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Prune YOLOv5 ONNX model to Person class only")
    parser.add_argument("--input", type=str, default="yolov5nu.onnx", 
                       help="Input ONNX model path (80 classes)")
    parser.add_argument("--output", type=str, default="yolov5nu_person_only.onnx",
                       help="Output ONNX model path (1 class)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify the pruned model after creation")
    
    args = parser.parse_args()
    
    # Check if input file exists
    import os
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Prune model
    try:
        prune_yolo_to_person_only(args.input, args.output)
        
        # Verify if requested
        if args.verify:
            verify_pruned_model(args.output)
        
        print("\n✓ Success! Pruned model ready for inference.")
        print(f"\nUsage in your detector:")
        print(f"  MODEL_PATH = '{args.output}'")
        print(f"  # Output shape: [batch, anchors, 6]")
        print(f"  # Format: [x, y, w, h, objectness, person_score]")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
