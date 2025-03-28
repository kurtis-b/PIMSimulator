import os
import torch
import onnx
from onnx import shape_inference

from stargan.model import Generator

def stargan_to_onnx(directory):
    # Load the pre-trained StarGAN generator model
    G = Generator()  # Adjust parameters as needed
    G.eval()

    # Define dummy input tensors with the appropriate shapes
    dummy_input_x = torch.randn(1, 3, 128, 128)  # Fixed batch size for compatibility
    dummy_input_c = torch.randn(1, 5)  # Fixed batch size for compatibility

    # Export the model to ONNX format
    onnx_file_path = os.path.join(directory, "stargan_generator.onnx")
    torch.onnx.export(
        G, 
        (dummy_input_x, dummy_input_c),  # Pass both inputs as a tuple
        onnx_file_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input_x', 'input_c'], 
        output_names=['output'], 
        # dynamic_axes=None  # Disable dynamic axes to avoid unsupported operations
    )

    # Perform shape inference on the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # Save the inferred model back to the same path
    onnx.save(inferred_model, onnx_file_path)

    print(f"Model has been converted to ONNX and saved at {onnx_file_path}")
    return onnx_file_path