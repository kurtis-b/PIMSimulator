import torch
import os
import onnx
from onnx import shape_inference

import dorn.model.dorn as dorn

def dorn_to_onnx(directory):
    # Instantiate the DORN model
    model = dorn.DORN(pretrained=False)

    # Set the model to evaluation mode
    model.eval()

    # Define a dummy input tensor (batch size = 1, 3 channels, height = 385, width = 513)
    dummy_input = torch.randn(1, 3, 385, 513)

    # Define the ONNX file path
    onnx_file_path = os.path.join(directory, "dorn.onnx")

    # Export the model to ONNX format
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['prob', 'label'],
        # dynamic_axes={'input': {0: 'batch_size'}, 'prob': {0: 'batch_size'}, 'label': {0: 'batch_size'}}
    )

    # Perform shape inference on the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # Save the inferred model back to the same path
    onnx.save(inferred_model, onnx_file_path)

    print(f"Model has been successfully exported to {onnx_file_path}")
    return onnx_file_path
