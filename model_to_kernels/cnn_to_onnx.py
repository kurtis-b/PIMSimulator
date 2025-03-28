import torch

import torchvision.models as models
import torch.onnx
import onnx
from onnx import shape_inference

def cnn_to_onnx(directory):
    # Load a pretrained ResNet-50 model
    model = models.resnet50(weights='DEFAULT')
    model.eval()  # Set the model to evaluation mode

    # Dummy input for the model (batch size: 1, 3 color channels, 224x224 image)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX format
    onnx_file_path = f"{directory}/resnet50.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file_path, 
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=14,    # ONNX version to export to
        do_constant_folding=True,  # Optimize constant folding for inference
        input_names=['input'],  # Input tensor name
        output_names=['output'],  # Output tensor name
        # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic batch size
    )

    # Perform shape inference on the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # Save the inferred model back to the same path
    onnx.save(inferred_model, onnx_file_path)

    print(f"Model has been converted to ONNX format and saved at {onnx_file_path}")
    return onnx_file_path