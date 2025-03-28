import torch
from transformers import ViTModel
import onnx
from onnx import shape_inference

def vit_to_onnx(directory):
    # Load a pretrained Vision Transformer (ViT) model
    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTModel.from_pretrained(model_name)
    model.eval()

    # Dummy input for the model
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 image

    # Export the model to ONNX format
    onnx_file_path = f"{directory}/vit.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        input_names=["input"],
        output_names=["output"],
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14
    )

    # Perform shape inference on the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # Save the inferred model back to the same path
    onnx.save(inferred_model, onnx_file_path)

    print(f"Model has been successfully converted to ONNX format and saved at {onnx_file_path}")
    return onnx_file_path