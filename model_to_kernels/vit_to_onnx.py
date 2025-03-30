import torch
from transformers import ViTModel, ViTImageProcessor
import onnx
from onnx import shape_inference
from PIL import Image
import requests

def vit_to_onnx(directory):
    # Load a pretrained Vision Transformer (ViT) model
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    model.eval()

    # Load the feature extractor for preprocessing
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Download a sample image from ILSVRC/imagenet-1k
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    dummy_input = inputs["pixel_values"] 

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