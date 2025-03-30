import torch
from transformers import GPT2Model, GPT2Tokenizer
from pathlib import Path
import onnx
from onnx import shape_inference
from datasets import load_dataset

def gpt_to_onnx(directory):
    # Load pretrained GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the model to evaluation mode
    model.eval()

    # Load a dataset from Hugging Face and use a sample as dummy input
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Calculate the median sequence length of the dataset
    sequence_lengths = [len(sample["text"].split()) for sample in dataset]
    median_length = sorted(sequence_lengths)[len(sequence_lengths) // 2]

    # Find the sample with length closest to the median
    closest_sample = min(dataset, key=lambda sample: abs(len(sample["text"].split()) - median_length))
    sample_text = closest_sample["text"]  # Extract the text of the closest sample)
    dummy_input = tokenizer(sample_text, return_tensors="pt")["input_ids"]

    # Define ONNX output path
    onnx_file_path = Path(directory) / "gpt2.onnx"

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        input_names=["input_ids"],
        output_names=["output"],
        # dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
        #             "output": {0: "batch_size", 1: "sequence_length"}},
        opset_version=14
    )

    # Perform shape inference on the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # Save the inferred model back to the same path
    onnx.save(inferred_model, onnx_file_path)

    print(f"Model has been converted to ONNX and saved at {onnx_file_path}")
    return onnx_file_path