import torch
import onnx
from onnx import shape_inference

import torch.nn as nn
import torch.onnx


# Define a simple RNN-T model
class RNNTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNTModel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, decoder_input):
        encoder_output, _ = self.encoder(encoder_input)
        decoder_output, _ = self.decoder(decoder_input)
        output = self.fc(decoder_output)
        return output

def rnnt_to_onnx(directory):
    # Instantiate the model
    input_size = 128
    hidden_size = 256
    output_size = 30
    model = RNNTModel(input_size, hidden_size, output_size)

    # Set the model to evaluation mode
    model.eval()

    # Dummy inputs for tracing
    encoder_input = torch.randn(1, 10, input_size)  # Batch size 1, sequence length 10
    decoder_input = torch.randn(1, 5, hidden_size)  # Batch size 1, sequence length 5

    # Export the model to ONNX format
    onnx_file_path = f"{directory}/rnnt.onnx"
    torch.onnx.export(
        model,
        (encoder_input, decoder_input),
        onnx_file_path,
        export_params=True,
        opset_version=14,
        input_names=["encoder_input", "decoder_input"],
        output_names=["output"],
        # dynamic_axes={
        #     "encoder_input": {0: "batch_size", 1: "sequence_length"},
        #     "decoder_input": {0: "batch_size", 1: "sequence_length"},
        #     "output": {0: "batch_size", 1: "sequence_length"},
        # },
    )

    # Perform shape inference on the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # Save the inferred model back to the same path
    onnx.save(inferred_model, onnx_file_path)

    print(f"Model has been converted to ONNX format and saved at {onnx_file_path}")
    return onnx_file_path