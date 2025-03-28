import torch

import torch.nn as nn
import torch.onnx
import onnx
from onnx import shape_inference

# Define a simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM gates
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, h_t), dim=1)

            i_t = torch.sigmoid(self.input_gate(combined))
            f_t = torch.sigmoid(self.forget_gate(combined))
            g_t = torch.tanh(self.cell_gate(combined))
            o_t = torch.sigmoid(self.output_gate(combined))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        output = self.fc(h_t)
        return output

def lstm_to_onnx(directory):
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 5
    num_layers = 1

    # Instantiate the model
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)

    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, input_size)  # Batch size = 1, Sequence length = 3

    # Save the model to ONNX format
    onnx_file_path = f"{directory}/lstm.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file_path, 
        input_names=["input"], 
        output_names=["output"], 
        # dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size"}},
        opset_version=14
    )

    # Perform shape inference on the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # Save the inferred model back to the same path
    onnx.save(inferred_model, onnx_file_path)

    print(f"Model has been converted to ONNX format and saved at {onnx_file_path}")
    return onnx_file_path