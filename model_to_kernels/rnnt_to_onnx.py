import torch
import onnx
from onnx import shape_inference

import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

import numpy as np
from lstm_to_onnx import lstm_to_onnx

class BaseDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, share_weight=False):
        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_proj = nn.Linear(hidden_size, output_size)

        if share_weight:
            self.embedding.weight = self.output_proj.weight

    def forward(self, inputs, length=None, hidden=None):

        embed_inputs = self.embedding(inputs)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.output_proj(outputs)

        return outputs, hidden

class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=True):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.output_proj = nn.Linear(2 * hidden_size if bidirectional else hidden_size,
                                     output_size,
                                     bias=True)

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.output_proj(outputs)

        return logits, hidden

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class Transducer(nn.Module):
    def __init__(self):
        super(Transducer, self).__init__()

        self.encoder = BaseEncoder(
            input_size=160,  # feature_dim
            hidden_size=320,
            output_size=320,
            n_layers=4,
            dropout=0.3,
            bidirectional=True
        )

        self.decoder = BaseDecoder(
            hidden_size=512,
            vocab_size=4232,  # vocab_size
            output_size=320,
            n_layers=1,
            dropout=0.3,
            share_weight=False
        )

        self.joint = JointNet(
            input_size=640,  # enc output_size + dec output_size
            inner_dim=512,
            vocab_size=4232  # vocab_size
        )

    def forward(self, inputs, inputs_length, targets, targets_length):

        enc_state, _ = self.encoder(inputs, inputs_length)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)

        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        logits = self.joint(enc_state, dec_state)

        return logits

def generate_dummy_inputs(num_samples=1):
    feature_dim = 160  # Assuming 160-dimensional features (e.g., log-Mel spectrograms)
    max_input_length = 500
    max_target_length = 50
    vocab_size = 4232

    # Generate random features
    features = np.random.rand(num_samples, max_input_length, feature_dim).astype(np.float32)

    # Generate random input lengths (ensuring they are <= max_input_length)
    inputs_length = np.random.randint(1, max_input_length + 1, size=(num_samples,)).astype(np.int64)

    # Generate random target sequences
    targets = np.random.randint(1, vocab_size, size=(num_samples, max_target_length)).astype(np.int64)

    # Generate random target lengths (ensuring they are <= max_target_length)
    targets_length = np.random.randint(1, max_target_length + 1, size=(num_samples,)).astype(np.int64)

    return features, inputs_length, targets, targets_length

# The classes used above were taken from an RNN-T implementation on GitHub:
# https://github.com/ZhengkunTian/rnn-transducer
def rnnt_to_onnx(directory):
    # Instantiate the model
    model = Transducer()

    # Set the model to evaluation mode
    model.eval()
    # Generate dummy inputs for the model
    features, inputs_length, targets, targets_length = generate_dummy_inputs() 

    # Convert numpy arrays to PyTorch tensors
    encoder_input = torch.tensor(features)
    inputs_length = torch.tensor(inputs_length)
    decoder_input = torch.tensor(targets)
    targets_length = torch.tensor(targets_length)

    # Export the model to ONNX format
    onnx_file_path = f"{directory}/rnnt.onnx"
    torch.onnx.export(
        model,
        (encoder_input, inputs_length, decoder_input, targets_length),
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

    rnnt_encoder_path = lstm_to_onnx(directory, 160, 320, 320, 4, "rnnt_encoder_part")
    rnnt_decoder_path = lstm_to_onnx(directory, 4232, 512, 320, 1, "rnnt_decoder_part")
    print(f"Model has been converted to ONNX format and saved at {onnx_file_path}")
    return [onnx_file_path, rnnt_encoder_path, rnnt_decoder_path]