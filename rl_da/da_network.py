import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
An LSTM-base Data Association network.
"""
class DANetwork(nn.Module):
    """
    :param input_dim Dimension of the input vector for each LSTM + Linear layer
    :param hidden_dim Dimension of the LSTM hidden state
    :param output_dim Dimension of the output vector for each LSTM + layer
    :param n_layers Number of LSTM and Linear layer pairs
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(DANetwork, self).__init__()
        self.input_dim_ = input_dim
        self.hidden_dim_ = hidden_dim
        self.output_dim_ = output_dim
        self.n_layers_ = n_layers

        self.lstms_ = nn.ModuleList([nn.LSTM(input_dim, hidden_dim) for _ in range(n_layers)])
        self.linears_ = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(n_layers)])

    """
    Makes forward propagation.
    :param input_seq A (n_layers, input_dim) tensor that represents sequence of inputs for LSTM cells
    :return A (n_layers, output_dim) tensor that represents outputs of each LSTM cell 
    """
    def forward(self, input_vectors):
        hidden_input = (torch.randn(1, 1, self.hidden_dim_),
                        torch.randn(1, 1, self.hidden_dim_))
        outputs = torch.zeros((self.n_layers_, self.output_dim_))

        for i in range(self.n_layers_):
            input_batch = input_vectors[i].unsqueeze(dim=0).unsqueeze(dim=0)
            out, hidden_input = self.lstms_[i](input_batch, hidden_input)
            outputs[i, :] = self.linears_[i](out[0, 0, :])

        return outputs

