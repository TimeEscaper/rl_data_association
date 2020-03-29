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
    :param seq_len Target sequence length
    """
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(DANetwork, self).__init__()
        self.input_dim_ = input_dim
        self.hidden_dim_ = hidden_dim
        self.output_dim_ = output_dim
        self.seq_len_ = seq_len

        #self.lstms_ = nn.ModuleList([nn.LSTM(input_dim, hidden_dim) for _ in range(n_layers)])
        #self.linears_ = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(n_layers)])
        self.lstm_ = nn.LSTM(input_dim, hidden_dim)
        self.linear_ = nn.Linear(hidden_dim, output_dim)

    """
    Makes forward propagation.
    :param input_seq A (seq_len, input_dim) tensor that represents sequence of inputs for LSTM cells
    :return A (seq_len, output_dim) tensor that represents outputs of each LSTM cell 
    """
    def forward(self, input_vectors):
        hidden_input = (torch.randn(1, 1, self.hidden_dim_),
                        torch.randn(1, 1, self.hidden_dim_))
        outputs = torch.zeros((self.seq_len_, self.output_dim_))

        input_batch = input_vectors.view(self.seq_len_, 1, self.input_dim_)

        output_batch, hidden_output = self.lstm_(input_batch, hidden_input)
        output_batch = output_batch.view(self.seq_len_, self.hidden_dim_)

        for i, output_vector in enumerate(output_batch):
            outputs[i, :] = F.softmax(self.linear_(output_vector), dim=-1)

        return outputs

