import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class TTE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1,
                 n_layers=1, dropout=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = nn.GRU(self.input_size, self.hidden_size,
                          self.n_layers, dropout=self.dropout)
        self.alpha = nn.Linear(self.hidden_size, self.output_size)
        self.beta = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, input_lengths):
        packed = pack_padded_sequence(inputs, input_lengths)
        rnn_outputs, _ = self.rnn(packed)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs)

        output = rnn_outputs

        alpha = self.alpha(output)
        alpha = torch.exp(alpha)

        beta = self.beta(output)
        beta = F.softplus(beta)

        return alpha, beta
