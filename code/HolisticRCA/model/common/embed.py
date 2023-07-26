import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, in_features, num_of_o11y_features, register_name):
        super(PositionalEmbedding, self).__init__()

        temp_in_features = in_features
        if temp_in_features % 2 == 1:
            temp_in_features += 1

        temp_num_of_o11y_features = num_of_o11y_features
        if temp_num_of_o11y_features % 2 == 1:
            temp_num_of_o11y_features += 1

        pe = torch.zeros(temp_in_features, temp_num_of_o11y_features).float()
        pe.require_grad = False

        position = torch.arange(0, temp_in_features).float().unsqueeze(1)
        div_term = (torch.arange(0, temp_num_of_o11y_features, 2).float() * -(math.log(10000.0) / temp_num_of_o11y_features)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.transpose(1, 0)[:num_of_o11y_features, :in_features]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe + x


class AlignEmbedding(nn.Module):
    def __init__(self, batch_size, feature_length, in_dim):
        super().__init__()
        add_embedding = torch.zeros([batch_size, feature_length, in_dim]).float()
        add_embedding.require_grad = False

        self.register_buffer('add_embedding', add_embedding)

    def forward(self, x):
        return x + self.add_embedding
