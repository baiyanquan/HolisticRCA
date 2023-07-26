import torch.nn as nn
import torch.nn.functional as F


class MapNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mapping_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.mapping_layer(x)
        x = F.elu(x)
        return x
