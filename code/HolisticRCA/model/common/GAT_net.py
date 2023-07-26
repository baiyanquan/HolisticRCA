import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout):
        super().__init__()
        self.conv1 = gnn.GATv2Conv(in_channels=in_channels,
                                   out_channels=out_channels,
                                   heads=heads,
                                   dropout=dropout,
                                   add_self_loops=False)
        self.conv2 = gnn.GATv2Conv(in_channels=out_channels * heads,
                                   out_channels=int(out_channels / heads),
                                   heads=heads,
                                   dropout=dropout,
                                   add_self_loops=False)
        self.dropout_ratio = dropout

    def forward(self, x, edge_index):
        batch_size = x.shape[0]
        x = x.view(x.shape[0] * x.shape[1], x.shape[2]).contiguous()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = x.view(batch_size, int(x.shape[0] / batch_size), x.shape[1]).contiguous()
        return x
