import torch
import torch.nn as nn
from HolisticRCA.model.common.GAT_net import GATNet
from HolisticRCA.util.ent_batch_graph import EntBatchGraph


class FeatureFusion(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data
        self.GAT_net = GATNet(in_channels=param_dict['eff_in_dim'],
                              out_channels=param_dict['eff_GAT_out_channels'],
                              heads=param_dict['eff_GAT_heads'],
                              dropout=param_dict['eff_GAT_dropout'])
        self.linear_dict = nn.ModuleDict()
        for ent_type in self.meta_data['ent_types']:
            index_pair = self.meta_data['ent_fault_type_index'][ent_type]
            self.linear_dict[ent_type] = nn.Linear(param_dict['eff_GAT_out_channels'], index_pair[1] - index_pair[0])

    def forward(self, batch_data):
        ent_batch_graph = EntBatchGraph(batch_data, self.meta_data).to(self.device_marker.device)
        x = ent_batch_graph.x['re']
        x = self.GAT_net(x, ent_batch_graph.edge_index)
        return x
