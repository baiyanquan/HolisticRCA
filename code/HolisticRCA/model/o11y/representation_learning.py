import torch
import torch.nn as nn
from HolisticRCA.model.common.embed import PositionalEmbedding


class RepresentationLearning(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data
        self.different_modal_mapping_dict = nn.ModuleDict()
        self.positional_embedding_dict = nn.ModuleDict()
        self.modal_transformer_encoder_layer_dict = nn.ModuleDict()
        for modal_type in self.meta_data['modal_types']:
            self.positional_embedding_dict[modal_type] = PositionalEmbedding(in_features=param_dict['window_size'],
                                                                             num_of_o11y_features=self.meta_data['o11y_length'][modal_type],
                                                                             register_name=f'{modal_type}_pe')
            self.different_modal_mapping_dict[modal_type] = nn.Linear(in_features=param_dict['window_size'],
                                                                      out_features=param_dict['orl_te_in_channels'])
            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=param_dict['orl_te_in_channels'],
                                                                   nhead=param_dict['orl_te_heads'])
            self.modal_transformer_encoder_layer_dict[modal_type] = nn.TransformerEncoder(transformer_encoder_layer, num_layers=param_dict['orl_te_layers'])

    def forward(self, batch_data):
        for modal_type in self.meta_data['modal_types']:
            batch_data[f'x_{modal_type}'] = batch_data[f'x_{modal_type}'].to(self.device_marker.device)
            batch_data[f'x_{modal_type}'] = self.positional_embedding_dict[modal_type](batch_data[f'x_{modal_type}']).contiguous()
            batch_data[f'x_{modal_type}'] = self.different_modal_mapping_dict[modal_type](batch_data[f'x_{modal_type}'])
            batch_data[f'x_{modal_type}'] = self.modal_transformer_encoder_layer_dict[modal_type](batch_data[f'x_{modal_type}'])
        return batch_data
