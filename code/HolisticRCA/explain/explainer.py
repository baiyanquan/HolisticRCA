import math
import torch
import torch.nn as nn
from torch.nn import Parameter, ParameterDict
from torch_geometric.nn import MessagePassing
from HolisticRCA.util.data_handler import copy_batch_data
from torch_geometric.utils import sort_edge_index
import numpy as np
from shared_util.logger import Logger


class Explainer(nn.Module):
    def __init__(self, model, meta_data, param_dict):
        super().__init__()
        self.coeffs = {
            'ent_edge_size': 0.005,
            'ent_edge_reduction': 'sum',
            'o11y_size': 1.0,
            'o11y_reduction': 'mean',
            'ent_edge_entropy': 1.0,
            'o11y_entropy': 0.1,
            'EPS': 1e-15,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() and param_dict['explainer_gpu'] else "cpu")
        self.model = model
        self.meta_data = meta_data
        self.param_dict = param_dict
        self.o11y_mask = self.hard_o11y_mask = None
        self.ent_edge_mask = self.hard_ent_edge_mask = None
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.logger = Logger(logging_level='DEBUG').logger

    def init_o11y_mask(self):
        mask = ParameterDict()
        for modal_type in self.meta_data['modal_types']:
            mask[modal_type] = Parameter(torch.FloatTensor(self.meta_data['o11y_length'][modal_type])).to(self.device)
            std = 0.1
            with torch.no_grad():
                mask[modal_type].normal_(1.0, std)
        self.o11y_mask = mask

    def init_ent_edge_mask(self, test_sample_data):
        num_edges = test_sample_data['ent_edge_index'].shape[2]
        num_entities = len(self.meta_data['ent_names'])
        std = nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (num_entities + num_entities)
        )
        mask = Parameter(torch.randn(num_edges, device=self.device) * std)
        self.ent_edge_mask = mask

    def set_masks(self, test_sample_data):
        edge_index = torch.squeeze(test_sample_data['ent_edge_index'])
        for module in self.model[2].GAT_net.modules():
            loop_mask = torch.full_like(edge_index[0], True, dtype=bool)
            if isinstance(module, MessagePassing):
                module.explain = True
                module._edge_mask = self.ent_edge_mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = True

    def clean_explainer(self):
        for module in self.model[2].GAT_net.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        self.o11y_mask = self.hard_o11y_mask = None
        self.ent_edge_mask = self.hard_ent_edge_mask = None

    def train_explainer(self, test_sample_data, original_y_pred, entity_type, entity_index):
        self.init_o11y_mask()
        self.init_ent_edge_mask(test_sample_data)
        self.set_masks(test_sample_data)
        parameters = []
        for key in self.o11y_mask.keys():
            parameters.append(self.o11y_mask[key])
        parameters.append(self.ent_edge_mask)
        optimizer = torch.optim.Adam(parameters, lr=self.param_dict['explainer_lr'], weight_decay=self.param_dict['explainer_weight_decay'])
        test_sample_data['ent_edge_index'][0] = sort_edge_index(test_sample_data['ent_edge_index'][0])
        for epoch in range(self.param_dict['explainer_epochs']):
            copy_sample_data = copy_batch_data(test_sample_data, self.device)
            optimizer.zero_grad()
            self.model.zero_grad()
            for modal_type in self.meta_data['modal_types']:
                x = copy_sample_data[f'x_{modal_type}']
                std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                z = torch.normal(mean=mean_tensor, std=std_tensor)
                copy_sample_data[f'x_{modal_type}'] = x + torch.mul(z.transpose(1, 2), (1 - self.o11y_mask[modal_type])).transpose(1, 2)

            out = self.model(copy_sample_data)
            y_pred = out[entity_type][entity_index]

            loss = self.explainer_loss(y_pred, original_y_pred)
            if epoch % 10 == 0:
                self.logger.info(f'[{epoch}/{self.param_dict["explainer_epochs"]}] | train_loss: {loss.item():.5f}')

            loss.backward()
            optimizer.step()

            if epoch == 0 and self.o11y_mask is not None:
                self.hard_o11y_mask = dict()
                for modal_type in self.meta_data['modal_types']:
                    self.hard_o11y_mask[modal_type] = self.o11y_mask[modal_type].grad != 0.0
            if epoch == 0 and self.ent_edge_mask is not None:
                self.hard_ent_edge_mask = self.ent_edge_mask.grad != 0.0

        related_ent_edge_sorted_mask, related_ent_edge_indices = torch.sort(self.ent_edge_mask[self.hard_ent_edge_mask].sigmoid(), descending=True)
        final_edge_index = test_sample_data['ent_edge_index'][self.hard_ent_edge_mask.repeat(2, 1).unsqueeze(0)].reshape(2, -1).t()[related_ent_edge_indices]
        ent_name_result, index_set = [], set()
        for i in range(final_edge_index.shape[0]):
            for j in [0, 1]:
                if final_edge_index[i][j].cpu().detach().item() not in index_set:
                    ent_name_result.append((related_ent_edge_sorted_mask[i].cpu().detach().item(), self.meta_data['ent_names'][final_edge_index[i][j]]))
                    index_set.add(final_edge_index[i][j].cpu().detach().item())

        related_o11y_sorted_mask = []
        related_o11y_names = []
        for modal_type in self.meta_data['modal_types']:
            mask, indices = torch.sort(self.o11y_mask[modal_type][self.hard_o11y_mask[modal_type]].sigmoid(), descending=True)
            related_o11y_sorted_mask.extend(mask.cpu().detach().tolist())
            related_o11y_names.extend(np.array(self.meta_data['o11y_names'][modal_type])[self.hard_o11y_mask[modal_type].cpu().detach()][indices.cpu().detach()].tolist())
        o11y_name_result = zip(related_o11y_sorted_mask, related_o11y_names)
        o11y_name_result = sorted(o11y_name_result, reverse=True)
        self.clean_explainer()
        return ent_name_result, o11y_name_result

    def explainer_loss(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)

        if self.hard_ent_edge_mask is not None:
            assert self.ent_edge_mask is not None
            m = self.ent_edge_mask[self.hard_ent_edge_mask].sigmoid()
            ent_edge_reduce = getattr(torch, self.coeffs['ent_edge_reduction'])
            loss = loss + self.coeffs['ent_edge_size'] * ent_edge_reduce(m)
            entropy = - m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['ent_edge_entropy'] * entropy.mean()

        if self.hard_o11y_mask is not None:
            assert self.o11y_mask is not None
            for modal_type in self.meta_data['modal_types']:
                if self.hard_o11y_mask[modal_type].any():
                    m = self.o11y_mask[modal_type][self.hard_o11y_mask[modal_type]].sigmoid()
                    o11y_reduce = getattr(torch, self.coeffs['o11y_reduction'])
                    loss = loss + self.coeffs['o11y_size'] * o11y_reduce(m) / len(self.meta_data['modal_types'])
                    entropy = - m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
                    loss = loss + self.coeffs['o11y_entropy'] * entropy.mean() / len(self.meta_data['modal_types'])
        return loss
