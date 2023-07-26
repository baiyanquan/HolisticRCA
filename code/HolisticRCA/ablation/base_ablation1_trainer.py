from abc import ABC
from shared_util.logger import Logger
import torch
from HolisticRCA.data_loader.rca_data_loader import RCADataLoader
from HolisticRCA.model.re.feature_integration import FeatureIntegration
from HolisticRCA.model.re.feature_fusion import FeatureFusion
from HolisticRCA.model.re.fault_classifier import FaultClassifier
from HolisticRCA.util.data_handler import rearrange_y
from shared_util.evaluation_metrics import *
import torch.nn as nn


class AblationRepresentationLearning(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data
        self.different_modal_mapping_dict = nn.ModuleDict()
        for modal_type in self.meta_data['modal_types']:
            self.different_modal_mapping_dict[modal_type] = nn.Linear(in_features=param_dict['window_size'],
                                                                      out_features=param_dict['orl_te_in_channels'])

    def forward(self, batch_data):
        for modal_type in self.meta_data['modal_types']:
            batch_data[f'x_{modal_type}'] = batch_data[f'x_{modal_type}'].to(self.device_marker.device)
            batch_data[f'x_{modal_type}'] = self.different_modal_mapping_dict[modal_type](batch_data[f'x_{modal_type}'])
        return batch_data


class BaseAblation1Trainer(ABC):
    def __init__(self, param_dict):
        self.param_dict = param_dict
        assert param_dict['orl_te_in_channels'] == param_dict['efi_in_dim'] and param_dict['efi_out_dim'] == param_dict['eff_in_dim']
        self.device = torch.device("cuda" if torch.cuda.is_available() and param_dict['gpu'] else "cpu")

        self.logger = Logger(logging_level='DEBUG').logger

        self.rca_data_loader = RCADataLoader(param_dict)
        self.rca_data_loader.load_data(f'{self.param_dict["dataset_path"]}')

        o11y_representation_learning = AblationRepresentationLearning(param_dict, self.rca_data_loader.meta_data)
        re_feature_integration = FeatureIntegration(param_dict, self.rca_data_loader.meta_data)
        re_feature_fusion = FeatureFusion(param_dict, self.rca_data_loader.meta_data)
        re_fault_classifier = FaultClassifier(param_dict, self.rca_data_loader.meta_data)
        self.model = torch.nn.Sequential(o11y_representation_learning, re_feature_integration, re_feature_fusion, re_fault_classifier).to(self.device)
        self.model_rank = []

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'], weight_decay=self.param_dict['weight_decay'])

        criterion_dict = dict()
        for ent_type in self.rca_data_loader.meta_data['ent_types']:
            pos_weight = torch.FloatTensor(self.rca_data_loader.meta_data['ent_fault_type_weight'][ent_type]).to(self.device)
            criterion_dict[ent_type] = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(self.param_dict['epochs']):
            self.model.train()
            train_loss = 0
            for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['train']):
                optimizer.zero_grad()
                y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
                out = self.model(batch_data)
                loss = 0
                for ent_type in self.rca_data_loader.meta_data['ent_types']:
                    loss += criterion_dict[ent_type](out[ent_type][torch.where(~(y[ent_type] == -1).all(1))[0]], y[ent_type][torch.where(~(y[ent_type] == -1).all(1))[0]])
                train_loss += batch_data['y'].shape[0] * loss.item()
                loss.backward()
                optimizer.step()
            train_loss /= len(self.rca_data_loader.data_loader['train'].dataset)
            self.logger.info(f'[{epoch}/{self.param_dict["epochs"]}] | train_loss: {train_loss:.5f}')

            self.model.eval()
            y_pred = dict()
            y_true = dict()
            if epoch % 10 == 0:
                with torch.no_grad():
                    for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['valid']):
                        y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
                        out = self.model(batch_data)
                        for ent_type in self.rca_data_loader.meta_data['ent_types']:
                            if ent_type not in y_pred.keys():
                                y_pred[ent_type] = []
                                y_true[ent_type] = []
                            y_pred[ent_type].extend((torch.sigmoid(out[ent_type][y[ent_type] != -1].reshape(-1, out[ent_type].shape[1])) > self.param_dict[f'{ent_type}_accuracy_th']).cpu().detach().numpy())
                            y_true[ent_type].extend(y[ent_type][y[ent_type] != -1].reshape(-1, y[ent_type].shape[1]).cpu().detach().numpy())
                self.output_evaluation_rca_d3_result(y_pred, y_true, 'valid')
        torch.save(self.model.state_dict(), self.param_dict["model_path"])

    def evaluate_rca_d3(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.param_dict["model_path"]))

        y_pred, y_true = dict(), dict()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader['test']):
                y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
                out = self.model(batch_data)
                for ent_type in self.rca_data_loader.meta_data['ent_types']:
                    fault_prob = torch.sigmoid(out[ent_type])
                    temp_y_pred = (fault_prob > self.param_dict[f'{ent_type}_accuracy_th']).cpu().detach().numpy()
                    temp_y_true = y[ent_type].cpu().detach().numpy()
                    if ent_type not in y_pred.keys():
                        y_pred[ent_type] = []
                        y_true[ent_type] = []
                    y_pred[ent_type].extend(temp_y_pred)
                    y_true[ent_type].extend(temp_y_true)

        self.output_evaluation_rca_d3_result(y_pred, y_true, 'test')

    def output_evaluation_rca_d3_result(self, y_pred, y_true, dataset_type):
        self.logger.info('----------')
        self.logger.info(f'evaluation dataset type: {dataset_type}')
        for ent_type in self.rca_data_loader.meta_data['ent_types']:
            ent_y_pred = np.array(y_pred[ent_type])
            ent_y_true = np.array(y_true[ent_type])
            fc_result = fault_type_classification(ent_y_pred, ent_y_true)
            convert = {
                'p': 'precision',
                'r': 'recall',
                'f1': 'f1'
            }
            for em in ['p', 'r', 'f1']:
                self.logger.info(f'{ent_type.ljust(8) + convert[em].ljust(9)} | micro: {fc_result["micro_" + convert[em] + "_score"]:.6f}; macro: {fc_result["macro_" + convert[em] + "_score"]:.6f}; score: {fc_result[convert[em] + "_score"]}')
        self.logger.info('----------')
