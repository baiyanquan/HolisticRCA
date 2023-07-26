import torch
from HolisticRCA.data_loader.mask_learning_data_loader import MaskLearningDataLoader
from HolisticRCA.base.base_rca_trainer import BaseRCATrainer
from HolisticRCA.util.data_handler import copy_batch_data
from HolisticRCA.util.data_handler import rearrange_y
from shared_util.evaluation_metrics import *
from HolisticRCA.explain.explainer import Explainer


class BaseLocalizer(BaseRCATrainer):
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.mask_learning_data_loader = MaskLearningDataLoader(param_dict)
        self.mask_learning_data_loader.load_data(f'{self.param_dict["dataset_path"]}')
        
    def predict(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.param_dict["model_path"]))
        explainer = Explainer(self.model, self.mask_learning_data_loader.meta_data, self.param_dict)
        result = dict()
        for ent_type in self.rca_data_loader.meta_data['ent_types']:
            if ent_type not in result.keys():
                result[ent_type] = {
                    'total': 0,
                    'd1': {
                        'AC@1_num': 0,
                        'AC@3_num': 0,
                        'AC@5_num': 0,
                    },
                    'd2': {
                        'AC@1_num': 0,
                        'AC@3_num': 0,
                        'AC@5_num': 0,
                    }
                }

        for batch_id, batch_data in enumerate(self.mask_learning_data_loader.data_loader):
            explain_batch_data = copy_batch_data(batch_data, self.device)
            y = rearrange_y(self.rca_data_loader.meta_data, batch_data['y'], self.device)
            root_cause_list = []
            for ent_type in y.keys():
                pos = np.nonzero(y[ent_type])
                for i in range(pos.shape[0]):
                    root_cause_list.append({'d1': self.mask_learning_data_loader.meta_data['ent_names'][(self.mask_learning_data_loader.meta_data['ent_type_index'][ent_type][0] + pos[i][0]).item()],
                                            'd2': self.mask_learning_data_loader.meta_data['fault_type_related_o11y_names'][(self.mask_learning_data_loader.meta_data['ent_fault_type_index'][ent_type][0] + pos[i][1]).item()],
                                            'level': ent_type,
                                            'fault_type': self.mask_learning_data_loader.meta_data['fault_type_list'][self.mask_learning_data_loader.meta_data['ent_fault_type_index'][ent_type][0] + pos[i][1]]})

            exact_root_cause = dict()
            for root_cause in root_cause_list:
                exact_root_cause = root_cause
                if root_cause['level'] == 'service':
                    break
            result[exact_root_cause['level']]['total'] += 1
            with torch.no_grad():
                out = self.model(batch_data)

            localization_result = {
                'd1': dict(),
                'd2': dict()
            }

            suspect_list = []
            for ent_type in self.rca_data_loader.meta_data['ent_types']:
                temp_y_pred = (torch.sigmoid(out[ent_type]) > self.param_dict[f'{ent_type}_accuracy_th']).cpu().detach().numpy()
                fault_prob = torch.sigmoid(out[ent_type])
                ent_fault_prob = torch.max(fault_prob, dim=1).values.cpu().detach().numpy()
                for ent_index in range(len(temp_y_pred)):
                    suspect_list.append((ent_index, ent_type, ent_fault_prob[ent_index]))
                    if temp_y_pred[ent_index].any():
                        suspect_list.pop()
                        trigger_ent = self.mask_learning_data_loader.meta_data['ent_names'][(self.mask_learning_data_loader.meta_data['ent_type_index'][ent_type][0] + ent_index)]
                        if trigger_ent not in localization_result['d1'].keys():
                            localization_result['d1'][trigger_ent] = 0
                        localization_result['d1'][trigger_ent] += ent_fault_prob[ent_index]
                        ent_name_result, o11y_name_result = explainer.train_explainer(explain_batch_data, torch.sigmoid(out[ent_type])[ent_index], ent_type, ent_index)
                        for ent_name_pair in ent_name_result:
                            if ent_name_pair[1] not in localization_result['d1'].keys():
                                localization_result['d1'][ent_name_pair[1]] = 0
                            localization_result['d1'][ent_name_pair[1]] += ent_name_pair[0] * ent_fault_prob[ent_index]
                        for o11y_name_pair in o11y_name_result:
                            if o11y_name_pair[1] not in localization_result['d2'].keys():
                                localization_result['d2'][o11y_name_pair[1]] = 0
                            localization_result['d2'][o11y_name_pair[1]] += o11y_name_pair[0] * ent_fault_prob[ent_index]
            suspect_index = 0
            suspect_list = sorted(suspect_list, key=lambda item: item[2], reverse=True)
            while len(localization_result['d1']) < 5 or len(localization_result['d2']) < 5:
                ent_name_result, o11y_name_result = explainer.train_explainer(explain_batch_data, torch.sigmoid(out[suspect_list[suspect_index][1]])[suspect_list[suspect_index][0]], suspect_list[suspect_index][1], suspect_list[suspect_index][0])
                for ent_name_pair in ent_name_result:
                    if ent_name_pair[1] not in localization_result['d1'].keys():
                        localization_result['d1'][ent_name_pair[1]] = 0
                    localization_result['d1'][ent_name_pair[1]] += ent_name_pair[0] * suspect_list[suspect_index][2]
                for o11y_name_pair in o11y_name_result:
                    if o11y_name_pair[1] not in localization_result['d2'].keys():
                        localization_result['d2'][o11y_name_pair[1]] = 0
                    localization_result['d2'][o11y_name_pair[1]] += o11y_name_pair[0] * suspect_list[suspect_index][2]
                suspect_index += 1

            localization_result['d1'] = sorted(localization_result['d1'].items(), key=lambda item: item[1], reverse=True)
            localization_result['d2'] = sorted(localization_result['d2'].items(), key=lambda item: item[1], reverse=True)

            self.logger.info('----------')
            self.logger.info(f'sample {batch_id}/{len(self.mask_learning_data_loader.data_loader)} | d1: {exact_root_cause["d1"]}; level: {exact_root_cause["level"]}; fault_type: {exact_root_cause["fault_type"]}')
            self.logger.info(f'sample {batch_id}/{len(self.mask_learning_data_loader.data_loader)} | predict d1: {localization_result["d1"][0:min(5, len(localization_result["d1"]))]}')
            self.logger.info(f'sample {batch_id}/{len(self.mask_learning_data_loader.data_loader)} | predict d2: {localization_result["d2"][0:min(5, len(localization_result["d2"]))]}')
            self.logger.info('----------')

            d1_hit, d2_hit = dict(), dict()
            k_list = [1, 3, 5]
            for k in k_list:
                d1_hit[k], d2_hit[k] = False, False

            for i in range(len(localization_result['d1'])):
                if exact_root_cause['d1'] in localization_result['d1'][i][0]:
                    for k in k_list:
                        if i < k:
                            d1_hit[k] = True
            for i in range(len(localization_result['d2'])):
                hit = False
                for exact_o11y_name in exact_root_cause['d2']['exact']:
                    if exact_root_cause['d1'] in localization_result['d2'][i][0] and exact_o11y_name in localization_result['d2'][i][0]:
                        hit = True
                for fuzzy_o11y_name in exact_root_cause['d2']['fuzzy']:
                    if fuzzy_o11y_name in localization_result['d2'][i][0]:
                        hit = True
                if hit:
                    for k in k_list:
                        if i < k:
                            d2_hit[k] = True

            for k in k_list:
                if d1_hit[k]:
                    result[exact_root_cause['level']]['d1'][f'AC@{k}_num'] += 1
                if d2_hit[k]:
                    result[exact_root_cause['level']]['d2'][f'AC@{k}_num'] += 1

        self.logger.info('----------')
        self.logger.info(f'evaluation')
        for ent_type in result.keys():
            self.logger.info(f'{ent_type.ljust(8)} d1 | AC@1: {result[ent_type]["d1"]["AC@1_num"] / result[ent_type]["total"]:.6f}; AC@3: {result[ent_type]["d1"]["AC@3_num"] / result[ent_type]["total"]:.6f}; AC@5: {result[ent_type]["d1"]["AC@5_num"] / result[ent_type]["total"]:.6f}')
            self.logger.info(f'{ent_type.ljust(8)} d2 | AC@1: {result[ent_type]["d2"]["AC@1_num"] / result[ent_type]["total"]:.6f}; AC@3: {result[ent_type]["d2"]["AC@3_num"] / result[ent_type]["total"]:.6f}; AC@5: {result[ent_type]["d2"]["AC@5_num"] / result[ent_type]["total"]:.6f}')
        self.logger.info('----------')
