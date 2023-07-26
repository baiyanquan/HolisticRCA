from shared_util.file_handler import FileHandler
from data_filter.Eadro_TT_and_SN.base.base_generator import BaseGenerator
from data_filter.Eadro_TT_and_SN.service.metric_generator import MetricGenerator
from data_filter.Eadro_TT_and_SN.service.trace_generator import TraceGenerator
from data_filter.Eadro_TT_and_SN.service.log_generator import LogGenerator
from data_filter.Eadro_TT_and_SN.service.ent_edge_index_generator import EntEdgeIndexGenerator
from data_filter.Eadro_TT_and_SN.base.base_sn_class import BaseSNClass
from data_filter.Eadro_TT_and_SN.base.base_tt_class import BaseTTClass
import numpy as np
import pickle


class DatasetGenerator(BaseGenerator):
    def __init__(self, base):
        super().__init__(base)
        self.meta_data = {
            "modal_types": ['metric', 'trace', 'log'],
            "ent_types": ['service'],
            'ent_names': self.base.all_entity_list,
            'ent_type_index': {
                "service": (0, len(self.base.all_entity_list)),
            },
            'ent_features': {
                "metric": [],
                "trace": [],
                "log": []
            },
            'max_ent_feature_num': {
                "node": {
                    "metric": 0,
                    "trace": 0,
                    "log": 0
                },
                "service": {
                    "metric": 0,
                    "trace": 0,
                    "log": 0
                },
                "pod": {
                    "metric": 0,
                    "trace": 0,
                    "log": 0
                },
            },
            'o11y_names': {
                "metric": [],
                "trace": [],
                "log": []
            },
            'o11y_length': {
                "metric": 0,
                "trace": 0,
                "log": 0
            },
            'ent_fault_type_index': {
                "service": (0, 3),
            },
            'ent_fault_type_weight': {
                "service": [1, 1, 1],
            },
            "fault_type_list": self.base.fault_type_list,
            "fault_type_related_o11y_names": self.base.fault_type_related_o11y_names
        }
        self.data = dict()

    def get_base_data(self, window_size, modal_type_list):
        modal_dict = dict()
        self.meta_data['modal_types'] = modal_type_list
        if 'metric' in modal_type_list:
            temp = MetricGenerator(self.base).get_metric(window_size)
            self.meta_data['ent_features']['metric'] = temp['entity_features']
            self.meta_data['o11y_names']['metric'] = temp['metric_names']
            self.meta_data['o11y_length']['metric'] = len(temp['metric_names'])
            modal_dict['metric'] = temp['metric_data']
        if 'trace' in modal_type_list:
            temp = TraceGenerator(self.base).get_trace(window_size)
            self.meta_data['ent_features']['trace'] = temp['entity_features']
            self.meta_data['o11y_names']['trace'] = temp['trace_names']
            self.meta_data['o11y_length']['trace'] = len(temp['trace_names'])
            modal_dict['trace'] = temp['trace_data']
        if 'log' in modal_type_list:
            temp = LogGenerator(self.base).get_log(window_size)
            self.meta_data['ent_features']['log'] = temp['entity_features']
            self.meta_data['o11y_names']['log'] = temp['log_names']
            self.meta_data['o11y_length']['log'] = len(temp['log_names'])
            modal_dict['log'] = temp['log_data']

        for ent_type in self.meta_data['ent_types']:
            start_end_index_pair = self.meta_data['ent_type_index'][ent_type]
            for i in range(start_end_index_pair[0], start_end_index_pair[1]):
                for modal_type in modal_type_list:
                    ent_feature_num = self.meta_data['ent_features'][modal_type][i][1][1] - self.meta_data['ent_features'][modal_type][i][1][0]
                    self.meta_data['max_ent_feature_num'][ent_type][modal_type] = max(ent_feature_num, self.meta_data['max_ent_feature_num'][ent_type][modal_type])

        ent_edge_index = EntEdgeIndexGenerator(self.base).get_edge_index(window_size)
        ground_truth_dict = self.ground_truth_dao.get_time_label_interval(window_size)
        y = dict()
        for data_type in ['train', 'valid', 'test']:
            y[data_type] = []
            for ground_truth in ground_truth_dict[data_type]:
                y[data_type].append(np.zeros((len(self.base.all_entity_list), len(self.base.fault_type_list))))
                if ground_truth[1]:
                    y[data_type][-1][ground_truth[4]][ground_truth[5] - 1] = 1
        zero_count, one_count = [0, 0, 0], [0, 0, 0]
        for train_label in y['train']:
            for i in range(0, 3):
                label_num = np.sum(train_label[:, i])
                one_count[i] += label_num
                zero_count[i] += train_label.shape[0] - label_num
        self.meta_data['ent_fault_type_weight']['service'] = [zero_count[i] / one_count[i] for i in range(0, 3)]

        for data_type in ['train', 'valid', 'test']:
            for modal_type in modal_type_list:
                self.data[f'x_{modal_type}_{data_type}'] = np.array(modal_dict[modal_type][data_type])
            self.data[f'ent_edge_index_{data_type}'] = np.array(ent_edge_index[data_type])
            self.data[f'y_{data_type}'] = np.array(y[data_type])

    def generate_rca_multimodal_dataset(self):
        modal_types = [
            ['metric', 'trace', 'log'],
        ]
        folder = FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/merge')
        for window_size in self.base.window_size_list:
            for modal_type_list in modal_types:
                self.get_base_data(window_size, modal_type_list)
                modal_name = modal_type_list[0]
                for i in range(1, len(modal_type_list)):
                    modal_name += f'_{modal_type_list[i]}'
                with open(f'{folder}/window_size_{window_size}.pkl', 'wb') as f:
                    pickle.dump({
                        'data': self.data,
                        'meta_data': self.meta_data
                    }, f)


if __name__ == '__main__':
    dataset_generator = DatasetGenerator(BaseSNClass())
    dataset_generator.generate_rca_multimodal_dataset()
