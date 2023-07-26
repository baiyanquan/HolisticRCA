from shared_util.file_handler import FileHandler
from data_filter.Eadro_TT_and_SN.base.base_generator import BaseGenerator
from data_filter.Eadro_TT_and_SN.base.base_sn_class import BaseSNClass
from data_filter.Eadro_TT_and_SN.base.base_tt_class import BaseTTClass
import numpy as np
import pickle


class MetricGenerator(BaseGenerator):
    def __init__(self, base):
        super().__init__(base)

    def get_metric_feature_info(self) -> (list, list):
        feature_index, entity_features, metric_name_list = 0, [], []
        for i in range(len(self.base.all_entity_list)):
            for metric_name in self.base.metric_name_list:
                metric_name_list.append(f'{self.base.all_entity_list[i]}/{metric_name}')
            entity_features.append((i, (feature_index * len(self.base.metric_name_list), (feature_index + 1) * len(self.base.metric_name_list))))
            feature_index += 1
        return entity_features, metric_name_list

    def generate_metric_data(self):
        for window_size in self.base.window_size_list:
            ground_truth_dict = self.ground_truth_dao.get_time_label_interval(window_size)
            raw_metric_dict = self.raw_metric_dao.load_metric_csv()

            metric_dict = dict()
            for dataset_type in ['train', 'valid', 'test', 'z-score']:
                metric_dict[dataset_type] = []
                for ground_truth in ground_truth_dict[dataset_type]:
                    temp = []
                    for entity in self.base.all_entity_list:
                        timestamp_list = raw_metric_dict[ground_truth[0]]['service'][entity].loc[:, 'timestamp'].tolist()
                        start_index = int((ground_truth[2] - timestamp_list[0]) / (timestamp_list[1] - timestamp_list[0]))
                        temp.extend(raw_metric_dict[ground_truth[0]]['service'][entity].query(f'{timestamp_list[start_index]} <= timestamp < {timestamp_list[start_index + window_size]}').loc[:, self.base.metric_name_list].values.T.tolist())
                    temp = np.array(temp).T
                    metric_dict[dataset_type].append(temp)
                metric_dict[dataset_type] = np.array(metric_dict[dataset_type])
            metric_dict = MetricGenerator.z_score_data(metric_dict)

            entity_features, metric_name_list = self.get_metric_feature_info()
            with open(FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/metric') + f'/metric_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump({
                    'metric_data': metric_dict,
                    'entity_features': entity_features,
                    'metric_names': metric_name_list
                }, f)

    def get_metric(self, window_size) -> np:
        with open(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/metric/metric_window_size_{window_size}.pkl', 'rb') as f:
            metric = pickle.load(f)
            return metric


if __name__ == '__main__':
    metric_generator = MetricGenerator(BaseSNClass())
    metric_generator.generate_metric_data()
