import sys

sys.path.append('/workspace/project/working/2023/HolisticRCA/code')

from data_filter.ICASSP_AIOps_challenge_2022.base.base_generator import BaseGenerator
import pickle
import numpy as np

from shared_util.file_handler import FileHandler


class MetricGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def load_raw_metric(self):
        return self.metric_dao.load_metric_csv()

    def generate_metric_data(self):
        all_metric_dict = dict()
        seq_len = 30
        raw_data = self.load_raw_metric()
        all_feature_list = []
        for feature in self.all_feature_list:
            if ':' not in feature:
                all_feature_list.append(f'{feature}:1')
            else:
                all_feature_list.append(feature)
        for dataset_type, raw_metric_dict in raw_data.items():
            all_metric_dict[dataset_type] = []
            for sample_index, raw_metric_df in raw_metric_dict.items():
                if len(raw_metric_df) >= seq_len:
                    all_metric_dict[dataset_type].append(raw_metric_df.loc[:seq_len - 1, all_feature_list].values.tolist())
                else:
                    temp_metric_df = raw_metric_df.loc[:, all_feature_list]
                    add = temp_metric_df.mean().to_frame().T
                    for t in range(seq_len - len(temp_metric_df)):
                        temp_metric_df = temp_metric_df.append(add)
                    all_metric_dict[dataset_type].append(temp_metric_df.values.tolist())
        test_index_list = []
        all_feature_list = []
        for feature in self.all_feature_list:
            if ':' not in feature:
                all_feature_list.append(f'{feature}:1')
            else:
                all_feature_list.append(f'{feature.split(":")[0]}:23')
        for sample_index, raw_metric_df in raw_data['test'].items():
            start_index = len(all_metric_dict['test'])
            test_index_list.append((start_index, start_index + len(raw_metric_df)))
            for i in range(len(raw_metric_df)):
                point_sample = np.tile(raw_metric_df.loc[i, all_feature_list].values.tolist(), (30, 1)).tolist()
                all_metric_dict['test'].append(point_sample)

        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/merge')
        with open(f'{folder}/test_index_list.pkl', 'wb') as f:
            pickle.dump(test_index_list, f)

        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/metric')
        with open(f'{folder}/all_metric.pkl', 'wb') as f:
            pickle.dump({
                'metric_data': all_metric_dict,
                'entity_features': self.entity_features,
                'metric_names': self.all_feature_list
            }, f)

    def get_metric(self):
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/metric')
        with open(f'{folder}/all_metric.pkl', 'rb') as f:
            metric = pickle.load(f)
            return metric


if __name__ == '__main__':
    metric_generator = MetricGenerator()
    metric_generator.generate_metric_data()
