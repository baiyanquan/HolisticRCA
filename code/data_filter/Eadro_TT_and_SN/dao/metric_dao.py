import pandas as pd
import numpy as np

from data_filter.Eadro_TT_and_SN.base.base_class import BaseClass
from data_filter.Eadro_TT_and_SN.base.base_sn_class import BaseSNClass
from data_filter.Eadro_TT_and_SN.base.base_tt_class import BaseTTClass
from data_filter.Eadro_TT_and_SN.util.time_interval import TimeInterval
from shared_util.file_handler import FileHandler


class MetricDao:
    def __init__(self, base: BaseClass):
        self.base = base

    def extract_metric_features(self):
        dataset_type_list = ['faulty', 'normal', 'z-score']
        for dataset_type in dataset_type_list:
            data_base_path = self.base.config.data_dict[self.base.dataset_name]['file'][dataset_type]['base_folder']
            for t in self.base.config.data_dict[self.base.dataset_name]['time'][dataset_type]:
                timestamp_list = TimeInterval.generate_timestamp_list(f'{data_base_path}/{self.base.dataset_name}.fault-{t}.json', self.base.sample_granularity)
                for entity in self.base.all_entity_list:
                    result_dict = {'timestamp': timestamp_list}
                    result_path = FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/raw_data/{t}/raw_metric') + f'/{entity}_metrics.csv'
                    raw_metric_df = pd.read_csv(f'{data_base_path}/{self.base.dataset_name}.{t}/metrics/{entity}.csv')

                    for metric_name in self.base.metric_name_list:
                        result_dict[metric_name] = []
                        for i in range(len(timestamp_list)):
                            result_dict[metric_name].append(np.mean(raw_metric_df.query(f'{timestamp_list[i]} <= timestamp < {timestamp_list[i] + timestamp_list[1] - timestamp_list[0]}').iloc[:, raw_metric_df.columns == metric_name].values))
                    result_df = pd.DataFrame(result_dict)
                    result_df.to_csv(result_path, index=False)

    def load_metric_csv(self):
        result_dict = dict()
        dataset_type_list = ['faulty', 'normal', 'z-score']
        for dataset_type in dataset_type_list:
            for t in self.base.config.data_dict[self.base.dataset_name]['time'][dataset_type]:
                result_dict[t] = {'service': dict()}
                for entity in self.base.all_entity_list:
                    result_dict[t]['service'][entity] = pd.read_csv(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/raw_data/{t}/raw_metric/{entity}_metrics.csv')
        return result_dict


if __name__ == '__main__':
    metric_dao = MetricDao(BaseSNClass())
    metric_dao.extract_metric_features()
