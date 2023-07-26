from shared_util.file_handler import FileHandler
from data_filter.Eadro_TT_and_SN.base.base_generator import BaseGenerator
from data_filter.Eadro_TT_and_SN.base.base_sn_class import BaseSNClass
from data_filter.Eadro_TT_and_SN.base.base_tt_class import BaseTTClass
import numpy as np
import pickle
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import os
import json


class LogGenerator(BaseGenerator):
    def __init__(self, base):
        super().__init__(base)

    @staticmethod
    def init_template_miner():
        drain_config = TemplateMinerConfig()
        drain_config.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'drain3.ini'))
        drain_config.profiling_enabled = True
        return TemplateMiner(config=drain_config)

    def extract_log_patterns(self):
        for window_size in self.base.window_size_list:
            ground_truth_dict = self.ground_truth_dao.get_time_label_interval(window_size)
            raw_log_dict = self.raw_log_dao.load_raw_log()

            log_template_miner = LogGenerator.init_template_miner()

            log_pattern_list = []
            for dataset_type in ['train', 'valid']:
                for ground_truth in ground_truth_dict[dataset_type]:
                    timestamp_list = raw_log_dict[ground_truth[0]]['timestamp']
                    start_index = int((ground_truth[2] - timestamp_list[0]) / (timestamp_list[1] - timestamp_list[0]))
                    for i in range(start_index, start_index + window_size):
                        for entity in self.base.all_entity_list:
                            for log in raw_log_dict[ground_truth[0]][entity][i]:
                                log_pattern = log_template_miner.add_log_message(log)['template_mined']
                                if log_pattern not in log_pattern_list:
                                    log_pattern_list.append(log_pattern)

            result_base_path = FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/analysis/log')
            with open(result_base_path + f'/log_patterns_window_size_{window_size}.json', 'w') as f:
                json.dump(log_pattern_list, f, indent=2)

    def generate_log_data(self):
        for window_size in self.base.window_size_list:
            result_dict = {
                'analysis': {
                    'idf': dict(),
                    'max_tf': dict(),
                    'max_tf_idf': dict()
                },
                'raw_tf_idf': {
                    'tf': dict(),
                    'tf_idf': dict()
                }
            }

            ground_truth_dict = self.ground_truth_dao.get_time_label_interval(window_size)
            raw_log_dict = self.raw_log_dao.load_raw_log()
            log_template_miner = LogGenerator.init_template_miner()
            with open(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/analysis/log/log_patterns_window_size_{window_size}.json') as f:
                log_pattern_list = json.load(f)

            temp_dict = dict()
            for log_pattern in log_pattern_list:
                temp_dict[log_pattern] = []

            for dataset_type in ['train', 'valid']:
                for ground_truth in ground_truth_dict[dataset_type]:
                    for log_pattern in log_pattern_list:
                        temp_dict[log_pattern].append(0)

                    timestamp_list = raw_log_dict[ground_truth[0]]['timestamp']
                    start_index = int((ground_truth[2] - timestamp_list[0]) / (timestamp_list[1] - timestamp_list[0]))
                    for i in range(start_index, start_index + window_size):
                        for j in range(len(self.base.all_entity_list)):
                            for k in range(len(raw_log_dict[ground_truth[0]][self.base.all_entity_list[j]][i])):
                                log = raw_log_dict[ground_truth[0]][self.base.all_entity_list[j]][i][k]
                                log_pattern = log_template_miner.add_log_message(log)['template_mined']
                                if log_pattern in log_pattern_list:
                                    temp_dict[log_pattern][-1] += 1

            total_list = [0 for _ in range(len(ground_truth_dict['train']) + len(ground_truth_dict['valid']))]
            for log_pattern in log_pattern_list:
                for i in range(len(total_list)):
                    total_list[i] += temp_dict[log_pattern][i]
            for log_pattern in log_pattern_list:
                for i in range(len(total_list)):
                    if total_list[i] == 0:
                        temp_dict[log_pattern][i] = 0
                    else:
                        temp_dict[log_pattern][i] = temp_dict[log_pattern][i] / total_list[i]

            for log_pattern in log_pattern_list:
                temp = np.array(temp_dict[log_pattern])
                idf = np.log(temp.shape[0] / (1 + np.count_nonzero(temp)))
                result_dict['analysis']['idf'][log_pattern] = idf
                result_dict['analysis']['max_tf'][log_pattern] = np.sort(temp)[-3:].mean()
                result_dict['analysis']['max_tf_idf'][log_pattern] = np.sort(temp)[-3:].mean() * idf
                result_dict['raw_tf_idf']['tf'][log_pattern] = temp.tolist()
                result_dict['raw_tf_idf']['tf_idf'][log_pattern] = (temp * idf).tolist()

            tf_idf_threshold = {
                'TT': 0.015,
                'SN': 0.025
            }
            selected_log_pattern_list = []
            for log_pattern in log_pattern_list:
                if result_dict['analysis']['max_tf_idf'][log_pattern] > tf_idf_threshold[self.base.dataset_name]:
                    selected_log_pattern_list.append(log_pattern)

            result_base_path = FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/analysis/log')
            with open(result_base_path + f'/tf_idf_info_window_size_{window_size}.json', 'w') as f:
                json.dump(result_dict, f, indent=2)

            with open(result_base_path + f'/selected_log_patterns_window_size_{window_size}.json', 'w') as f:
                json.dump(selected_log_pattern_list, f, indent=2)

            log_template_miner = LogGenerator.init_template_miner()
            log_dict = dict()
            for dataset_type in ['train', 'valid', 'test', 'z-score']:
                log_dict[dataset_type] = []
                for ground_truth in ground_truth_dict[dataset_type]:
                    log_dict[dataset_type].append(np.zeros(shape=(window_size, len(self.base.all_entity_list) * len(selected_log_pattern_list))))
                    timestamp_list = raw_log_dict[ground_truth[0]]['timestamp']
                    start_index = int((ground_truth[2] - timestamp_list[0]) / (timestamp_list[1] - timestamp_list[0]))
                    for i in range(start_index, start_index + window_size):
                        for j in range(len(self.base.all_entity_list)):
                            for k in range(len(raw_log_dict[ground_truth[0]][self.base.all_entity_list[j]][i])):
                                log = raw_log_dict[ground_truth[0]][self.base.all_entity_list[j]][i][k]
                                log_pattern = log_template_miner.add_log_message(log)['template_mined']
                                if log_pattern in selected_log_pattern_list:
                                    index = selected_log_pattern_list.index(log_pattern)
                                    log_dict[dataset_type][-1][i - start_index][j * len(selected_log_pattern_list) + index] += 1
            log_dict = LogGenerator.z_score_data(log_dict)
            feature_index, entity_features, log_name_list = 0, [], []
            for i in range(len(self.base.all_entity_list)):
                for k in range(len(selected_log_pattern_list)):
                    log_name_list.append(f'{self.base.all_entity_list[i]}/<log pattern {k}>')
                entity_features.append((i, (feature_index * len(selected_log_pattern_list), (feature_index + 1) * len(selected_log_pattern_list))))
                feature_index += 1

            with open(FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/log') + f'/log_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump({
                    'log_data': log_dict,
                    'entity_features': entity_features,
                    'log_names': log_name_list
                }, f)

    def get_log(self, window_size) -> np:
        with open(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/log/log_window_size_{window_size}.pkl', 'rb') as f:
            log = pickle.load(f)
            return log


if __name__ == '__main__':
    log_generator = LogGenerator(BaseSNClass())
    log_generator.extract_log_patterns()
    log_generator.generate_log_data()
