from data_filter.Eadro_TT_and_SN.base.base_class import BaseClass
from data_filter.Eadro_TT_and_SN.base.base_sn_class import BaseSNClass
from data_filter.Eadro_TT_and_SN.base.base_tt_class import BaseTTClass
import json
from sklearn.model_selection import train_test_split
from shared_util.file_handler import FileHandler


class GroundTruthDao:
    def __init__(self, base: BaseClass):
        self.base = base

    def get_ground_truth(self, window_size):
        result_list = []
        data_base_path = self.base.config.data_dict[self.base.dataset_name]['file']['faulty']['base_folder']
        sliding_num = 5

        for t in self.base.config.data_dict[self.base.dataset_name]['time']['faulty']:
            with open(f'{data_base_path}/{self.base.dataset_name}.fault-{t}.json') as f:
                temp_list = json.load(f)['faults']
                for i in range(len(temp_list)):
                    temp_list[i]['start'] = int(temp_list[i]['start'])
                    service_name = temp_list[i]['name'].replace('-1', '').replace('_1', '').replace('socialnetwork-', '').replace('nginx-thrift', 'nginx-web-server').replace('dockercomposemanifests_', '')
                    if service_name not in self.base.valid_network_entity_list and (temp_list[i]['fault'] == 'network_delay' or temp_list[i]['fault'] == 'network_loss'):
                        continue
                    for j in range(sliding_num):
                        result_list.append((t, 1, temp_list[i]['start'] + self.base.sample_granularity * j, temp_list[i]['start'] + self.base.sample_granularity * (j + window_size),
                                            self.base.all_entity_list.index(service_name),
                                            self.base.fault_type_list.index(temp_list[i]['fault']) + 1))
        return result_list

    def get_normal_interval(self, window_size):
        result_list = []
        data_base_path = self.base.config.data_dict[self.base.dataset_name]['file']['normal']['base_folder']
        for t in self.base.config.data_dict[self.base.dataset_name]['time']['normal']:
            with open(f'{data_base_path}/{self.base.dataset_name}.fault-{t}.json') as f:
                temp = json.load(f)
                interval = self.base.sample_granularity * window_size
                result_list.extend([(t, 0, i, i + interval, None, None) for i in range(int(temp['start']), int(temp['end']) - 2 * interval, interval)])
        return result_list

    def get_z_score_interval(self, window_size):
        result_list = []
        data_base_path = self.base.config.data_dict[self.base.dataset_name]['file']['z-score']['base_folder']
        for t in self.base.config.data_dict[self.base.dataset_name]['time']['z-score']:
            with open(f'{data_base_path}/{self.base.dataset_name}.fault-{t}.json') as f:
                temp = json.load(f)
                interval = self.base.sample_granularity * window_size
                result_list.extend([(t, 0, i, i + interval, None, None) for i in range(int(temp['start']), int(temp['end']) - 2 * interval, interval)])
        return result_list

    def slice_dataset(self):
        for window_size in self.base.window_size_list:
            interval_list = self.get_ground_truth(window_size)

            train_valid, test = train_test_split(interval_list, test_size=0.2, random_state=409, shuffle=True)
            train, valid = train_test_split(train_valid, test_size=0.25, random_state=409, shuffle=True)

            folder = FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/time_interval_and_label')
            with open(f'{folder}/time_interval_window_size_{window_size}.json', 'w') as f:
                json.dump({
                    'train': train,
                    'valid': valid,
                    'test': test,
                    'z-score': self.get_z_score_interval(window_size)
                }, f, indent=2)

    def get_time_label_interval(self, window_size):
        with open(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/time_interval_and_label/time_interval_window_size_{window_size}.json') as f:
            time_label_dict = json.load(f)
        return time_label_dict


if __name__ == '__main__':
    ground_truth_dao = GroundTruthDao(BaseSNClass())
    ground_truth_dao.slice_dataset()
    ground_truth_dao.slice_dataset()
