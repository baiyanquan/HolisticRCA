from data_filter.Eadro_TT_and_SN.base.base_class import BaseClass
from data_filter.Eadro_TT_and_SN.util.time_interval import TimeInterval
from data_filter.Eadro_TT_and_SN.base.base_sn_class import BaseSNClass
from data_filter.Eadro_TT_and_SN.base.base_tt_class import BaseTTClass
from shared_util.file_handler import FileHandler
from shared_util.time_handler import TimeHandler
import json
import bisect
import pickle


class LogDao:
    def __init__(self, base: BaseClass):
        self.base = base

    def find_log_timestamp(self, log):
        if self.base.dataset_name == 'SN':
            temp = log.split('] ')
            return TimeHandler.datetime_to_timestamp(temp[0][1:].replace('Apr', '04').split('.')[0]), temp[1].split(' ', 2)[2]
        if self.base.dataset_name == 'TT':
            return TimeHandler.datetime_to_timestamp(log[:19]), log[97:]

    def extract_log_features(self):
        dataset_type_list = ['faulty', 'normal', 'z-score']
        for dataset_type in dataset_type_list:
            data_base_path = self.base.config.data_dict[self.base.dataset_name]['file'][dataset_type]['base_folder']
            for t in self.base.config.data_dict[self.base.dataset_name]['time'][dataset_type]:
                result_path = FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/raw_data/{t}/raw_log') + f'/raw_log.pkl'
                timestamp_list = TimeInterval.generate_timestamp_list(f'{data_base_path}/{self.base.dataset_name}.fault-{t}.json', self.base.sample_granularity)
                result_dict = {'timestamp': timestamp_list}
                for entity in self.base.all_entity_list:
                    result_dict[entity] = [[] for _ in timestamp_list]
                with open(f'{data_base_path}/{self.base.dataset_name}.{t}/logs.json') as f:
                    log_data = json.load(f)
                for entity, log_list in log_data.items():
                    for log in log_list:
                        log_timestamp, log_content = self.find_log_timestamp(log)
                        bucket_index = int((log_timestamp - timestamp_list[0]) / (timestamp_list[1] - timestamp_list[0]))
                        result_dict[entity][bucket_index].append(log_content)
                with open(result_path, 'wb') as f:
                    pickle.dump(result_dict, f)

    def load_raw_log(self):
        result_dict = dict()
        dataset_type_list = ['faulty', 'normal', 'z-score']
        for dataset_type in dataset_type_list:
            for t in self.base.config.data_dict[self.base.dataset_name]['time'][dataset_type]:
                with open(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/raw_data/{t}/raw_log/raw_log.pkl', 'rb') as f:
                    result_dict[t] = pickle.load(f)
        return result_dict


if __name__ == '__main__':
    log_dao = LogDao(BaseSNClass())
    log_dao.extract_log_features()
    log_dao.load_raw_log()
