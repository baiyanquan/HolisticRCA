import numpy as np
import json
import pickle
from tqdm import tqdm

from shared_util.file_handler import FileHandler
from shared_util.time_handler import TimeHandler
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator


class TimeIntervalLabelGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def get_ground_truth(self, ground_truth_dict, index):
        return {
            'timestamp': ground_truth_dict['timestamp'][index],
            'level': ground_truth_dict['level'][index],
            'cmdb_id': ground_truth_dict['cmdb_id'][index],
            'fault_type': self.fault_type_list.index(ground_truth_dict['fault_type'][index]) + 1
        }

    @staticmethod
    def get_date_timestamp_list(date_str: str) -> list:
        date_start_timestamp = TimeHandler.datetime_to_timestamp(date_str + ' 00:00:00')
        return list(range(date_start_timestamp, date_start_timestamp + 24 * 60 * 60, 60))

    def get_ground_truth_label(self, ground_truth):
        label = np.zeros(len(self.all_entity_list))
        label[self.all_entity_list.index(ground_truth['cmdb_id'])] = ground_truth['fault_type']
        if ground_truth['level'] == 'service':
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}-0')] = ground_truth['fault_type']
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}-1')] = ground_truth['fault_type']
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}-2')] = ground_truth['fault_type']
            label[self.all_entity_list.index(f'{ground_truth["cmdb_id"]}2-0')] = ground_truth['fault_type']
        return label

    def slice_ground_truth_timestamp(self, date, cloud_bed, ground_truth_timestamp, window_size, sliding_ratio):
        interval_list = []
        start_timestamp = TimeHandler.datetime_to_timestamp(date + ' 00:00:00')
        c_ts = ground_truth_timestamp - (ground_truth_timestamp - start_timestamp) % 60
        s_ts = c_ts - int(window_size * sliding_ratio) * 60
        e_ts = s_ts + window_size * 60
        interval_list.append((date, cloud_bed, s_ts, e_ts))
        return interval_list

    def generate_time_interval_label(self):
        window_size_bar = tqdm(self.window_size_list)
        for window_size in window_size_bar:
            faulty_time_interval, faulty_y = {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}
            faulty_entity_type, faulty_template, faulty_cmdb_id, faulty_root_cause_type = {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}, {'train_valid': [], 'test': []}

            for dataset_type in ['train_valid', 'test']:
                train_ground_truth_timestamp_dict = dict()
                for date, cloud_dict in self.ground_truth_dao.get_ground_truth(dataset_type).items():
                    for cloud_bed in cloud_dict.keys():
                        train_ground_truth_timestamp_dict[f'{date}/{cloud_bed}'] = []

                        for i in range(len(cloud_dict[cloud_bed]['timestamp'])):
                            ground_truth = self.get_ground_truth(cloud_dict[cloud_bed], i)
                            train_ground_truth_timestamp_dict[f'{date}/{cloud_bed}'].append(ground_truth['timestamp'])
                            temp_time_interval_list = self.slice_ground_truth_timestamp(date, cloud_bed, ground_truth['timestamp'], window_size, 0.5)
                            faulty_time_interval[dataset_type].extend(temp_time_interval_list)
                            faulty_y[dataset_type].extend([self.get_ground_truth_label(ground_truth) for i in range(len(temp_time_interval_list))])
                            faulty_entity_type[dataset_type].append(ground_truth['level'])
                            faulty_template[dataset_type].append(ground_truth['cmdb_id'].replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '').replace('-3', '').replace('-4', '').replace('-5', '').replace('-6', ''))
                            faulty_cmdb_id[dataset_type].append(ground_truth['cmdb_id'])
                            faulty_root_cause_type[dataset_type].append(self.fault_type_list[ground_truth['fault_type'] - 1])

            folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/time_interval_and_label')
            with open(f'{folder}/time_interval_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump({
                    'time_interval': {
                        'train_valid': faulty_time_interval['train_valid'],
                        'test': faulty_time_interval['test']
                    },
                    'y': {
                        'train_valid': faulty_y['train_valid'],
                        'test': faulty_y['test']
                    },
                    'entity_type': {
                        'train_valid': faulty_entity_type['train_valid'],
                        'test': faulty_entity_type['test']
                    },
                    'template': {
                        'train_valid': faulty_template['train_valid'],
                        'test': faulty_template['test']
                    },
                    'cmdb_id': {
                        'train_valid': faulty_cmdb_id['train_valid'],
                        'test': faulty_cmdb_id['test']
                    },
                    'root_cause_type': {
                        'train_valid': faulty_root_cause_type['train_valid'],
                        'test': faulty_root_cause_type['test']
                    }
                }, f)
            window_size_bar.set_description("Time interval and label generating".format(window_size))

    def get_time_interval_label(self, window_size) -> dict:
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/time_interval_and_label')
        with open(f'{folder}/time_interval_window_size_{window_size}.pkl', 'rb') as f:
            time_interval_label = pickle.load(f)
            return time_interval_label


if __name__ == '__main__':
    time_interval_label_generator = TimeIntervalLabelGenerator()
    time_interval_label_generator.generate_time_interval_label()
    time_interval_label_generator.get_time_interval_label(9)
