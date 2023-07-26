import sys

sys.path.append('/workspace/project/working/2023/HolisticRCA/code')

import numpy as np
import json
import pickle
from tqdm import tqdm

from shared_util.file_handler import FileHandler
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator
from data_filter.CCF_AIOps_challenge_2022.service.time_interval_label_generator import TimeIntervalLabelGenerator


class TraceGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def load_raw_trace(self):
        return self.raw_trace_dao.load_trace_csv()

    def calculate_trace_statistic(self):
        statistic_dict = dict()
        data_dict = dict()

        raw_data = self.load_raw_trace()
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['span_features']
                    for feature_name in feature_df.keys():
                        if feature_name == 'timestamp':
                            continue
                        exact_feature_name = TraceGenerator.extract_entity_feature_name(feature_name)
                        if exact_feature_name not in data_dict.keys():
                            statistic_dict[exact_feature_name] = 0
                            data_dict[exact_feature_name] = []
                        data_dict[exact_feature_name].extend(raw_data[dataset_type][date][cloud_bed]['span_features'][feature_name].tolist())

        for feature_name in statistic_dict.keys():
            trace_data = data_dict[feature_name]
            median = np.nanmedian(trace_data)
            percentile_1 = np.nanpercentile(trace_data, 1)
            percentile_99 = np.nanpercentile(trace_data, 99)
            q1 = np.nanpercentile(trace_data, 25)
            q3 = np.nanpercentile(trace_data, 75)
            mean = np.nanmean(trace_data)
            std = np.nanstd(trace_data)
            valid_ratio = (np.count_nonzero(~np.isnan(trace_data))) / len(list(trace_data))

            statistic_dict[feature_name] = {
                'mean': mean,
                'std': std,
                'percentile_1': percentile_1,
                'q1': q1,
                'median': median,
                'q3': q3,
                'percentile_99': percentile_99,
                'valid_ratio': valid_ratio
            }

        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/analysis/trace')
        with open(f'{folder}/statistic.json', 'w') as f:
            json.dump(statistic_dict, f, indent=2)

    def z_score_trace_data(self):
        raw_data = self.load_raw_trace()

        file_dict = self.config.data_dict['file']
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/trace/statistic.json', 'r') as f:
            statistic_dict = json.load(f)

        for dataset_type, dataset_detail_dict in file_dict.items():
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    feature_df = raw_data[dataset_type][date][cloud_bed]['span_features']
                    for feature_name in feature_df.keys():
                        if feature_name == 'timestamp':
                            continue
                        exact_feature_name = TraceGenerator.extract_entity_feature_name(feature_name)
                        raw_trace_feature_data = raw_data[dataset_type][date][cloud_bed]['span_features'][feature_name]

                        iqr = statistic_dict[exact_feature_name]['q3'] - statistic_dict[exact_feature_name]['q1']
                        median = statistic_dict[exact_feature_name]['median']

                        if iqr != 0:
                            update_trace_feature_data = (raw_trace_feature_data - median) / iqr
                            for i in range(len(update_trace_feature_data)):
                                raw_data[dataset_type][date][cloud_bed]['span_features'].loc[i, feature_name] = update_trace_feature_data[i]

        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/trace')
        with open(f'{folder}/all_trace_features.pkl', 'wb') as f:
            pickle.dump(raw_data, f)

    def generate_trace_data(self):
        all_trace_dict = dict()
        with open(f'{self.config.param_dict["temp_data_storage"]}/dataset/trace/all_trace_features.pkl', 'rb') as f:
            temp_dict = pickle.load(f)
            for date_cloud_bed_data in temp_dict.values():
                for date, cloud_bed_data in date_cloud_bed_data.items():
                    all_trace_dict[date] = cloud_bed_data

        def get_time_interval_trace_data(st, et, data_frame):
            return np.array(data_frame.query(f'{st} <= timestamp < {et}').iloc[:, data_frame.columns != "timestamp"].values)

        window_size_bar = tqdm(self.window_size_list)
        for window_size in window_size_bar:
            trace_dict = dict()

            entity_features = []
            trace_name_list = []
            record_features = True

            for node in self.config.data_dict['setting']['metric']['node_order']:
                entity_features.append((node, (0, 0)))

            for service in self.config.data_dict['setting']['metric']['service_order']:
                entity_features.append((service, (0, 0)))
            for data_type in ['train_valid', 'test']:
                time_interval_label_list = TimeIntervalLabelGenerator().get_time_interval_label(window_size)['time_interval'][data_type]
                trace_dict[data_type] = []
                for time_interval in time_interval_label_list:
                    feature_index = 0
                    trace_data = all_trace_dict[time_interval[0]][time_interval[1]]['span_features']

                    temp = get_time_interval_trace_data(time_interval[2], time_interval[3], trace_data)
                    temp_name_list = list(trace_data.columns[trace_data.columns != "timestamp"])

                    data = []
                    for service in self.config.data_dict['setting']['metric']['service_order']:
                        pod_list = TraceGenerator.rename_service2pod(service)
                        for pod in pod_list:
                            pod_related_trace_name_list = []
                            for position_type in ['upstream', 'current', 'downstream']:
                                for feature_type in ['<intensity>', '<duration>']:
                                    feature_name = f'{feature_type}; cmdb_id: {pod}; type: {position_type}'
                                    if feature_name not in temp_name_list:
                                        continue
                                    data.append(temp[:, temp_name_list.index(feature_name)])
                                    pod_related_trace_name_list.append(feature_name)
                            if record_features:
                                entity_features.append((pod, (feature_index, feature_index + len(pod_related_trace_name_list))))
                                feature_index += len(pod_related_trace_name_list)
                                trace_name_list.extend(pod_related_trace_name_list)

                    trace_dict[data_type].append(np.array(data).transpose())
                    record_features = False

            folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/trace')
            with open(f'{folder}/trace_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump({
                    'trace_data': trace_dict,
                    'entity_features': entity_features,
                    'trace_names': trace_name_list
                }, f)
            window_size_bar.set_description("Trace dataset generating".format(window_size))

    @staticmethod
    def extract_entity_feature_name(feature_name):
        cmdb_id = feature_name.split(';')[1].replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '')
        return f'{feature_name.split(";")[0]};{cmdb_id};{feature_name.split(";")[2]}'

    def get_trace(self, window_size) -> dict:
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/trace')
        with open(f'{folder}/trace_window_size_{window_size}.pkl', 'rb') as f:
            trace = pickle.load(f)
            return trace


if __name__ == '__main__':
    trace_generator = TraceGenerator()
    trace_generator.calculate_trace_statistic()
    trace_generator.z_score_trace_data()
    trace_generator.generate_trace_data()
    test = trace_generator.get_trace(9)
