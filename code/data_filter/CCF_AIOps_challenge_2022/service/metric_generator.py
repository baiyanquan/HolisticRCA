import sys

sys.path.append('/workspace/project/working/2023/HolisticRCA/code')

import numpy as np
import json
import pickle
from tqdm import tqdm

from shared_util.file_handler import FileHandler
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator
from data_filter.CCF_AIOps_challenge_2022.service.time_interval_label_generator import TimeIntervalLabelGenerator


class MetricGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def load_raw_metric(self):
        return self.raw_metric_dao.load_metric_csv()

    def calculate_common_statistic(self):
        common_statistic_dict = dict()
        data_dict = dict()

        raw_data = self.load_raw_metric()
        file_dict = self.config.data_dict['file']
        for dataset_type, dataset_detail_dict in file_dict.items():
            if dataset_type == 'test':
                continue
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    resource_type_list = ['node', 'container', 'service', 'istio']
                    for resource_type in resource_type_list:
                        if resource_type not in data_dict.keys():
                            common_statistic_dict[resource_type] = dict()
                            data_dict[resource_type] = dict()
                        entity_list = self.raw_metric_dao.get_entity_list(resource_type)
                        for entity in entity_list:
                            merged_entity = MetricGenerator.merge_entity(resource_type, entity)
                            if merged_entity not in data_dict[resource_type].keys():
                                common_statistic_dict[resource_type][merged_entity] = dict()
                                data_dict[resource_type][merged_entity] = dict()
                            metric_name_list = raw_data[dataset_type][date][cloud_bed][resource_type][entity].keys()
                            for metric_name in metric_name_list:
                                if metric_name == 'timestamp':
                                    continue
                                if metric_name not in data_dict[resource_type][merged_entity].keys():
                                    common_statistic_dict[resource_type][merged_entity][metric_name] = 0
                                    data_dict[resource_type][merged_entity][metric_name] = []
                                metric_data = raw_data[dataset_type][date][cloud_bed][resource_type][entity][metric_name]
                                metric_data = MetricGenerator.diff_metric(metric_name, metric_data.tolist())
                                data_dict[resource_type][merged_entity][metric_name].extend(metric_data)

        for resource_type, metric_dict in common_statistic_dict.items():
            for entity, entity_metric_dict in metric_dict.items():
                for metric_name in entity_metric_dict.keys():
                    metric_data = data_dict[resource_type][entity][metric_name]
                    common_statistic_dict[resource_type][entity][metric_name] = MetricGenerator.calculate_statistic(metric_data)

        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/analysis/metric')
        with open(f'{folder}/common_statistic.json', 'w') as f:
            json.dump(common_statistic_dict, f, indent=2)

    def z_score_metric_data(self):
        raw_data = self.load_raw_metric()

        file_dict = self.config.data_dict['file']
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/metric/common_statistic.json', 'r') as f:
            common_statistic_dict = json.load(f)
        for dataset_type, dataset_detail_dict in file_dict.items():
            for date in dataset_detail_dict['date']:
                for cloud_bed in dataset_detail_dict['cloud_bed']:
                    if date == '2022-03-24' and cloud_bed in ['cloudbed-1', 'cloudbed-2']:
                        continue
                    resource_type_list = ['node', 'container', 'service', 'istio']
                    for resource_type in resource_type_list:
                        entity_list = self.raw_metric_dao.get_entity_list(resource_type)
                        for entity in entity_list:
                            merged_entity = MetricGenerator.merge_entity(resource_type, entity)
                            metric_name_list = raw_data[dataset_type][date][cloud_bed][resource_type][entity].keys()
                            for metric_name in metric_name_list:
                                if metric_name == 'timestamp':
                                    continue
                                raw_metric_data = raw_data[dataset_type][date][cloud_bed][resource_type][entity][metric_name]
                                raw_metric_data = np.array(MetricGenerator.diff_metric(metric_name, raw_metric_data))
                                common_mean = common_statistic_dict[resource_type][merged_entity][metric_name]['mean']
                                common_std = common_statistic_dict[resource_type][merged_entity][metric_name]['std']
                                raw_metric_data = np.array(raw_metric_data)
                                if not (np.isnan(common_mean) or np.isnan(common_std) or common_std == 0):
                                    update_metric_data = (raw_metric_data - common_mean) / common_std
                                    update_metric_data[np.isnan(update_metric_data)] = 0
                                for i in range(update_metric_data.shape[0]):
                                    raw_data[dataset_type][date][cloud_bed][resource_type][entity].loc[i, metric_name] = update_metric_data[i]

        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/metric')
        with open(f'{folder}/all_metric.pkl', 'wb') as f:
            pickle.dump(raw_data, f)

    def generate_metric_data(self):
        all_metric_dict = dict()
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/metric')
        with open(f'{folder}/all_metric.pkl', 'rb') as f:
            temp_dict = pickle.load(f)
            for date_cloud_bed_data in temp_dict.values():
                for date, cloud_bed_data in date_cloud_bed_data.items():
                    all_metric_dict[date] = cloud_bed_data

        def get_time_interval_metric_data(st, et, data_frame):
            return np.array(data_frame.query(f'{st} <= timestamp < {et}').iloc[:, data_frame.columns != "timestamp"].values)

        window_size_bar = tqdm(self.window_size_list)
        for window_size in window_size_bar:
            metric_dict = dict()
            entity_features = []
            metric_name_list = []
            record_features = True
            for data_type in ['train_valid', 'test']:
                time_interval_label_list = TimeIntervalLabelGenerator().get_time_interval_label(window_size)['time_interval'][data_type]
                metric_dict[data_type] = []
                for time_interval in time_interval_label_list:
                    feature_index = 0
                    metric_data = all_metric_dict[time_interval[0]][time_interval[1]]
                    node_data_list = []
                    for node in self.config.data_dict['setting']['metric']['node_order']:
                        temp = get_time_interval_metric_data(time_interval[2], time_interval[3], metric_data['node'][node])
                        node_data_list.append(temp)
                        if record_features:
                            entity_features.append((node, (feature_index, feature_index + temp.shape[-1])))
                            feature_index += temp.shape[-1]
                            temp_name_list = list(metric_data['node'][node].columns[metric_data['node'][node].columns != "timestamp"])
                            metric_name_list.extend([f'{node}/{temp_name}' for temp_name in temp_name_list])
                    node_data = np.concatenate(node_data_list, axis=-1)

                    service_data_list = []
                    for service in self.config.data_dict['setting']['metric']['service_order']:
                        temp = get_time_interval_metric_data(time_interval[2], time_interval[3], metric_data['service'][service])
                        service_data_list.append(temp)
                        if record_features:
                            entity_features.append((service, (feature_index, feature_index + temp.shape[-1])))
                            feature_index += temp.shape[-1]
                            temp_name_list = list(metric_data['service'][service].columns[metric_data['service'][service].columns != "timestamp"])
                            metric_name_list.extend([f'{service}/{temp_name}' for temp_name in temp_name_list])
                    service_data = np.concatenate(service_data_list, axis=-1)

                    pod_data_list = []
                    for pod in self.config.data_dict['setting']['metric']['pod_order']:
                        container_data = get_time_interval_metric_data(time_interval[2], time_interval[3], metric_data['container'][pod])
                        temp = container_data
                        pod_data_list.append(temp)
                        if record_features:
                            entity_features.append((pod, (feature_index, feature_index + temp.shape[-1])))
                            feature_index += temp.shape[-1]
                            temp_name_list = list(metric_data['container'][pod].columns[metric_data['container'][pod].columns != "timestamp"])
                            metric_name_list.extend([f'{pod}/{temp_name}' for temp_name in temp_name_list])
                    pod_data = np.concatenate(pod_data_list, axis=-1)

                    metric_dataset_item = np.concatenate((node_data, service_data, pod_data), axis=-1)
                    metric_dict[data_type].append(metric_dataset_item)
                    record_features = False

                folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/metric')
                with open(f'{folder}/metric_window_size_{window_size}.pkl', 'wb') as f:
                    pickle.dump({
                        'metric_data': metric_dict,
                        'entity_features': entity_features,
                        'metric_names': metric_name_list
                    }, f)
            window_size_bar.set_description("Metric dataset generating".format(window_size))

    @staticmethod
    def calculate_statistic(metric_data):
        median = np.nanmedian(metric_data)
        percentile_1 = np.nanpercentile(metric_data, 1)
        percentile_99 = np.nanpercentile(metric_data, 99)
        q1 = np.nanpercentile(metric_data, 25)
        q3 = np.nanpercentile(metric_data, 75)
        mean = np.nanmean(metric_data)
        std = np.nanstd(metric_data)
        clip_data = np.clip(metric_data, percentile_1, percentile_99)
        clip_mean = np.nanmean(clip_data)
        clip_std = np.nanstd(clip_data)
        valid_ratio = (np.count_nonzero(~np.isnan(metric_data))) / len(list(metric_data))

        return {
            'clip_mean': clip_mean,
            'clip_std': clip_std,
            'percentile_1': percentile_1,
            'q1': q1,
            'median': median,
            'q3': q3,
            'percentile_99': percentile_99,
            'valid_ratio': valid_ratio,
            'mean': mean,
            'std': std
        }

    @staticmethod
    def merge_entity(resource_type, entity):
        if resource_type == 'container' or resource_type == 'istio':
            entity = entity.replace('2-0', '').replace('-0', '').replace('-1', '').replace('-2', '')
        elif resource_type == 'node':
            entity = entity.split('-')[0]
        return entity

    def eliminate_env_effect(self, date, cloud_bed, resource_type, entity, metric_name, metric_data):
        with open(f'{self.config.param_dict["temp_data_storage"]}/analysis/metric/env_statistic.json', 'r') as f:
            env_statistic_dict = json.load(f)
        env_mean = env_statistic_dict[f'{date}_{cloud_bed}'][resource_type][entity][metric_name]['mean']

        name_list = [
            "Memory/system.mem.used",
            "Memory/system.mem.real.used",
            "Memory/system.mem.usable",
            "Disk/system.disk.free",
            "Disk/system.disk.used",
            "Memory/kpi_container_memory_usage_MB",
            "Memory/kpi_container_memory_working_set_MB",
            "Memory/kpi_container_memory_rss",
            "Memory/kpi_container_memory_mapped_file"
            "Disk/kpi_container_fs_reads_MB",
            "Disk/kpi_container_fs_usage_MB",
            "Disk/kpi_container_fs_writes_MB",
        ]
        if metric_name in name_list and not np.isnan(env_mean):
            metric_data = metric_data - env_mean
        return metric_data

    @staticmethod
    def diff_metric(metric_name, metric_data):
        diff_name_list = [
            "Memory/system.mem.used",
            "Memory/system.mem.real.used",
            "Memory/system.mem.usable",
            "Disk/system.disk.free",
            "Disk/system.disk.used",
            "Memory/kpi_container_memory_usage_MB",
            "Memory/kpi_container_memory_working_set_MB",
            "Memory/kpi_container_memory_rss",
            "Memory/kpi_container_memory_mapped_file"
            "Disk/kpi_container_fs_reads_MB",
            "Disk/kpi_container_fs_usage_MB",
            "Disk/kpi_container_fs_writes_MB",
            "Thread/kpi_container_threads",
        ]
        if metric_name in diff_name_list:
            metric_data = np.array(metric_data)
            metric_data = np.diff(metric_data)
            metric_data = np.append(metric_data, metric_data[-1])
        return metric_data

    def get_metric(self, window_size) -> dict:
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/metric')
        with open(f'{folder}/metric_window_size_{window_size}.pkl', 'rb') as f:
            metric = pickle.load(f)
            return metric


if __name__ == '__main__':
    metric_generator = MetricGenerator()
    metric_generator.calculate_common_statistic()
    metric_generator.z_score_metric_data()
    metric_generator.generate_metric_data()
