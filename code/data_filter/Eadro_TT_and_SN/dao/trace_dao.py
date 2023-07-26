import pandas as pd

from data_filter.Eadro_TT_and_SN.base.base_class import BaseClass
from data_filter.Eadro_TT_and_SN.util.time_interval import TimeInterval
from data_filter.Eadro_TT_and_SN.base.base_sn_class import BaseSNClass
from data_filter.Eadro_TT_and_SN.base.base_tt_class import BaseTTClass
from shared_util.file_handler import FileHandler
import numpy as np
import json


class TraceDao:
    def __init__(self, base: BaseClass):
        self.base = base
        self.span_contains_service_count_dict = dict()

    def extract_trace_features(self):
        self.span_contains_service_count_dict = dict()
        dataset_type_list = ['faulty', 'normal', 'z-score']
        for dataset_type in dataset_type_list:
            data_base_path = self.base.config.data_dict[self.base.dataset_name]['file'][dataset_type]['base_folder']
            for t in self.base.config.data_dict[self.base.dataset_name]['time'][dataset_type]:
                result_path = FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/raw_data/{t}/raw_trace') + f'/span_features.csv'
                timestamp_list = TimeInterval.generate_timestamp_list(f'{data_base_path}/{self.base.dataset_name}.fault-{t}.json', self.base.sample_granularity)
                self.process_traces(timestamp_list, f'{data_base_path}/{self.base.dataset_name}.{t}/spans.json', result_path)
        count_dict_save_path = FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/analysis/trace') + f'/span_service_count.json'
        with open(count_dict_save_path, 'w') as f:
            json.dump(self.span_contains_service_count_dict, f, indent=2)

    def process_traces(self, timestamp_list: list, data_path: str, result_path: str):
        trace_pattern_list = [f'cmdb_id: {entity}' for entity in self.base.all_entity_list]
        span_feature_dict = {'timestamp': timestamp_list}

        for trace_pattern in trace_pattern_list:
            span_feature_dict[f'<intensity>; {trace_pattern}; type: upstream'] = [0 for _ in timestamp_list]
            span_feature_dict[f'<duration>; {trace_pattern}; type: upstream'] = [[] for _ in timestamp_list]
            span_feature_dict[f'<intensity>; {trace_pattern}; type: current'] = [0 for _ in timestamp_list]
            span_feature_dict[f'<duration>; {trace_pattern}; type: current'] = [[] for _ in timestamp_list]
            span_feature_dict[f'<intensity>; {trace_pattern}; type: downstream'] = [0 for _ in timestamp_list]
            span_feature_dict[f'<duration>; {trace_pattern}; type: downstream'] = [[] for _ in timestamp_list]

        with open(data_path) as f:
            trace_list = json.load(f)

        for trace in trace_list:
            bucket_index = ''
            span_id_index_dict = dict()
            child_span_dict = dict()
            for i in range(len(trace['spans'])):
                if len(trace['spans'][i]['references']) == 0:
                    bucket_index = int((int(trace['spans'][i]['startTime'] / 1000000) - timestamp_list[0] - 3600 * 8) / (timestamp_list[1] - timestamp_list[0]))
                span_id_index_dict[trace['spans'][i]['spanID']] = i
                for reference in trace['spans'][i]['references']:
                    if reference['refType'] == 'CHILD_OF':
                        if reference['spanID'] not in child_span_dict.keys():
                            child_span_dict[reference['spanID']] = []
                        child_span_dict[reference['spanID']].append(trace['spans'][i]['spanID'])
            if bucket_index == '' or bucket_index >= len(timestamp_list) or bucket_index < 0:
                continue

            for span in trace['spans']:
                entity = trace['processes'][span['processID']]['serviceName']
                if entity not in self.span_contains_service_count_dict.keys():
                    self.span_contains_service_count_dict[entity] = 0
                self.span_contains_service_count_dict[entity] += 1

                span_feature_dict[f'<intensity>; cmdb_id: {entity}; type: current'][bucket_index] += 1
                span_feature_dict[f'<duration>; cmdb_id: {entity}; type: current'][bucket_index].append(span['duration'])
                for reference in span['references']:
                    if reference['refType'] == 'CHILD_OF' and reference['spanID'] in span_id_index_dict.keys():
                        span_feature_dict[f'<intensity>; cmdb_id: {entity}; type: upstream'][bucket_index] += 1
                        span_feature_dict[f'<duration>; cmdb_id: {entity}; type: upstream'][bucket_index].append(trace['spans'][span_id_index_dict[reference['spanID']]]['duration'])
                if span['spanID'] in child_span_dict.keys():
                    for span_id in child_span_dict[span['spanID']]:
                        span_feature_dict[f'<intensity>; cmdb_id: {entity}; type: downstream'][bucket_index] += 1
                        span_feature_dict[f'<duration>; cmdb_id: {entity}; type: downstream'][bucket_index].append(trace['spans'][span_id_index_dict[span_id]]['duration'])

        for trace_pattern in trace_pattern_list:
            for feature_type in ['upstream', 'current', 'downstream']:
                span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}'] = [np.mean(duration_list) if len(duration_list) > 0 else 0 for duration_list in span_feature_dict[f'<duration>; {trace_pattern}; type: {feature_type}']]

        span_feature_df = pd.DataFrame(span_feature_dict)
        span_feature_df.to_csv(result_path, index=False)

    def load_trace_csv(self):
        result_dict = dict()
        dataset_type_list = ['faulty', 'normal', 'z-score']
        for dataset_type in dataset_type_list:
            for t in self.base.config.data_dict[self.base.dataset_name]['time'][dataset_type]:
                result_dict[t] = pd.read_csv(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/raw_data/{t}/raw_trace/span_features.csv')

        return result_dict


if __name__ == '__main__':
    trace_dao = TraceDao(BaseSNClass())
    trace_dao.extract_trace_features()
