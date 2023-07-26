from shared_util.file_handler import FileHandler
from data_filter.Eadro_TT_and_SN.base.base_generator import BaseGenerator
from data_filter.Eadro_TT_and_SN.base.base_sn_class import BaseSNClass
from data_filter.Eadro_TT_and_SN.base.base_tt_class import BaseTTClass
import numpy as np
import pickle


class TraceGenerator(BaseGenerator):
    def __init__(self, base):
        super().__init__(base)

    def get_trace_feature_info(self) -> (list, list):
        trace_pattern_list = [f'cmdb_id: {entity}' for entity in self.base.all_entity_list]
        feature_index, entity_features, trace_name_list = 0, [], []
        for i in range(len(trace_pattern_list)):
            trace_name_list.append(f'<intensity>; {trace_pattern_list[i]}; type: upstream')
            trace_name_list.append(f'<duration>; {trace_pattern_list[i]}; type: upstream')
            trace_name_list.append(f'<intensity>; {trace_pattern_list[i]}; type: current')
            trace_name_list.append(f'<duration>; {trace_pattern_list[i]}; type: current')
            trace_name_list.append(f'<intensity>; {trace_pattern_list[i]}; type: downstream')
            trace_name_list.append(f'<duration>; {trace_pattern_list[i]}; type: downstream')
            entity_features.append((i, (feature_index * 6, (feature_index + 1) * 6)))
            feature_index += 1
        return entity_features, trace_name_list

    def calculate_mean_and_std(self, data_dict):
        statistic_dict = dict()

        for i in range(len(self.base.all_entity_list)):
            statistic_dict[self.base.all_entity_list[i]] = {
                'train_valid_test': dict(),
                'train_valid': dict(),
                'normal': dict()
            }

        train_valid_test = np.concatenate([data_dict['train'], data_dict['valid'], data_dict['test']], axis=0)
        train_valid_test = train_valid_test.reshape(-1, train_valid_test.shape[2])
        for i in range(len(self.base.all_entity_list)):
            statistic_dict[self.base.all_entity_list[i]]['train_valid_test'] = {
                'mean': np.mean(train_valid_test, axis=0)[i * 6:(i + 1) * 6],
                'std': np.std(train_valid_test, axis=0)[i * 6:(i + 1) * 6],
                'max': np.max(train_valid_test, axis=0)[i * 6:(i + 1) * 6],
            }

        train_valid = np.concatenate([data_dict['train'], data_dict['valid']], axis=0)
        train_valid = train_valid.reshape(-1, train_valid.shape[2])
        for i in range(len(self.base.all_entity_list)):
            statistic_dict[self.base.all_entity_list[i]]['train_valid'] = {
                'mean': np.mean(train_valid, axis=0)[i * 6:(i + 1) * 6],
                'std': np.std(train_valid, axis=0)[i * 6:(i + 1) * 6],
                'max': np.max(train_valid, axis=0)[i * 6:(i + 1) * 6],
            }

        normal = np.concatenate([data_dict['z-score']], axis=0)
        normal = normal.reshape(-1, normal.shape[2])
        for i in range(len(self.base.all_entity_list)):
            statistic_dict[self.base.all_entity_list[i]]['normal'] = {
                'mean': np.mean(normal, axis=0)[i * 6:(i + 1) * 6],
                'std': np.std(normal, axis=0)[i * 6:(i + 1) * 6],
                'max': np.max(normal, axis=0)[i * 6:(i + 1) * 6],
            }

        return statistic_dict

    def generate_trace_data(self):
        entity_features, trace_name_list = self.get_trace_feature_info()
        for window_size in self.base.window_size_list:
            ground_truth_dict = self.ground_truth_dao.get_time_label_interval(window_size)
            raw_trace_dict = self.raw_trace_dao.load_trace_csv()

            trace_dict = dict()
            for dataset_type in ['train', 'valid', 'test', 'z-score']:
                trace_dict[dataset_type] = []
                for ground_truth in ground_truth_dict[dataset_type]:
                    timestamp_list = raw_trace_dict[ground_truth[0]].loc[:, 'timestamp'].tolist()
                    start_index = int((ground_truth[2] - timestamp_list[0]) / (timestamp_list[1] - timestamp_list[0]))
                    trace_dict[dataset_type].append(raw_trace_dict[ground_truth[0]].query(f'{timestamp_list[start_index]} <= timestamp < {timestamp_list[start_index + window_size]}').loc[:, trace_name_list].values)
                trace_dict[dataset_type] = np.array(trace_dict[dataset_type])

            analysis_trace_dict_before = {2: dict(), 3: dict()}
            for dataset_type in ['train', 'valid', 'test', 'z-score']:
                analysis_trace_dict_before[2][dataset_type] = []
                analysis_trace_dict_before[3][dataset_type] = []
                for i in range(len(ground_truth_dict[dataset_type])):
                    entity, fault_type = ground_truth_dict[dataset_type][i][4], ground_truth_dict[dataset_type][i][5]
                    if entity is not None and fault_type in analysis_trace_dict_before.keys():
                        analysis_trace_dict_before[fault_type][dataset_type].append((trace_dict[dataset_type][i, :, entity * 6:(entity + 1) * 6], self.base.all_entity_list[entity]))

            trace_dict = TraceGenerator.z_score_data(trace_dict)
            statistic_dict = self.calculate_mean_and_std(trace_dict)

            analysis_trace_dict_after = {2: dict(), 3: dict()}
            for dataset_type in ['train', 'valid', 'test', 'z-score']:
                analysis_trace_dict_after[2][dataset_type] = []
                analysis_trace_dict_after[3][dataset_type] = []
                for i in range(len(ground_truth_dict[dataset_type])):
                    entity, fault_type = ground_truth_dict[dataset_type][i][4], ground_truth_dict[dataset_type][i][5]
                    if entity is not None and fault_type in analysis_trace_dict_before.keys():
                        analysis_trace_dict_after[fault_type][dataset_type].append((trace_dict[dataset_type][i, :, entity * 6:(entity + 1) * 6], self.base.all_entity_list[entity], statistic_dict[self.base.all_entity_list[entity]]))

            with open(FileHandler.set_folder(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/trace') + f'/trace_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump({
                    'trace_data': trace_dict,
                    'entity_features': entity_features,
                    'trace_names': trace_name_list
                }, f)

    def get_trace(self, window_size) -> np:
        with open(f'{self.base.config.param_dict["temp_data_storage"][self.base.dataset_name]}/dataset/trace/trace_window_size_{window_size}.pkl', 'rb') as f:
            trace = pickle.load(f)
            return trace


if __name__ == '__main__':
    trace_generator = TraceGenerator(BaseSNClass())
    trace_generator.generate_trace_data()
