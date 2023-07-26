from shared_util.file_handler import FileHandler
from data_filter.CCF_AIOps_challenge_2022.base.base_generator import BaseGenerator
from data_filter.CCF_AIOps_challenge_2022.service.time_interval_label_generator import TimeIntervalLabelGenerator

import copy
import pickle


class EntEdgeIndexGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def extract_topology(self):
        result_dict = dict()

        topology_dict = self.topology_dao.load_topology_edge_index()
        for window_size in self.window_size_list:
            result_dict[window_size] = dict()
            window_value = TimeIntervalLabelGenerator().get_time_interval_label(window_size)
            result_dict[window_size] = dict()
            for data_type in ['train_valid', 'test']:
                result_dict[window_size][data_type] = []
                for time_interval in window_value['time_interval'][data_type]:
                    result_dict[window_size][data_type].append(copy.deepcopy(topology_dict[f'{time_interval[0]}/{time_interval[1]}']))

        return result_dict

    def save_ent_edge_index(self):
        topology = self.extract_topology()

        for window_size in self.window_size_list:
            relation_dict = {
                'train_valid': topology[window_size]['train_valid'],
                'test': topology[window_size]['test']
            }
            folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/ent_edge_index')
            with open(f'{folder}/ent_edge_index_window_size_{window_size}.pkl', 'wb') as f:
                pickle.dump(relation_dict, f)

    def get_ent_edge_index(self, window_size):
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/ent_edge_index')
        with open(f'{folder}/ent_edge_index_window_size_{window_size}.pkl', 'rb') as f:
            ent_edge_index = pickle.load(f)
            return ent_edge_index


if __name__ == '__main__':
    ent_edge_index_generator = EntEdgeIndexGenerator()
    ent_edge_index_generator.save_ent_edge_index()
