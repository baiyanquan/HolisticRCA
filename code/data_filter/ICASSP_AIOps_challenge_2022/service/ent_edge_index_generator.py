from shared_util.file_handler import FileHandler
from data_filter.ICASSP_AIOps_challenge_2022.base.base_generator import BaseGenerator

import pickle
import numpy as np


class EntEdgeIndexGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()

    def extract_edge_index(self):
        result_dict = dict()
        edge_index = [[], []]
        for s_index in range(0, 3):
            edge_index[0].append(s_index)
            edge_index[1].append(s_index)

        for dataset_type in ['train_valid', 'test']:
            if dataset_type == 'train_valid':
                sample_index = self.ground_truth_dao.get_train_valid_ground_truth()['sample_index']
            else:
                sample_index = list(range(0, 600))
            result_dict[dataset_type] = [edge_index for _ in sample_index]
        return result_dict

    def save_ent_edge_index(self):
        edge_index = self.extract_edge_index()

        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/ent_edge_index')
        with open(f'{folder}/ent_edge_index.pkl', 'wb') as f:
            pickle.dump(edge_index, f)

    def get_ent_edge_index(self):
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/ent_edge_index')
        with open(f'{folder}/ent_edge_index.pkl', 'rb') as f:
            ob_relation = pickle.load(f)
            return ob_relation


if __name__ == '__main__':
    ent_edge_index_generator = EntEdgeIndexGenerator()
    ent_edge_index_generator.save_ent_edge_index()
