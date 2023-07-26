from data_filter.Eadro_TT_and_SN.base.base_generator import BaseGenerator
import numpy as np


class EntEdgeIndexGenerator(BaseGenerator):
    def __init__(self, base):
        super().__init__(base)

    def get_edge_index(self, window_size) -> dict:
        result_dict = dict()

        ground_truth_dict = self.ground_truth_dao.get_time_label_interval(window_size)
        for dataset_type in ['train', 'valid', 'test']:
            result_dict[dataset_type] = np.array([self.base.ent_edge_index_list for _ in range(len(ground_truth_dict[dataset_type]))])
        return result_dict
