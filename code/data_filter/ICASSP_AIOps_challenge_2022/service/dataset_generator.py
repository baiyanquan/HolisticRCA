from shared_util.file_handler import FileHandler
from data_filter.ICASSP_AIOps_challenge_2022.base.base_generator import BaseGenerator
from data_filter.ICASSP_AIOps_challenge_2022.service.metric_generator import MetricGenerator
from data_filter.ICASSP_AIOps_challenge_2022.service.ent_edge_index_generator import EntEdgeIndexGenerator
from data_filter.ICASSP_AIOps_challenge_2022.util.dataset_handler import DatasetHandler


class DatasetGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.meta_data = {
            "modal_types": ['metric'],
            "ent_types": ['rc1', 'rc2', 'rc3'],
            'ent_names': ['rc1', 'rc2', 'rc3'],
            'ent_type_index': {
                'rc1': (0, 1),
                'rc2': (1, 2),
                'rc3': (2, 3)
            },
            'ent_features': {
                "metric": self.entity_features
            },
            'max_ent_feature_num': {
                "rc1": {
                    "metric": self.entity_features[0][1][1] - self.entity_features[0][1][0],
                },
                "rc2": {
                    "metric": self.entity_features[1][1][1] - self.entity_features[1][1][0],
                },
                "rc3": {
                    "metric": self.entity_features[2][1][1] - self.entity_features[2][1][0],
                }
            },
            'o11y_names': {
                "metric": self.all_feature_list
            },
            'o11y_length': {
                "metric": len(self.all_feature_list)
            },
            'ent_fault_type_index': {
                "rc1": (0, 1),
                "rc2": (0, 1),
                "rc3": (0, 1),
            },
            'ent_fault_type_weight': {
                "rc1": [1.0],
                "rc2": [1.0],
                "rc3": [1.006],
            }
        }

    def get_base_data(self):
        modal_dict = {
            'metric': MetricGenerator().get_metric()['metric_data']
        }
        ent_edge_index = EntEdgeIndexGenerator().get_ent_edge_index()
        y = {
            'train_valid': self.ground_truth_dao.get_train_valid_ground_truth()['y'],
            'test': self.ground_truth_dao.get_test_ground_truth()
        }
        return modal_dict, ent_edge_index, y

    def generate_rca_dataset(self):
        folder = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/merge')
        modal_dict, ent_edge_index, y = self.get_base_data()
        DatasetHandler.split_and_save_dataset(modal_type_list=['metric'],
                                              modal_data=modal_dict,
                                              ent_edge_index=ent_edge_index,
                                              valid_ratio=0.2,
                                              y=y,
                                              meta_data=self.meta_data,
                                              save_file_path=f'{folder}/rca.pkl')


if __name__ == '__main__':
    dataset_generator = DatasetGenerator()
    dataset_generator.generate_rca_dataset()
    ...
