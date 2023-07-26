import os
import json


class DataConfig:
    def __init__(self):
        self.data_dict = dict()
        self.param_dict = dict()

        self.set_data_dict()
        self.set_param_dict()

    def set_data_dict(self):
        data_base_path = '/workspace/dataset/2022_ICASSP_AIOps_challenge/'
        self.data_dict = {
            'file': {
                'train_valid': {
                    'base_folder': data_base_path + 'train',
                },
                'test': {
                    'base_folder': data_base_path + 'test',
                },
                'analysis': {
                    'base_folder': data_base_path + 'analysis'
                }
            },
            'ground_truth': {
                'train_valid': data_base_path + 'train_label.csv',
                'test': data_base_path + 'test_label.csv'
            },
            'o11y_relation': data_base_path + 'causality_graph.json'
        }

    def set_param_dict(self):
        data_base_path = '/workspace/project/working/2023/HolisticRCA'
        self.param_dict = {
            'logging': {
                'level': 'DEBUG'
            },
            'temp_data_storage': f'{data_base_path}/temp_data/2022_ICASSP_AIOps_challenge'
        }
