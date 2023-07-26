import os
import json


class Config:
    def __init__(self):
        self.data_dict = dict()
        self.param_dict = dict()

        self.set_data_dict()
        self.set_param_dict()

    def set_data_dict(self):
        data_base_path = '/workspace/project/working/2023/HolisticRCA'
        self.data_dict = {
            'data_storage': f'{data_base_path}/temp_data'
        }

    def set_param_dict(self):
        data_base_path = '/workspace/project/working/2023/HolisticRCA'
        self.param_dict = {
            'logging': {
                'level': 'DEBUG'
            },
            'save_path': f'{data_base_path}'
        }
