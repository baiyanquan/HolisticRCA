import os
import json


class DataConfig:
    def __init__(self):
        self.data_dict = dict()
        self.param_dict = dict()

        self.set_data_dict()
        self.set_param_dict()

    def set_data_dict(self):
        tt_base_path = '/workspace/dataset/2023_Eadro_TT/'
        sn_base_path = '/workspace/dataset/2023_Eadro_SN/'
        self.data_dict = {
            'TT': {
                'time': {
                    'faulty': [
                        '2022-04-17T212101D2022-04-17T230842',
                        '2022-04-18T102520D2022-04-18T121301',
                        '2022-04-18T121515D2022-04-18T140256',
                        '2022-04-18T150729D2022-04-18T165511',
                        '2022-04-18T165807D2022-04-18T184548',
                        '2022-04-18T184759D2022-04-18T203539',
                        '2022-04-18T203805D2022-04-18T222547',
                        '2022-04-18T222801D2022-04-19T001541',
                        '2022-04-19T001753D2022-04-19T020534'
                    ],
                    'z-score': [
                        '2022-04-21T105158D2022-04-21T130705',
                    ],
                    'normal': [
                        '2022-04-21T153246D2022-04-21T174753'
                    ]
                },
                'file': {
                    'faulty': {
                        'base_folder': tt_base_path + 'data',
                    },
                    'z-score': {
                        'base_folder': tt_base_path + 'no fault',
                    },
                    'normal': {
                        'base_folder': tt_base_path + 'no fault',
                    },
                }
            },
            'SN': {
                'time': {
                    'faulty': [
                        '2022-04-17T181245D2022-04-17T183616',
                        '2022-04-17T183729D2022-04-17T190100',
                        '2022-04-17T190213D2022-04-17T192544',
                        '2022-04-17T192658D2022-04-17T195031'
                    ],
                    'z-score': [
                        '2022-04-20T182405D2022-04-20T184806',
                        '2022-04-21T105249D2022-04-21T111651',
                    ],
                    'normal': [
                        '2022-04-21T153302D2022-04-21T155703'
                    ]
                },
                'file': {
                    'faulty': {
                        'base_folder': sn_base_path + 'data',
                    },
                    'z-score': {
                        'base_folder': sn_base_path + 'no fault',
                    },
                    'normal': {
                        'base_folder': sn_base_path + 'no fault',
                    },
                }
            }
        }

    def set_param_dict(self):
        data_base_path = '/workspace/project/working/2023/HolisticRCA'
        self.param_dict = {
            'logging': {
                'level': 'DEBUG'
            },
            'temp_data_storage': {
                'TT': f'{data_base_path}/temp_data/2023_Eadro_TT',
                'SN': f'{data_base_path}/temp_data/2023_Eadro_SN'
            },
            'fault_interval': {
                'TT': 600,
                'SN': 120
            },
            'sample_granularity': {
                'TT': 30,
                'SN': 5
            }
        }
