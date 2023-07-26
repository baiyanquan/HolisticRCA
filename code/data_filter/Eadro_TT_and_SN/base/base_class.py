from abc import ABC
from data_filter.Eadro_TT_and_SN.config.dataset_config import DataConfig
from shared_util.logger import Logger


class BaseClass(ABC):
    def __init__(self):
        self.dataset_name = ''
        self.fault_interval = ''
        self.sample_granularity = ''
        self.config = DataConfig()
        self.logger = Logger(self.config.param_dict['logging']['level']).logger
        self.window_size_list = [6, 8, 10, 12, 14]
        self.all_entity_list = []
        self.valid_network_entity_list = []
        self.ent_edge_index_list = []
        self.metric_name_list = [
            'cpu_usage_system', 'cpu_usage_total', 'cpu_usage_user',
            'memory_usage', 'memory_working_set', 'rx_bytes', 'tx_bytes'
        ]
        self.fault_type_list = [
            'cpu_load',
            'network_delay',
            'network_loss',
        ]
        self.fault_type_related_o11y_names = {
            0: {
                "exact": ['cpu_usage_system', 'cpu_usage_total', 'cpu_usage_user'],
                "fuzzy": []
            },
            1: {
                "exact": [],
                "fuzzy": ['rx_bytes', 'tx_bytes', "<intensity>", "<duration>"]
            },
            2: {
                "exact": [],
                "fuzzy": ['rx_bytes', 'tx_bytes', "<intensity>", "<duration>"]
            }
        }
