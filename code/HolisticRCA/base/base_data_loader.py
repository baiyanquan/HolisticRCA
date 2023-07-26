from abc import ABC, abstractmethod
from HolisticRCA.config.config import Config


class BaseDataLoader(ABC):
    def __init__(self, param_dict):
        self.config = Config()
        self.param_dict = param_dict
        self.meta_data = dict()
        self.data_loader = dict()

    @abstractmethod
    def load_data(self, data_path):
        raise NotImplementedError
