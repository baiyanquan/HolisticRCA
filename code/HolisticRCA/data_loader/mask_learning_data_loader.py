import pickle
from torch.utils.data import DataLoader

from HolisticRCA.base.base_data_loader import BaseDataLoader
from HolisticRCA.dataset.rca_dataset import RCADataset


class MaskLearningDataLoader(BaseDataLoader):
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def load_data(self, data_path):
        with open(f'{data_path}', 'rb') as f:
            temp = pickle.load(f)
        self.meta_data = temp['meta_data']
        data = dict()
        dataset_type = 'test'
        for modal_type in self.meta_data['modal_types']:
            data[f'x_{modal_type}'] = temp['data'][f'x_{modal_type}_{dataset_type}'].transpose((0, 2, 1))
        data[f'ent_edge_index'] = temp['data'][f'ent_edge_index_{dataset_type}']
        data[f'y'] = temp['data'][f'y_{dataset_type}']
        self.data_loader = DataLoader(RCADataset(data), batch_size=1, shuffle=False)
