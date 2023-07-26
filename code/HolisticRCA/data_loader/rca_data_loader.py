import pickle
from torch.utils.data import DataLoader

from HolisticRCA.base.base_data_loader import BaseDataLoader
from HolisticRCA.dataset.rca_dataset import RCADataset


class RCADataLoader(BaseDataLoader):
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def load_data(self, data_path):
        with open(f'{data_path}', 'rb') as f:
            temp = pickle.load(f)
        self.meta_data = temp['meta_data']
        data = dict()
        for dataset_type in ['train', 'valid', 'test']:
            data[dataset_type] = dict()
            for modal_type in self.meta_data['modal_types']:
                data[dataset_type][f'x_{modal_type}'] = temp['data'][f'x_{modal_type}_{dataset_type}'].transpose((0, 2, 1))
            data[dataset_type][f'ent_edge_index'] = temp['data'][f'ent_edge_index_{dataset_type}']
            data[dataset_type][f'y'] = temp['data'][f'y_{dataset_type}']

            shuffle = False
            if dataset_type == 'train':
                shuffle = True

            if dataset_type == 'train' or dataset_type == 'valid':
                self.data_loader[dataset_type] = DataLoader(RCADataset(data[dataset_type]),
                                                            batch_size=self.param_dict['batch_size'],
                                                            shuffle=shuffle)
            else:
                self.data_loader[dataset_type] = DataLoader(RCADataset(data[dataset_type]),
                                                            batch_size=self.param_dict['batch_size'],
                                                            shuffle=shuffle)
