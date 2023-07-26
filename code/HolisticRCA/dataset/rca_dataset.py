import torch
from torch.utils.data import Dataset
import numpy as np


class RCADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        item = dict()
        for key in self.data.keys():
            if key == 'ent_edge_index':
                item[key] = torch.LongTensor(self.data[key][idx])
            else:
                item[key] = torch.FloatTensor(self.data[key][idx].astype(np.float64))
        return item
