import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, feature, adj_matrix, label):
        self.feature = feature
        self.adj_matrix = adj_matrix
        self.label = label

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.feature[idx])
        graph = torch.FloatTensor(self.adj_matrix[idx])
        label = torch.tensor(self.label[idx]).float()
        return x, graph, label
