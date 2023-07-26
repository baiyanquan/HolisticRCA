import torch
from torch_sparse import SparseTensor


class BaseBatchGraph:
    def __init__(self, batch_data, meta_data):
        self.meta_data = meta_data
        self.edge_index = ''
        self.batch_size = 0
        self.x = dict()
        self.x_batch = []

    def generate_batch_edge_index(self, edge_index, num_of_nodes):
        edge_index = edge_index.transpose(1, 2).contiguous()
        for i in range(self.batch_size):
            edge_index[i] += i * num_of_nodes
        self.edge_index = edge_index.view(edge_index.shape[0] * edge_index.shape[1], edge_index.shape[2]).t().contiguous()
        self.edge_index = SparseTensor(row=self.edge_index[0], col=self.edge_index[1])

    def generate_x_batch(self, num_of_nodes):
        for i in range(self.batch_size):
            for _ in range(num_of_nodes):
                self.x_batch.append(i)
        self.x_batch = torch.tensor(self.x_batch, dtype=torch.long)

    def to(self, device):
        for key in self.x.keys():
            self.x[key] = self.x[key].to(device)
        self.edge_index = self.edge_index.to(device)
        self.x_batch = self.x_batch.to(device)
        return self
