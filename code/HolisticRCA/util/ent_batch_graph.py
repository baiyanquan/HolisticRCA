from HolisticRCA.base.base_batch_graph import BaseBatchGraph


class EntBatchGraph(BaseBatchGraph):
    def __init__(self, batch_data, meta_data):
        super().__init__(batch_data, meta_data)
        self.x['re'] = batch_data['x_ent']
        self.edge_index = batch_data['ent_edge_index']
        self.batch_size = self.edge_index.shape[0]

        num_of_nodes = len(self.meta_data['ent_names'])
        self.generate_x_batch(num_of_nodes)
        self.generate_batch_edge_index(self.edge_index, num_of_nodes)
