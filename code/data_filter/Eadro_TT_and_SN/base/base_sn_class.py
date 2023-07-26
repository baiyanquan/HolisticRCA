from data_filter.Eadro_TT_and_SN.base.base_class import BaseClass


class BaseSNClass(BaseClass):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'SN'
        self.fault_interval = self.config.param_dict['fault_interval']['SN']
        self.sample_granularity = self.config.param_dict['sample_granularity']['SN']
        self.all_entity_list = [
            'social-graph-service', 'compose-post-service', 'post-storage-service',
            'user-timeline-service', 'url-shorten-service', 'user-service',
            'media-service', 'text-service', 'unique-id-service', 'user-mention-service',
            'home-timeline-service', "nginx-web-server"
        ]
        ent_edge_info = {
            "compose-post-service": [
                "home-timeline-service", "media-service", "post-storage-service",
                "text-service", "unique-id-service", "user-service", "user-timeline-service"
            ],
            "home-timeline-service": [
                "post-storage-service", "social-graph-service"
            ],
            "social-graph-service": ["user-service"],
            "text-service": ["url-shorten-service", "user-mention-service"],
            "nginx-web-server": [
                "compose-post-service", "home-timeline-service",
                "social-graph-service", "user-service"
            ]
        }
        self.valid_network_entity_list = [
            'social-graph-service', 'compose-post-service', 'post-storage-service',
            'user-timeline-service', 'url-shorten-service', 'user-service',
            'media-service', 'text-service', 'unique-id-service', 'user-mention-service',
            'home-timeline-service', "nginx-web-server"
        ]
        self.ent_edge_index_list = [[], []]
        for i in range(len(self.all_entity_list)):
            self.ent_edge_index_list[0].append(i)
            self.ent_edge_index_list[1].append(i)
        for start_edge, end_edge_list in ent_edge_info.items():
            for end_edge in end_edge_list:
                self.ent_edge_index_list[0].append(self.all_entity_list.index(start_edge))
                self.ent_edge_index_list[1].append(self.all_entity_list.index(end_edge))
                self.ent_edge_index_list[1].append(self.all_entity_list.index(start_edge))
                self.ent_edge_index_list[0].append(self.all_entity_list.index(end_edge))
