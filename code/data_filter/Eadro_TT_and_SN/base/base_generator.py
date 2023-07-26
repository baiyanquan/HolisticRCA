from data_filter.Eadro_TT_and_SN.base.base_class import BaseClass
from data_filter.Eadro_TT_and_SN.dao.metric_dao import MetricDao
from data_filter.Eadro_TT_and_SN.dao.trace_dao import TraceDao
from data_filter.Eadro_TT_and_SN.dao.log_dao import LogDao
from data_filter.Eadro_TT_and_SN.dao.ground_truth_dao import GroundTruthDao
import numpy as np


class BaseGenerator:
    def __init__(self, base: BaseClass):
        self.base = base
        self.ground_truth_dao = GroundTruthDao(base)
        self.raw_metric_dao = MetricDao(base)
        self.raw_trace_dao = TraceDao(base)
        self.raw_log_dao = LogDao(base)

    @staticmethod
    def z_score_data(data_dict):
        z_score = np.concatenate([data_dict['train'], data_dict['valid']], axis=0)
        z_score = z_score.reshape(-1, z_score.shape[2])

        mean, std = [], []
        for i in range(z_score.shape[1]):
            mean.append(np.mean(z_score[:, i][z_score[:, i] != 0]))
            std.append(np.mean(z_score[:, i][z_score[:, i] != 0]))
        mean, std = np.nan_to_num(np.array(mean), nan=0), np.nan_to_num(np.array(std), nan=0)
        for dataset_type in ['train', 'valid', 'test']:
            data_dict[dataset_type] = (data_dict[dataset_type] - mean) / (std + 1e-8)
        return data_dict
