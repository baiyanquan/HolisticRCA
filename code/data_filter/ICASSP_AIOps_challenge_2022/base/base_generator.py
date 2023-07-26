from data_filter.ICASSP_AIOps_challenge_2022.base.base_class import BaseClass
from data_filter.ICASSP_AIOps_challenge_2022.dao.metric_dao import MetricDao
from data_filter.ICASSP_AIOps_challenge_2022.dao.ground_truth_dao import GroundTruthDao
from data_filter.ICASSP_AIOps_challenge_2022.dao.o11y_relation_dao import O11yRelationDao


class BaseGenerator(BaseClass):
    def __init__(self):
        super().__init__()
        self.ground_truth_dao = GroundTruthDao()
        self.metric_dao = MetricDao()
        self.o11y_relation_dao = O11yRelationDao()
