from abc import ABC
from data_filter.ICASSP_AIOps_challenge_2022.config.dataset_config import DataConfig
from shared_util.logger import Logger
import pandas as pd


class BaseClass(ABC):
    def __init__(self):
        self.config = DataConfig()
        self.logger = Logger(self.config.param_dict['logging']['level']).logger

        self.root_cause1_feature = [
            'feature0', 'feature1', 'feature2', 'feature11', 'feature12',
            'feature13', 'feature15', 'feature16', 'feature17', 'feature18',
            'feature19',
            'feature60',
            'featureY_if', 'featureX_if', 'featureY_mean', 'featureX_mean',

            'feature20_edge', 'feature20_distance', 'length'
        ]

        self.root_cause23_feature = [
            'feature0:mean', 'feature1:mean', 'feature2:mean', 'feature11:mean', 'feature12:mean',
            'feature13:mean', 'feature14:mean', 'feature15:mean', 'feature16:mean', 'feature17:mean',
            'feature18:mean', 'feature19:mean', 'feature20_distance:mean', 'feature60:mean',
            'featureY_if:mean', 'featureX_if:mean', 'featureY_mean:mean', 'featureX_mean:mean'
        ]

        self.all_feature_list = []
        self.all_feature_list.extend(list(set(self.root_cause1_feature) - set(self.root_cause23_feature)))
        self.all_feature_list.extend(list(set(self.root_cause1_feature) & set(self.root_cause23_feature)))
        self.all_feature_list.extend(list(set(self.root_cause23_feature) - set(self.root_cause1_feature)))
        self.entity_features = [('rc1', (0, len(self.root_cause1_feature))),
                                ('rc2', (len(self.all_feature_list) - len(self.root_cause23_feature), len(self.all_feature_list))),
                                ('rc3', (len(self.all_feature_list) - len(self.root_cause23_feature), len(self.all_feature_list)))]
        self.rc1_rename_dict = {
            'feature_edge': 'feature20_edge',
            'feature_distance': 'feature20_distance'
        }
        self.baseline_rc1_df = {
            'train_valid': pd.read_csv(f'{self.config.data_dict["file"]["analysis"]["base_folder"]}/train_mean.csv',
                                       index_col=0),
            'test': pd.read_csv(f'{self.config.data_dict["file"]["analysis"]["base_folder"]}/new_sample_mean.csv',
                                index_col=0)
        }
        for dataset_type in self.baseline_rc1_df.keys():
            self.baseline_rc1_df[dataset_type].rename(columns=self.rc1_rename_dict, inplace=True)

        self.baseline_rc23_df = {
            'train_valid': pd.read_csv(f'{self.config.data_dict["file"]["analysis"]["base_folder"]}/train_for_ml.csv',
                                       index_col=0).groupby('sample_index').mean(numeric_only=True).dropna(),
            'test': pd.read_csv(f'{self.config.data_dict["file"]["analysis"]["base_folder"]}/test_for_ml.csv',
                                index_col=0).groupby('sample_index').mean(numeric_only=True)
        }
        self.baseline_rc23_df['test'].fillna(self.baseline_rc23_df['test'].mean(), inplace=True)

        self.baseline_rc1_statistic_df = self.baseline_rc1_df['train_valid'].describe().loc[['mean', 'std']]
        self.baseline_rc23_original_statistic_df = {
            'train_valid': self.baseline_rc23_df['train_valid'].describe().loc[['mean', 'std']],
            'test': self.baseline_rc23_df['test'].describe().loc[['mean', 'std']],
            'final': pd.concat([self.baseline_rc23_df['train_valid'], self.baseline_rc23_df['test']])
        }
        self.baseline_rc23_statistic_df = {
            'mean': pd.concat([self.baseline_rc23_df['train_valid'], self.baseline_rc23_df['test']]).mean(),
            'std': pd.concat([self.baseline_rc23_df['train_valid'], self.baseline_rc23_df['test']]).std(ddof=0)
        }
