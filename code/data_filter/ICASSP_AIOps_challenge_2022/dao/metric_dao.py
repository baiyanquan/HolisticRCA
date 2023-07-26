import sys

sys.path.append('/workspace/project/working/2023/HolisticRCA/code')

import pandas as pd
import numpy as np

from data_filter.ICASSP_AIOps_challenge_2022.base.base_class import BaseClass
from data_filter.ICASSP_AIOps_challenge_2022.dao.ground_truth_dao import GroundTruthDao

from data_filter.ICASSP_AIOps_challenge_2022.util.feature_process import FeatureProcess
from shared_util.file_handler import FileHandler
import pickle


class MetricDao(BaseClass):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_sample_index_list(dataset_type):
        if dataset_type == 'train_valid':
            return GroundTruthDao().get_train_valid_ground_truth()['sample_index']
        else:
            return list(range(0, 600))

    @staticmethod
    def load_metric_df(raw_metric_df):
        metric_df = raw_metric_df.drop(columns='Date & Time', axis=1)
        metric_df = metric_df.drop(columns=['feature3_5', 'feature3_6', 'feature3_7', 'feature3_8'], axis=1)
        metric_df['feature60'] = raw_metric_df['feature60'].apply(FeatureProcess.feature60_process)
        metric_df['feature20_edge'] = raw_metric_df.loc[:, raw_metric_df.columns.str.contains('feature20')].apply(FeatureProcess.feature20_edge_count, axis=1)
        metric_df['feature20_distance'] = raw_metric_df.loc[:, raw_metric_df.columns.str.contains('feature20')].apply(FeatureProcess.feature20_distance_count, axis=1)
        metric_df[['featureY_if', 'featureX_if']] = raw_metric_df.apply(FeatureProcess.feature20_xy_inference, axis=1, result_type='expand')
        metric_df[['featureY_mean', 'featureX_mean', 'featureY_min', 'featureX_min']] = raw_metric_df.apply(FeatureProcess.feature_xy_process, axis=1, result_type='expand')
        metric_df[['feature20_valid', 'featureX_valid', 'featureY_valid']] = raw_metric_df.apply(FeatureProcess.feature20_xy_valid, axis=1, result_type='expand')
        metric_df['length'] = [len(raw_metric_df) for _ in range(len(raw_metric_df))]
        return metric_df

    def calculate_mean(self):
        result_base_path = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/analysis')

        for dataset_type in ['train_valid', 'test']:
            mean_df = pd.DataFrame()
            feature_length = []
            for sample_index in MetricDao.get_sample_index_list(dataset_type):
                raw_metric_df = pd.read_csv(f'{self.config.data_dict["file"][dataset_type]["base_folder"]}/{sample_index}.csv')
                metric_df = MetricDao.load_metric_df(raw_metric_df)
                for column in metric_df.columns.tolist():
                    mean_df.loc[sample_index, column] = metric_df.loc[:, column].mean()
                feature_length.append(len(metric_df))
            mean_df['length'] = feature_length
            if dataset_type == 'train_valid':
                pad_data = mean_df.fillna(method='ffill', axis=0)
                back_data = mean_df.fillna(method='backfill', axis=0)
                mean_df = (pad_data + back_data) / 2
            else:
                front_part = mean_df[0:597]
                pad_front_part = front_part.fillna(method='ffill', axis=0)
                back_front_part = front_part.fillna(method='backfill', axis=0)
                front_part = (pad_front_part + back_front_part) / 2

                end_part = mean_df[596:600]
                end_part = end_part.fillna(method='ffill', axis=0)[1:]

                mean_df = front_part.append(end_part)

            mean_df.to_csv(f'{result_base_path}/{dataset_type}_sample_mean.csv')

    def generate_metric_csv(self):
        for dataset_type in ['train_valid', 'test']:
            result_base_path = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/raw_data/{dataset_type}/raw_metric')
            sample_index_list = MetricDao.get_sample_index_list(dataset_type)
            for sample_index in sample_index_list:
                raw_metric_df = pd.read_csv(f'{self.config.data_dict["file"][dataset_type]["base_folder"]}/{sample_index}.csv')
                raw_metric_df = MetricDao.load_metric_df(raw_metric_df)

                feature_dict = dict()
                for column in self.root_cause1_feature:
                    feature_dict[f'{column}:1'] = raw_metric_df.loc[:, column]
                    if column in self.baseline_rc1_statistic_df.columns:
                        feature_dict[f'{column}:1'].fillna(self.baseline_rc1_df[dataset_type].loc[sample_index, column], inplace=True)
                        feature_dict[f'{column}:1'] = (feature_dict[f'{column}:1'] - self.baseline_rc1_statistic_df.loc['mean', column]) / self.baseline_rc1_statistic_df.loc['std', column]
                    else:
                        if sample_index in self.baseline_rc23_df[dataset_type].index:
                            feature_dict[f'{column}:1'].fillna(self.baseline_rc23_df[dataset_type].loc[sample_index, column],inplace=True)
                        else:
                            feature_dict[f'{column}:1'].fillna(self.baseline_rc23_statistic_df['mean'].loc[column], inplace=True)
                        feature_dict[f'{column}:1'] = (feature_dict[f'{column}:1'] - self.baseline_rc23_statistic_df['mean'].loc[column]) / self.baseline_rc23_statistic_df['std'].loc[column]
                for column in self.root_cause23_feature:
                    if sample_index in self.baseline_rc23_df[dataset_type].index.tolist():
                        feature_dict[column] = np.array([self.baseline_rc23_df[dataset_type].loc[sample_index, column.split(':')[0]] for _ in range(len(raw_metric_df))])
                    else:
                        feature_dict[column] = np.array([self.baseline_rc23_original_statistic_df[dataset_type].loc['mean', column.split(':')[0]] for _ in range(len(raw_metric_df))])
                    feature_dict[column] = (feature_dict[column] - self.baseline_rc23_statistic_df['mean'].loc[column.split(':')[0]]) / self.baseline_rc23_statistic_df['std'].loc[column.split(':')[0]]

                    raw_column = column.split(':')[0]
                    if sample_index in self.baseline_rc23_df[dataset_type].index.tolist():
                        feature_dict[f'{raw_column}:23'] = raw_metric_df.loc[:, raw_column].interpolate()
                        feature_dict[f'{raw_column}:23'].fillna(self.baseline_rc23_df[dataset_type].loc[sample_index, raw_column], inplace=True)
                    else:
                        feature_dict[f'{raw_column}:23'] = np.array([self.baseline_rc23_original_statistic_df[dataset_type].loc['mean', column.split(':')[0]] for _ in range(len(raw_metric_df))])
                    feature_dict[f'{raw_column}:23'] = (feature_dict[f'{raw_column}:23'] - self.baseline_rc23_statistic_df['mean'].loc[raw_column]) / self.baseline_rc23_statistic_df['std'].loc[raw_column]

                metric_df = pd.DataFrame(feature_dict)
                metric_df.to_csv(f'{result_base_path}/{sample_index}.csv')
                print(f'sample_index: {sample_index}')

    def find_special_samples(self):
        special_sample_dict = {
            'rc1_fluctuate': [],
            'rc2_fluctuate': [],
            'rc3_fluctuate': [],
            'rc23_null': [142, 152, 153, 284, 305, 355, 361, 373, 420]
        }
        result_base_path = FileHandler.set_folder(f'{self.config.param_dict["temp_data_storage"]}/dataset/merge')

        # rc1
        feature13_std = []
        feature15_std = []
        sample_index_list = MetricDao.get_sample_index_list('test')
        for sample_index in sample_index_list:
            raw_metric_df = pd.read_csv(f'{self.config.data_dict["file"]["test"]["base_folder"]}/{sample_index}.csv')
            feature13_std.append(raw_metric_df['feature13'].std())
            feature15_std.append(raw_metric_df['feature15'].std())
        feature13_std, feature15_std = np.array(feature13_std), np.array(feature15_std)
        feature13_std[np.isnan(feature13_std)] = 0
        feature15_std[np.isnan(feature15_std)] = 0
        feature13_std = feature13_std / np.max(feature13_std)
        feature15_std = feature15_std / np.max(feature15_std)
        feature_fil = feature13_std + feature15_std
        special_sample_dict['rc1_fluctuate'] = np.where(np.array(feature_fil) > 0.6)[0].tolist()

        baseline_rc23_df = pd.read_csv(f'{self.config.data_dict["file"]["analysis"]["base_folder"]}/test_for_ml.csv', index_col=0)
        data_test = baseline_rc23_df.groupby('sample_index').mean()
        data_test.fillna(data_test.mean(), inplace=True)

        for sample_index, df in baseline_rc23_df.groupby('sample_index'):
            df.fillna(df.interpolate(), inplace=True)
            df.fillna(data_test.mean(), inplace=True)
            if df['feature19'].std() > 7:
                special_sample_dict['rc2_fluctuate'].append(sample_index)
            if df['feature13'].std() > 18000:
                special_sample_dict['rc3_fluctuate'].append(sample_index)

        with open(f'{result_base_path}/special_samples.pkl', 'wb') as f:
            pickle.dump(special_sample_dict, f)

    def load_metric_csv(self):
        result_dict = dict()
        for dataset_type in ['train_valid', 'test']:
            result_dict[dataset_type] = dict()
            sample_index_list = MetricDao.get_sample_index_list(dataset_type)
            for sample_index in sample_index_list:
                result_dict[dataset_type][sample_index] = pd.read_csv(f'{self.config.param_dict["temp_data_storage"]}/raw_data/{dataset_type}/raw_metric/{sample_index}.csv', index_col=0)
        return result_dict


if __name__ == '__main__':
    metric_dao = MetricDao()
    # metric_dao.calculate_mean()
    metric_dao.generate_metric_csv()
    metric_dao.find_special_samples()
