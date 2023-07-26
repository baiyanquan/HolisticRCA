from data_filter.ICASSP_AIOps_challenge_2022.base.base_class import BaseClass
import pandas as pd
import numpy as np


class GroundTruthDao(BaseClass):
    def __init__(self):
        super().__init__()

    def get_train_valid_ground_truth(self):
        result_dict = {
            'sample_index': [],
            'y': []
        }

        raw_label_dict = pd.read_csv(f'{self.config.data_dict["ground_truth"]["train_valid"]}').to_dict('list')
        for i in range(len(raw_label_dict['sample_index'])):
            result_dict['sample_index'].append(raw_label_dict['sample_index'][i])
            temp_y = [0, 0, 0]
            if raw_label_dict['root-cause(s)'][i] == 'rootcause1':
                temp_y[0] = 1
            if raw_label_dict['root-cause(s)'][i] == 'rootcause2':
                temp_y[1] = 1
            elif raw_label_dict['root-cause(s)'][i] == 'rootcause3':
                temp_y[2] = 1
            elif raw_label_dict['root-cause(s)'][i] == 'rootcause2&rootcause3':
                temp_y[1] = 1
                temp_y[2] = 1
            result_dict['y'].append(temp_y)

        feature_select = ['feature0', 'feature1', 'feature2', 'feature11', 'feature12',
                          'feature13', 'feature14', 'feature15', 'feature16', 'feature17',
                          'feature18', 'feature19', 'feature60', 'feature20_distance',
                          'featureY_if', 'featureX_if', 'featureY_mean', 'featureX_mean']
        baseline_rc23_df = pd.read_csv(f'{self.config.data_dict["file"]["analysis"]["base_folder"]}/train_for_ml.csv', index_col=0)
        data_train = baseline_rc23_df.groupby('sample_index').mean(numeric_only=True).loc[:, feature_select]
        total_index_set = set(data_train.index.tolist())
        valid_train_index_set = set(data_train.dropna().index.tolist())
        for sample_index in total_index_set - valid_train_index_set:
            index = result_dict['sample_index'].index(sample_index)
            result_dict['y'][index] = [result_dict['y'][index][0], result_dict['y'][index][1], -1]

        return result_dict

    def get_test_ground_truth(self):
        result_list = []
        raw_label_dict = pd.read_csv(f'{self.config.data_dict["ground_truth"]["test"]}').to_dict('list')
        for i in range(len(raw_label_dict['ID'])):
            result_list.append([raw_label_dict['Root1'][i], raw_label_dict['Root2'][i], raw_label_dict['Root3'][i]])
        return result_list


if __name__ == '__main__':
    ground_truth_dao = GroundTruthDao()
    ground_truth_dao.get_train_valid_ground_truth()

