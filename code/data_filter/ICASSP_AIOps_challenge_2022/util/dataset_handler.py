import copy

from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import random
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE


random.seed(409)


class DatasetHandler:
    @staticmethod
    def split_and_save_dataset(modal_type_list: list,
                               modal_data: dict,
                               ent_edge_index: dict,
                               valid_ratio: float,
                               y: dict,
                               meta_data: dict,
                               save_file_path: str):
        result_dict = {
            'data': dict(),
            'meta_data': meta_data
        }

        test_size = valid_ratio
        random_state = 409
        add_mean_metric, add_y = [], []
        mean_metric_train_valid = [np.mean(i, axis=0).tolist() for i in modal_data['metric']['train_valid']]

        def remove_invalid_samples(var_x, var_y):
            index = np.where(var_y != -1)[0]
            return np.array(var_x)[index].tolist(), var_y[index]

        base_x, base_y = remove_invalid_samples(mean_metric_train_valid, np.array(y['train_valid'])[:, 0])
        smote_metric_train_valid, smote_y_train_valid = BorderlineSMOTE(random_state=409).fit_resample(base_x, base_y)
        for i in range(len(smote_metric_train_valid)):
            if smote_metric_train_valid[i] not in mean_metric_train_valid:
                add_mean_metric.append([smote_metric_train_valid[i] for _ in range(30)])
                add_y.append([smote_y_train_valid[i], -1, -1])

        base_x, base_y = remove_invalid_samples(mean_metric_train_valid, np.array(y['train_valid'])[:, 1])
        smote_metric_train_valid, smote_y_train_valid = SMOTEENN(random_state=409).fit_resample(base_x, base_y)
        for i in range(len(smote_metric_train_valid)):
            if smote_metric_train_valid[i] not in mean_metric_train_valid:
                add_mean_metric.append([smote_metric_train_valid[i] for _ in range(30)])
                add_y.append([-1, smote_y_train_valid[i], -1])

        base_x, base_y = remove_invalid_samples(mean_metric_train_valid, np.array(y['train_valid'])[:, 2])
        smote_metric_train_valid, smote_y_train_valid = SMOTEENN(random_state=409).fit_resample(base_x, base_y)
        for i in range(len(smote_metric_train_valid)):
            if smote_metric_train_valid[i] not in mean_metric_train_valid:
                add_mean_metric.append([smote_metric_train_valid[i] for _ in range(30)])
                add_y.append([-1, -1, smote_y_train_valid[i]])

        modal_data['metric']['train_valid'].extend(add_mean_metric)
        y['train_valid'].extend(add_y)

        for i in range(0, 3):
            rc_label = np.array(y['train_valid'])[:, i]
            rc_label = rc_label[rc_label != -1]
            print(f'rc{i + 1} weight: {np.sum(rc_label) / (rc_label.shape[0] - np.sum(rc_label))}, 1: {np.sum(rc_label)}, total: {rc_label.shape[0]}')

        train_valid_list, test_list = [], []
        for modal_type in modal_type_list:
            train_valid_list.append(modal_data[modal_type]['train_valid'])
            test_list.append(modal_data[modal_type]['test'])

        for _ in range(600, len(modal_data['metric']['test'])):
            y['test'].append([0, 0, 0])

        train_valid_list.append([ent_edge_index['train_valid'][0] for _ in range(len(y['train_valid']))])
        test_list.append([ent_edge_index['test'][0] for _ in range(len(y['test']))])
        train_valid_list.append(y['train_valid'])
        test_list.append(y['test'])

        train_valid = train_test_split(*tuple(train_valid_list), test_size=test_size, random_state=random_state, shuffle=True)

        data = dict()
        data['train'], data['valid'], data['test'] = train_valid[::2], train_valid[1::2], test_list
        # data['train'] = DatasetHandler.over_sampling(data['train'])

        for data_type in ['train', 'valid', 'test']:
            for i in range(len(modal_type_list)):
                result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = np.array(data[data_type][i])
            result_dict['data'][f'ent_edge_index_{data_type}'] = data[data_type][len(modal_type_list)]
            temp_y = np.array(data[data_type][len(modal_type_list) + 1])
            result_dict['data'][f'y_{data_type}'] = temp_y.reshape((temp_y.shape[0], temp_y.shape[1], 1))

        with open(save_file_path, 'wb') as f:
            pickle.dump(result_dict, f, protocol=4)

    @staticmethod
    def over_sampling(train):
        x = np.array(train).transpose()
        y_train = train[-1]

        temp_y = []
        for label in y_train:
            for i in range(len(label)):
                if label[i] == 1:
                    temp_y.append(i)
        temp_y = np.array(temp_y)

        k = 0
        index_dict = dict()
        for i in range(0, 3):
            index_dict[i] = np.where(temp_y == i)[0]
            k = max(k, index_dict[i].shape[0])
        for i in range(0, 3):
            index_dict[i] = random.choices(population=index_dict[i].tolist(), k=k)

        indices = []
        for i in range(0, 3):
            indices.extend(index_dict[i])
        random.shuffle(indices)
        indices = np.array(indices)

        x = x[indices]
        return np.array(x).transpose().tolist()
