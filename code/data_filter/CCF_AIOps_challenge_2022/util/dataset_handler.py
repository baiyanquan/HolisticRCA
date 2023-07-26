from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import random


random.seed(409)


class DatasetHandler:
    @staticmethod
    def split_and_save_dataset(modal_type_list: list,
                               modal_data: dict,
                               ent_edge_index: dict,
                               valid_ratio: float,
                               y: dict,
                               multi_class_label_format: bool,
                               num_of_fault_types: int,
                               meta_data: dict,
                               save_file_path: str):
        result_dict = {
            'data': dict(),
            'meta_data': meta_data
        }

        test_size = valid_ratio
        random_state = 409

        train_valid_list, test_list = [], []
        for modal_type in modal_type_list:
            train_valid_list.append(modal_data[modal_type]['train_valid'])
            test_list.append(modal_data[modal_type]['test'])
        train_valid_list.append(ent_edge_index['train_valid'])
        test_list.append(ent_edge_index['test'])
        train_valid_list.append(y['train_valid'])
        test_list.append(y['test'])

        train_valid = train_test_split(*tuple(train_valid_list),
                                       test_size=test_size,
                                       random_state=random_state,
                                       shuffle=True)

        data = dict()
        data['train'], data['valid'], data['test'] = train_valid[::2], train_valid[1::2], test_list
        data['train'] = DatasetHandler.over_sampling(data['train'])

        for data_type in ['train', 'valid', 'test']:
            for i in range(len(modal_type_list)):
                result_dict['data'][f'x_{modal_type_list[i]}_{data_type}'] = np.array(data[data_type][i], dtype=object)
            result_dict['data'][f'ent_edge_index_{data_type}'] = data[data_type][len(modal_type_list)]

            if multi_class_label_format:
                data[data_type][len(modal_type_list) + 1] = DatasetHandler.label_to_multi_class_format(data[data_type][len(modal_type_list) + 1], num_of_fault_types)

            result_dict['data'][f'y_{data_type}'] = np.array(data[data_type][len(modal_type_list) + 1])

        with open(save_file_path, 'wb') as f:
            pickle.dump(result_dict, f, protocol=4)

    @staticmethod
    def label_to_multi_class_format(raw_y, num_of_fault_types=15):
        y = []
        raw_y = np.array(raw_y)
        for i in range(raw_y.shape[0]):
            y.append([])
            for j in range(raw_y.shape[1]):
                y[-1].append(np.zeros(num_of_fault_types))
                if raw_y[i][j] != 0:
                    y[-1][-1][int(raw_y[i][j] - 1)] = 1
        return np.array(y)

    @staticmethod
    def over_sampling(train):
        x = np.array(train, dtype=object).transpose().tolist()
        y_train = train[-1]

        temp_y = []
        fault_type_list = []
        for label in y_train:
            fault_type = 'None'
            if np.count_nonzero(label) > 1:
                fault_type = f'service:{int(np.max(label))}'
            elif np.count_nonzero(label) == 1:
                fault_type = f'{int(np.max(label))}'
            if fault_type not in fault_type_list:
                fault_type_list.append(fault_type)
            temp_y.append(fault_type_list.index(fault_type))
        x, temp_y = np.array(x, dtype=object), np.array(temp_y, dtype=object)

        k = 0
        index_dict = dict()
        for i in range(len(fault_type_list)):
            index_dict[i] = np.where(temp_y == i)[0]
            k = max(k, index_dict[i].shape[0])
        for i in range(len(fault_type_list)):
            index_dict[i] = random.choices(population=index_dict[i].tolist(), k=k)

        indices = []
        for i in range(len(fault_type_list)):
            indices.extend(index_dict[i])
        random.shuffle(indices)
        indices = np.array(indices)

        x = x[indices]
        return np.array(x).transpose().tolist()
