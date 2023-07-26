import numpy as np
import pandas as pd
import warnings


class FeatureProcess:
    @staticmethod
    def feature20_edge_count(feature20_series):
        edge_list = [0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23, 24, 25, 30, 31]
        feature20_edge_count = 0
        for num in feature20_series:
            if not np.isnan(num) and num in edge_list:
                feature20_edge_count += 1
        if feature20_series.isnull().all():
            feature20_edge_count = np.nan
        return feature20_edge_count

    @staticmethod
    def feature20_distance_count(feature20_series):
        coordinate = []
        for m in feature20_series:
            coordinate.append((m - 8 * (m // 8), m // 8))
        distance = 0
        for a in coordinate:
            for b in coordinate:
                distance = distance + ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        return distance

    @staticmethod
    def feature60_process(s):
        if isinstance(s, str):
            s = eval(s.replace(';', ','))
            s = round(np.mean(s), 2)
        return s

    @staticmethod
    def feature_xy_process(feature_xy_series):
        feature_y_list = ['feature28', 'feature36', 'feature44', 'feature52']
        feature_x_list = ['feature61', 'feature69', 'feature77', 'feature85']
        feature_x = pd.DataFrame()
        feature_y = pd.DataFrame()
        for feature_x_name in feature_x_list:
            feature_x = feature_x.append(feature_xy_series.loc[feature_xy_series.index.str.contains(feature_x_name)])
        for feature_y_name in feature_y_list:
            feature_y = feature_y.append(feature_xy_series.loc[feature_xy_series.index.str.contains(feature_y_name)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if len(feature_x) == 0:
                feature_x_mean = np.nan
                feature_x_min = np.nan
            else:
                feature_x_mean = np.nanmean(feature_x)
                feature_x_min = np.nanmin(feature_x)
            if len(feature_y) == 0:
                feature_y_mean = np.nan
                feature_y_min = np.nan
            else:
                feature_y_mean = np.nanmean(feature_y)
                feature_y_min = np.nanmin(feature_y)
        return [feature_y_mean, feature_x_mean, feature_y_min, feature_x_min]

    @staticmethod
    def feature20_xy_inference(feature20_xy_series):
        feature_if_dict = {}
        feature20 = feature20_xy_series.loc[feature20_xy_series.index.str.contains('feature20')]
        feature_y_list = ['feature28', 'feature36', 'feature44', 'feature52']
        feature_x_list = ['feature61', 'feature69', 'feature77', 'feature85']

        for feature_v_list in [feature_y_list, feature_x_list]:
            for feature_v_name in feature_v_list:
                feature_v = feature20_xy_series.loc[feature20_xy_series.index.str.contains(feature_v_name)]
                c_v_list = []
                for i, value in zip(feature20, feature_v):
                    if not (np.isnan(i) and np.isnan(value)):
                        c_v = [i - 8 * (i // 8), i // 8, value]
                        c_v_list.append(c_v)
                interference = 0
                for item in c_v_list:
                    for other_item in c_v_list:
                        if other_item != item:
                            distance = ((item[0] - other_item[0]) ** 2 + (item[1] - other_item[1]) ** 2) ** 0.5
                            if distance == 0:
                                distance = 1
                            interference += other_item[2] / distance
                if len(c_v_list) == 0:
                    interference = np.nan
                else:
                    interference = interference / len(c_v_list)
                    interference = round(interference, 2)
                feature_if_dict[feature_v_name] = interference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feature_y_if = np.nanmean([feature_if_dict[k] for k in feature_y_list])
            feature_x_if = np.nanmean([feature_if_dict[k] for k in feature_x_list])
        return [feature_y_if, feature_x_if]

    @staticmethod
    def feature20_xy_valid(feature20_xy_series):
        feature20 = ['feature20']
        feature_y_list = ['feature28', 'feature36', 'feature44', 'feature52']
        feature_x_list = ['feature61', 'feature69', 'feature77', 'feature85']

        result_list = []
        for feature_v_list in [feature20, feature_y_list, feature_x_list]:
            valid = False
            for feature_v_name in feature_v_list:
                feature_v = feature20_xy_series.loc[feature20_xy_series.index.str.contains(feature_v_name)]
                for value in feature_v:
                    if not np.isnan(value):
                        valid = True
            if valid:
                result_list.append(1)
            else:
                result_list.append(0)
        return result_list
