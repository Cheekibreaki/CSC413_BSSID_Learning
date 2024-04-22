from sklearn.preprocessing import MinMaxScaler,RobustScaler
import pandas as pd
import numpy as np
import math
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataHelper(object):

    def __init__(self):
        self.base_dir = os.getcwd()
        self.WAP_SIZE = 520
        self.LONGITUDE = 520
        self.LATITUDE = 521
        self.FLOOR = 522
        self.BUILDINGID = 523
        self.normY = self.NormY()

    #input data separated before

    def set_config(self,wap_size,long,lat,floor,building_id):
        self.WAP_SIZE = wap_size
        self.LONGITUDE = long
        self.LATITUDE = lat
        self.FLOOR = floor
        self.BUILDINGID = building_id

    class NormY(object):
        long_max=None
        long_min=None
        lati_max=None
        lati_min=None
        long_scale=None
        lati_scale=None

        def __init__(self):
            pass

        def fit(self,long,lati):
            self.long_max=max(long)
            self.long_min=min(long)
            self.lati_max=max(lati)
            self.lati_min=min(lati)
            self.long_scale=self.long_max-self.long_min
            self.lati_scale=self.lati_max-self.lati_min

        def normalizeY(self,longitude_arr, latitude_arr):

            longitude_arr = np.reshape(longitude_arr, [-1, 1])
            latitude_arr = np.reshape(latitude_arr, [-1, 1])
            long=(longitude_arr-self.long_min)/self.long_scale
            lati=(latitude_arr-self.lati_min)/self.lati_scale
            return long,lati

        def reverse_normalizeY(self,longitude_arr, latitude_arr):

            longitude_arr = np.reshape(longitude_arr, [-1, 1])
            latitude_arr = np.reshape(latitude_arr, [-1, 1])
            long=(longitude_arr*self.long_scale)+self.long_min
            lati=(latitude_arr*self.lati_scale)+self.lati_min
            return long,lati

    def normalizeX(self,arr,b=2.8):
        # return normalizeX_powed_noise(arr,rate=noise_rate).reshape([-1,520])
        # return normalizeX_zero_to_one(arr).reshape([-1, 520])
        return self.normalizeX_powed(arr,b).reshape([-1, self.WAP_SIZE])

    def normalizeX_powed(self,arr, b):
        res = np.copy(arr).astype(np.float)
        for i in range(np.shape(res)[0]):
            for j in range(np.shape(res)[1]):
                if (res[i][j] > 50) | (res[i][j] == None) | (res[i][j] < -95):
                    res[i][j] = 0
                elif (res[i][j] >= 0):
                    res[i][j] = 1

                else:
                    res[i][j] = ((95 + res[i][j]) / 95.0) ** b
                # res[i][j] = (0.01 * (110 + res[i][j])) ** 2.71828
        return res



        # return normx.transform(arr).reshape([-1,520])
    def load_data_perspective(self,file_name):
        data_frame = pd.read_csv(file_name)
        data_x = data_frame.get_values().T[:self.WAP_SIZE].T
        data_y = data_frame.get_values().T[[self.LONGITUDE, self.LATITUDE, self.FLOOR, self.BUILDINGID], :].T
        return data_x, data_y

    def split_data_perspective(self, data_x, data_y):
        unique_floors = np.unique(data_y[:, 2])  # Assuming 3rd column of data_y is the floor
        data_lists = {
            'train': {'x': [], 'y': []},
            'valid': {'x': [], 'y': []},
            'test': {'x': [], 'y': []}
        }

        for floor in unique_floors:
            floor_mask = data_y[:, 2] == floor
            floor_data_x = data_x[floor_mask]
            floor_data_y = data_y[floor_mask]

            unique_coords = np.unique(floor_data_y[:, [0, 1]],
                                      axis=0)  # Assume columns 0 and 1 are longitude and latitude
            # Shuffle unique coordinates to randomize the distribution
            np.random.shuffle(unique_coords)

            # Split unique coordinates into sets ensuring unique distribution
            num_coords = len(unique_coords)
            if num_coords < 3:  # Not enough data to split meaningfully
                data_lists['train']['x'].extend(floor_data_x)
                data_lists['train']['y'].extend(floor_data_y)
            else:
                split_idx1 = int(num_coords * 0.7)  # 70% for training
                split_idx2 = int(num_coords * 0.9)  # Next 20% for validation, last 10% for test

                train_coords = unique_coords[:split_idx1]
                valid_coords = unique_coords[split_idx1:split_idx2]
                test_coords = unique_coords[split_idx2:]

                for coords_set, key in zip([train_coords, valid_coords, test_coords], ['train', 'valid', 'test']):
                    for coords in coords_set:
                        coords_mask = (floor_data_y[:, 0] == coords[0]) & (floor_data_y[:, 1] == coords[1])
                        data_lists[key]['x'].append(floor_data_x[coords_mask])
                        data_lists[key]['y'].append(floor_data_y[coords_mask])

        # Concatenate all data from lists and convert to numpy arrays
        for key in data_lists:
            data_lists[key]['x'] = np.concatenate(data_lists[key]['x'], axis=0)
            data_lists[key]['y'] = np.concatenate(data_lists[key]['y'], axis=0)

            # Save to single CSV files
            df = pd.DataFrame(np.hstack([data_lists[key]['x'], data_lists[key]['y']]),
                              columns=[f'feature_{i}' for i in range(data_lists[key]['x'].shape[1])] + ['longitude',
                                                                                                        'latitude',
                                                                                                        'floor',
                                                                                                        'building_id'])
            df.to_csv(os.path.join(self.base_dir, f'UJIndoorLoc_Unique_lonlat_{key}.csv'), index=False)

        print("Data split and saved to CSV files: train.csv, valid.csv, test.csv.")



    # def split_data_perspective(self, data_x, data_y):
    #     # Unique floors
    #     unique_floors = np.unique(data_y[:, 2])  # Assuming 3rd column of data_y is the floor
    #
    #     # Initialize empty lists to collect data splits
    #     train_x_list = []
    #     train_y_list = []
    #     valid_x_list = []
    #     valid_y_list = []
    #     test_x_list = []
    #     test_y_list = []
    #
    #     # Splitting data for each floor and collecting them
    #     for floor in unique_floors:
    #         # Select data for the current floor
    #         floor_mask = data_y[:, 2] == floor
    #         floor_data_x = data_x[floor_mask]
    #         floor_data_y = data_y[floor_mask]
    #
    #         # Splitting into training, validation, and test sets (70%, 20%, 10%)
    #         train_x, temp_x, train_y, temp_y = train_test_split(floor_data_x, floor_data_y, test_size=0.3,
    #                                                             random_state=42)
    #         valid_x, test_x, valid_y, test_y = train_test_split(temp_x, temp_y, test_size=1 / 3,
    #                                                             random_state=42)  # ~1/3 of 30% is 10%
    #
    #         # Append to lists
    #         train_x_list.append(train_x)
    #         train_y_list.append(train_y)
    #         valid_x_list.append(valid_x)
    #         valid_y_list.append(valid_y)
    #         test_x_list.append(test_x)
    #         test_y_list.append(test_y)
    #
    #     # Concatenate all data from lists
    #     train_x = np.concatenate(train_x_list, axis=0)
    #     train_y = np.concatenate(train_y_list, axis=0)
    #     valid_x = np.concatenate(valid_x_list, axis=0)
    #     valid_y = np.concatenate(valid_y_list, axis=0)
    #     test_x = np.concatenate(test_x_list, axis=0)
    #     test_y = np.concatenate(test_y_list, axis=0)
    #
    #     # Save to single CSV files
    #     train_df = pd.DataFrame(np.hstack([train_x, train_y]),
    #                             columns=[f'feature_{i}' for i in range(train_x.shape[1])] + ['longitude', 'latitude',
    #                                                                                          'floor', 'building_id'])
    #     valid_df = pd.DataFrame(np.hstack([valid_x, valid_y]),
    #                             columns=[f'feature_{i}' for i in range(valid_x.shape[1])] + ['longitude', 'latitude',
    #                                                                                          'floor', 'building_id'])
    #     test_df = pd.DataFrame(np.hstack([test_x, test_y]),
    #                            columns=[f'feature_{i}' for i in range(test_x.shape[1])] + ['longitude', 'latitude',
    #                                                                                        'floor', 'building_id'])
    #
    #     train_df.to_csv(os.path.join(self.base_dir, 'UTSndoorLoc_train.csv'), index=False)
    #     valid_df.to_csv(os.path.join(self.base_dir, 'UTSndoorLoc_valid.csv'), index=False)
    #     test_df.to_csv(os.path.join(self.base_dir, 'UTSndoorLoc_test.csv'), index=False)
    #
    #     print("Data split and saved to CSV files: train.csv, valid.csv, test.csv.")
    #
    # # Assume other methods and class implementations continue here


    def load_data_all(self,train,valid,test):
        return self.load_data_perspective(train),self.load_data_perspective(valid),self.load_data_perspective(test)

if __name__ == '__main__':
    data_helper = DataHelper()
    base_dir = os.getcwd()
    test_csv_path = os.path.join(base_dir,
                                 '/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/validationData.csv')
    valid_csv_path = os.path.join(base_dir,
                                  '/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/validationData.csv')
    train_csv_path = os.path.join(base_dir,
                                  '/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/trainingData.csv')
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path, valid_csv_path,
                                                                                         test_csv_path)
    data_helper.split_data_perspective(train_x, train_y)
