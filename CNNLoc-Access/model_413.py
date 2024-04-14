from encoder_model import EncoderDNN
import data_helper_413
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"]='0'
from keras.backend.tensorflow_backend import set_session
# config=tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.9
# set_session(tf.Session(config=config))


base_dir= os.getcwd()
test_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/validationData.csv')
valid_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/validationData.csv')
train_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/trainingData.csv')


def filter_building(x, y, building_id):
    # Initialize lists to store filtered results
    filtered_x = []
    filtered_y = []

    # Loop over all samples and filter based on the building ID
    for i in range(y.shape[0]):  # Ensure looping over correct range
        if y[i, 3] == building_id:  # Check building ID at the correct index
            filtered_x.append(x[i, :])  # Add the corresponding x vector
            filtered_y.append(y[i])  # Add the entire y vector

    # Convert lists to numpy arrays before returning
    filtered_x = np.array(filtered_x)
    filtered_y = np.array(filtered_y)

    return filtered_x, filtered_y


class NN(object):

    def __init__(self):
        self.normalize_valid_x= None
        self.normalize_x= None
        self.normalize_y= None
        self.normalize_valid_y= None

    def _preprocess(self, x, y, valid_x, valid_y):
        #self.normY = data_helper_413.NormY()
        self.normalize_x = data_helper.normalizeX(x)
        self.normalize_valid_x = data_helper.normalizeX(valid_x)

        data_helper.normY.fit(y[:, 0], y[:, 1])
        self.longitude_normalize_y, self.latitude_normalize_y = data_helper.normY.normalizeY(y[:, 0], y[:, 1])
        self.floorID_y = y[:, 2]
        self.buildingID_y = y[:, 3]

        self.longitude_normalize_valid_y, self.latitude_normalize_valid_y = data_helper.normY.normalizeY(valid_y[:, 0],valid_y[:, 1])
        self.floorID_valid_y = valid_y[:, 2]
        self.buildingID_valid_y = valid_y[:, 3]


    def _knn_process(self):
        knn = KNeighborsClassifier(n_neighbors=4)

        # Train the classifier
        knn.fit(self.normalize_x, self.floorID_y)

        # Predict the floor numbers for the testing set
        Y_pred = knn.predict(self.normalize_x)
        accuracy = accuracy_score(self.floorID_y, Y_pred)
        #
        # Y_pred = knn.predict(self.normalize_valid_x)
        # accuracy = accuracy_score(self.floorID_valid_y, Y_pred)
        print(accuracy)


if __name__ == '__main__':
    data_helper = data_helper_413.DataHelper()
    #data_helper.set_config(wap_size=589,long=589,lat=590,floor=591,building_id=592)
    (train_x, train_y), (valid_x, valid_y),(test_x,test_y) = data_helper.load_data_all(train_csv_path, valid_csv_path,test_csv_path)
    (train_x,train_y) = filter_building(train_x,train_y,1)
    (valid_x, valid_y) = filter_building(valid_x, valid_y, 1)
    nn_model = NN()
    nn_model._preprocess(train_x[:2000],train_y[:2000],valid_x[:400],valid_y[:400])
    nn_model._knn_process()
