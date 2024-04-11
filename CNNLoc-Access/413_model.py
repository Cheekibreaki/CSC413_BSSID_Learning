from encoder_model import EncoderDNN
import data_helper_413
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]='0'
from keras.backend.tensorflow_backend import set_session
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
set_session(tf.Session(config=config))


base_dir= os.getcwd()
test_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/validationData.csv')
valid_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/validationData.csv')
train_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/trainingData.csv')




class NN(object):

    def __init__(self):
        self.normalize_valid_x= None
        self.normalize_x= None
        self.normalize_y= None
        self.normalize_valid_y= None


    def _preprocess(self, x, y, valid_x, valid_y):
        self.normY = data_helper_413.NormY()
        self.normalize_x = data_helper_413.normalizeX(x)
        self.normalize_valid_x = data_helper_413.normalizeX(valid_x)

        self.normY.fit(y[:, 0], y[:, 1])
        self.longitude_normalize_y, self.latitude_normalize_y = self.normY.normalizeY(y[:, 0], y[:, 1])
        self.floorID_y = y[:, 2]
        self.buildingID_y = y[:, 3]

        self.longitude_normalize_valid_y, self.latitude_normalize_valid_y = self.normY.normalizeY(valid_y[:, 0],valid_y[:, 1])
        self.floorID_valid_y = valid_y[:, 2]
        self.buildingID_valid_y = valid_y[:, 3]



if __name__ == '__main__':
    (train_x, train_y), (valid_x, valid_y),(test_x,test_y) = data_helper_413.load_data_all(train_csv_path, valid_csv_path,test_csv_path)
    nn_model = NN()
    nn_model._preprocess(train_x[:2000],train_y[:2000],valid_x[:400],valid_y[:400])
