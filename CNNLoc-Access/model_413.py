from encoder_model import EncoderDNN
import data_helper_413
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

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

class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.WAP_SIZE = 520
        self.LONGITUDE = 520
        self.LATITUDE = 521
        self.FLOOR = 522
        self.BUILDINGID = 523
        self.Floor_classes = 4

        self.normalize_valid_x= None
        self.normalize_x= None
        self.normalize_y= None
        self.normalize_valid_y= None

        self.weights = nn.Parameter(torch.randn(self.WAP_SIZE, self.WAP_SIZE)) # Assuming the input dimension is 100 and output is 4x100
        self.fc1 = nn.Linear(self.WAP_SIZE, 260)  # First fully connected layer
        self.fc2 = nn.Linear(260, self.Floor_classes)  # Output layer
    def forward(self, x):
        # Apply a softmax to weights to simulate a stochastic matrix where each row sums to 1
        # soft_permutation_matrix = F.softmax(self.weights, dim=1)
        # Matrix multiplication to reorder x in a soft manner
        #x = torch.matmul(soft_permutation_matrix, x.unsqueeze(-1)).squeeze(-1)
        # Use ReLu
        x = F.relu(self.fc1(x))  # Activation function after first fully connected layer
        x = self.fc2(x)  # Activation function after second fully connected layer

        #output = F.softmax(logits, dim=1)  # Applying softmax to get probabilities for each class
        output = x
        return output

    # def __init__(self):
    #     super(NN, self).__init__()
    #     self.WAP_SIZE = 520
    #     self.LONGITUDE = 520
    #     self.LATITUDE = 521
    #     self.FLOOR = 522
    #     self.BUILDINGID = 523
    #     self.Floor_classes = 4
    #
    #     self.normalize_valid_x = None
    #     self.normalize_x = None
    #     self.normalize_y = None
    #     self.normalize_valid_y = None
    #
    #     # Define the weights for the permutation matrix
    #     self.weights = nn.Parameter(torch.randn(self.WAP_SIZE, self.WAP_SIZE))
    #
    #     # Convolutional layers
    #
    #     self.conv1 = nn.Conv2d(1, 1, kernel_size=(30, 4), padding=(1, 1), stride=(10, 1))
    #
    #     # Final fully connected layers
    #     self.fc1 = nn.Linear(33,
    #                          11)  # Adjust the input features depending on the output of the last conv layer
    #     self.fc2 = nn.Linear(11, self.Floor_classes)
    #
    # def forward(self, x):
    #     # Apply softmax to simulate a stochastic matrix where each row sums to 1
    #     soft_permutation_matrix = F.softmax(self.weights, dim=1)
    #     x_reordered = torch.matmul(soft_permutation_matrix, x.unsqueeze(-1)).squeeze(-1)
    #
    #     # Reshape x_reordered for convolution
    #     # Here you need to ensure the total dimensions match the reshaped dimensions
    #     x = x_reordered.view(-1, 1, 130, 4)  # Batch size, Channels, Height, Width
    #
    #     # Convolutional layers with activation and pooling
    #     x = F.relu(self.conv1(x))
    #
    #     # Flatten the output for the fully connected layer
    #     x = torch.flatten(x, 1)
    #
    #     # Fully connected layers
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #
    #     return x




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




from torch.nn import BCEWithLogitsLoss

def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), torch.nn.functional.one_hot(targets, num_classes=model.Floor_classes).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            train_correct += (predicted == targets.max(1)[1]).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100 * train_correct / total)  # Update for multi-class classification

        # Validation phase
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.float(), torch.nn.functional.one_hot(targets, num_classes=model.Floor_classes).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                k = targets.max(1)
                val_correct += (predicted == targets.max(1)[1]).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * val_correct / total)

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2f}%')

    return model, train_losses, val_losses, train_accs, val_accs


if __name__ == '__main__':
    # Setup

    num_epochs = 200
    learning_rate = 0.0002

    # Assuming data_helper functions and filter_building return appropriate numpy arrays
    nn_model = NN()
    # nn_model.to(device)
    data_helper = data_helper_413.DataHelper()
    data_helper.set_config(wap_size=nn_model.WAP_SIZE, long=nn_model.LONGITUDE, lat=nn_model.LATITUDE, floor=nn_model.FLOOR,
                           building_id=nn_model.BUILDINGID)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path, valid_csv_path,
                                                                                         test_csv_path)
    (train_x, train_y) = filter_building(train_x, train_y, 1)
    (valid_x, valid_y) = filter_building(valid_x, valid_y, 1)

    # Assuming train_x and train_y are numpy arrays, you need to convert them to tensors
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y[:, 2], dtype=torch.long)  # Assuming column 2 is the target

    valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32)
    valid_y_tensor = torch.tensor(valid_y[:, 2], dtype=torch.long)  # Same assumption

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    valid_dataset = TensorDataset(valid_x_tensor, valid_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=64)

    # Train the model
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(nn_model, train_loader, val_loader, num_epochs, learning_rate)












