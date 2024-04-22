
import data_helper_413
import os
from torch.nn import CrossEntropyLoss
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
import math
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from keras.backend.tensorflow_backend import set_session
# config=tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.9
# set_session(tf.Session(config=config))


base_dir= os.getcwd()

test_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/TestData.csv')
valid_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/ValuationData.csv')
train_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UJIndoorLoc/TrainingData.csv')

# test_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UOFT_WIFI/TestData.csv')
# valid_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UOFT_WIFI/ValidataionData.csv')
# train_csv_path=os.path.join(base_dir,'/home/jiping/Projects/BSSID_CNNLOC/CNNLoc-Access/UOFT_WIFI/TrainingData.csv')




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

    # def __init__(self):
    #     super(NN, self).__init__()
    #     self.WAP_SIZE = 520
    #     self.LONGITUDE = 520
    #     self.LATITUDE = 521
    #     self.FLOOR = 522
    #     self.BUILDINGID = 523
    #     self.FLOOR_CLASSES = 5
    #     self.EMBEDDING_SIZE = 6
    #
    #     self.normalize_x = None
    #     self.normalize_valid_x = None
    #     self.normalize_test_x = None
    #
    #     self.longitude_normalize_y = None
    #     self.latitude_normalize_y = None
    #     self.floorID_y = None
    #     self.buildingID_y = None
    #
    #     self.longitude_normalize_valid_y = None
    #     self.latitude_normalize_valid_y = None
    #     self.floorID_valid_y = None
    #     self.buildingID_valid_y = None
    #
    #     self.longitude_normalize_test_y = None
    #     self.latitude_normalize_test_y = None
    #     self.floorID_test_y = None
    #     self.buildingID_test_y = None
    #
    #     self.fc1 = nn.Linear(self.WAP_SIZE , 200)
    #     self.fc2 = nn.Linear(200, 200)
    #     self.fc3 = nn.Linear(200, self.FLOOR_CLASSES)
    #
    #
    # def forward(self, x):
    #     x = nn.ReLU()(self.fc1(x))
    #     x = nn.ReLU()(self.fc2(x))
    #     out = self.fc3(x)
    #     return out


    # def __init__(self, input_dim, hidden_dim, output_dim):
    #     super(FullyConnectedModel, self).__init__()
    #     self.fc1 = nn.Linear(input_dim, hidden_dim)
    #     self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    #     self.fc3 = nn.Linear(hidden_dim, output_dim)


    #2DCNN
    # def __init__(self):
    #     super(NN, self).__init__()
    #     self.WAP_SIZE = 520
    #     self.LONGITUDE = 520
    #     self.LATITUDE = 521
    #     self.FLOOR = 522
    #     self.BUILDINGID = 523
    #     self.FLOOR_CLASSES = 4
    #     self.EMBEDDING_SIZE = 6
    #
    #     self.normalize_x = None
    #     self.normalize_valid_x = None
    #     self.normalize_test_x = None
    #
    #     self.longitude_normalize_y = None
    #     self.latitude_normalize_y = None
    #     self.floorID_y = None
    #     self.buildingID_y = None
    #
    #     self.longitude_normalize_valid_y = None
    #     self.latitude_normalize_valid_y = None
    #     self.floorID_valid_y = None
    #     self.buildingID_valid_y = None
    #
    #     self.longitude_normalize_test_y = None
    #     self.latitude_normalize_test_y = None
    #     self.floorID_test_y = None
    #     self.buildingID_test_y = None
    #
    #     self.conv1 = nn.Conv2d(1, 16, kernel_size=(30, 4), padding=(1, 1), stride=(10, 1))
    #     self.bn1 = nn.BatchNorm2d(16)
    #     self.dropout1 = nn.Dropout(p=0.5)
    #     self.fc1 = nn.Linear(528, 120)
    #     self.bn2 = nn.BatchNorm1d(120)
    #     self.dropout2 = nn.Dropout(p=0.5)
    #     self.fc2 = nn.Linear(120, self.FLOOR_CLASSES)
    #
    #
    # def forward(self, x):
    #     x = x.view(-1, 1, 130, 4)
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = self.dropout1(x)
    #     x = torch.flatten(x, 1)
    #     x = F.relu(self.bn2(self.fc1(x)))
    #     x = self.dropout2(x)
    #     x = self.fc2(x)
    #     return x


    #  #Attention
    def __init__(self):
        super(NN, self).__init__()
        self.WAP_SIZE = 520
        self.LONGITUDE = 520
        self.LATITUDE = 521
        self.FLOOR = 522
        self.BUILDINGID = 523
        self.FLOOR_CLASSES = 4
        self.EMBEDDING_SIZE = 6

        self.normalize_x = None
        self.normalize_valid_x = None
        self.normalize_test_x = None

        self.longitude_normalize_y = None
        self.latitude_normalize_y = None
        self.floorID_y = None
        self.buildingID_y = None

        self.longitude_normalize_valid_y = None
        self.latitude_normalize_valid_y = None
        self.floorID_valid_y = None
        self.buildingID_valid_y = None

        self.longitude_normalize_test_y = None
        self.latitude_normalize_test_y = None
        self.floorID_test_y = None
        self.buildingID_test_y = None





        self.num_heads = 3
        self.head_dim = self.EMBEDDING_SIZE // self.num_heads
        assert self.EMBEDDING_SIZE % self.num_heads == 0, "Embedding size must be divisible by num_heads"

        self.embedding = nn.Embedding(self.WAP_SIZE, self.EMBEDDING_SIZE)
        self.Wq = nn.Parameter(torch.randn(self.EMBEDDING_SIZE, self.EMBEDDING_SIZE))
        self.Wk = nn.Parameter(torch.randn(self.EMBEDDING_SIZE, self.EMBEDDING_SIZE))
        self.Wv = nn.Parameter(torch.randn(self.EMBEDDING_SIZE, self.EMBEDDING_SIZE))
        self.Wo = nn.Parameter(torch.randn(self.num_heads * self.head_dim, self.EMBEDDING_SIZE))

        self.layer_norm1 = nn.LayerNorm(self.EMBEDDING_SIZE)  # Layer normalization after attention
        self.layer_norm2 = nn.LayerNorm(self.EMBEDDING_SIZE)
        self.fc1 = nn.Linear(self.EMBEDDING_SIZE * self.WAP_SIZE, self.EMBEDDING_SIZE)  # First fully connected layer
        self.fc2 = nn.Linear(self.EMBEDDING_SIZE, self.FLOOR_CLASSES)

    def forward(self, x):
        indices = torch.arange(0, self.WAP_SIZE, dtype=torch.long, device=x.device)
        embedded = self.embedding(indices)
        x = x.unsqueeze(-1)  # Shape: [batch_size, WAP_SIZE, 1]
        x = x * embedded
        x = x.expand(-1, -1, self.EMBEDDING_SIZE)

        Q = torch.matmul(x, self.Wq)
        K = torch.matmul(x, self.Wk)
        V = torch.matmul(x, self.Wv)

        Q = Q.view(-1, self.WAP_SIZE, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(-1, self.WAP_SIZE, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(-1, self.WAP_SIZE, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(-1, self.WAP_SIZE, self.num_heads * self.head_dim)
        output = torch.matmul(context, self.Wo)  # Final output projection

        output = self.layer_norm1(output)  # Apply layer normalization after attention output

        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))  # Apply ReLU after the first fully connected layer
# Apply layer normalization after the first FC layer
        output = self.layer_norm2(output)
        output = self.fc2(output)
        # output = F.softmax(output, dim=-1)  # Apply softmax across the last dimension

        return output


    def _preprocess(self, x, y, valid_x, valid_y,test_x,test_y):

        self.normalize_x = data_helper.normalizeX(x)
        self.normalize_valid_x = data_helper.normalizeX(valid_x)
        self.normalize_test_x = data_helper.normalizeX(test_x)

        data_helper.normY.fit(y[:, 0], y[:, 1])
        self.longitude_normalize_y, self.latitude_normalize_y = data_helper.normY.normalizeY(y[:, 0], y[:, 1])
        self.floorID_y = y[:, 2]
        self.buildingID_y = y[:, 3]

        data_helper.normY.fit(valid_y[:, 0], valid_y[:, 1])
        self.longitude_normalize_valid_y, self.latitude_normalize_valid_y = data_helper.normY.normalizeY(valid_y[:, 0],valid_y[:, 1])
        self.floorID_valid_y = valid_y[:, 2]

        self.buildingID_valid_y = valid_y[:, 3]

        data_helper.normY.fit(test_y[:, 0], test_y[:, 1])
        self.longitude_normalize_test_y, self.latitude_normalize_test_y = data_helper.normY.normalizeY(test_y[:, 0],
                                                                                                         test_y[:, 1])
        self.floorID_test_y = test_y[:, 2]
        self.buildingID_test_y = test_y[:, 3]

        # Assuming train_x and train_y are numpy arrays, you need to convert them to tensors
        train_x_tensor = torch.tensor(self.normalize_x, dtype=torch.float32)
        train_y_tensor = torch.tensor(self.floorID_y, dtype=torch.long)
        valid_x_tensor = torch.tensor(self.normalize_valid_x, dtype=torch.float32)
        valid_y_tensor = torch.tensor(self.floorID_valid_y, dtype=torch.long)
        test_x_tensor = torch.tensor(self.normalize_test_x, dtype=torch.float32)
        test_y_tensor = torch.tensor(self.floorID_test_y, dtype=torch.long)


        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        valid_dataset = TensorDataset(valid_x_tensor, valid_y_tensor)
        test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(valid_dataset, batch_size=64)
        self.test_loader = DataLoader(test_dataset,batch_size=64)
        unique_ids1, counts1 = torch.unique(train_y_tensor, return_counts=True)
        unique_ids2, counts2 = torch.unique(valid_y_tensor, return_counts=True)
        unique_ids3, counts3 = torch.unique(test_y_tensor, return_counts=True)
        self.FLOOR_CLASSES = max(counts1.numel(), counts2.numel(), counts3.numel())
        print("FLOOR_CLASSES", self.FLOOR_CLASSES)


from torch.nn import BCEWithLogitsLoss


def compute_accuracy(model):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients for validation data
        for inputs, labels in model.test_loader:
            inputs = inputs.float()
            labels = torch.nn.functional.one_hot(labels, num_classes=model.FLOOR_CLASSES).float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.max(1)[1]).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train_model(model,num_epochs, learning_rate):

    criterion = BCEWithLogitsLoss()
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        # Training phase

        train_loss, train_correct, total = 0, 0, 0
        for inputs, targets in model.train_loader:

            # inputs, targets = inputs.float(), torch.nn.functional.one_hot(targets, num_classes=model.FLOOR_CLASSES).float()
            targets = targets.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            #train_correct += (predicted == targets.max(1)[1]).sum().item()
            train_correct += (predicted == targets).sum().item()

        train_losses.append(train_loss / len(model.train_loader))
        train_accs.append(100 * train_correct / total)  # Update for multi-class classification

        # Validation phase
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in model.val_loader:
                inputs, targets = inputs.float(), torch.nn.functional.one_hot(targets, num_classes=model.FLOOR_CLASSES).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                k = targets.max(1)
                val_correct += (predicted == targets.max(1)[1]).sum().item()

        val_losses.append(val_loss / len(model.val_loader))
        val_accs.append(100 * val_correct / total)

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2f}%')

        # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()



    return model, train_losses, val_losses, train_accs, val_accs


if __name__ == '__main__':
    # Setup

    num_epochs = 150
    learning_rate = 0.0001

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
    (test_x, test_y) = filter_building(test_x, test_y, 1)


    nn_model._preprocess(train_x, train_y, valid_x, valid_y, test_x, test_y)
    # #




    # Train the model
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(nn_model, num_epochs, learning_rate)
    test_accuracy = compute_accuracy(trained_model)
    print(f'Test Accuracy: {test_accuracy:.2f}%')












