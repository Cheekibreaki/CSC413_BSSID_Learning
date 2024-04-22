import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import data_helper_413

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

base_dir = os.getcwd()
test_csv_path = os.path.join(base_dir, 'UJIndoorLoc/TestData.csv')
valid_csv_path = os.path.join(base_dir, 'UJIndoorLoc/ValuationData.csv')
train_csv_path = os.path.join(base_dir, 'UJIndoorLoc/TrainingData.csv')


def filter_building(x, y, building_id):
    filtered_x, filtered_y = [], []
    for i in range(y.shape[0]):
        if y[i, 3] == building_id:
            filtered_x.append(x[i, :])
            filtered_y.append(y[i])
    return np.array(filtered_x), np.array(filtered_y)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.WAP_SIZE = 520
        self.LONGITUDE = 520
        self.LATITUDE = 521
        self.FLOOR = 522
        self.BUILDINGID = 523
        self.Floor_classes = 4

        self.normalize_valid_x = None
        self.normalize_x = None
        self.normalize_y = None
        self.normalize_valid_y = None

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(30, 4), padding=(1, 1), stride=(10, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(528, 120)
        self.bn2 = nn.BatchNorm1d(120)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, self.Floor_classes)

    def forward(self, x):
        x = x.view(-1, 1, 130, 4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == targets).sum().item()
            total += targets.size(0)

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100 * train_correct / total)

        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                total += targets.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * val_correct / total)

        scheduler.step()

        print(
            f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2f}%')

    return model, train_losses, val_losses, train_accs, val_accs


def evaluate_test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_loss /= len(test_loader)
    test_acc = 100 * test_correct / total
    return test_loss, test_acc


def main():
    num_epochs = 200
    learning_rate = 0.001

    nn_model = NN()
    data_helper = data_helper_413.DataHelper()
    data_helper.set_config(wap_size=nn_model.WAP_SIZE, long=nn_model.LONGITUDE, lat=nn_model.LATITUDE,
                           floor=nn_model.FLOOR,
                           building_id=nn_model.BUILDINGID)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path, valid_csv_path,
                                                                                         test_csv_path)
    (train_x, train_y) = filter_building(train_x, train_y, 1)
    (valid_x, valid_y) = filter_building(valid_x, valid_y, 1)
    (test_x, test_y) = filter_building(test_x, test_y, 1)

    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y[:, 2], dtype=torch.long)
    valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32)
    valid_y_tensor = torch.tensor(valid_y[:, 2], dtype=torch.long)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y[:, 2], dtype=torch.long)

    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    valid_dataset = TensorDataset(valid_x_tensor, valid_y_tensor)
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    nn_model, train_losses, val_losses, train_accs, val_accs = train_model(nn_model, train_loader, val_loader,
                                                                           num_epochs, learning_rate)

    test_loss, test_acc = evaluate_test(nn_model, test_loader)

    min_val_loss = min(val_losses)
    max_val_acc = max(val_accs)

    print(f'Final Train Loss: {train_losses[-1]:.4f}')
    print(f'Final Train Acc: {train_accs[-1]:.2f}%')
    print(f'Final Val Loss: {val_losses[-1]:.4f}')
    print(f'Final Val Acc: {val_accs[-1]:.2f}%')
    print(f'Min Val Loss: {min_val_loss:.4f}')
    print(f'Max Val Acc: {max_val_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    main()