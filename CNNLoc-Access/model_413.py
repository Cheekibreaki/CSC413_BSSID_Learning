import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math

import data_helper_413

device = torch.device("cpu")

base_dir = os.getcwd()
test_csv_path = os.path.join(base_dir, 'UJIndoorLoc/validationData.csv')
valid_csv_path = os.path.join(base_dir, 'UJIndoorLoc/validationData.csv')
train_csv_path = os.path.join(base_dir, 'UJIndoorLoc/trainingData.csv')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.encoding[:x.size(0), :].detach()

class TransformerModel(nn.Module):
    def __init__(self, WAP_SIZE, num_classes, d_model=512, nhead=8, num_encoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
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
        
        self.num_classes = num_classes
        self.pos_encoder = PositionalEncoding(d_model).to(device)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers).to(device)
        self.embedding = nn.Linear(WAP_SIZE, d_model).to(device)
        self.output_layer = nn.Linear(d_model, num_classes).to(device)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.WAP_SIZE)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_layer(output.mean(dim=1))
        return output

def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

            val_losses.append(val_loss / len(val_loader))

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    return model, train_losses, val_losses

if __name__ == '__main__':
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 64

    model = TransformerModel(WAP_SIZE=520, num_classes=5).to(device)
    
    data_helper = data_helper_413.DataHelper()
    data_helper.set_config(wap_size=model.WAP_SIZE, long=model.LONGITUDE, lat=model.LATITUDE, floor=model.FLOOR,
                           building_id=model.BUILDINGID)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path, valid_csv_path,
                                                                                         test_csv_path)

    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y[:, 2], dtype=torch.long)

    valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32)
    valid_y_tensor = torch.tensor(valid_y[:, 2], dtype=torch.long)

    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    valid_dataset = TensorDataset(valid_x_tensor, valid_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size)

    trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, learning_rate)