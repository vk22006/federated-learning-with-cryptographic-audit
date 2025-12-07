# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kishore V

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(561, 100)  # HAR dataset has 561 features
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 6)    # 6 activities
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Load local partition of dataset
def load_data(client_id):
    X = np.load(f"data/X_train_client{client_id}.npy")
    y = np.load(f"data/y_train_client{client_id}.npy")
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(Xt, yt)
    return DataLoader(dataset, batch_size=32, shuffle=True)

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.model = model
        self.trainloader = trainloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # one local epoch
            for data, target in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.trainloader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        return float(0.0), len(self.trainloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1])  # client id from command line
    trainloader = load_data(client_id)
    model = Net()
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(model, trainloader),
    )
