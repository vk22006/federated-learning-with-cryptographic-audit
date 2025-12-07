# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kishore V

import torch.nn as nn

class HARNet(nn.Module):
    def __init__(self, input_dim=561, hidden_dim=128, num_classes=6):
        super(HARNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
