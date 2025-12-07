# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kishore V

import numpy as np
import os
from sklearn.model_selection import train_test_split

# Path to UCI HAR Dataset (after unzipping)
DATASET_PATH = "dataset/UCI_HAR_Dataset/UCI_HAR_Dataset/"

def load_data():
    # Load train data
    X_train = np.loadtxt(os.path.join(DATASET_PATH, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(DATASET_PATH, "train", "y_train.txt")).astype(int) - 1

    # Load test data
    X_test = np.loadtxt(os.path.join(DATASET_PATH, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(DATASET_PATH, "test", "y_test.txt")).astype(int) - 1

    return X_train, y_train, X_test, y_test

def partition_data(X, y, num_clients=5):
    """Split training data into `num_clients` parts"""
    size = len(X) // num_clients
    client_data = []
    for i in range(num_clients):
        start, end = i * size, (i + 1) * size
        client_data.append((X[start:end], y[start:end]))
    return client_data

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    client_data = partition_data(X_train, y_train, num_clients=5)

    os.makedirs("data", exist_ok=True)

    # Save each client's local dataset
    for i, (Xc, yc) in enumerate(client_data, 1):
        np.save(f"data/X_train_client{i}.npy", Xc)
        np.save(f"data/y_train_client{i}.npy", yc)

    # Save test set (for global evaluation later)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)

    print("Data prepared and saved in 'data/' folder")
