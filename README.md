![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange?logo=pytorch)
![Flower](https://img.shields.io/badge/Federated%20Learning-FLWR-green)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAR-yellow)
![Audit](https://img.shields.io/badge/Crypto%20Audit-SHA256-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)



# Federated Learning for Privacy-Preserving Model Training

*A decentralized training simulation with secure aggregation concepts and cryptographic audit logging.*

This project demonstrates how distributed clients can collaboratively train a machine learning model **without sharing raw data**, using the **Flower (FLWR) Federated Learning framework**. It uses the UCI HAR dataset split across five clients and introduces **SHA-256 audit logging** for tamper-evident update tracking. Accuracy improves consistently across all federated rounds. 

---

## 📌 Features

* Federated Learning with **FedAvg**
* Conceptual **secure aggregation** for privacy
* **SHA-256 audit logging** to verify model update integrity
* Uses **UCI HAR** dataset (561 features, 6 activity classes)
* Five-client decentralized training setup
* Accuracy improves from **78.11% → 92.76%** over 5 rounds 

---

## 📂 Dataset

The model is trained on the **UCI Human Activity Recognition (HAR)** dataset, which contains accelerometer and gyroscope signals from smartphones.
Each sample has **561 features** and one of six activity labels.
The training set is partitioned into **five subsets**, simulating federated IoT clients. 

---

## 🧠 Model Architecture

Each client trains a lightweight neural network implemented in PyTorch:

* Input: 561-dimensional feature vector
* Hidden layer: 100 neurons + ReLU
* Output: 6-class softmax

This architecture is defined in the `Net` model. 

---

## 🔄 Federated Learning Workflow

From the project methodology: 

1. Server sends global model to all clients
2. Clients train locally for one epoch
3. Clients send updated weights
4. Server aggregates using **FedAvg**
5. Repeat for 5 rounds

---

## 📊 Results

Results from the evaluation table: 

| Round | Accuracy (%) | Improvement |
| ----- | ------------ | ----------- |
| 1     | 78.11        | –           |
| 2     | 87.22        | +9.11       |
| 3     | 90.03        | +2.81       |
| 4     | 91.62        | +1.59       |
| 5     | 92.76        | +1.14       |

The results show steady accuracy improvement as global updates aggregate client contributions.

---

## 🔐 Audit Logging

The project includes SHA-256 hashing of model weights after each round, creating a **tamper-evident audit trail**.
This provides blockchain-style integrity without requiring an actual blockchain implementation.
(See `audit.py` for details.) 

---

## ▶️ How to Run

### 1. Prepare the dataset

```bash
python prepare_data.py
```

### 2. Start the federated server

```bash
python server.py
```

### 3. Launch each client (open 5 terminals)

```bash
python client.py 1
python client.py 2
python client.py 3
python client.py 4
python client.py 5
```

---

## Future Improvements

* Real blockchain integration for on-chain audit
* Implement secure aggregation (AES/SMPC)
* Support Non-IID data distributions
* More complex neural models

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

If you want a **shorter GitHub description**, a **banner**, or **badges**, I can generate those too.

