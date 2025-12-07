# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kishore V

import hashlib
import torch
from model import HARNet

def save_model_hash(model, round_num, logfile="audit.log"):
    # Save model state dict hash
    torch.save(model.state_dict(), f"model_round_{round_num}.pt")
    with open(f"model_round_{round_num}.pt", "rb") as f:
        model_bytes = f.read()
        model_hash = hashlib.sha256(model_bytes).hexdigest()

    with open(logfile, "a") as f:
        f.write(f"Round {round_num}: {model_hash}\n")
    print(f"Audit log updated: Round {round_num}, Hash={model_hash}")
