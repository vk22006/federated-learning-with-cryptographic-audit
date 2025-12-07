# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kishore V

import flwr as fl

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # FIX: Use ServerConfig instead of dict
    config = fl.server.ServerConfig(num_rounds=5)

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=config,
    )
