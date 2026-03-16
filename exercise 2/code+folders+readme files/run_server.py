# run_server.py

import flwr as fl
import json
import argparse
import torch
from scaffold_strategy import ScaffoldStrategy
from utils import get_central_test_loader, set_parameters, test_model
from model import CustomFashionModel

NUM_ROUNDS = 50

def make_serializable(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=str, required=True, help="Alpha identifier like alpha01, alpha1, alpha10")
    args = parser.parse_args()
    alpha = args.alpha

    print(f"🚀 Starting server for {alpha} on localhost:8080...")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testloader = get_central_test_loader()

    # Define centralized evaluation function
    def evaluate_fn(server_round, parameters, config):
        model = CustomFashionModel().to(DEVICE)
        set_parameters(model, parameters)
        loss, accuracy = test_model(model, testloader, DEVICE)
        return loss, {"accuracy": accuracy}

    # Start Flower server with evaluation
    strategy = ScaffoldStrategy(evaluate_fn=evaluate_fn)

    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy
    )

    # Save server history
    history_dict = {
        "num_rounds": NUM_ROUNDS,
        "loss": history.losses_distributed,
        "metrics": history.metrics_distributed,
    }

    with open(f"server_history_{alpha}.json", "w") as f:
        json.dump(history_dict, f, indent=4, default=make_serializable)

    print(f"✅ Server history saved to server_history_{alpha}.json")

if __name__ == "__main__":
    main()
