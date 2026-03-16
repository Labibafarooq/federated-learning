# run_server.py (with ScaffoldStrategy)
import json
import flwr as fl
from scaffold_strategy import ScaffoldStrategy
from flwr.server import ServerConfig
from config import NUM_ROUNDS
from flwr.common import parameters_to_ndarrays

def make_serializable(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

def main():
    server_address = "localhost:8080"
    num_rounds = NUM_ROUNDS

    # Initialize strategy and control variates
    strategy = ScaffoldStrategy(initial_parameters=None)

    config = ServerConfig(num_rounds=num_rounds)

    history = fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )

    history_dict = {
        "num_rounds": num_rounds,
        "loss": history.losses_distributed,
        "metrics": history.metrics_distributed,
    }

    with open("server_history.json", "w") as f:
        json.dump(history_dict, f, default=make_serializable, indent=4)

    print("✅ Server history saved to server_history.json")

if __name__ == "__main__":
    main()
