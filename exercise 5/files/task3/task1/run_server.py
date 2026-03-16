# run_server.py
import json
import flwr as fl
from FedAvgStrategy import FedAvgStrategy
from flwr.server import ServerConfig
from config import NUM_ROUNDS  # ✅ Import from config.py

def make_serializable(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

def main():
    server_address = "localhost:8080"
    num_rounds = NUM_ROUNDS  # ✅ Use shared value

    strategy = FedAvgStrategy(initial_parameters=None)
    config = ServerConfig(num_rounds=num_rounds)

    print("Config type:", type(config))
    print("Config num_rounds:", config.num_rounds)

    history = fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )

    # Convert the history object to a dictionary
    # Convert the history object to a dictionary, including num_rounds
    history_dict = {
    "num_rounds": num_rounds,  # ✅ Add number of rounds
    "loss": history.losses_distributed,
    "metrics": history.metrics_distributed,
}

    with open("server_history.json", "w") as f:
        json.dump(history_dict, f, default=make_serializable, indent=4)

    print("✅ Server history saved to server_history.json")

if __name__ == "__main__":
    main()
