#main_server file:
import flwr as fl
import json
from FedAvgStrategy import FedAvgStrategy  # Correct import
from config import NUM_ROUNDS


loss_history = []

strategy = FedAvgStrategy()  # Use your custom strategy here

def main():
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # Save loss history to JSON after training
    with open("loss_history.json", "w") as f:
        json.dump(loss_history, f)

if __name__ == "__main__":
    main()
